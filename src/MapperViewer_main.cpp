#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include "glm/glm.hpp"
#include "lib_cg_sandbox/core/cg_sandbox.h"
#include "mesh_processing/mesh_processing.h"
#include <Eigen/Dense>

#include "util/cameraman.cpp"

// toggleable options
bool view_depth = false;

struct CameraIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
    int pixels_x;
    int pixels_y;
    double depth_scale;
    double short_depth_scale;
    mat3x3 intrinsic_matrix;
    mat3x3 intrinsic_matrix_inverse;
} intrinsics;


// Globals
Aspect<Camera> main_camera;
struct FrameMetadata {
    // 1341847980.790000 -0.6832 2.6909 1.7373 0.0003 0.8617 -0.5072 -0.0145 1341847980.786879 depth/1341847980.786879.png 1341847980.786856 rgb/1341847980.786856.png
    vec3 world_position;
    double t;
    std::string depth_file;
    std::string rgb_file;
    GLuint rgb_tex;
    GLuint depth_tex;
    Image<vec4> rgb_image;
    Image<float> depth_image;

    mat3x3 rotation; // extrinsic_matrix is [R|t]
    mat4x4 extrinsic_matrix;

    mat3x3 intrinsic_matrix;
    mat3x3 intrinsic_matrix_inverse;
};
std::vector<FrameMetadata> frame_metadata;


vec3 intrinsic_point(double u, double v, double z)
{
    return vec3(((intrinsics.pixels_x - 1) * u - intrinsics.cx) * z/intrinsics.fx,
	        ((intrinsics.pixels_y - 1) * v - intrinsics.cy) * z/intrinsics.fy, z);
}
vec3 frame_to_world(const FrameMetadata &md, vec3 uvz)
{
    // vec3 cameraspace_point = intrinsics.intrinsic_matrix_inverse * uvz;
    // return (md.extrinsic_matrix_inverse * vec4(cameraspace_point, 1)).xyz();

    //---I think fastfusion uses an incorrect notion of the extrinsic matrix...
    return md.world_position + md.rotation * intrinsic_point(uvz.x(), uvz.y(), uvz.z());
}




Image<vec4> cv_matrix_to_image_rgba(cv::Mat &cv_img)
{
    auto img = Image<vec4>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; i++) {
        for (int j = 0; j < cv_img.cols; j++) {
            auto rgb = cv_img.at<cv::Vec3b>(i, j);
            img(i,j) = vec4(rgb[0]/256.f, rgb[1]/256.f, rgb[2]/256.f, 1);
        }
    }
    return img;
}
Image<float> depth_to_distance_image(cv::Mat &cv_img)
{
    auto img = Image<float>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; i++) {
        for (int j = 0; j < cv_img.cols; j++) {
            auto color = cv_img.at<unsigned short>(i, j);
            float scaled = color / intrinsics.short_depth_scale;
            img(i,j) = scaled;
        }
    }
    return img;
}
Image<float> depth_to_image(cv::Mat &cv_img)
{
    auto img = Image<float>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; i++) {
        for (int j = 0; j < cv_img.cols; j++) {
            auto color = cv_img.at<unsigned short>(i, j);
            float scaled = color * 1.f/(2 << 16);
            img(i,j) = scaled;
        }
    }
    return img;
}




struct Scene : public IBehaviour {
    int frame_A;
    int frame_B;

    GLShaderProgram depth_map_shader;
    GLuint depth_map_dummy_vao;
    GLuint depth_map_dummy_vbo;


    bool ground_truth;
    bool draw_point_clouds;
    bool draw_depth_maps;

    Image<float> computed_depth_map;
    Image<float> computed_depth_map_scaled;
    GLuint computed_depth_map_texture;

    void update_computed_depth_map() {
        glDeleteTextures(1, &computed_depth_map_texture);
        for (int i = 0; i < computed_depth_map.height(); i++) {
            for (int j = 0; j < computed_depth_map.width(); j++) {
                computed_depth_map_scaled(i, j) = computed_depth_map(i, j) * (5000.f/(2<<16));
            }
        }
        computed_depth_map_texture = computed_depth_map_scaled.texture();
    }

    Scene() {
        ground_truth = true;

        computed_depth_map_texture = 0;
        computed_depth_map = Image<float>(640, 480); //---hardcoded size
        computed_depth_map_scaled = Image<float>(640, 480);
        computed_depth_map.clear(1.f);
        for (int i = 0; i < computed_depth_map.height(); i++) {
            for (int j = 0; j < computed_depth_map.width(); j++) {
                computed_depth_map(i, j) = 3.f + sin(i/200.f) + cos(j/250.f);
            }
        }
        update_computed_depth_map();

        frame_A = 0;
        frame_B = 0;
        draw_point_clouds = false;
        draw_depth_maps = true;

        depth_map_shader = GLShaderProgram();
        depth_map_shader.add_shader(GLShader(VertexShader, "mapper_shaders/depth_map.vert"));
        depth_map_shader.add_shader(GLShader(TessControlShader, "mapper_shaders/depth_map.tcs"));
        depth_map_shader.add_shader(GLShader(TessEvaluationShader, "mapper_shaders/depth_map.tes"));
        depth_map_shader.add_shader(GLShader(FragmentShader, "mapper_shaders/depth_map.frag"));
        depth_map_shader.link();

        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        vec3 dummy_data = vec3::zero();
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3), (const void *) &dummy_data, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (const void *) 0);
        glEnableVertexAttribArray(0);

        depth_map_dummy_vao = vao;
        depth_map_dummy_vbo = vbo;
    }
    void keyboard_handler(KeyboardEvent e) {
        if (e.action == KEYBOARD_PRESS) {
            if (e.key.code == KEY_M) {
                frame_A = (frame_A + 1) % frame_metadata.size();
            }
            if (e.key.code == KEY_N) {
                frame_A -= 1;
                if (frame_A < 0) frame_A = frame_metadata.size()-1;
            }
            if (e.key.code == KEY_V) {
                view_depth = !view_depth;
            }
            if (e.key.code == KEY_O) { // select frame B
                frame_B = frame_A;
            }
            if (e.key.code == KEY_P) {
                draw_point_clouds = !draw_point_clouds;
            }
            if (e.key.code == KEY_I) {
                draw_depth_maps = !draw_depth_maps;
            }
            if (e.key.code == KEY_G) {
                ground_truth = !ground_truth;
            }
        }
    }

    void draw_point_cloud_frame(int frame_index, int Nx, int Ny, Image<float> *depth_map=nullptr, float sphere_size=0.01) {
        // If depth_map is not null, then use an alternative depth map.
        auto &md = frame_metadata[frame_index];
	Image<float> *depth = depth_map == nullptr ? &md.depth_image : depth_map;
        for (int i = 0; i < Ny; i++) {
            for (int j = 0; j < Nx; j++) {
                float u = i*1.f/(Ny-1);
                float v = j*1.f/(Nx-1);
                int row = int(u*intrinsics.pixels_y);
                int col = int(v*intrinsics.pixels_x);
                float z = (*depth)(row, col);
                if (z < 1e-5) continue;
                vec3 p = frame_to_world(md, vec3(v,u,z));
                vec4 color = vec4(md.rgb_image(row, col).xyz(), 1);
                world->graphics.paint.sphere(p, sphere_size, color);
            }
        }
    }

    void draw_depth_map(int frame_index, int use_depth_texture=0, float depth_scale=1.f) {
        // If use_depth_texture is not zero, the depth texture is replaced.
        //
        auto &md = frame_metadata[frame_index];

        auto &program = depth_map_shader;
        program.bind();
        glUniform1i(program.uniform_location("rgb_image"), 0);
        glUniform1i(program.uniform_location("depth_image"), 1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, md.rgb_tex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, use_depth_texture==0 ? md.depth_tex : use_depth_texture);
        auto vp_matrix = main_camera->view_projection_matrix();
        glUniformMatrix4fv(program.uniform_location("vp_matrix"), 1, GL_FALSE, (const GLfloat *) &vp_matrix);

        // std::cout << program.uniform_location("center") << "\n"; getchar();
        glUniform1i(program.uniform_location("N"), 100); //tessellation
        glUniform3fv(program.uniform_location("center"), 1, (const GLfloat *) &md.world_position);
        glUniformMatrix3fv(program.uniform_location("rotation"), 1, GL_FALSE, (const GLfloat *) &md.rotation);
        glUniform1i(program.uniform_location("pixels_x"), intrinsics.pixels_x);
        glUniform1i(program.uniform_location("pixels_y"), intrinsics.pixels_y);
        glUniform1f(program.uniform_location("fx"), intrinsics.fx);
        glUniform1f(program.uniform_location("fy"), intrinsics.fy);
        glUniform1f(program.uniform_location("cx"), intrinsics.cx);
        glUniform1f(program.uniform_location("cy"), intrinsics.cy);
        glUniform1f(program.uniform_location("depth_scale"), depth_scale);

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glPatchParameteri(GL_PATCH_VERTICES, 1);
        glBindVertexArray(depth_map_dummy_vao);
        glDrawArrays(GL_PATCHES, 0, 1);

        glBindVertexArray(0);
        program.unbind();
    }

    void update() {
        
        auto &paint = world->graphics.paint;

        std::vector<vec3> positions;
        for (auto md : frame_metadata) {
            positions.push_back(md.world_position);
            paint.sphere(md.world_position, 0.02, vec4(1,0,0,1));
        }
        paint.chain(positions, 10, vec4(0,0,0,1));

        #if 1
        int frustum_frame_indices[2] = {frame_A, frame_B};
        vec4 frustum_colors[2] = {vec4(1,0,0,1), vec4(0,0,1,1)};
        for (int i = 0; i < 2; i++) {
            auto md = frame_metadata[frustum_frame_indices[i]];
            vec3 up = md.rotation * vec3(0,1,0);
            vec3 right = md.rotation * vec3(1,0,0);

            float Z = 0.33;
            vec3 bl = intrinsic_point(0,0, Z);
            vec3 tr = intrinsic_point(1,1, Z);
            float w = tr.x() - bl.x();
            float h = tr.y() - bl.y();

            vec3 world_bl = frame_to_world(md, vec3(0,0,Z));
            paint.image_3D(view_depth ? md.depth_tex : md.rgb_tex, world_bl, up, right, w, h, 1);

            if (draw_point_clouds && ground_truth) draw_point_cloud_frame(frustum_frame_indices[i], 100, 100);

            vec3 p, q;
            p = intrinsic_point(0,0,Z);
            q = md.rotation * p;
            vec4 color = frustum_colors[i];
            paint.line(md.world_position, md.world_position + q, 5, color);
            p = intrinsic_point(1,0,Z);
            q = md.rotation * p;
            paint.line(md.world_position, md.world_position + q, 5, color);
            p = intrinsic_point(1,1,Z);
            q = md.rotation * p;
            paint.line(md.world_position, md.world_position + q, 5, color);
            p = intrinsic_point(0,1,Z);
            q = md.rotation * p;
            paint.line(md.world_position, md.world_position + q, 5, color);
        }
        if (!ground_truth && draw_point_clouds) draw_point_cloud_frame(frame_A, 100, 100, &computed_depth_map);

        #endif
    }
    void post_render_update() {
        if (draw_depth_maps) {
            if (ground_truth) {
                draw_depth_map(frame_A, 0, (2<<16)/5000.f);
                draw_depth_map(frame_B, 0, (2<<16)/5000.f);
            } else {
                draw_depth_map(frame_A, computed_depth_map_texture,(2<<16)/5000.f);
            }
        }
    }
};



class App : public IGC::Callbacks {
public:
    App(World &_world);

    void close();
    void loop();
    void keyboard_handler(KeyboardEvent e);
    void mouse_handler(MouseEvent e);
    void window_handler(WindowEvent e);
    
    World &world;
    GLShaderProgram shader_image;
};


App::App(World &_world) : world{_world}
{
    Entity cameraman = create_cameraman(world);
    cameraman.get<Transform>()->position = vec3(0,0,2);
    main_camera = cameraman.get<Camera>();
    Entity e = world.entities.add();
    auto mesh = world.add<Scene>(e);

    // Set the camera intrinsics (TUM https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect)
    intrinsics.fx = 535.4;
    intrinsics.fy = 539.2;
    intrinsics.cx = 320.1;
    intrinsics.cy = 247.6;
    intrinsics.pixels_x = 640;
    intrinsics.pixels_y = 480;
    intrinsics.short_depth_scale = 5000;
    intrinsics.depth_scale = intrinsics.short_depth_scale * 1.f/(2 << 16);
    intrinsics.intrinsic_matrix = mat3x3(intrinsics.fx,0,0, 0,intrinsics.fy,0, intrinsics.cx,intrinsics.cy,1);
    auto tmp = mat4x4(vec3(0), intrinsics.intrinsic_matrix).inverse(); //---need mat3x3 inverse...
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intrinsics.intrinsic_matrix_inverse.entry(i, j) = tmp.entry(i, j);
        }
    }

    FILE *file = fopen("data/rgbd_tum/associate.txt", "r");
    assert(file != NULL);
    char line[1024];
    int skip_counter = 0;
    int get_nth_image = 50;
    while (fgets(line, 1024, file)) {
        bool skip = skip_counter != 0;
        skip_counter = (skip_counter + 1) % get_nth_image;
        if (skip) continue;
        float t;
        float x, y, z;
        float qw, qx, qy, qz;
        float depth_time;
        char depth_filename[1024];
        float rgb_time;
        char rgb_filename[1024];
        
// 1341847980.790000 -0.6832 2.6909 1.7373 0.0003 0.8617 -0.5072 -0.0145 1341847980.786879 depth/1341847980.786879.png 1341847980.786856 rgb/1341847980.786856.png
        size_t num_matched = sscanf(line, "%f %f %f %f %f %f %f %f %f %s %f %s\n",
            &t, &x, &y, &z, &qx, &qy, &qz, &qw, &depth_time, depth_filename, &rgb_time, &rgb_filename
        );
        assert(num_matched == 12);

        FrameMetadata md;
        auto rot = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
        mat3x3 R;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R.entry(i, j) = rot(i, j);
            }
        }
        md.rotation = R;
        vec3 T = vec3(x, y, z);
        mat4x4 ext = mat4x4(T, R);

        md.t = t;
        md.depth_file = std::string(depth_filename);
        md.rgb_file = std::string(rgb_filename);
        md.extrinsic_matrix = ext;
        md.world_position = T; //---fastfusion

        std::cout << "Loaded frame metadata" << "\n";
        std::cout << "    \"" << md.depth_file << "\"\n";
        std::cout << "    \"" << md.rgb_file << "\"\n";
        std::cout << "    " << md.t << "\n";
        // getchar();

        frame_metadata.push_back(md);
    }
    fclose(file);
    std::cout << "Loaded " << frame_metadata.size() << " frames.\n";
    // getchar();
    
    
    // Change to the coordinates system of the first camera.
    vec3 c1_pos = frame_metadata[0].world_position;
    mat3x3 c1_rotation = frame_metadata[0].rotation;
    for (auto &md : frame_metadata) {
        md.world_position = c1_rotation.transpose() * (md.world_position - c1_pos);
        md.rotation = c1_rotation.transpose() * md.rotation;
        md.extrinsic_matrix = mat4x4(md.world_position, md.rotation);
    }
    // fastfusion and the TUM dataset uses an inverted coordinate system, so flip it here.
    // NOTE: OpenGL rgb & depth images are actually upside down.
    for (auto &md : frame_metadata) {
        auto flipmatrix = mat3x3(-1,0,0,  0,-1,0,  0,0,1);
        md.world_position = flipmatrix * md.world_position;
        md.rotation = flipmatrix * md.rotation;
        md.extrinsic_matrix = mat4x4(md.world_position, md.rotation);
    }
        
    // Upload all rgb images to the GPU.
    for (auto &md : frame_metadata) {
        // std::cout << name << "\n"; getchar();
        // cv::Mat cv_img = cv::imread("data/rgbd_tum/rgb/1341847980.722988.png", -1);
        {
            std::string name = "data/rgbd_tum/" + std::string(md.rgb_file);
            cv::Mat cv_img = cv::imread(name, -1);
            assert(!cv_img.empty());
            auto img = cv_matrix_to_image_rgba(cv_img);
            md.rgb_tex = img.texture();
            md.rgb_image = img;
        }
        {
            std::string name = "data/rgbd_tum/" + std::string(md.depth_file);
            cv::Mat cv_img = cv::imread(name, -1);
            assert(!cv_img.empty());
            auto img = depth_to_distance_image(cv_img);
            md.depth_image = img;
            // md.depth_tex = depth_to_image(cv_img).texture(); // normalized to 0...1
            md.depth_tex = depth_to_image(cv_img).texture(); // normalized to 0...1
        }
    }
}

void App::close()
{
}
void App::loop()
{
}
void App::keyboard_handler(KeyboardEvent e)
{
    if (e.action == KEYBOARD_PRESS) {
        if (e.key.code == KEY_Q) {
            exit(EXIT_SUCCESS);
        }
    }
}
void App::mouse_handler(MouseEvent e)
{
}
void App::window_handler(WindowEvent e)
{
}


int main(void)
{
    srand(time(0));

    printf("[main] Creating context...\n");
    auto context = IGC::Context("Context");
    printf("[main] Creating world...\n");
    auto world = World(context);
    printf("[main] Creating app...\n");
    auto app = App(world);
    printf("[main] Adding app callbacks...\n");
    context.add_callbacks(&app);

    printf("[main] Entering loop...\n");
    context.enter_loop();
    context.close();
}
