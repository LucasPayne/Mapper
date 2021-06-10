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
    GLuint depth_distance_tex;
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
vec3 world_to_frame(const FrameMetadata &md, vec3 p)
{
    vec3 q = md.rotation.transpose() * (p - md.world_position);
    float u = (intrinsics.fx*q.x()/q.z() + intrinsics.cx)/(intrinsics.pixels_x - 1);
    float v = (intrinsics.fy*q.y()/q.z() + intrinsics.cy)/(intrinsics.pixels_y - 1);
    return vec3(u, v, q.z());
}
vec3 reproject(const FrameMetadata &from_md, const FrameMetadata &to_md, vec3 uvz)
{
    vec3 p = frame_to_world(from_md, uvz);
    return world_to_frame(to_md, p);
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
    Image<float> computed_depth_map_scratch;
    GLuint computed_depth_map_texture;
    Image<vec4> reprojection_image;
    GLuint reprojection_texture;

    Image<float> difference_image;
    GLuint difference_texture;
    
    float timer; //depth map shader timer

    // Cost function parameters
    float lambda;
    float theta;

    float diff(vec4 a, vec4 b) {
        return max(fabs(a.x() - b.x()), max(fabs(a.y() - b.y()), fabs(a.z() - b.z())));
    }

    void iterate() {
        auto &depth = ground_truth ? frame_metadata[frame_A].depth_image : computed_depth_map;

        // Minimize reprojection error.
        for (int i = 0; i < 480; i++) {
            for (int j = 0; j < 640; j++) {

                float u = (j+0.5f)*1.f/(depth.width()-1);
                float v = (i+0.5f)*1.f/(depth.height()-1);
                float z0 = depth(i, j);
                vec4 color = frame_metadata[frame_A].rgb_image(i, j);

                float c1 = 0.5 / (2*theta);

                auto f = [&](float z) {
                    vec3 r = reproject(frame_metadata[frame_A], frame_metadata[frame_B], vec3(u, v, z));
                    int r_i = int(r.y() * (reprojection_image.height()-1));
                    int r_j = int(r.x() * (reprojection_image.width()-1));
                    if (r_i < 0 || r_j < 0 || r_i > reprojection_image.height()-1 || r_j > reprojection_image.width()-1) {
                        return INFINITY;
                    }
                    vec4 reproj_color = frame_metadata[frame_B].rgb_image(r_i, r_j);
                    float penalty = c1*(z - z0)*(z - z0);
                    return lambda * diff(reproj_color, color);
                    // return lambda * diff(reproj_color, color) + penalty;
                };
                //--brute force
                float min_f = INFINITY;
                float min_z = INFINITY;
                int N = 100;
                float d0 = 0.f;
                float d1 = 10.f;
                for (int K = 0; K <= N; K++) {
                    float z = d0 + (d1 - d0)*K*1.f/(N-1);
                    float fz = f(z);
                    if (fz < min_f) {
                        min_f = fz;
                        min_z = z;
                    }
                }
                computed_depth_map(i, j) = min_z != INFINITY ? min_z : 0.f;
            }
        }
    }

    float cost() {
        // Evaluate the cost function.
        int width = 640;
        int height = 480; //---
        float weight = 1.f / (width * height);
        float total = 0.f;
        int num_valid_pixels = 0;
        // Reprojected intensity difference integral.
        // Note: The integration domain is [0,1]^2, so pixel rectangularness is not considered in the integrals.
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                vec4 a = frame_metadata[frame_A].rgb_image(i, j);
                vec4 b = reprojection_image(i, j);
                if (b.w() > 1e-5) { // this component is 0 if the reprojection failed.
                    float d = diff(a, b);
                    total += lambda * weight * d;
                    //--------- store in difference image
                    difference_image(i, j) = d;
                    num_valid_pixels ++;
                } else {
                    difference_image(i, j) = 0;
                }
            }
        }
        // Rescale so that the integral is over the valid pixels only.
        if (num_valid_pixels != 0) total *= (width * height) * 1.f/num_valid_pixels;
        else total = 10000000; //---...
        
        // Total-variation regularizer integral.
        auto &depth = ground_truth ? frame_metadata[frame_A].depth_image : computed_depth_map;
        float dx = 1.f / width;
        float dy = 1.f / height;
        // boundary is not included in the integral
        for (int i = 1; i < height-1; i++) {
            for (int j = 1; j < width-1; j++) {
                float ddx = 0.5 * width * (depth(i, j+1) - depth(i, j-1));
                float ddy = 0.5 * height * (depth(i+1, j) - depth(i-1, j));
                float grad_norm = sqrt(ddx*ddx + ddy*ddy);
                total += weight * grad_norm;
            }
        }
        return total;
    }

    void update_computed_depth_map() {
        for (int i = 0; i < computed_depth_map.height(); i++) {
            for (int j = 0; j < computed_depth_map.width(); j++) {
                computed_depth_map_scratch(i, j) = computed_depth_map(i, j) * (5000.f/(2<<16));
            }
        }
        computed_depth_map_texture = computed_depth_map_scratch.texture();
        reprojection_texture = reprojection_image.texture();
        difference_texture = difference_image.texture();
    }

    Scene() {
        lambda = 1000.f;
        theta = 0.00025f;
        ground_truth = true;

        timer = 0.f;
        computed_depth_map_texture = 0;
        computed_depth_map = Image<float>(480, 640); //---hardcoded size
        computed_depth_map_scratch = Image<float>(480, 640); // processing buffer
        computed_depth_map.clear(2);
        reprojection_texture = 0;
        reprojection_image = Image<vec4>(480, 640);
        reprojection_image.clear(vec4(1,0,1,1));

        difference_texture = 0;
        difference_image = Image<float>(480, 640);
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
                frame_B = (frame_B + 1) % frame_metadata.size();
            }
            if (e.key.code == KEY_N) {
                frame_B -= 1;
                if (frame_B < 0) frame_B = frame_metadata.size()-1;
            }
            // if (e.key.code == KEY_V) {
            //     view_depth = !view_depth;
            // }
            if (e.key.code == KEY_O) {
                frame_A = frame_B;
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
            if (e.key.code == KEY_V) {
                iterate();
            }
            if (e.key.code == KEY_C) {
                computed_depth_map.clear(2);
            }
            if (e.key.code == KEY_T) {
                timer = 0.f;
            }
            if (e.key.code == KEY_Y) {
                // introduce artificial noise
                for (int i = 0; i < 480; i++) {
                    for (int j = 0; j < 640; j++) {
                        float x = 0.2;
                        frame_metadata[frame_A].depth_image(i, j) += 2*x*(frand()-0.5f);
                    }
                }
		// frame_metadata[frame_A].depth_distance_tex = frame_metadata[frame_A].depth_image.texture();
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

    void compute_reprojection(Image<float> &depth) {
        for (int i = 0; i < depth.height(); i++) {
            for (int j = 0; j < depth.width(); j++) {
                float u = (j+0.5f)*1.f/(depth.width()-1);
                float v = (i+0.5f)*1.f/(depth.height()-1);
                float Z = depth(i, j);
                vec3 uvz_b = reproject(frame_metadata[frame_A], frame_metadata[frame_B], vec3(u, v, Z));
                
                // vec4 reproj_color = frame_metadata[frame_B].rgb_image.bilinear(uvz_b.x(), uvz_b.y());
                int r_i = int(uvz_b.y() * (reprojection_image.height()-1));
                int r_j = int(uvz_b.x() * (reprojection_image.width()-1));
                vec4 reproj_color;
                if (r_i < 0 || r_j < 0 || r_i > reprojection_image.height()-1 || r_j > reprojection_image.width()-1) {
                    reproj_color = vec4(0,0,0,0);
                } else {
                    reproj_color = frame_metadata[frame_B].rgb_image(r_i, r_j);
                }
                reprojection_image(i, j) = reproj_color;
            }
        }
        // for (int i = 0; i < reprojection_image.height(); i++)
        //     for (int j = 0; j < reprojection_image.width(); j++)
        //         reprojection_image(i, j) = frame_metadata[frame_A].rgb_image(i, j);
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
        glUniform1i(program.uniform_location("N"), 1000); //tessellation
        glUniform1f(program.uniform_location("time"), timer);
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
        timer += dt;
        auto &paint = world->graphics.paint;

        std::vector<vec3> positions;
        for (int i = 0; i < frame_metadata.size(); i++) {
            auto &md = frame_metadata[i];
            positions.push_back(md.world_position);
            paint.sphere(md.world_position, 0.02, (i==frame_A || i==frame_B) ? vec4(1,1,1,1) : vec4(1,0,0,1));
        }
        paint.chain(positions, 1, vec4(0,0,0,1));

        #if 1
        int frustum_frame_indices[2] = {frame_A, frame_B};
        vec4 frustum_colors[2] = {vec4(0,0,0,1), vec4(1,1,1,1)};
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
            paint.line(md.world_position, md.world_position + q, 2, color);
            p = intrinsic_point(1,0,Z);
            q = md.rotation * p;
            paint.line(md.world_position, md.world_position + q, 2, color);
            p = intrinsic_point(1,1,Z);
            q = md.rotation * p;
            paint.line(md.world_position, md.world_position + q, 2, color);
            p = intrinsic_point(0,1,Z);
            q = md.rotation * p;
            paint.line(md.world_position, md.world_position + q, 2, color);
        }
        if (!ground_truth && draw_point_clouds) draw_point_cloud_frame(frame_A, 100, 100, &computed_depth_map);

        compute_reprojection(ground_truth ? frame_metadata[frame_A].depth_image : computed_depth_map);
        update_computed_depth_map();

        #if 0
        paint.sprite(frame_metadata[frame_A].rgb_tex, vec2(0,0.2), 0.2, -0.2);
        paint.sprite(frame_metadata[frame_B].rgb_tex, vec2(0.2,0.2), 0.2, -0.2);
        if (ground_truth) {
            paint.depth_sprite(frame_metadata[frame_A].depth_tex, vec2(0,0.4), 0.2, -0.2);
            paint.depth_sprite(frame_metadata[frame_B].depth_tex, vec2(0.2,0.4), 0.2, -0.2);
        } else {
            paint.depth_sprite(computed_depth_map_texture, vec2(0,0.4), 0.2, -0.2);
        }
        paint.sprite(reprojection_texture, vec2(0.8,0.2), 0.2, -0.2);
        paint.depth_sprite(difference_texture, vec2(0.6,0.2), 0.2, -0.2);
        #else 
        paint.sprite(reprojection_texture, vec2(0,0.4), 0.5, -0.4);
        paint.depth_sprite(difference_texture, vec2(0.5,0.4), 0.5, -0.4);
        paint.depth_sprite(ground_truth ? frame_metadata[frame_A].depth_tex : computed_depth_map_texture, vec2(0,0.8), 0.5, -0.4);
        #endif


        printf("Cost: %.6f\n", cost());

        #endif
    }
    void post_render_update() {
        if (draw_depth_maps) {
            if (ground_truth) {
                draw_depth_map(frame_A, 0, (2<<16)/5000.f);
                // draw_depth_map(frame_B, 0, (2<<16)/5000.f);
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
    int get_nth_image = 36;
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
            md.depth_distance_tex = md.depth_image.texture();
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
