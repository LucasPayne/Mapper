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
    Image<vec4> depth_image;

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


// Camera
struct CameraController : public IBehaviour {
    float azimuth;
    float angle;

    float strafe_speed;
    float forward_speed;
    float lift_speed;

    bool view_with_mouse;
    float key_view_speed_horizontal;
    float key_view_speed_vertical;

    float min_angle;
    float max_angle;

    #define BASE_MOUSE_SENSITIVITY 1.22
    // mouse_sensitivity multiplies the base sensitivity.
    float mouse_sensitivity;

    CameraController() {}

    inline void lock_angle() {
        if (angle < min_angle) angle = min_angle;
        if (angle > max_angle) angle = max_angle;
    }

    void keyboard_handler(KeyboardEvent e) {
        if (e.action == KEYBOARD_PRESS) {
            if (e.key.code == KEY_E) {
                view_with_mouse = !view_with_mouse;
            }
        }
    }
    void mouse_handler(MouseEvent e) {
        if (view_with_mouse) {
            if (e.action == MOUSE_MOVE) {
                azimuth -= BASE_MOUSE_SENSITIVITY * mouse_sensitivity * e.cursor.dx;
                angle += BASE_MOUSE_SENSITIVITY * mouse_sensitivity * e.cursor.dy;
                lock_angle();
            }
        }
        if (e.action == MOUSE_SCROLL) {
            strafe_speed *= 1.f + (dt * e.scroll_y);
            forward_speed *= 1.f + (dt * e.scroll_y);
            lift_speed *= 1.f + (dt * e.scroll_y);
        }
    }
    void update() {
        auto t = entity.get<Transform>();
        const KeyboardKeyCode up = KEY_W;
        const KeyboardKeyCode down = KEY_S;
        const KeyboardKeyCode left = KEY_A;
        const KeyboardKeyCode right = KEY_D;
        const KeyboardKeyCode view_up = KEY_K;
        const KeyboardKeyCode view_down = KEY_J;
        const KeyboardKeyCode view_left = KEY_H;
        const KeyboardKeyCode view_right = KEY_L;

        float forward_movement = 0;
        float side_movement = 0;
        float lift = 0;
        if (world->input.keyboard.down(up)) forward_movement += forward_speed;
        if (world->input.keyboard.down(down)) forward_movement -= forward_speed;
        if (world->input.keyboard.down(left)) side_movement -= strafe_speed;
        if (world->input.keyboard.down(right)) side_movement += strafe_speed;

        if (!view_with_mouse) {
            if (world->input.keyboard.down(view_left)) azimuth += key_view_speed_horizontal * dt;
            if (world->input.keyboard.down(view_right)) azimuth -= key_view_speed_horizontal * dt;
            if (world->input.keyboard.down(view_down)) angle -= key_view_speed_vertical * dt;
            if (world->input.keyboard.down(view_up)) angle += key_view_speed_vertical * dt;
        }

        if (world->input.keyboard.down(KEY_SPACE)) lift += lift_speed;
        if (world->input.keyboard.down(KEY_LEFT_SHIFT)) lift -= lift_speed;

        lock_angle();
        float cos_azimuth = cos(azimuth);
        float sin_azimuth = sin(azimuth);
        vec3 forward = vec3(-sin_azimuth, 0, -cos_azimuth);
        vec3 side = vec3(cos_azimuth, 0, -sin_azimuth);

        t->position += dt*(side_movement*side + forward_movement*forward);
        t->position.y() += dt*lift;

        Quaternion q1 = Quaternion::from_axis_angle(vec3(0,1,0), azimuth);
        Quaternion q2 = Quaternion::from_axis_angle(side, angle);
        t->rotation = q2 * q1;
    }
    void init() {
        float speed = 4;
        strafe_speed = speed;
        forward_speed = speed;
        lift_speed = speed;
        key_view_speed_horizontal = 2;
        key_view_speed_vertical = 1.5;
        azimuth = 0;
        angle = 0;
        min_angle = -M_PI/2.0 + 0.15;
        max_angle = M_PI/2.0 - 0.15;
        view_with_mouse = true;
        mouse_sensitivity = 2;
    }
};
Entity create_cameraman(World &world)
{
    Entity cameraman = world.entities.add();
    auto camera = cameraman.add<Camera>(0.1, 300, 0.1, 0.566);
    auto t = cameraman.add<Transform>();
    CameraController *controller = world.add<CameraController>(cameraman);
    controller->init();
    return cameraman;
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
Image<vec4> depth_to_distance_image(cv::Mat &cv_img)
{
    auto img = Image<vec4>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; i++) {
        for (int j = 0; j < cv_img.cols; j++) {
            auto color = cv_img.at<unsigned short>(i, j);
            float scaled = color / intrinsics.depth_scale;
            img(i,j) = vec4(scaled, scaled, scaled, 1);
        }
    }
    return img;
}
Image<vec4> depth_to_image(cv::Mat &cv_img)
{
    auto img = Image<vec4>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; i++) {
        for (int j = 0; j < cv_img.cols; j++) {
            auto color = cv_img.at<unsigned short>(i, j);
            float scaled = color * 1.f/(2 << 16);
            img(cv_img.rows-1-i,j) = vec4(scaled, scaled, scaled, 1);
        }
    }
    return img;
}




struct Mesh : public IBehaviour {
    int frame_number;

    Mesh() {
        frame_number = 0;
    }
    void keyboard_handler(KeyboardEvent e) {
        if (e.action == KEYBOARD_PRESS) {
            if (e.key.code == KEY_M) {
                frame_number = (frame_number + 1) % frame_metadata.size();
            }
            if (e.key.code == KEY_N) {
                frame_number -= 1;
                if (frame_number < 0) frame_number = frame_metadata.size()-1;
            }
            if (e.key.code == KEY_V) {
                view_depth = !view_depth;
            }
        }
    }

    void update() {
        
        auto &paint = world->graphics.paint;

        std::vector<vec3> positions;
        for (auto md : frame_metadata) {
            positions.push_back(md.world_position);
            paint.sphere(md.world_position, 0.02, vec4(1,0,0,1));
        }
        paint.chain(positions, 10, vec4(0,0,0,1));

        int Nx = 60;
        int Ny = 60;
        float sphere_size = 0.01;
        for (int K = 0; K <= frame_number; K++) {
            auto &md = frame_metadata[K];
            for (int i = 0; i < Ny; i++) {
                for (int j = 0; j < Nx; j++) {
                    float u = i*1.f/(Ny-1);
                    float v = j*1.f/(Nx-1);
                    int row = int(u*intrinsics.pixels_y);
                    int col = int(v*intrinsics.pixels_x);
                    float z = md.depth_image(row, col).x();
                    if (z < 1e-5) continue;
                    vec3 p = frame_to_world(md, vec3(v,u,z));
                    // float dist = (frame_metadata[frame_number].rotation.transpose() * (p - frame_metadata[frame_number].position)).z();
                    // float gray = exp(-dist*0.5);
                    vec4 color = vec4(md.rgb_image(row, col).xyz(), 1);
                    paint.sphere(p, sphere_size, color);
                }
            }
        }

        {
            auto md = frame_metadata[frame_number];
            vec3 up = md.rotation * vec3(0,1,0);
            vec3 right = md.rotation * vec3(1,0,0);

            float Z = 1;
            vec3 bl = intrinsic_point(0,0, Z);
            vec3 tr = intrinsic_point(1,1, Z);
            float w = tr.x() - bl.x();
            float h = tr.y() - bl.y();

            vec3 world_bl = frame_to_world(md, vec3(0,0,Z));
            paint.image_3D(view_depth ? md.depth_tex : md.rgb_tex, world_bl, up, right, w, h);
    // // DEBUG comparison with fastfusion
    // for (auto md : frame_metadata) {
        // std::cout << md.extrinsic_matrix << "\n";
        // std::cout << md.rotation << "\n";
        // std::cout << md.world_position << "\n";
        // getchar();
        std::cout << md.world_position << "\n";
        std::cout << intrinsic_point(0,0,1) << "\n";
        std::cout << md.rotation * intrinsic_point(0,0,1) << "\n";
    
        
      //  getchar();
    // }
        vec3 p, q;
        p = intrinsic_point(0,0,1);
        q = md.rotation * p;
        paint.line(md.world_position, md.world_position + q, 5, vec4(0,0,1,1));
        p = intrinsic_point(1,0,1);
        q = md.rotation * p;
        paint.line(md.world_position, md.world_position + q, 5, vec4(0,0,1,1));
        p = intrinsic_point(1,1,1);
        q = md.rotation * p;
        paint.line(md.world_position, md.world_position + q, 5, vec4(0,0,1,1));
        p = intrinsic_point(0,1,1);
        q = md.rotation * p;
        paint.line(md.world_position, md.world_position + q, 5, vec4(0,0,1,1));

            // paint.line(md.world_position, world_bl, 5, vec4(0,0,1,1));
            // paint.line(md.world_position, world_bl + w*right, 5, vec4(0,0,1,1));
            // paint.line(md.world_position, world_bl + h*up, 5, vec4(0,0,1,1));
            // paint.line(md.world_position, world_bl + w*right + h*up, 5, vec4(0,0,1,1));
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
    auto mesh = world.add<Mesh>(e);

    // Set the camera intrinsics (TUM https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect)
    intrinsics.fx = 535.4;
    intrinsics.fy = 539.2;
    intrinsics.cx = 320.1;
    intrinsics.cy = 247.6;
    intrinsics.pixels_x = 640;
    intrinsics.pixels_y = 480;
    intrinsics.depth_scale = 5000;
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
            md.depth_tex = depth_to_image(cv_img).texture();
            md.depth_image = img;
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
