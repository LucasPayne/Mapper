#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "glm/glm.hpp"
#include "lib_cg_sandbox/core/cg_sandbox.h"
#include "mesh_processing/mesh_processing.h"


// Globals
Aspect<Camera> main_camera;
struct FrameMetadata {
    // 1341847980.790000 -0.6832 2.6909 1.7373 0.0003 0.8617 -0.5072 -0.0145 1341847980.786879 depth/1341847980.786879.png 1341847980.786856 rgb/1341847980.786856.png
    vec3 position;
    vec3 rotation;
    double t;
    std::string depth_file;
    std::string rgb_file;
};
std::vector<FrameMetadata> frame_metadata;


// Mesh loading.
Entity create_mesh_object(World &world,
                          MLModel &model,
                          const std::string &mat_path)
{
    if (!model.has_normals) model.compute_phong_normals();

    Entity e = world.entities.add();
    auto t = e.add<Transform>(0,0,0);
    auto box = BoundingBox(model.positions);
    auto sphere = box.bounding_sphere();
    
    Resource<GeometricMaterial> gmat = world.graphics.shading.geometric_materials.load("shaders/triangle_mesh/triangle_mesh.gmat");
    Resource<Material> mat = world.graphics.shading.materials.load(mat_path);

    VertexArrayData vad;
    MLModel_to_VertexArrayData(model, vad);

    auto va = world.resources.add<VertexArray>();
    *va = VertexArray::from_vertex_array_data(vad);

    // Resource<VertexArray> model_vertex_array = world.assets.models.load(model_path);
    auto gmat_instance = GeometricMaterialInstance(gmat, va);
    auto mat_instance = MaterialInstance(mat);
    auto drawable = e.add<Drawable>(gmat_instance, mat_instance);
    drawable->raw_bounding_sphere = sphere;
    return e;
}
Entity create_mesh_object(World &world,
                          const std::string &model_path,
                          const std::string &mat_path)
{
    auto model = MLModel::load(model_path);
    return create_mesh_object(world, model, mat_path);
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






struct Mesh : public IBehaviour {
    SurfaceGeometry *geom;
    Image<vec4> img;
    GLuint img_tex;

    Mesh(SurfaceGeometry *_geom) : geom{_geom}
    {
        cv::Mat cv_img = cv::imread("images/test.png", -1);
        assert(!img.empty());

        img = Image<vec4>(cv_img.rows, cv_img.cols);
        for (int i = 0; i < cv_img.rows; i++) {
            for (int j = 0; j < cv_img.cols; j++) {
                auto rgb = cv_img.at<cv::Vec3b>(i, j);
                img(cv_img.rows-1-i,j) = vec4(rgb[0]/256.f, rgb[1]/256.f, rgb[2]/256.f, 1);
            }
        }
        // img(2,2) = vec4(1,0,0,1);
        // img(4,4) = vec4(0,1,0,1);
        img_tex = img.texture();
    }
    void update() {
        auto &paint = world->graphics.paint;
        paint.wireframe(*geom, mat4x4::identity(), 0.001);
        paint.image_3D(img_tex, vec3::zero(), vec3(0,1,0), vec3(sin(total_time),0,cos(total_time)), 3, 3);
        paint.image_3D(img_tex, vec3(0, 5, 0), vec3(0,1,0), vec3(sin(total_time),0,cos(total_time)), 3, 3);
        paint.image_3D(img_tex, vec3(5, 0, 0), vec3(0,1,0), vec3(sin(total_time),0,cos(total_time)), 3, 3);


        std::vector<vec3> positions;
        for (auto md : frame_metadata) {
            positions.push_back(md.position);
        }
        paint.chain(positions, 10, vec4(0,0,0,1));
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
    // shader_image = GLShaderProgram();
    // shader_image->add_shader(GLShader(VertexShader, "shaders/shader_image.vert"));
    // shader_image->add_shader(GLShader(FragmentShader, "shaders/shader_image.frag"));
    // shader_image->link();

    // auto v = vec3(1,1,1);
    //
    Entity cameraman = create_cameraman(world);
    cameraman.get<Transform>()->position = vec3(0,0,2);
    main_camera = cameraman.get<Camera>();

    // Entity obj = create_mesh_object(world, "models/dragon.off", "shaders/uniform_color.mat");
    // obj.get<Transform>()->position = vec3(0,0,0);
    // obj.get<Drawable>()->material.properties.set_vec4("albedo", 0.8,0.8,0.8,1);

    auto model = MLModel::load("models/20mm_cube.stl");
    auto model_geom = new SurfaceGeometry();
    model_geom->add_model(model);
    for (auto v : model_geom->vertices()) {
        model_geom->vertex_positions[v] *= 0.02;
    }
    Entity e = world.entities.add();
    auto mesh = world.add<Mesh>(e, model_geom);


    FILE *file = fopen("data/rgbd_tum/associate.txt", "r");
    assert(file != NULL);
    char line[1024];
    while (fgets(line, 1024, file)) {
        float t;
        float x1, y1, z1;
        float w;
        float x2, y2, z2;
        float depth_time;
        char depth_filename[1024];
        float rgb_time;
        char rgb_filename[1024];
        
// 1341847980.790000 -0.6832 2.6909 1.7373 0.0003 0.8617 -0.5072 -0.0145 1341847980.786879 depth/1341847980.786879.png 1341847980.786856 rgb/1341847980.786856.png
        size_t num_matched = sscanf(line, "%f %f %f %f %f %f %f %f %f %s %f %s",
            &t, &x1, &y1, &y2, &w, &x2, &y2, &z2, &depth_time, depth_filename, &rgb_time, &rgb_filename
        );
        assert(num_matched == 12);

        FrameMetadata md;
        md.position = vec3(x1, y1, z1);
        md.rotation = vec3(x2, y2, z2);
        md.t = t;
        md.depth_file = std::string(depth_filename);
        md.rgb_file = std::string(rgb_filename);
        frame_metadata.push_back(md);

        std::cout << "Loaded frame metadata" << "\n";
        std::cout << "    " << md.position << ", " << md.rotation << "\n";
        std::cout << "    " << md.depth_file << "\n";
        std::cout << "    " << md.rgb_file << "\n";
        // getchar();
    }
    fclose(file);


    // cv::Mat img = cv::imread("images/test.png", -1);
    // assert(!img.empty());
    // cv::namedWindow("ExampleWindow", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Example1", img);
    // cv::waitKey(0);
    // cv::destroyWindow("Example1");
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
