#
# src/
# Builds executables:
#    MapperViewer
#

include_directories(../lib_cg_sandbox ../lib_cg_sandbox/tools ../lib_cg_sandbox/core ../lib_cg_sandbox/lib)
add_executable(MapperViewer
    MapperViewer_main.cpp
)
target_compile_options(MapperViewer PRIVATE -Wall -g)

target_link_libraries(MapperViewer
    ${OpenCV_LIBS}
    QtCore QtGui QtOpenGL
    GL GLU glut GLEW dl
)
# target_link_libraries(MapperViewer Eigen3::Eigen)

target_link_libraries(MapperViewer cg_sandbox)
