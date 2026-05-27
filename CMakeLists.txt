cmake_minimum_required(VERSION 3.20)
project(prism LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vendor/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(prism
    src/main.cpp
    src/audio.cpp
    src/ml_thread.cpp
    src/model.cpp
    src/lpc.cpp
    src/pth_reader.cpp
    vendor/yin.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(prism PRIVATE
    include
    vendor
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLFW_INCLUDE_DIRS}
    /usr/include/onnxruntime
)

target_link_libraries(prism PRIVATE
    OpenGL::GL
    ${GLFW_LIBRARIES}
    onnxruntime
    dl
    pthread
)

target_compile_options(prism PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
