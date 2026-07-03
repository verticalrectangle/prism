cmake_minimum_required(VERSION 3.20)
project(prism LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
# Native PipeWire duplex for low-latency monitoring; miniaudio is the fallback.
pkg_check_modules(PIPEWIRE libpipewire-0.3)

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
    src/audio_pw.cpp
    src/ml_thread.cpp
    src/dsp.cpp
    src/rmvpe_onnx.cpp
    src/hubert_encoder.cpp
    src/rvc_onnx.cpp
    src/pth_reader.cpp
    src/hf_api.cpp
    vendor/yin.cpp
    ${IMGUI_SOURCES}
)

add_executable(prism-test
    tools/test_dsp.cpp
    src/dsp.cpp
)
target_include_directories(prism-test PRIVATE include)
target_compile_options(prism-test PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

target_include_directories(prism PRIVATE
    include
    vendor
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLFW_INCLUDE_DIRS}
    ${FFTW3F_INCLUDE_DIRS}
    /usr/include/onnxruntime
)

target_link_libraries(prism PRIVATE
    OpenGL::GL
    ${GLFW_LIBRARIES}
    ${FFTW3F_LIBRARIES}
    onnxruntime
    dl
    pthread
)

add_executable(prism-test-pipeline
    tools/test_pipeline.cpp
    src/ml_thread.cpp
    src/dsp.cpp
    src/rmvpe_onnx.cpp
    src/hubert_encoder.cpp
    vendor/yin.cpp
)
target_include_directories(prism-test-pipeline PRIVATE
    include
    vendor
    ${FFTW3F_INCLUDE_DIRS}
    /usr/include/onnxruntime
)
target_link_libraries(prism-test-pipeline PRIVATE
    ${FFTW3F_LIBRARIES}
    onnxruntime
    pthread
)
target_compile_options(prism-test-pipeline PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

add_executable(prism-test-rmvpe
    tools/test_rmvpe.cpp
    src/rmvpe_onnx.cpp
)
target_include_directories(prism-test-rmvpe PRIVATE
    include
    ${FFTW3F_INCLUDE_DIRS}
    /usr/include/onnxruntime
)
target_link_libraries(prism-test-rmvpe PRIVATE
    ${FFTW3F_LIBRARIES}
    onnxruntime
    pthread
)
target_compile_options(prism-test-rmvpe PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

target_compile_options(prism PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

if(PIPEWIRE_FOUND)
    target_compile_definitions(prism PRIVATE HAVE_PIPEWIRE)
    target_include_directories(prism PRIVATE ${PIPEWIRE_INCLUDE_DIRS})
    target_link_libraries(prism PRIVATE ${PIPEWIRE_LIBRARIES})
endif()

