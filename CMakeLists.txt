cmake_minimum_required(VERSION 3.15)
project(Lyra CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build configuration ---------------------------------------------------
# Both Debug and Release are supported. Release matters a lot here: the operator
# kernels in src/ops.cpp are tight scalar loops, and -O3 is worth roughly another
# 20x on top of the algorithmic work in Stage 6.
#
#   Release (default):  cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
#                       cmake --build build-release
#   Debug:              cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
#                       cmake --build build
#
# In VS Code's CMake Tools, pick the variant with the "CMake: Select Variant"
# command (status bar shows the active one) -- it sets CMAKE_BUILD_TYPE for you.
#
# Single-config generators (Makefiles, Ninja) leave CMAKE_BUILD_TYPE empty when
# unset, which silently means "no optimization flags at all" -- worse than Debug,
# and the trap this project was already in. Default to Release instead. Multi-config
# generators (Xcode, Visual Studio) pick the config at build time, so skip them.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
             Debug Release RelWithDebInfo MinSizeRel)
message(STATUS "Lyra build type: ${CMAKE_BUILD_TYPE}")

find_package(Protobuf CONFIG REQUIRED)

add_executable(lyra
    src/main.cpp
    src/ops.cpp
    third_party/onnx.proto3.pb.cc
)

target_include_directories(lyra PRIVATE third_party)
target_link_libraries(lyra PRIVATE protobuf::libprotobuf)