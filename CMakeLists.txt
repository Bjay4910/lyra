cmake_minimum_required(VERSION 3.15)
project(Lyra CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)

add_executable(lyra
    src/main.cpp
    src/ops.cpp
    third_party/onnx.proto3.pb.cc
)

target_include_directories(lyra PRIVATE third_party)
target_link_libraries(lyra PRIVATE protobuf::libprotobuf)