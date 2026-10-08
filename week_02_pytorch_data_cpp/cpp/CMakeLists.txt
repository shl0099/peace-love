cmake_minimum_required(VERSION 3.16)

project(w02_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hello hello.cpp)

target_compile_options(
    hello
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(csv_validator csv_validator.cpp)

target_compile_options(
    csv_validator
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
