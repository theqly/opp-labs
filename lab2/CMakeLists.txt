cmake_minimum_required(VERSION 3.22.1)
project(lab2 C)

set(SRC parallel.c)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include/x86_64-linux-gnu/mpich")
target_link_libraries(${PROJECT_NAME} m)



