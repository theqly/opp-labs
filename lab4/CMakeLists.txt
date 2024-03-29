cmake_minimum_required(VERSION 3.5.22)
project(lab4)

set(SRC main.c)

add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include/x86_64-linux-gnu/mpich")
target_link_libraries(${PROJECT_NAME} m)