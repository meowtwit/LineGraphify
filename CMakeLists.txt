cmake_minimum_required(VERSION 3.16)
project(LineGraphifyCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(linegraphify_core src/linegraphify_core.cpp)
target_link_libraries(linegraphify_core PRIVATE ${OpenCV_LIBS})
