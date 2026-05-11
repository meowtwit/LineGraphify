cmake_minimum_required(VERSION 3.16)
project(LineGraphifyCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(formula_generator src/formula_generator.cpp)

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_executable(linegraphify_core src/linegraphify_core.cpp)
    target_link_libraries(linegraphify_core PRIVATE ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV C++ package not found; building formula_generator only.")
endif()
