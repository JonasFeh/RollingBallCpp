cmake_minimum_required(VERSION 3.12)
project(OpenCVProjects)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file (GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file (GLOB_RECURSE MY_HEADERS CONFIGURE_DEPENDS "include/*.h")

set (OpenCVProjects_INCLUDE_DIRS "")
foreach (_headerFile ${MY_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND OpenCVProjects_INCLUDE_DIRS ${_dir})
endforeach()
list (REMOVE_DUPLICATES OpenCVProjects_INCLUDE_DIRS)
message(STATUS ${OpenCVProjects_INCLUDE_DIRS})
message(STATUS ${MY_SOURCES})

add_executable(OpenCVProjectsExe ${MY_SOURCES})
target_include_directories(OpenCVProjectsExe PRIVATE ${OpenCVProjects_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

message(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
message(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(OpenCVProjectsExe ${OpenCV_LIBS})
