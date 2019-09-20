project(test_package CXX)
cmake_minimum_required(VERSION 2.8.11)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(${PROJECT_NAME} test_package.cpp)
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
