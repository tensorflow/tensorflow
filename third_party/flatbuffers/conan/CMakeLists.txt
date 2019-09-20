cmake_minimum_required(VERSION 2.8)

message(STATUS "Conan FlatBuffers Wrapper")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

if (WIN32 AND MSVC AND FLATBUFFERS_BUILD_SHAREDLIB)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32 AND MSVC AND FLATBUFFERS_BUILD_SHAREDLIB)

include(${CMAKE_SOURCE_DIR}/CMakeListsOriginal.txt)
