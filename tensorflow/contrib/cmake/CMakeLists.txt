# Minimum CMake required
cmake_minimum_required(VERSION 3.1)

# Project
project(tensorflow C CXX)

# Actual source is the ../../.. directory
get_filename_component(tf_contrib_source_dir ${tensorflow_SOURCE_DIR} PATH)
get_filename_component(tf_tf_source_dir ${tf_contrib_source_dir} PATH)
get_filename_component(tensorflow_source_dir ${tf_tf_source_dir} PATH)

# [CLEANUP] Not sure if this is needed (copied from Protobuf)
# CMake policies
cmake_policy(SET CMP0022 NEW)

# Options
option(tensorflow_VERBOSE "Enable for verbose output" OFF)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_ENABLE_GRPC_SUPPORT "Enable gRPC support" ON)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)
option(tensorflow_BUILD_ALL_KERNELS "Build all OpKernels" ON)
option(tensorflow_BUILD_CONTRIB_KERNELS "Build OpKernels from tensorflow/contrib/..." ON)


#Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option for
# targets that link ${CMAKE_THREAD_LIBS_INIT}.
find_package (Threads)

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

# Location where external projects will be downloaded
set (DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
     CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DEIGEN_AVOID_STL_ARRAY)
if(WIN32)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC -D__VERSION__=\"MSVC\")
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} /MP)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fno-exceptions -std=c++11")
endif()

# External dependencies
include(gif)
include(png)
include(jpeg)
include(eigen)
include(jsoncpp)
include(farmhash)
include(highwayhash)
include(protobuf)
find_package(ZLIB REQUIRED)

set(tensorflow_EXTERNAL_LIBRARIES
    ${gif_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${highwayhash_STATIC_LIBRARIES}
    ${protobuf_STATIC_LIBRARIES}
    ${ZLIB_LIBRARIES}
)
set(tensorflow_EXTERNAL_DEPENDENCIES
  gif_copy_headers_to_destination png_copy_headers_to_destination jpeg_copy_headers_to_destination jsoncpp farmhash_copy_headers_to_destination highwayhash_copy_headers_to_destination protobuf eigen)

include_directories(
    # Source and generated code.
    ${tensorflow_source_dir}
    ${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${gif_INCLUDE_DIR}
    ${png_INCLUDE_DIR}
    ${jpeg_INCLUDE_DIR}
    ${eigen_INCLUDE_DIRS}
    ${jsoncpp_INCLUDE_DIR}
    ${farmhash_INCLUDE_DIR}
    ${highwayhash_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
)

if(tensorflow_ENABLE_SSL_SUPPORT)
  include(boringssl)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${boringssl_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES boringssl)
  include_directories(${boringssl_INCLUDE_DIR})
endif()
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(grpc)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${grpc_STATIC_LIBRARIES})
  list(APPEND tensorflow_EXTERNAL_DEPENDENCIES grpc)
  include_directories(${GRPC_INCLUDE_DIRS})
endif()
if(WIN32)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES wsock32 ws2_32 shlwapi)
endif()
if(UNIX)
  list(APPEND tensorflow_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
endif()

# Let's get to work!
include(tf_core_framework.cmake)
include(tf_tools.cmake)
# NOTE: Disabled until issue #3996 is fixed.
# include(tf_stream_executor.cmake)
include(tf_core_cpu.cmake)
include(tf_models.cmake)
include(tf_core_ops.cmake)
include(tf_core_direct_session.cmake)
if(tensorflow_ENABLE_GRPC_SUPPORT)
  include(tf_core_distributed_runtime.cmake)
endif()
include(tf_core_kernels.cmake)
include(tf_cc_ops.cmake)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials.cmake)
endif()
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tf_python.cmake)
endif()
