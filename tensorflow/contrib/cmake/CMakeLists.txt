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
option(tensorflow_BUILD_TESTS "Build tests" ON)
option(tensorflow_ENABLE_SSL_SUPPORT "Enable boringssl support" OFF)
option(tensorflow_BUILD_CC_EXAMPLE "Build the C++ tutorial example" ON)
option(tensorflow_BUILD_PYTHON_BINDINGS "Build the Python bindings" ON)

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

# External dependencies
include(gif)
include(png)
include(jpeg)
include(eigen)
include(jsoncpp)
if(tensorflow_ENABLE_SSL_SUPPORT)
  include(boringssl)
endif()
include(farmhash)
include(highwayhash)
include(protobuf)
include(grpc)

# Let's get to work!
include(tf_core_framework.cmake)
# NOTE: Disabled until issue #3996 is fixed.
# include(tf_stream_executor.cmake)
include(tf_core_cpu.cmake)
include(tf_models.cmake)
include(tf_core_ops.cmake)
include(tf_core_direct_session.cmake)
include(tf_core_distributed_runtime.cmake)
include(tf_core_kernels.cmake)
include(tf_cc_ops.cmake)
include(tf_tools.cmake)
if(tensorflow_BUILD_CC_EXAMPLE)
  include(tf_tutorials.cmake)
endif()
if(tensorflow_BUILD_PYTHON_BINDINGS)
  include(tf_python.cmake)
endif()

if (tensorflow_BUILD_TESTS)
  include(tests.cmake)
endif (tensorflow_BUILD_TESTS)

include(install.cmake)
