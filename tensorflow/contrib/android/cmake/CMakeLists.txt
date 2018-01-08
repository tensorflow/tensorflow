#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.4.1)
include(ExternalProject)

# TENSORFLOW_ROOT_DIR:
#    root directory of tensorflow repo
#    used for shared source files and pre-built libs
get_filename_component(TENSORFLOW_ROOT_DIR ../../../.. ABSOLUTE)
set(PREBUILT_DIR ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/gen)

add_library(lib_proto STATIC IMPORTED )
set_target_properties(lib_proto PROPERTIES IMPORTED_LOCATION
    ${PREBUILT_DIR}/protobuf/lib/libprotobuf.a)

add_library(lib_nsync STATIC IMPORTED )
set_target_properties(lib_nsync PROPERTIES IMPORTED_LOCATION
    ${TARGET_NSYNC_LIB}/lib/libnsync.a)

add_library(lib_tf STATIC IMPORTED )
set_target_properties(lib_tf PROPERTIES IMPORTED_LOCATION
        ${PREBUILT_DIR}/lib/libtensorflow-core.a)
# Change to compile flags should be replicated into bazel build file
# TODO: Consider options other than -O2 for binary size.
#       e.g. -Os for gcc, and -Oz for clang.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIS_SLIM_BUILD \
                     -std=c++11 -fno-rtti -fno-exceptions \
                     -O2 -Wno-narrowing -fomit-frame-pointer \
                     -mfpu=neon -mfloat-abi=softfp -fPIE -fPIC \
                     -ftemplate-depth=900 \
                     -DGOOGLE_PROTOBUF_NO_RTTI \
                     -DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
                              -Wl,--allow-multiple-definition \
                              -Wl,--whole-archive \
                              -fPIE -pie -v")
file(GLOB tensorflow_inference_sources
     ${CMAKE_CURRENT_SOURCE_DIR}/../jni/*.cc)
file(GLOB java_api_native_sources
     ${TENSORFLOW_ROOT_DIR}/tensorflow/java/src/main/native/*.cc)

add_library(tensorflow_inference SHARED
            ${tensorflow_inference_sources}
            ${TENSORFLOW_ROOT_DIR}/tensorflow/c/tf_status_helper.cc
            ${TENSORFLOW_ROOT_DIR}/tensorflow/c/checkpoint_reader.cc
            ${TENSORFLOW_ROOT_DIR}/tensorflow/c/test_op.cc
            ${TENSORFLOW_ROOT_DIR}/tensorflow/c/c_api.cc
            ${java_api_native_sources})

# Include libraries needed for hello-jni lib
target_link_libraries(tensorflow_inference
                      android
                      dl
                      log
                      m
                      z
                      lib_tf
                      lib_proto
                      lib_nsync)

include_directories(
    ${PREBUILT_DIR}/proto
    ${PREBUILT_DIR}/protobuf/include
    ${PREBUILT_DIR}/nsync/public
    ${TENSORFLOW_ROOT_DIR}/tensorflow/contrib/makefile/downloads/eigen
    ${TENSORFLOW_ROOT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
