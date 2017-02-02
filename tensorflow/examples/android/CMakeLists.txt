#
# Copyright (C)  The Android Open Source Project
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

set(CMAKE_VERBOSE_MAKEFILE on)

# TENSORFLOW_SRC_DIR:
#    root directory of tensorflow repo
#    used for shared source files and pre-built libs
get_filename_component(TENSORFLOW_SRC_DIR ../../.. ABSOLUTE)
set(PREBUILT_LIB_PATH ${TENSORFLOW_SRC_DIR}/tensorflow/contrib/makefile/gen)

add_library(lib_proto STATIC IMPORTED )
set_target_properties(lib_proto PROPERTIES IMPORTED_LOCATION
    ${PREBUILT_LIB_PATH}/protobuf/lib/libprotobuf.a)

add_library(lib_tf STATIC IMPORTED )
set_target_properties(lib_tf PROPERTIES IMPORTED_LOCATION
        ${PREBUILT_LIB_PATH}/lib/libtensorflow-core.a)

# Include libraries needed for com_tensorflow_demo lib
include_directories(
    ./jni
    ../../contrib/android/jni
    ${TENSORFLOW_SRC_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
    ${TENSORFLOW_SRC_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/include
    ${TENSORFLOW_SRC_DIR}/tensorflow/contrib/makefile/gen/host_obj
    ${TENSORFLOW_SRC_DIR}/tensorflow/contrib/makefile/downloads/eigen
    ${TENSORFLOW_SRC_DIR}/tensorflow/contrib/makefile/downloads
    ${TENSORFLOW_SRC_DIR}
    ${TENSORFLOW_SRC_DIR}/protobuf/src)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -frtti -fno-exceptions -fpic -O3 -mfpu=neon -DTF_LEAN_BINARY")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition -Wl,--whole-archive")

add_library(tensorflow_demo SHARED
                jni/imageutils_jni.cc
                jni/yuv2rgb.cc
                jni/rgb2yuv.cc
                ../../contrib/android/jni/jni_utils.cc
                ../../contrib/android/jni/tensorflow_inference_jni.cc)

target_link_libraries(tensorflow_demo
                         android
                         log
                         jnigraphics
                         m
                         atomic
                         z
                         lib_tf
                         lib_proto)
