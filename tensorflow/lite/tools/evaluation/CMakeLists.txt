#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For TensorFlow Lite evaluation tools cross-compilation purposes a natively pre-compiled 'protoc'
# compiler is required. If found, it is added as a custom target. When compiling
# natively, 'protoc' target is already available in protobuf CMake configuration.
if(${CMAKE_CROSSCOMPILING})
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${TFLITE_SOURCE_DIR}/tools/cmake/native_tools)
    include(native_utils)
    
    find_native_compiler("protoc")
    add_executable(protoc IMPORTED)
    set_property(TARGET protoc PROPERTY
                IMPORTED_LOCATION $CACHE{PROTOC-BIN})
    # When cross-compiling, the following needs to be disabled in order to avoid
    # naming clash with the 'protoc' target that is part of the protobuf CMake project
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Do not cross-compile protoc and libprotoc")
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Turn off protobuf unit tests")
find_package(protobuf REQUIRED)

get_filename_component(TF_ROOT_DIR ${TFLITE_SOURCE_DIR} DIRECTORY)
get_filename_component(REPO_ROOT_DIR ${TF_ROOT_DIR} DIRECTORY)

file(GLOB PROTOBUF_DEFINITION_FILES "${TFLITE_SOURCE_DIR}/tools/evaluation/proto/*.proto")

set(PROTOBUF_INPUT_DIRECTORY ${REPO_ROOT_DIR})
set(PROTOBUF_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(PROTO_INCLUDE_PATH "tensorflow/lite/tools/evaluation/proto" )

foreach(FILE ${PROTOBUF_DEFINITION_FILES})
    set(PROTOBUF_ARGUMENTS "--proto_path=${PROTOBUF_INPUT_DIRECTORY}" "--cpp_out=${PROTOBUF_OUTPUT_DIRECTORY}" "${FILE}")
    get_filename_component(PROTO_NAME ${FILE} NAME_WE)
    set (PROTO_OUTPUT_NAME "${PROTOBUF_OUTPUT_DIRECTORY}/${PROTO_INCLUDE_PATH}/${PROTO_NAME}.pb.cc" )
    add_custom_command(OUTPUT ${PROTO_OUTPUT_NAME}
                       COMMAND protoc ${PROTOBUF_ARGUMENTS}
                       DEPENDS protoc)   
    list(APPEND PROTO_GEN_OUTPUTS ${PROTO_OUTPUT_NAME})            
endforeach()

add_library(proto_gen_outputs ${PROTO_GEN_OUTPUTS})

# Headers corresponding to the specific libprotobuf library version are required.
# ${CMAKE_BINARY_DIR} contains the generated .pb.cc and .pb.h files (/tensorflow/lite/tools/evaluation/proto).
target_include_directories(proto_gen_outputs
    PUBLIC
        ${protobuf_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR})

find_package(nsync REQUIRED)
populate_tflite_source_vars("tools/evaluation/stages" TFLITE_EVALUATION_STAGES_SRCS RECURSE)

list(APPEND TFLITE_EVALUATION_BINARY_SRCS
    ${TF_SOURCE_DIR}/core/platform/default/logging.cc
    ${TF_SOURCE_DIR}/core/platform/default/env_time.cc
    ${TF_SOURCE_DIR}/core/platform/default/mutex.cc
    ${TF_SOURCE_DIR}/core/lib/jpeg/jpeg_mem.cc
    ${TF_SOURCE_DIR}/core/lib/jpeg/jpeg_handle.cc
    ${TFLITE_SOURCE_DIR}/profiling/time.cc
    ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
    ${TFLITE_SOURCE_DIR}/tools/evaluation/evaluation_delegate_provider.cc
    ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/task_executor_main.cc
    ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/task_executor.cc
    ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
    ${TFLITE_SOURCE_DIR}/nnapi/sl/SupportLibrary.cc
    ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
    ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
    ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
)
list(APPEND TFLITE_EVALUATION_BINARY_SRCS ${TFLITE_CMD_LINE_TOOLS_SRCS})
list(APPEND TFLITE_EVALUATION_BINARY_SRCS ${TFLITE_EVALUATION_STAGES_SRCS})
list(APPEND TFLITE_EVALUATION_BINARY_SRCS ${TFLITE_EVALUATION_PROTO_SRCS})
if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_EVALUATION_BINARY_SRCS ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc)
endif()
if(TFLITE_ENABLE_XNNPACK)
    list(APPEND TFLITE_EVALUATION_BINARY_SRCS ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc)
endif()

add_executable(coco_object_detection_run_eval
    ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/coco_object_detection/run_eval.cc
    ${TFLITE_EVALUATION_BINARY_SRCS}
)
target_link_libraries(coco_object_detection_run_eval proto_gen_outputs )

add_executable(imagenet_image_classification_run_eval
    ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc
    ${TFLITE_EVALUATION_BINARY_SRCS}
)
target_link_libraries(imagenet_image_classification_run_eval proto_gen_outputs)

add_executable(inference_diff_run_eval
    ${TFLITE_SOURCE_DIR}/tools/evaluation/tasks/inference_diff/run_eval.cc
    ${TFLITE_EVALUATION_BINARY_SRCS}
)
target_link_libraries(inference_diff_run_eval proto_gen_outputs)

list(APPEND EVALTOOLS
    coco_object_detection_run_eval
    imagenet_image_classification_run_eval
    inference_diff_run_eval
)

foreach(evaltool ${EVALTOOLS})
    target_link_libraries(${evaltool}
        tensorflow-lite
        ${TFLITE_BENCHMARK_LIBS}
        absl::base
        absl::strings
        absl::flat_hash_map
        nsync_cpp
        libprotobuf
        "-ljpeg"
        ${CMAKE_DL_LIBS}
    )
    target_compile_options(${evaltool}
        PRIVATE
        ${TFLITE_BENCHMARK_CC_OPTIONS})
endforeach()

