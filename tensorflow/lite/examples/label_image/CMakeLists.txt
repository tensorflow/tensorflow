#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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


# The label_image example for Tensorflow Lite.

populate_source_vars("${TFLITE_SOURCE_DIR}/examples/label_image"
  TFLITE_LABEL_IMAGE_SRCS
  FILTER "_test\\.cc$"
)
list(APPEND TFLITE_LABEL_IMAGE_SRCS
  ${XLA_SOURCE_DIR}/xla/tsl/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
)

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
    ${TFLITE_SOURCE_DIR}/core/acceleration/configuration/c/xnnpack_plugin.cc
  )
else()
  set(TFLITE_LABEL_IMAGE_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_LABEL_IMAGE_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_LABEL_IMAGE_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

if(TFLITE_ENABLE_EXTERNAL_DELEGATE)
  list(APPEND TFLITE_LABEL_IMAGE_SRCS
          ${TFLITE_SOURCE_DIR}/tools/delegates/external_delegate_provider.cc)
endif()

include_directories(label_image
  PUBLIC
  ${CMAKE_BINARY_DIR}
)

add_executable(label_image
  ${TFLITE_LABEL_IMAGE_SRCS}
)
if(TFLITE_ENABLE_LABEL_IMAGE)
  set_target_properties(label_image PROPERTIES EXCLUDE_FROM_ALL FALSE)
  if(TFLITE_ENABLE_INSTALL)
    install(TARGETS label_image)
  endif()  # TFLITE_ENABLE_INSTALL
else()
  set_target_properties(label_image PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()  # TFLITE_ENABLE_LABEL_IMAGE
target_compile_options(label_image
  PRIVATE
    ${TFLITE_LABEL_IMAGE_CC_OPTIONS}
)
target_link_libraries(label_image
  tensorflow-lite
  profiling_info_proto
  protobuf
)
