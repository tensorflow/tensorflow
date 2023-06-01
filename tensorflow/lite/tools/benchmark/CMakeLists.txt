#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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


# The benchmark tool for Tensorflow Lite.

populate_source_vars("${TFLITE_SOURCE_DIR}/tools/benchmark"
  TFLITE_BENCHMARK_SRCS
  FILTER "(_test|_plus_flex_main|_performance_options.*)\\.cc$"
)
list(APPEND TFLITE_BENCHMARK_SRCS
  ${TF_SOURCE_DIR}/tsl/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/kernels/internal/utils/sparsity_format_converter.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_usage_monitor.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/root_profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/profiler.cc
  ${TFLITE_SOURCE_DIR}/profiling/telemetry/telemetry.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/model_loader.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
  ${TFLITE_SOURCE_DIR}/tools/utils.cc
)

list(APPEND TFLITE_BENCHMARK_LIBS
  tensorflow-lite
)

# TODO(b/171007016): Enable performance options on Windows.
if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  list(APPEND TFLITE_BENCHMARK_SRCS
    ${TFLITE_SOURCE_DIR}/tools/benchmark/benchmark_performance_options.cc
  )
endif()

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_BENCHMARK_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
    ${TFLITE_SOURCE_DIR}/core/acceleration/configuration/c/xnnpack_plugin.cc)
else()
  set(TFLITE_BENCHMARK_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(TFLITE_ENABLE_EXTERNAL_DELEGATE)
  list(APPEND TFLITE_BENCHMARK_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/external_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_EXTERNAL_DELEGATE

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_BENCHMARK_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
  list(APPEND TFLITE_BENCHMARK_LIBS
    ${ANDROID_LOG_LIB}
    absl::strings
  )
endif()  # Android

if(TFLITE_ENABLE_GPU)
  list(APPEND TFLITE_BENCHMARK_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  )
endif()  # TFLITE_ENABLE_GPU

add_executable(benchmark_model
  EXCLUDE_FROM_ALL
  ${TFLITE_BENCHMARK_SRCS}
)
target_compile_options(benchmark_model
  PRIVATE
    ${TFLITE_BENCHMARK_CC_OPTIONS}
)
target_link_libraries(benchmark_model
    ${TFLITE_BENCHMARK_LIBS}
)
