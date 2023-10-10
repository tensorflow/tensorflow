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

find_package(googletest REQUIRED)
find_package(google_benchmark REQUIRED)
find_package(nsync REQUIRED)
find_package(re2 REQUIRED)

# Generate the mutable schema_generated.h header for tests.
set(SCHEMA_FILE ${TFLITE_SOURCE_DIR}/schema/schema.fbs)
set(SCHEMA_GENERATED_ROOT ${CMAKE_CURRENT_BINARY_DIR}/schema)
set(SCHEMA_GENERATED_PATH ${SCHEMA_GENERATED_ROOT}/tensorflow/lite/schema/mutable)
set(SCHEMA_GENERATED_FILE ${SCHEMA_GENERATED_PATH}/schema_generated.h)

# Use the util function in flatbuffer to generate the schema header.
include(BuildFlatBuffers)

# For cross-compilation purposes a natively compiled 'flatc' compiler is required
if(${CMAKE_CROSSCOMPILING})
  set(FLATC_PATHS
      ${TFLITE_HOST_TOOLS_DIR}
      ${TFLITE_HOST_TOOLS_DIR}/bin
      ${TFLITE_HOST_TOOLS_DIR}/flatbuffers-flatc/bin
      )
  find_program(FLATC_BIN flatc PATHS ${FLATC_PATHS})
  if(${FLATC_BIN} STREQUAL "FLATC_BIN-NOTFOUND")
    message(FATAL_ERROR "Natively compiled 'flatc' compiler has not been found in the following\
    locations: ${FLATC_PATHS}")
  else()
    message(STATUS "Pre-built 'flatc' compiler for cross-compilation purposes found: ${FLATC_BIN}")
    set(FLATBUFFERS_FLATC_EXECUTABLE ${FLATC_BIN})
  endif()
else()
  set(FLATBUFFERS_FLATC_EXECUTABLE ${CMAKE_BINARY_DIR}/flatbuffers-flatc/bin/flatc)
endif()

set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS
  -c
  --gen-object-api
  --gen-mutable
)
build_flatbuffers(
  "${SCHEMA_FILE}"
  ""
  mutable_schema_file
  flatbuffers-flatc
  "${SCHEMA_GENERATED_PATH}"
  ""
  ""
)

set(DELEGATE_PROVIDERS_SUPP
  ${TFLITE_SOURCE_DIR}/nnapi/sl/SupportLibrary.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
)

set(DELEGATE_PROVIDERS
  ${DELEGATE_PROVIDERS_SUPP}
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  # List of delegates referenced as options in the tensorflow/lite/CMakeLists.txt
  ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
)

if(TFLITE_ENABLE_EXTERNAL_DELEGATE)
  list(APPEND DELEGATE_PROVIDERS ${TFLITE_SOURCE_DIR}/tools/delegates/external_delegate_provider.cc)
endif()

set(TEST_FRAMEWORK_SRC
  ${TFLITE_SOURCE_DIR}/delegates/nnapi/acceleration_test_list.cc
  ${TFLITE_SOURCE_DIR}/delegates/nnapi/acceleration_test_util.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/schema/schema_conversion_utils.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${DELEGATE_PROVIDERS}
  ${TFLITE_SOURCE_DIR}/tools/optimize/model_utils.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/operator_property.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/quantization_utils.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
  ${TFLITE_SOURCE_DIR}/tools/versioning/op_version.cc
  ${TSL_SOURCE_DIR}/tsl/platform/default/env_time.cc
  ${TSL_SOURCE_DIR}/tsl/platform/default/logging.cc
  ${TSL_SOURCE_DIR}/tsl/platform/default/mutex.cc
  internal/test_util.cc
  acceleration_test_util.cc
  acceleration_test_util_internal.cc
  subgraph_test_util.cc
  test_delegate_providers.cc
  test_util.cc
)
if(NOT _TFLITE_ENABLE_NNAPI)
  list(APPEND TEST_FRAMEWORK_SRC
    ${TFLITE_SOURCE_DIR}/nnapi/nnapi_util.cc
  )
endif()
if(NOT TFLITE_ENABLE_GPU)
  list(APPEND TEST_FRAMEWORK_SRC
    ${TFLITE_SOURCE_DIR}/tools/versioning/op_signature.cc
  )
endif()

set(TEST_FRAMEWORK_OPTIONS "")
if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TEST_FRAMEWORK_SRC
    ${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
    ${TFLITE_SOURCE_DIR}/core/acceleration/configuration/c/xnnpack_plugin.cc)
else()
  list(APPEND TEST_FRAMEWORK_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()

# Base library to be later linked with the gtest OR gtest_main library
add_library(tensorflow-lite-test-base ${TEST_FRAMEWORK_SRC})
target_link_libraries(tensorflow-lite-test-base
  gmock
  nsync_cpp
  re2
  tensorflow-lite
  benchmark
)
add_dependencies(tensorflow-lite-test-base mutable_schema_file)
target_include_directories(tensorflow-lite-test-base PUBLIC ${SCHEMA_GENERATED_ROOT})
target_compile_options(tensorflow-lite-test-base PUBLIC ${TEST_FRAMEWORK_OPTIONS})

add_library(tensorflow-lite-test-gtest-main INTERFACE)
target_link_libraries(tensorflow-lite-test-gtest-main INTERFACE
  tensorflow-lite-test-base
  gtest_main
)

set(TEST_FRAMEWORK_MAIN_SRC test_main.cc)
add_library(tensorflow-lite-test-external-main ${TEST_FRAMEWORK_MAIN_SRC})
target_link_libraries(tensorflow-lite-test-external-main
  # We need the nnapi related symbols later.
  # Please check:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/BUILD#L96
  -Wl,--whole-archive
  tensorflow-lite-test-base
  -Wl,--no-whole-archive
  gtest
)

macro(add_kernel_test TEST_SRC TEST_LIB)
  string(REPLACE "/" "-" TEST_NAME ${TEST_SRC})
  string(REPLACE ".cc" "" TEST_NAME ${TEST_NAME})

  add_executable(${TEST_NAME} ${TEST_SRC})
  target_link_libraries(${TEST_NAME}
    ${TEST_LIB}
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
    WORKING_DIRECTORY ${TENSORFLOW_SOURCE_DIR}
  )
  set_tests_properties(${TEST_NAME} PROPERTIES LABELS "plain")

  if(_TFLITE_ENABLE_NNAPI OR TFLITE_ENABLE_EXTERNAL_DELEGATE OR TFLITE_ENABLE_XNNPACK)
    set(DELEGATE_TEST "${TEST_NAME}_delegate")
    add_test(
      NAME ${DELEGATE_TEST}
      COMMAND cmake -DTEST_EXECUTABLE=$<TARGET_FILE:${TEST_NAME}> -P run-tests.cmake
    )
    set_tests_properties(${DELEGATE_TEST} PROPERTIES LABELS "delegate")
  endif()
endmacro()

# Tests where main() is provided by the file referenced in TEST_FRAMEWORK_MAIN_SRC
set(TEST_WITH_EXTERNAL_MAIN_LIST
  internal/averagepool_quantized_test.cc
  internal/batch_to_space_nd_test.cc
  internal/conv_per_channel_quantized_16x8_test.cc
  internal/depthwiseconv_float_test.cc
  internal/depthwiseconv_per_channel_quantized_16x8_test.cc
  internal/depthwiseconv_per_channel_quantized_test.cc
  internal/depthwiseconv_quantized_test.cc
  internal/log_quantized_test.cc
  internal/logsoftmax_quantized_test.cc
  internal/maxpool_quantized_test.cc
  internal/non_max_suppression_test.cc
  internal/per_channel_dequantize_test.cc
  internal/quantization_util_test.cc
  internal/resize_bilinear_test.cc
  internal/resize_nearest_neighbor_test.cc
  internal/softmax_quantized_test.cc
  internal/strided_slice_logic_test.cc
  internal/tensor_test.cc
  internal/tensor_utils_test.cc
  internal/transpose_utils_test.cc
  acceleration_test_util_internal_test.cc
  activations_test.cc
  add_n_test.cc
  add_test.cc
  arg_min_max_test.cc
  audio_spectrogram_test.cc
  basic_rnn_test.cc
  batch_matmul_test.cc
  batch_to_space_nd_test.cc
  bidirectional_sequence_lstm_test.cc
  bidirectional_sequence_rnn_test.cc
  broadcast_to_test.cc
  call_once_test.cc
  cast_test.cc
  ceil_test.cc
  comparisons_test.cc
  complex_support_test.cc
  concatenation_test.cc
  conv3d_test.cc
  conv_mem_test.cc
  conv_test.cc
  cumsum_test.cc
  densify_test.cc
  depth_to_space_test.cc
  depthwise_conv_hybrid_test.cc
  depthwise_conv_test.cc
  dequantize_test.cc
  detection_postprocess_test.cc
  div_test.cc
  elementwise_test.cc
  embedding_lookup_sparse_test.cc
  embedding_lookup_test.cc
  exp_test.cc
  expand_dims_test.cc
  fake_quant_test.cc
  fill_test.cc
  floor_div_test.cc
  floor_mod_test.cc
  floor_test.cc
  fully_connected_test.cc
  gather_nd_test.cc
  gather_test.cc
  hashtable_lookup_test.cc
  hashtable_ops_test.cc
  if_test.cc
  l2norm_test.cc
  local_response_norm_test.cc
  log_softmax_test.cc
  logical_test.cc
  lsh_projection_test.cc
  lstm_eval_test.cc
  lstm_test.cc
  matrix_diag_test.cc
  matrix_set_diag_test.cc
  maximum_minimum_test.cc
  mfcc_test.cc
  mirror_pad_test.cc
  mul_test.cc
  multinomial_test.cc
  neg_test.cc
  non_max_suppression_test.cc
  numeric_verify_test.cc
  one_hot_test.cc
  pack_test.cc
  pad_test.cc
  pooling_test.cc
  pow_test.cc
  quant_basic_lstm_test.cc
  quantize_test.cc
  random_ops_test.cc
  range_test.cc
  rank_test.cc
  reduce_test.cc
  reshape_test.cc
  resize_bilinear_test.cc
  resize_nearest_neighbor_test.cc
  reverse_sequence_test.cc
  reverse_test.cc
  rfft2d_test.cc
  round_test.cc
  scatter_nd_test.cc
  segment_sum_test.cc
  select_test.cc
  shape_test.cc
  skip_gram_test.cc
  slice_test.cc
  softmax_test.cc
  space_to_batch_nd_test.cc
  space_to_depth_test.cc
  sparse_to_dense_test.cc
  split_test.cc
  split_v_test.cc
  squared_difference_test.cc
  squeeze_test.cc
  strided_slice_test.cc
  sub_test.cc
  svdf_test.cc
  # We don't always have the delegate environment(e.g nnapi)
  # So, skip "test_delegate_providers_test.cc".
  #test_delegate_providers_test.cc
  tile_test.cc
  topk_v2_test.cc
  transpose_conv_test.cc
  transpose_test.cc
  unidirectional_sequence_lstm_test.cc
  unidirectional_sequence_rnn_test.cc
  unique_test.cc
  unpack_test.cc
  variable_ops_test.cc
  where_test.cc
  while_test.cc
  zeros_like_test.cc
)
# Tests where the main() provided by the GoogleTest framework
set(TEST_WITH_GTEST_MAIN_LIST
  cpu_backend_gemm_test.cc
  cpu_backend_threadpool_test.cc
  eigen_support_test.cc
  kernel_util_test.cc
  optional_tensor_test.cc
  subgraph_test_util_test.cc
  test_util_test.cc
)

foreach(test_src IN LISTS TEST_WITH_EXTERNAL_MAIN_LIST)
  add_kernel_test(${test_src} tensorflow-lite-test-external-main)
endforeach()

foreach(test_src IN LISTS TEST_WITH_GTEST_MAIN_LIST)
  add_kernel_test(${test_src} tensorflow-lite-test-gtest-main)
endforeach()

# Copy the test utility that facilitates cross-compiled kernel tests run with launch arguments
if(${CMAKE_CROSSCOMPILING})
  configure_file(
    ${TFLITE_SOURCE_DIR}/tools/cmake/test_utils/run-tests.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/run-tests.cmake
    COPYONLY
  )
endif()
