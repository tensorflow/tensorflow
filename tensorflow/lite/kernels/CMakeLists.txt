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
find_package(nsync REQUIRED)
find_package(re2 REQUIRED)

# Generate the mutable schema_generated.h header for tests.
set(SCHEMA_FILE ${TFLITE_SOURCE_DIR}/schema/schema.fbs)
set(SCHEMA_GENERATED_ROOT ${CMAKE_CURRENT_BINARY_DIR}/schema)
set(SCHEMA_GENERATED_PATH ${SCHEMA_GENERATED_ROOT}/tensorflow/lite/schema/mutable)
set(SCHEMA_GENERATED_FILE ${SCHEMA_GENERATED_PATH}/schema_generated.h)

# Use the util function in flatbuffer to generate the schema header.
include(BuildFlatBuffers)
set(FLATBUFFERS_FLATC_EXECUTABLE ${CMAKE_BINARY_DIR}/flatbuffers-flatc/bin/flatc)
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

set(TEST_FRAMEWORK_SRC
  ${TFLITE_SOURCE_DIR}/delegates/nnapi/acceleration_test_list.cc
  ${TFLITE_SOURCE_DIR}/delegates/nnapi/acceleration_test_util.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/schema/schema_conversion_utils.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/model_utils.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/operator_property.cc
  ${TFLITE_SOURCE_DIR}/tools/optimize/quantization_utils.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
  ${TFLITE_SOURCE_DIR}/tools/versioning/op_version.cc
  ${TF_SOURCE_DIR}/core/platform/default/env_time.cc
  ${TF_SOURCE_DIR}/core/platform/default/logging.cc
  ${TF_SOURCE_DIR}/core/platform/default/mutex.cc
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
set(TEST_FRAMEWORK_MAIN_SRC test_main.cc)

add_library(tensorflow-lite-test ${TEST_FRAMEWORK_SRC})
target_link_libraries(tensorflow-lite-test
  gmock
  gtest
  nsync_cpp
  re2
  tensorflow-lite
)
add_dependencies(tensorflow-lite-test mutable_schema_file)
target_include_directories(tensorflow-lite-test PUBLIC ${SCHEMA_GENERATED_ROOT})

add_library(tensorflow-lite-test-with-main ${TEST_FRAMEWORK_MAIN_SRC})
target_link_libraries(tensorflow-lite-test-with-main
  # We need the nnapi related symbols later.
  # Please check:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/BUILD#L96
  -Wl,--whole-archive
  tensorflow-lite-test
  -Wl,--no-whole-archive
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
endmacro()

# The tflite kernel test
set(TEST_LIST
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
  random_standard_normal_test.cc
  random_uniform_test.cc
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
# The test which has main() function.
set(TEST_WITH_MAIN_LIST
  cpu_backend_gemm_test.cc
  cpu_backend_threadpool_test.cc
  eigen_support_test.cc
  kernel_util_test.cc
  optional_tensor_test.cc
  subgraph_test_util_test.cc
  test_util_test.cc
)

foreach(test_src IN LISTS TEST_LIST)
  add_kernel_test(${test_src} tensorflow-lite-test-with-main)
endforeach()

foreach(test_src IN LISTS TEST_WITH_MAIN_LIST)
  add_kernel_test(${test_src} tensorflow-lite-test)
endforeach()
