/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/builtin_op_data.h"
#include <gtest/gtest.h>

namespace tflite {

// Builtin op data is just a set of data definitions, so the only meaningful
// test we can run is whether we can create the structs we expect to find.
// Testing each struct's members might be possible, but it seems unnecessary
// until we've locked down the API. The build rule has copts set to ignore the
// unused variable warning, since this is just a compilation test.
TEST(IntArray, CanCompileStructs) {
  TfLitePadding padding = kTfLitePaddingSame;
  TfLitePaddingValues padding_values;
  TfLiteFusedActivation fused_activation = kTfLiteActRelu;
  TfLiteConvParams conv_params;
  TfLitePoolParams pool_params;
  TfLiteDepthwiseConvParams depthwise_conv_params;
  TfLiteSVDFParams svdf_params;
  TfLiteRNNParams rnn_params;
  TfLiteSequenceRNNParams sequence_rnn_params;
  TfLiteFullyConnectedWeightsFormat fully_connected_weights_format =
      kTfLiteFullyConnectedWeightsFormatDefault;
  TfLiteFullyConnectedParams fully_connected_params;
  TfLiteLSHProjectionType projection_type = kTfLiteLshProjectionDense;
  TfLiteLSHProjectionParams projection_params;
  TfLiteSoftmaxParams softmax_params;
  TfLiteConcatenationParams concatenation_params;
  TfLiteAddParams add_params;
  TfLiteSpaceToBatchNDParams space_to_batch_nd_params;
  TfLiteBatchToSpaceNDParams batch_to_space_nd_params;
  TfLiteMulParams mul_params;
  TfLiteSubParams sub_params;
  TfLiteDivParams div_params;
  TfLiteL2NormParams l2_norm_params;
  TfLiteLocalResponseNormParams local_response_norm_params;
  TfLiteLSTMKernelType lstm_kernel_type = kTfLiteLSTMBasicKernel;
  TfLiteLSTMParams lstm_params;
  TfLiteResizeBilinearParams resize_bilinear_params;
  TfLitePadParams pad_params;
  TfLitePadV2Params pad_v2_params;
  TfLiteReshapeParams reshape_params;
  TfLiteSkipGramParams skip_gram_params;
  TfLiteSpaceToDepthParams space_to_depth_params;
  TfLiteDepthToSpaceParams depth_to_space_params;
  TfLiteCastParams cast_params;
  TfLiteCombinerType combiner_type = kTfLiteCombinerTypeSqrtn;
  TfLiteEmbeddingLookupSparseParams lookup_sparse_params;
  TfLiteGatherParams gather_params;
  TfLiteTransposeParams transpose_params;
  TfLiteReducerParams reducer_params;
  TfLiteSplitParams split_params;
  TfLiteSplitVParams split_v_params;
  TfLiteSqueezeParams squeeze_params;
  TfLiteStridedSliceParams strided_slice_params;
  TfLiteArgMaxParams arg_max_params;
  TfLiteArgMinParams arg_min_params;
  TfLiteTransposeConvParams transpose_conv_params;
  TfLiteSparseToDenseParams sparse_to_dense_params;
  TfLiteShapeParams shape_params;
  TfLiteRankParams rank_params;
  TfLiteFakeQuantParams fake_quant_params;
  TfLitePackParams pack_params;
  TfLiteUnpackParams unpack_params;
  TfLiteOneHotParams one_hot_params;
  TfLiteBidirectionalSequenceRNNParams bidi_sequence_rnn_params;
  TfLiteBidirectionalSequenceLSTMParams bidi_sequence_lstm_params;
}

}  // namespace tflite
