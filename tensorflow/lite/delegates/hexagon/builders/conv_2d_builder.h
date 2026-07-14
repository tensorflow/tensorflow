/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_CONV_2D_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_CONV_2D_BUILDER_H_

#include <vector>

#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

// Stores quantization data for Conv/TransposeConv nodes.
// This information is used to handle the per-channel quantized weights & biases
// correctly in the Hexagon delegate.
struct PerChannelQuantData {
  // This is initialized while processing quantized weights, and acts as an
  // input to Hexagon Conv nodes.
  OpBuilder* channel_scales_node = nullptr;
  // Scale information is obtained from TfLiteAffineQuantization in the weights
  // tensor.
  float* scales_data = nullptr;
  int num_scale_values = 1;
  // Number of splits to workaround DepthwiseConv accuracy issue.
  // See Conv2dOpBuilder.should_split_dwconv_ for details.
  int splits = 0;
  std::vector<OpBuilder*> channel_scales_nodes;
};

class Conv2dOpBuilder : public OpBuilder {
 public:
  explicit Conv2dOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

  ~Conv2dOpBuilder() override;

 private:
  TfLiteStatus InitializeWeightsNodes(const TfLiteIntArray* inputs,
                                      const TfLiteIntArray* outputs,
                                      TfLiteContext* context,
                                      const int input_depth);

  TfLiteStatus InitializeBiasNodes(const TfLiteIntArray* inputs,
                                   const TfLiteIntArray* outputs,
                                   TfLiteContext* context);

  void BuildStandardConv(const TfLiteIntArray* inputs,
                         const TfLiteTensor& output_data_tensor,
                         OpBuilder* data_min_const, OpBuilder* data_max_const,
                         OpBuilder* conv_output_min_const,
                         OpBuilder* conv_output_max_const,
                         OpBuilder* stride_node,
                         const TfLitePadding padding_type,
                         TensorID* output_tensor, TensorID* output_min_tensor,
                         TensorID* output_max_tensor);
  void BuildDilatedDwConv(const TfLiteIntArray* inputs,
                          const TfLiteTensor& data_tensor,
                          const TfLiteTensor& output_data_tensor,
                          OpBuilder* data_min_const, OpBuilder* data_max_const,
                          OpBuilder* conv_output_min_const,
                          OpBuilder* conv_output_max_const,
                          OpBuilder* stride_node, int stride_height,
                          const TfLitePadding padding_type,
                          TensorID* output_tensor, TensorID* output_min_tensor,
                          TensorID* output_max_tensor);
  void BuildSplittedDwConv(
      const TfLiteIntArray* inputs, const TfLiteTensor& data_tensor,
      const TfLiteTensor& output_data_tensor, OpBuilder* data_min_const,
      OpBuilder* data_max_const, OpBuilder* conv_output_min_const,
      OpBuilder* conv_output_max_const, OpBuilder* stride_node,
      const TfLitePadding padding_type, TensorID* output_tensor,
      TensorID* output_min_tensor, TensorID* output_max_tensor);

  TensorID node_output_;
  std::vector<float> transposed_weights_;
  std::vector<int> stride_shape_;
  std::vector<int> weight_shape_;
  OpBuilder* weights_min_node_ = nullptr;
  OpBuilder* weights_max_node_ = nullptr;
  OpBuilder* bias_min_node_ = nullptr;
  OpBuilder* bias_max_node_ = nullptr;

  // TODO(b/228874753)
  // We are seeing accuray issues on DepthwiseSupernode_8x8p32to8 in the
  // following case:
  // * kernel size is 5x5
  // * stride size is 2x2
  // * per channel quantized
  // * input depth more than 32
  //
  // To workaround the issue, the DepthwiseSupernode_8x8p32to8 is splitted
  // into 32 channel batches and concatenated afterwards.
  // Input tensor, weights, bias and channel scales are splitted into 32
  // channel sizes and fed to multiple DepthwiseSupernode_8x8p32to8 ops.
  // The results are stitched back with a Concat op.
  //
  // Checks if it has DepthwiseSupernode_8x8p32to8 accuracy issues.
  void CheckShouldSplitDwConv(TfLiteType weights_type, int input_depth,
                              bool is_per_channel_quant,
                              int channel_multiplier);
  // Split weights into multiple 32-channel nodes.
  // `converted_data` is MSB flipped int8 weight values.
  void SplitWeightsForDwConv(const std::vector<uint8_t>& converted_data,
                             int input_depth, int channel_multiplier);
  // Split bias into 32 element batches.
  // `preprocessed_bias_data` is the output of ProcessPerChannelQuantizedBias.
  void SplitBiasForDwConv(std::vector<int>& preprocessed_bias_data);
  bool should_split_dwconv_ = false;
  std::vector<TensorID> data_nodes_;
  std::vector<OpBuilder*> bias_nodes_;
  std::vector<OpBuilder*> weights_nodes_;

  // Modified only if node has per-channel quantized weights/biases.
  PerChannelQuantData per_channel_quant_;

  // Only used for dilated Depthwise Conv.
  std::vector<int> dilation_factors_h_w_;
  std::vector<int> space_to_batch_paddings_;
  std::vector<int> batch_to_space_crops_;
};

// ProcessPerChannelQuantizedWeights & ProcessPerChannelQuantizedBias can be
// used to pre-process per-channel quantized weights & biases for Hexagon.
// NOTE: ProcessPerChannelQuantizedWeights should be run before
// ProcessPerChannelQuantizedBias. This is becase we set PerChannelQuantData
// based on the weights tensor, which is utilized while preprocessing bias.

TfLiteStatus ProcessPerChannelQuantizedWeights(
    const TfLiteTensor& weights_tensor, TfLiteContext* context,
    float* weights_min, float* weights_max, GraphBuilder* graph_builder,
    PerChannelQuantData* per_channel_quant);

TfLiteStatus ProcessPerChannelQuantizedBias(
    const TfLiteTensor& data_tensor, const TfLiteTensor& bias_tensor,
    const int bias_tensor_idx, TfLiteContext* context, float* bias_min,
    float* bias_max, GraphBuilder* graph_builder,
    PerChannelQuantData* per_channel_quant,
    std::vector<int>* preprocessed_bias_data,
    OpBuilder** bias_const_node = nullptr);

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_HEXAGON_BUILDERS_CONV_2D_BUILDER_H_
