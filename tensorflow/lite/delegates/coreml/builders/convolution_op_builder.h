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
#ifndef TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_CONVOLUTION_OP_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_CONVOLUTION_OP_BUILDER_H_

#include <string>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {

enum class ConvolutionType { kConv, kDepthwiseConv, kTransposeConv };

// Layer that provides convolution and depthwise convolution.
class ConvolutionOpBuilder : public OpBuilder {
 public:
  explicit ConvolutionOpBuilder(GraphBuilder* graph_builder,
                                ConvolutionType conv_type)
      : OpBuilder(graph_builder), conv_type_(conv_type) {}

  const std::string& DebugName() override;

  CoreML::Specification::NeuralNetworkLayer* Build() override;

  TfLiteStatus PopulateSubgraph(TfLiteContext* context) override;

  void SetOutputChannels(uint64_t output_channels);

  void SetNGroups(uint64_t n_groups);

  void SetWeights(TfLiteTensor* weights);

  void SetBias(TfLiteTensor* bias);

  void SetOutputShape(TfLiteTensor* output_shape);

  void SetParams(void* builtin_data);

  TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                              TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  void FillCoreMLWeights();
  void FillCoreMLBias();

  // Transpose TFLite kernel weights to CoreML kernel weights.
  // Should be called after setting CoreML's kernel shapes.
  void TransposeKernelWeights();

  uint64_t output_channels_;
  uint64_t n_groups_ = 1;

  ConvolutionType conv_type_;

  // using default dilation_factor (1, 1)
  // CoreML ConvolutionLayerParams.isDeconvolution == false
  TfLiteTensor* weights_ = nullptr;
  TfLiteTensor* bias_ = nullptr;
  // Only used for TransposeConv.
  TfLiteTensor* output_shape_ = nullptr;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_CONVOLUTION_OP_BUILDER_H_
