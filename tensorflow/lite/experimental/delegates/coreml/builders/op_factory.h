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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_FACTORY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_FACTORY_H_

#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace delegates {
namespace coreml {
class GraphBuilder;
class OpBuilder;

OpBuilder* CreateAddOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateAveragePool2dOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateConcatenationOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateConvolutionOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateDepthwiseConvolutionOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateFullyConnectedOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateHardSwishOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateLogisticOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateMaxPool2dOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateMeanOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateMirrorPadOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateMulOpBuilder(GraphBuilder* graph_builder);
// PAD handles PAD and PADV2 together.
OpBuilder* CreatePadOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateReluOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateReluN1To1OpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateRelu6OpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateReshapeOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateResizeBilinearOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateSoftmaxOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateTanhOpBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateTransposeConvolutionOpBuilder(GraphBuilder* graph_builder);

OpBuilder* CreateActivationLayerBuilder(GraphBuilder* graph_builder);
OpBuilder* CreateThresholdLayerBuilder(GraphBuilder* graph_builder);
// Dummy Opbuilder for nodes that are claimed but not used. ex) FP16 dequantize
OpBuilder* CreateDummyOpBuilder(GraphBuilder* graph_builder);

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_COREML_BUILDERS_OP_FACTORY_H_
