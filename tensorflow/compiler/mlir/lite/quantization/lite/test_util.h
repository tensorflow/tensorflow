/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TEST_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TEST_UTIL_H_

#include <cstdarg>

#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

namespace mlir {
namespace lite {
namespace internal {
// Test model with a single convolution.
// Floating point weights of the model are all integers and lie in
// range[-127, 127]. The weights have been put in such a way that each
// channel has at least one weight as -127 and one weight as 127.
// The activations are all in range: [-128, 127]
// This means all bias computations should result in 1.0 scale.
extern const char* kConvModelWithMinus128Plus127Weights;

// Test model with single convolution where all weights are integers between
// [0, 10] weights are randomly distributed. It is not guaranteed that min max
// for weights are going to appear in each channel.
// Activations have min = 0, max = 10.
extern const char* kConvModelWith0Plus10Weights;

// Test model where no bias is in the conv.
extern const char* kConvModelWithNoBias;

// A floating point model with a single softmax. The input tensor has min
// and max in range [-5, 5], not necessarily -5 or +5.
extern const char* kSingleSoftmaxModelMinMinus5MaxPlus5;

// A floating point model with a single average pool. The input tensor has min
// and max in range [-5, 5], not necessarily -5 or +5.
extern const char* kSingleAvgPoolModelMinMinus5MaxPlus5;

// Test model with a weights variable that is shared between a convolution layer
// and an add operation.
extern const char* kModelWithSharedWeights;

// Test model with Add followed by a reshape. Model has 2 inputs for add.
extern const char* kMultiInputAddWithReshape;

// Test gather operation with quantized input.
extern const char* kQuantizedWithGather;

// Test model with a tf.constant input to tf.add. Model has 2 inputs one
// constant and other placeholder.
extern const char* kConstInputAddModel;

// A float test model with concat that has [0, 5] and [0, 10] for inputs and [0,
// 10] as output.
extern const char* kFloatConcatMax5Max10Max10;

// Test model with broadcast_to op.
extern const char* kModelWithBroadcastToOp;

// Test model with a custom op.
extern const char* kModelWithCustomOp;

// Test model with a argmax op.
extern const char* kModelWithArgMaxOp;

// Test model with a fully connected op.
extern const char* kModelWithFCOp;

// Test model with a gather_nd op.
extern const char* kModelWithGatherNDOp;

// Test model with a Where op.
extern const char* kModelWithWhereOp;

// Test model with mixed quantizable and un-quantizable ops.
// reshape->custom->custom->squeeze.
extern const char* kModelMixed;

// Test model with mixed quantizable and
// and un-quantizable ops for
// activations in 16-bit.
extern const char* kModelMixed16x8;

// Test model with split op.
extern const char* kModelSplit;

// Test model with pack op.
extern const char* kModelPack;

// Test model with LSTM op that has layer norm, has projection, without
// peephole, without cifg.
extern const char* kLstmCalibrated;
extern const char* kLstmQuantized;

// Test model with LSTM op that has peephole, without layer norm, without
// projection, without cifg.
extern const char* kLstmCalibrated2;
extern const char* kLstmQuantized2;

extern const char* kUnidirectionalSequenceLstmCalibrated;
extern const char* kUnidirectionalSequenceLstmQuantized;

// Test model with a minimum op.
extern const char* kModelWithMinimumOp;

// Test model with a maximum op.
extern const char* kModelWithMaximumOp;

// Test model with a transpose op.
extern const char* kModelWithTranspose;

// Test model with SVDF op.
extern const char* kSvdfCalibrated;
extern const char* kSvdfQuantized;

// Test model with an unpack op.
extern const char* kModelWithUnpack;

// Test QAT model with fc op.
extern const char* kQatModelWithFc;

// Test calibrated model with resource variables.
extern const char* kModelWithResourceVarsCalibrated;

// An error reporter that fails on testing.
class FailOnErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override;
};
}  // namespace internal
}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TEST_UTIL_H_
