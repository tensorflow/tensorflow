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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_TEST_UTIL_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_TEST_UTIL_H_

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
namespace optimize {
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

// A floating point model with a single softmax. The input tensor has min
// and max in range [-5, 5], not necessarily -5 or +5.
extern const char* kSingleSoftmaxModelMinMinus5MaxPlus5;

// A floating point model with a single average pool. The input tensor has min
// and max in range [-5, 5], not necessarily -5 or +5.
extern const char* kSingleAvgPoolModelMinMinus5MaxPlus5;

// An error reporter that fails on testing.
class FailOnErrorReporter : public ErrorReporter {
 public:
  int Report(const char* format, va_list args) override;
};
}  // namespace internal
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_TEST_UTIL_H_
