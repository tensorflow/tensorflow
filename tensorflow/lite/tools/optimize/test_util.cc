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
#include "tensorflow/lite/tools/optimize/test_util.h"

#include <gtest/gtest.h>

namespace tflite {
namespace optimize {
namespace internal {
const char* kConvModelWithMinus128Plus127Weights =
    "single_conv_weights_min_minus_127_max_plus_127.bin";

const char* kConvModelWith0Plus10Weights =
    "single_conv_weights_min_0_max_plus_10.bin";

const char* kSingleSoftmaxModelMinMinus5MaxPlus5 =
    "single_softmax_min_minus_5_max_plus_5.bin";

const char* kSingleAvgPoolModelMinMinus5MaxPlus5 =
    "single_avg_pool_min_minus_5_max_plus_5.bin";

const char* kModelWithSharedWeights = "weight_shared_between_convs.bin";
const char* kMultiInputAddWithReshape = "multi_input_add_reshape.bin";
const char* kQuantizedWithGather = "quantized_with_gather.bin";

const char* kConstInputAddModel = "add_with_const_input.bin";

const char* kFloatConcatMax5Max10Max10 = "concat.bin";

const char* kModelWithCustomOp = "custom_op.bin";

int FailOnErrorReporter::Report(const char* format, va_list args) {
  char buf[1024];
  vsnprintf(buf, sizeof(buf), format, args);
  EXPECT_TRUE(false) << "Error happened: " << buf;
  return 0;
}
}  // namespace internal
}  // namespace optimize
}  // namespace tflite
