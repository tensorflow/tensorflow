/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {

void TestMemoryThreshold(const std::string& model_path,
                         size_t threshold_in_kb) {
  // The Im2Col optimization is only applied on mobile platforms, so only
  // validate on such platforms.
  if (!IsMobilePlatform()) {
    return;
  }

  // The model has a conv op will require a huge temporary tensor if
  // im2col is performed and it's possible to cause OOM on devices. To prevent
  // this from happening, a size cap (i.e. kMaxIm2colBufferSizeMobile) of
  // to-be-allocated im2col data is used to determine whether to disable
  // im2col. This test will check the memory footprint before/after
  // interpreter Invoke to ensure the size cap is correctly enforced on mobile
  // platforms.
  auto model = FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;

  // Note that we explicitly set 1 thread here to avoid extra memory footprint
  // caused by multithreading, which will make the memory usage threshold
  // check later more reliable.
  ASSERT_EQ(InterpreterBuilder(*model, ops::builtin::BuiltinOpResolver())(
                &interpreter, /*num_threads*/ 1),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Memory required for all tensors should be smaller than the  threshold.
  int64_t accumulate_tensor_memory = 0;
  for (int i = 0; i < interpreter->tensors_size(); ++i) {
    accumulate_tensor_memory += interpreter->tensor(i)->bytes;
  }
  EXPECT_LE(accumulate_tensor_memory, threshold_in_kb * 1024);
}

TEST(ConvMemUsage, HugeIm2ColData) {
  TestMemoryThreshold(
      // The model has a conv op will require a temporary tensor of ~3.5GB if
      // im2col is performed.
      "tensorflow/lite/testdata/conv_huge_im2col.bin",
      /*threshold_in_kb=*/3 * 1024 * 1024);
}

TEST(Conv3DMemUsage, HugeIm2ColData) {
  TestMemoryThreshold(
      // The model has a Conv3D op will require a temporary tensor of ~1.3GB if
      // im2col is performed.If not, it will use about 450MB.
      "tensorflow/lite/testdata/conv3d_huge_im2col.bin",
      /*threshold_in_kb=*/1 * 1024 * 1024);
}

}  // namespace tflite
