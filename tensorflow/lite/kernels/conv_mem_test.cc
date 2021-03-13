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
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {

TEST(ConvMemUsage, HugeIm2ColData) {
  // The model has a conv op will require a temporary tensor of ~3.5GB if
  // im2col is performed and it's possible to cause OOM on devices. To prevent
  // this from happening, a size cap (i.e. kMaxIm2colBufferSize) of
  // to-be-allocated im2col data is used to determine whether to disable im2col.
  // This test will check the memory footprint before/after interpreter Invoke
  // to ensure the size cap is correctly enforced.
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/conv_huge_im2col.bin");
  ASSERT_TRUE(model);

  const auto mem_before = profiling::memory::GetMemoryUsage();
  std::unique_ptr<Interpreter> interpreter;

  // Note that we explicitly set 1 thread here to avoid extra memory footprint
  // caused by multithreading, which will make the memory usage threshold check
  // later more reliable.
  ASSERT_EQ(InterpreterBuilder(*model, ops::builtin::BuiltinOpResolver())(
                &interpreter, /*num_threads*/ 1),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  const auto mem_after = profiling::memory::GetMemoryUsage();
  TFLITE_LOG(INFO) << "HugeIm2ColData Memory usage info: "
                   << mem_after - mem_before;

  // The "3GB" threshold but still < 3.5GB is chosen to suit different testing
  // configurations, such as MSan/TSan related tests where extra system-level
  // memory footprint usage might be counted as well.
  EXPECT_LE((mem_after - mem_before).max_rss_kb, 3 * 1024 * 1024);
}

}  // namespace tflite
