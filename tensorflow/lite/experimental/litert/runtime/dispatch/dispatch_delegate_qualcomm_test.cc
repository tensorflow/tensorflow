// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"

namespace litert {

namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

static constexpr absl::string_view kNpuFile = kQualcommModelFileName;
static constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";

TEST(DispatchDelegate, Qualcomm) {
  auto runtime =
      testing::TflRuntime::CreateFromTflFileWithByteCode(kTfliteFile, kNpuFile);
  ASSERT_TRUE(runtime) << "Failed to initialize tflite interpreter";
  auto& rt = **runtime;
  auto& interpreter = rt.Interp();

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  auto dispatch_delegate_options = CreateDispatchDelegateOptionsPtr();
  LiteRtDispatchDelegateAddAllocBaseOption(dispatch_delegate_options.get(),
                                           rt.AllocBase());
  auto dispatch_delegate =
      CreateDispatchDelegatePtr(std::move(dispatch_delegate_options));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Get the list of signatures and check it.
  auto signature_defs = interpreter.signature_keys();
  ASSERT_EQ(signature_defs.size(), 0);

  tflite::impl::SignatureRunner* runner =
      interpreter.GetSignatureRunner(/*signature_key=*/nullptr);
  ASSERT_NE(runner, nullptr);

  EXPECT_EQ(runner->AllocateTensors(), kTfLiteOk);

  // Fill model inputs.
  ASSERT_STREQ(runner->input_names()[0], "arg0");
  auto input_0_tensor = runner->input_tensor("arg0");
  ASSERT_NE(input_0_tensor, nullptr);
  auto* input_0 = input_0_tensor->data.f;
  std::memcpy(input_0, kTestInput0Tensor, sizeof(kTestInput0Tensor));

  ASSERT_STREQ(runner->input_names()[1], "arg1");
  auto input_1_tensor = runner->input_tensor("arg1");
  ASSERT_NE(input_1_tensor, nullptr);
  auto* input_1 = input_1_tensor->data.f;
  std::memcpy(input_1, kTestInput1Tensor, sizeof(kTestInput1Tensor));

  EXPECT_EQ(runner->Invoke(), kTfLiteOk);

  // Check model output.
  ASSERT_STREQ(runner->output_names()[0], "tfl.custom");
  auto output_tensor = runner->output_tensor("tfl.custom");
  ASSERT_NE(output_tensor, nullptr);
  const float* output = output_tensor->data.f;
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
  }
  EXPECT_THAT(absl::MakeConstSpan(output, kTestOutputSize),
              Pointwise(FloatNear(1e-5), kTestOutputTensor));
}

}  // namespace

}  // namespace litert
