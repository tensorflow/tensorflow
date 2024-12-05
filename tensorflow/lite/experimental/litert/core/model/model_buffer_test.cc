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

#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace litert::internal {
namespace {

static constexpr absl::string_view kNpuFile = kGoogleTensorModelFileName;
static constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";

TEST(GetModelBufWithByteCode, CreateInterpreter) {
  auto model_with_byte_code =
      GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                              testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);

  auto alloc = std::make_unique<tflite::MemoryAllocation>(
      model_with_byte_code->Data(), model_with_byte_code->Size(),
      tflite::DefaultErrorReporter());

  auto fb_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(alloc->base()), alloc->bytes());
  ASSERT_NE(fb_model, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*fb_model, resolver)(&interpreter);
  EXPECT_NE(interpreter, nullptr);
}

TEST(GetModelBufWithByteCode, CheckMetadata) {
  auto model_with_byte_code =
      GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                              testing::GetTestFilePath(kNpuFile));
  ASSERT_TRUE(model_with_byte_code);

  auto model = LoadModelFromBuffer(*model_with_byte_code);

  auto byte_code_buffer = model->get()->FindMetadata(kByteCodeMetadataKey);
  ASSERT_TRUE(byte_code_buffer);
}

}  // namespace
}  // namespace litert::internal
