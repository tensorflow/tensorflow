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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define _LITERT_ASSERT_RESULT_OK_ASSIGN(decl, expr, result) \
  auto result = (expr);                                     \
  ASSERT_TRUE(result.HasValue());                           \
  decl = result.Value();

#define LITERT_ASSERT_RESULT_OK_ASSIGN(decl, expr) \
  _LITERT_ASSERT_RESULT_OK_ASSIGN(decl, expr,      \
                                  _CONCAT_NAME(_result, __COUNTER__))

#define _LITERT_ASSERT_RESULT_OK_MOVE(decl, expr, result) \
  auto result = (expr);                                   \
  ASSERT_TRUE(result.HasValue());                         \
  decl = std::move(result.Value());

#define LITERT_ASSERT_RESULT_OK_MOVE(decl, expr) \
  _LITERT_ASSERT_RESULT_OK_MOVE(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

#define LITERT_ASSERT_STATUS_HAS_CODE(expr, code) \
  {                                               \
    LiteRtStatus status = (expr);                 \
    ASSERT_EQ(status, code);                      \
  }

#define LITERT_ASSERT_STATUS_OK(expr) \
  LITERT_ASSERT_STATUS_HAS_CODE(expr, kLiteRtStatusOk);

namespace litert {
namespace testing {

std::string GetTestFilePath(absl::string_view filename);

Expected<std::vector<char>> LoadBinaryFile(absl::string_view filename);

Model LoadTestFileModel(absl::string_view filename);

void TouchTestFile(absl::string_view filename, absl::string_view dir);

bool ValidateTopology(const std::vector<Op>& ops);

// Get a buffer that is the concatenation of given tflite file and
// npu byte code file. Adds metadata containing the offset/size of npu byte
// code. Input tflite model custom ops should contain only the function name in
// custom options this will normalize with the correct format. NOTE only the
// case where all ops share the same offset is currently implemented.
Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    absl::string_view tfl_file, absl::string_view npu_file);

class TflRuntime {
 public:
  using Ptr = std::unique_ptr<TflRuntime>;
  static Expected<Ptr> CreateFromTflFile(absl::string_view filename);
  static Expected<Ptr> CreateFromTflFileWithByteCode(
      absl::string_view tfl_filename, absl::string_view npu_filename);

  tflite::Interpreter& Interp() { return *interp_; }

  BufferRef<uint8_t> ModelBuf() const {
    return BufferRef<uint8_t>(alloc_->base(), alloc_->bytes());
  }

  const void* AllocBase() const { return alloc_->base(); }

  const ::tflite::Model* Model() const { return fb_model_->GetModel(); }

 private:
  std::unique_ptr<::tflite::Interpreter> interp_;
  std::unique_ptr<::tflite::FlatBufferModel> fb_model_;
  std::unique_ptr<::tflite::Allocation> alloc_;
  OwningBufferRef<uint8_t> model_buf_;
};

}  // namespace testing
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_
