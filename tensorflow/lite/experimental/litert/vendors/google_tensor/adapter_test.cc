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

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/adapter.h"

#include <sys/types.h>

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert {
namespace google_tensor {

TEST(AdapterTest, CreateSuccess) {
  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }
  ASSERT_TRUE(adapter_result.HasValue());
}

TEST(AdapterTest, CreateFailure) {
  auto kLibDarwinnCompilerNoLib = "libcompiler_api_wrapper_no_lib.so";
  auto adapter_result = Adapter::Create(kLibDarwinnCompilerNoLib);
  ASSERT_FALSE(adapter_result.HasValue());
}

TEST(AdapterTest, CompileSuccess) {
  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }

  auto model = litert::testing::LoadTestFileModel("mul_simple.tflite");
  LiteRtModel litert_model = model.Get();

  LITERT_LOG(LITERT_INFO, "%s", "Serializing model");
  litert::OwningBufferRef buf;

  // Using weak pointer to link the data to the buffer.
  auto [data, size, offset] = buf.GetWeak();

  const auto opts = litert::SerializationOptions::Defaults();
  auto status =
      LiteRtSerializeModel(litert_model, &data, &size, &offset, false, opts);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to serialize model");
  }

  absl::string_view buffer_str(reinterpret_cast<const char*>(buf.Data()),
                               buf.Size());

  ASSERT_FALSE(buffer_str.empty());
  LITERT_LOG(LITERT_INFO, "buffer_str size: %d", buffer_str.size());
  LITERT_LOG(LITERT_INFO, "Compling model...");
  absl::string_view soc_model = "P25";
  litert::google_tensor::Flags flags;
  flags.clear();
  std::string compiled_code;
  auto compile_status = adapter_result.Value()->api().compile(
      buffer_str, soc_model, flags, &compiled_code);
  ASSERT_OK(compile_status);
  ASSERT_FALSE(compiled_code.empty());
}

}  // namespace google_tensor
}  // namespace litert
