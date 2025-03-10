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

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
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

}  // namespace google_tensor
}  // namespace litert
