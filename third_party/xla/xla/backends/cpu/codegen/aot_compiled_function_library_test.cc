/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

namespace {

int add(int a, int b) { return a + b; }

TEST(AotCompiledFunctionLibraryTest, ResolveFunction) {
  const std::string function_ptr_name = "Add";

  AotCompiledFunctionLibrary::FunctionPtr function_ptr =
      reinterpret_cast<AotCompiledFunctionLibrary::FunctionPtr>(&add);

  std::unique_ptr<FunctionLibrary> function_library =
      std::make_unique<AotCompiledFunctionLibrary>(
          absl::flat_hash_map<std::string,
                              AotCompiledFunctionLibrary::FunctionPtr>{
              {function_ptr_name, function_ptr}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto add_function,
      function_library->ResolveFunction<decltype(add)>(function_ptr_name));

  int a = 1;
  int b = 2;

  EXPECT_EQ(add_function(a, b), 3);
}

TEST(AotCompiledFunctionLibraryTest, ResolveNonExistentFunction) {
  const std::string function_ptr_name = "NonExistentFunction";

  std::unique_ptr<FunctionLibrary> function_library =
      std::make_unique<AotCompiledFunctionLibrary>(
          absl::flat_hash_map<std::string,
                              AotCompiledFunctionLibrary::FunctionPtr>{{}});

  EXPECT_FALSE(
      function_library->ResolveFunction<decltype(add)>(function_ptr_name).ok());
}

}  // namespace

}  // namespace xla::cpu
