// Copyright 2025 Google LLC.
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

#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

namespace litert {
namespace {

TEST(EnvironmentTest, Default) {
  auto env = litert::Environment::Create({});
  EXPECT_TRUE(env);
}

TEST(EnvironmentTest, Options) {
  constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));
  EXPECT_TRUE(env);
}

TEST(EnvironmentTest, CompiledModelBasic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  auto compiled_model = CompiledModel::Create(env, model);
  EXPECT_TRUE(compiled_model);
}

TEST(EnvironmentTest, StringLifeCycle) {
  std::string dispatch_library_dir = "/data/local/tmp";
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          absl::string_view(dispatch_library_dir),
      },
  };

  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));

  EXPECT_TRUE(env);

  // Change the string value but the environment should still have a copy.
  dispatch_library_dir = "";

  // Create Model and check signatures.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create CompiledModel.
  auto compiled_model = CompiledModel::Create(*env, model);
  EXPECT_TRUE(compiled_model);
}

}  // namespace
}  // namespace litert
