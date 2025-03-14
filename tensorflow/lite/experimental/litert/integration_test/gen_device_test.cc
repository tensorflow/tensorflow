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

#include <filesystem>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/integration_test/gen_device_test_lib.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

ABSL_FLAG(std::string, model_path, "",
          "Tflite models to test. This can be a single tflite model or a "
          "directory containing multiple tflite models.");
ABSL_FLAG(std::string, dispatch_library_dir, "/data/local/tmp/",
          "Path to the dispatch library.");
ABSL_FLAG(std::string, hw, "cpu", "Which accelerator to use.");

namespace litert::test {
namespace {

// UTILS ///////////////////////////////////////////////////////////////////////

bool IsTfliteModel(const std::filesystem::path& path) {
  return std::filesystem::is_regular_file(path) &&
         path.extension() == ".tflite";
}

std::vector<std::string> GetModelPaths(const std::string& model_path_str) {
  std::filesystem::path model_path = model_path_str;
  std::vector<std::string> models;
  if (std::filesystem::is_directory(model_path)) {
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
      if (IsTfliteModel(entry.path())) {
        models.push_back(entry.path().generic_string());
      }
    }
    return models;
  }

  if (IsTfliteModel(model_path)) {
    return {model_path.generic_string()};
  }

  return {};
}

std::string ModelName(const std::filesystem::path& path) {
  return path.filename().replace_extension().generic_string();
}

}  // namespace

// FIXTURES ////////////////////////////////////////////////////////////////////

class GenDeviceTestFixt : public ::testing::Test {};

// A test that simply calls the model and ensures it doesn't crash.
// Works with any accelerator.
template <class InvokerT>
class InvokeOnceTest : public GenDeviceTestFixt {
 public:
  InvokeOnceTest(std::string model_path, std::string dispatch_library_dir)
      : model_path_(std::move(model_path)),
        dispatch_library_dir_(std::move(dispatch_library_dir)) {}

  // Opens model and initializes the underlying invoker.
  void SetUp() override {
    const std::vector<litert::Environment::Option> environment_options = {
        litert::Environment::Option{
            litert::Environment::OptionTag::DispatchLibraryDir,
            absl::string_view(dispatch_library_dir_),
        },
    };
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto env, litert::Environment::Create(environment_options));

    LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                                litert::Model::CreateFromFile(model_path_));

    invoker_ = std::make_unique<InvokerT>(std::move(env), std::move(model));
    invoker_->MaybeSkip();
    ASSERT_NO_FATAL_FAILURE(invoker_->Setup());
  }

  void TestBody() override { ASSERT_NO_FATAL_FAILURE(invoker_->Run()); }

 private:
  std::string model_path_;
  std::string dispatch_library_dir_;

  CmInvoker::Ptr invoker_;
};

// REGISTRATION ////////////////////////////////////////////////////////////////

// Registers tests dynamically based on the hw flag and the model_path flag.
void ParseTests() {
  auto model_path_flag = absl::GetFlag(FLAGS_model_path);
  // Provide a sensible default based on linux/android.
  if (model_path_flag.empty()) {
#if defined(__ANDROID__)
    model_path_flag = "/data/local/tmp/";
#else
    // Set this on linux for smoke check linux presubmit.
    model_path_flag = testing::GetLiteRtPath(
        "integration_test/single_op_models/add_f32.tflite");
#endif
  }
  const auto model_paths = GetModelPaths(model_path_flag);
  const auto hw = absl::GetFlag(FLAGS_hw);
  const auto dispatch_library_dir = absl::GetFlag(FLAGS_dispatch_library_dir);

  LITERT_LOG(LITERT_INFO, "hw: %s", hw.c_str());
  LITERT_LOG(LITERT_INFO, "model_path: %s", model_path_flag.c_str());
  LITERT_LOG(LITERT_INFO, "dispatch_library_dir: %s",
             dispatch_library_dir.c_str());

  if (model_paths.empty()) {
    LITERT_LOG(LITERT_WARNING, "No models found to test.");
    return;
  }

  for (const auto& model_path : model_paths) {
    const auto test_name = absl::StrFormat("%s_%s", ModelName(model_path), hw);
    ::testing::RegisterTest("GenDeviceTest", test_name.c_str(), nullptr,
                            nullptr, __FILE__, __LINE__,
                            [=]() -> GenDeviceTestFixt* {
                              if (hw == "npu") {
                                return new InvokeOnceTest<CmNpuInvoker>(
                                    model_path, dispatch_library_dir);
                              } else {
                                return new InvokeOnceTest<CmCpuInvoker>(
                                    model_path, dispatch_library_dir);
                              }
                            });
  }
}

}  // namespace litert::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  litert::test::ParseTests();
  return RUN_ALL_TESTS();
}
