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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"

#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

// Note that these tests are not meant to be completely exhaustive, but to test
// error propagation.

namespace tflite {
namespace acceleration {
namespace {

class ValidatorTest : public ::testing::Test {
 protected:
  std::string GetTestTmpDir() {
    const char* from_env = getenv("TEST_TMPDIR");
    if (from_env) {
      return from_env;
    }
    return "/data/local/tmp";
  }

  std::string Path(const std::string rootdir, const std::string& filename) {
    return rootdir + "/" + filename;
  }

  void WriteFile(const std::string& path, const unsigned char* data,
                 size_t length) {
    (void)unlink(path.c_str());
    std::string contents(reinterpret_cast<const char*>(data), length);
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f << contents;
    f.close();
    ASSERT_EQ(chmod(path.c_str(), 0500), 0);
  }

  std::string ValidationModelPath() {
    return Path(GetTestTmpDir(), "mobilenet_quant_with_validation.tflite");
  }
  std::string PlainModelPath() {
    return Path(GetTestTmpDir(), "mobilenet_quant.tflite");
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(WriteFile(
        ValidationModelPath(),
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len));
    ASSERT_NO_FATAL_FAILURE(WriteFile(
        PlainModelPath(), g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len));
  }
};

TEST_F(ValidatorTest, HappyPath) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(CreateComputeSettings(fbb));
  const ComputeSettings* settings =
      flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer());

  Validator validator(ValidationModelPath(), settings);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results), kMinibenchmarkSuccess);
  EXPECT_TRUE(results.ok);
  EXPECT_EQ(results.delegate_error, 0);
}

TEST_F(ValidatorTest, NoValidationSubgraph) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(CreateComputeSettings(fbb));
  const ComputeSettings* settings =
      flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer());

  Validator validator(PlainModelPath(), settings);
  EXPECT_EQ(validator.CheckModel(), kMinibenchmarkValidationSubgraphNotFound);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results),
            kMinibenchmarkValidationSubgraphNotFound);
}

TEST_F(ValidatorTest, InvalidModel) {
  std::string path = Path(GetTestTmpDir(), "foo.tflite");
  // Drop 12k to introduce a truncated model. Last ~11k are associated files.
  ASSERT_NO_FATAL_FAILURE(WriteFile(
      path, g_tflite_acceleration_embedded_mobilenet_validation_model,
      g_tflite_acceleration_embedded_mobilenet_validation_model_len - 12000));
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(CreateComputeSettings(fbb));
  const ComputeSettings* settings =
      flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer());

  Validator validator(path, settings);
  EXPECT_EQ(validator.CheckModel(), kMinibenchmarkModelBuildFailed);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results), kMinibenchmarkModelBuildFailed);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
