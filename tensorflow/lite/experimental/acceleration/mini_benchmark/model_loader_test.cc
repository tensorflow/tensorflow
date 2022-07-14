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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_loader.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

namespace tflite {
namespace acceleration {
namespace {

class ModelLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_path_ = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);
  }
  std::string model_path_;
};

TEST_F(ModelLoaderTest, CreateFromModelPath) {
  std::unique_ptr<ModelLoader> model_loader =
      ModelLoader::CreateFromFdOrPath(model_path_);
  ASSERT_NE(model_loader, nullptr);
  EXPECT_THAT(model_loader->Init(), kMinibenchmarkSuccess);
}

TEST_F(ModelLoaderTest, CreateFromFdPath) {
  int fd = open(model_path_.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(fd, &stat_buf), 0);
  auto model_loader = std::make_unique<ModelLoader>(fd, 0, stat_buf.st_size);
  close(fd);

  ASSERT_NE(model_loader, nullptr);
  EXPECT_THAT(model_loader->Init(), kMinibenchmarkSuccess);
}

TEST_F(ModelLoaderTest, CreateFromFdOrModelPath) {
  int fd = open(model_path_.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(fd, &stat_buf), 0);
  std::string path = absl::StrFormat("fd:%d:%zu:%zu", fd, 0, stat_buf.st_size);
  auto model_loader = ModelLoader::CreateFromFdOrPath(path);
  close(fd);

  ASSERT_NE(model_loader, nullptr);
  EXPECT_THAT(model_loader->Init(), kMinibenchmarkSuccess);
}

TEST_F(ModelLoaderTest, InvalidFdPath) {
  int fd = open(model_path_.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  struct stat stat_buf = {0};
  ASSERT_EQ(fstat(fd, &stat_buf), 0);
  std::string path = absl::StrFormat("fd:%d:%zu", fd, 0);
  auto model_loader = ModelLoader::CreateFromFdOrPath(path);
  close(fd);

  EXPECT_EQ(model_loader, nullptr);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
