/* Copyright 2021 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/tsl/profiler/lib/profiler_factory.h"

#include <functional>
#include <utility>

#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/profiler/lib/profiler_interface.h"
#include "tensorflow/tsl/profiler/protobuf/profiler_options.pb.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

class TestProfiler : public ProfilerInterface {
 public:
  Status Start() override { return OkStatus(); }
  Status Stop() override { return OkStatus(); }
  Status CollectData(tensorflow::profiler::XSpace*) override {
    return OkStatus();
  }
};

std::unique_ptr<ProfilerInterface> TestFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return absl::make_unique<TestProfiler>();
}

TEST(ProfilerFactoryTest, FactoryFunctionPointer) {
  ClearRegisteredProfilersForTest();
  RegisterProfilerFactory(&TestFactoryFunction);
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

TEST(ProfilerFactoryTest, FactoryLambda) {
  ClearRegisteredProfilersForTest();
  RegisterProfilerFactory([](const tensorflow::ProfileOptions& options) {
    return absl::make_unique<TestProfiler>();
  });
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

std::unique_ptr<ProfilerInterface> NullFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return nullptr;
}

TEST(ProfilerFactoryTest, FactoryReturnsNull) {
  ClearRegisteredProfilersForTest();
  RegisterProfilerFactory(&NullFactoryFunction);
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_TRUE(profilers.empty());
}

class FactoryClass {
 public:
  explicit FactoryClass(void* ptr) : ptr_(ptr) {}
  FactoryClass(const FactoryClass&) = default;  // copyable
  FactoryClass(FactoryClass&&) = default;       // movable

  std::unique_ptr<ProfilerInterface> CreateProfiler(
      const tensorflow::ProfileOptions& options) const {
    return absl::make_unique<TestProfiler>();
  }

 private:
  void* ptr_ TF_ATTRIBUTE_UNUSED = nullptr;
};

TEST(ProfilerFactoryTest, FactoryClassCapturedByLambda) {
  ClearRegisteredProfilersForTest();
  static int token = 42;
  FactoryClass factory(&token);
  RegisterProfilerFactory([factory = std::move(factory)](
                              const tensorflow::ProfileOptions& options) {
    return factory.CreateProfiler(options);
  });
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
