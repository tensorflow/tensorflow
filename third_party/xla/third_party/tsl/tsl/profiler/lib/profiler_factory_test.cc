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
#include "tsl/profiler/lib/profiler_factory.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

class TestProfiler : public ProfilerInterface {
 public:
  absl::Status Start() override { return absl::OkStatus(); }
  absl::Status Stop() override { return absl::OkStatus(); }
  absl::Status CollectData(tensorflow::profiler::XSpace*) override {
    return absl::OkStatus();
  }
};

std::unique_ptr<ProfilerInterface> TestFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return std::make_unique<TestProfiler>();
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
    return std::make_unique<TestProfiler>();
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
    return std::make_unique<TestProfiler>();
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

class TestMultiPassProfiler : public MultiPassProfilerInterface {
 public:
  bool NeedMorePasses() override { return false; }
  absl::Status StartPass() override { return absl::OkStatus(); }
  absl::Status PushRange(absl::string_view name) override {
    return absl::OkStatus();
  }
  absl::Status PopRange() override { return absl::OkStatus(); }
  absl::Status StopPass() override { return absl::OkStatus(); }

  absl::Status Start() override { return absl::OkStatus(); }
  absl::Status Stop() override { return absl::OkStatus(); }
  absl::Status CollectData(tensorflow::profiler::XSpace*) override {
    return absl::OkStatus();
  }
};

std::unique_ptr<MultiPassProfilerInterface> TestMultiPassFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return absl::make_unique<TestMultiPassProfiler>();
}

TEST(ProfilerFactoryTest, MultiPassFactoryFunctionPointer) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(&TestMultiPassFactoryFunction);
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

TEST(ProfilerFactoryTest, MultiPassFactoryLambda) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(
      [](const tensorflow::ProfileOptions& options) {
        return absl::make_unique<TestMultiPassProfiler>();
      });
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

std::unique_ptr<MultiPassProfilerInterface> NullMultiPassFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return nullptr;
}

TEST(ProfilerFactoryTest, MultiPassFactoryReturnsNull) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(&NullMultiPassFactoryFunction);
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  EXPECT_TRUE(profilers.empty());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
