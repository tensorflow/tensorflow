/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/types.h"

namespace {

using ::stablehlo::quantization::io::CreateTmpDir;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

// A test-only derived class of `tsl::Env` which is broken. Used to cause
// failure for the `CreateTmpDir` function. Each of the overridden member
// functions implements a dummy functionality just to be able to create an
// instance of this class.
class TestEnvBrokenFileSystem : public tsl::Env {
 public:
  TestEnvBrokenFileSystem() = default;

  bool MatchPath(const tsl::string& path, const tsl::string& pattern) override {
    return false;
  }

  void SleepForMicroseconds(int64_t micros) override {}

  tsl::string GetRunfilesDir() override { return tsl::string("dummy_path"); }

  int32_t GetCurrentThreadId() override { return 0; }

  tsl::Thread* StartThread(const tsl::ThreadOptions& thread_options,
                           const tsl::string& name,
                           absl::AnyInvocable<void()> fn) override {
    return nullptr;
  }

  bool GetCurrentThreadName(tsl::string* name) override { return false; }

  void SchedClosure(absl::AnyInvocable<void()> closure) override {}

  void SchedClosureAfter(int64_t micros,
                         absl::AnyInvocable<void()> closure) override {}

  absl::Status LoadDynamicLibrary(const char* library_filename,
                                  void** handle) override {
    return tsl::OkStatus();
  }

  absl::Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                    void** symbol) override {
    return tsl::OkStatus();
  }

  tsl::string FormatLibraryFileName(const tsl::string& name,
                                    const tsl::string& version) override {
    return tsl::string("dummy_path");
  }

  absl::Status GetFileSystemForFile(const std::string& fname,
                                    tsl::FileSystem** result) override {
    return absl::InternalError("Broken file system");
  }

 private:
  // This is the part that essentially breaks the `CreateTmpDir` function
  // because it doesn't provide any available temp dirs.
  void GetLocalTempDirectories(std::vector<tsl::string>* list) override {}
};

TEST(IoTest, CreateTmpDirReturnsValidTmpPath) {
  absl::StatusOr<std::string> tmp_dir = CreateTmpDir();

  ASSERT_THAT(tmp_dir, IsOk());

  auto* const env = tsl::Env::Default();
  EXPECT_THAT(env->FileExists(*tmp_dir), IsOk());
}

TEST(IoTest, CreateTmpDirWhenInvalidPathReturnsInternalError) {
  TestEnvBrokenFileSystem test_env{};
  absl::StatusOr<std::string> tmp_dir = CreateTmpDir(&test_env);

  EXPECT_THAT(tmp_dir, StatusIs(absl::StatusCode::kInternal,
                                HasSubstr("Failed to create tmp dir")));
}

}  // namespace
