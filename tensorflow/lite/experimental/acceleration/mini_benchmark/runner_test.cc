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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"

#include <dlfcn.h>
#include <signal.h>

#include <fstream>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#ifdef __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_unit_test_entry_points.h"
#endif  // __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

extern "C" {
int FunctionInBinary(int argc, char** argv) { return 2; }
}  // extern "C"

namespace tflite {
namespace acceleration {
namespace {

std::string GetTestTmpDir() {
  const char* dir = getenv("TEST_TMPDIR");
  if (!dir) {
    dir = "/data/local/tmp";
  }
  return dir;
}

typedef int (*EntryPoint)(int, char**);

void* LoadEntryPointModule() {
  std::string path = GetTestTmpDir() + "/librunner_unit_test_entry_points.so";
  void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  EXPECT_TRUE(module) << dlerror();
  return module;
}

EntryPoint Load(const char* name) {
#ifdef __ANDROID__
  void* module = LoadEntryPointModule();
  if (!module) {
    return nullptr;
  }
#else   // !__ANDROID__
  auto module = RTLD_DEFAULT;
#endif  // __ANDROID__
  void* symbol = dlsym(module, name);
  return reinterpret_cast<EntryPoint>(symbol);
}

struct RunnerTest : ::testing::Test {
 protected:
  void SetUp() override {
#ifdef __ANDROID__
    // We extract the test files here as that's the only way to get the right
    // architecture when building tests for multiple architectures.
    ASSERT_NO_FATAL_FAILURE(WriteFile(
        "librunner_unit_test_entry_points.so",
        g_tflite_acceleration_embedded_runner_unit_test_entry_points,
        g_tflite_acceleration_embedded_runner_unit_test_entry_points_len));
#endif  // __ANDROID__
  }
  void WriteFile(const std::string& filename, const unsigned char* data,
                 size_t length) {
    std::string path = GetTestTmpDir() + "/" + filename;
    (void)unlink(path.c_str());
    std::string contents(reinterpret_cast<const char*>(data), length);
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f << contents;
    f.close();
    ASSERT_EQ(chmod(path.c_str(), 0500), 0);
  }
  void Init(const char* symbol_name) {
    EntryPoint function = Load(symbol_name);
    ASSERT_TRUE(function);
    runner =
        std::make_unique<ProcessRunner>(GetTestTmpDir(), symbol_name, function);
    ASSERT_EQ(runner->Init(), kMinibenchmarkSuccess);
  }
  int exitcode = 0;
  int signal = 0;
  std::string output;
  std::unique_ptr<ProcessRunner> runner;
};

// These tests are only for Android. They are also disabled on 64-bit arm
// because the 64-bit arm emulator doesn't have a shell that works with popen().
// These tests are run on x86 emulators.
#if !defined(__aarch64__)
TEST_F(RunnerTest, LoadSymbol) {
  EntryPoint JustReturnZero = Load("JustReturnZero");
  ASSERT_TRUE(JustReturnZero);
#ifdef __ANDROID__
  Dl_info dl_info;
  int status = dladdr(reinterpret_cast<void*>(JustReturnZero), &dl_info);
  ASSERT_TRUE(status) << dlerror();
  ASSERT_TRUE(dl_info.dli_fname) << dlerror();

  void* this_module =
      dlopen(dl_info.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  ASSERT_TRUE(this_module);
  void* symbol = dlsym(this_module, "JustReturnZero");
  EXPECT_TRUE(symbol);
#endif  // __ANDROID__
}

TEST_F(RunnerTest, JustReturnZero) {
  ASSERT_NO_FATAL_FAILURE(Init("JustReturnZero"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 0);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, ReturnOne) {
  ASSERT_NO_FATAL_FAILURE(Init("ReturnOne"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 1);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, ReturnSuccess) {
  ASSERT_NO_FATAL_FAILURE(Init("ReturnSuccess"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

#ifdef __ANDROID__
TEST_F(RunnerTest, SigKill) {
  ASSERT_NO_FATAL_FAILURE(Init("SigKill"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 0);
  EXPECT_EQ(signal, SIGKILL);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, WriteOk) {
  ASSERT_NO_FATAL_FAILURE(Init("WriteOk"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "ok\n");
}

TEST_F(RunnerTest, Write10kChars) {
  ASSERT_NO_FATAL_FAILURE(Init("Write10kChars"));
  EXPECT_EQ(runner->Run({}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output.size(), 10000);
}

TEST_F(RunnerTest, ArgsArePassed) {
  ASSERT_NO_FATAL_FAILURE(Init("WriteArgs"));
  EXPECT_EQ(runner->Run({"foo", "bar", "baz"}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "foo\nbar\nbaz\n");
}
#endif  // __ANDROID__

TEST_F(RunnerTest, NullFunctionPointer) {
  ProcessRunner runner("foo", "bar", nullptr);
  EXPECT_EQ(runner.Init(), kMinibenchmarkPreconditionNotMet);
  EXPECT_EQ(runner.Run({}, &output, &exitcode, &signal),
            kMinibenchmarkPreconditionNotMet);
}

#ifdef __ANDROID__
TEST_F(RunnerTest, SymbolLookupFailed) {
  ProcessRunner runner(GetTestTmpDir(), "FunctionInBinary", FunctionInBinary);
  EXPECT_EQ(runner.Init(), kMinibenchmarkSuccess);
  EXPECT_EQ(runner.Run({}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed)
      << output;
  EXPECT_EQ(exitcode, kMinibenchmarkRunnerMainSymbolLookupFailed) << output;
}
#endif  // __ANDROID__
#endif  // !__aarch64__

}  // namespace
}  // namespace acceleration
}  // namespace tflite
