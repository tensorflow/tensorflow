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

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/schema/mutable/schema_generated.h"
#ifdef __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_unit_test_entry_points.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#endif  // __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

extern "C" {
int FunctionInBinary(int argc, char** argv) { return 2; }
}  // extern "C"

namespace tflite {
namespace acceleration {
namespace {

typedef int (*EntryPoint)(int, char**);
using flatbuffers::FlatBufferBuilder;

struct RunnerTest : ::testing::Test {
 protected:
  void* LoadEntryPointModule() {
    void* module =
        dlopen(entry_point_file.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
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

  void SetUp() override {
#ifdef __ANDROID__
    // We extract the test files here as that's the only way to get the right
    // architecture when building tests for multiple architectures.
    entry_point_file = MiniBenchmarkTestHelper::DumpToTempFile(
        "librunner_unit_test_entry_points.so",
        g_tflite_acceleration_embedded_runner_unit_test_entry_points,
        g_tflite_acceleration_embedded_runner_unit_test_entry_points_len);
    ASSERT_TRUE(!entry_point_file.empty());
#endif  // __ANDROID__
  }

  void Init(const char* symbol_name) {
    EntryPoint function = Load(symbol_name);
    ASSERT_TRUE(function);
    runner = std::make_unique<ProcessRunner>(::testing::TempDir(), symbol_name,
                                             function);
    ASSERT_EQ(runner->Init(), kMinibenchmarkSuccess);
  }

  FlatBufferBuilder CreateTestModel() {
    ModelT model;
    model.description = "test";
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(CreateModel(fbb, &model));
    return fbb;
  }
  int exitcode = 0;
  int signal = 0;
  std::string output;
  std::unique_ptr<ProcessRunner> runner;
  std::string entry_point_file;
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
  EXPECT_EQ(runner->Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 0);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, ReturnOne) {
  ASSERT_NO_FATAL_FAILURE(Init("ReturnOne"));
  EXPECT_EQ(runner->Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 1);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, ReturnSuccess) {
  ASSERT_NO_FATAL_FAILURE(Init("ReturnSuccess"));
  EXPECT_EQ(runner->Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, NullFunctionPointer) {
  ProcessRunner runner("foo", "bar", nullptr);
  EXPECT_EQ(runner.Init(), kMinibenchmarkPreconditionNotMet);
  EXPECT_EQ(runner.Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkPreconditionNotMet);
}

#ifdef __ANDROID__
TEST_F(RunnerTest, SigKill) {
  ASSERT_NO_FATAL_FAILURE(Init("SigKill"));
  EXPECT_EQ(runner->Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed);
  EXPECT_EQ(exitcode, 0);
  EXPECT_EQ(signal, SIGKILL);
  EXPECT_EQ(output, "");
}

TEST_F(RunnerTest, WriteOk) {
  ASSERT_NO_FATAL_FAILURE(Init("WriteOk"));
  EXPECT_EQ(runner->Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "ok\n");
}

TEST_F(RunnerTest, Write10kChars) {
  ASSERT_NO_FATAL_FAILURE(Init("Write10kChars"));
  EXPECT_EQ(runner->Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output.size(), 10000);
}

TEST_F(RunnerTest, ArgsArePassed) {
  ASSERT_NO_FATAL_FAILURE(Init("WriteArgs"));
  EXPECT_EQ(
      runner->Run(nullptr, {"foo", "bar", "baz"}, &output, &exitcode, &signal),
      kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output, "foo\nbar\nbaz\n");
}

TEST_F(RunnerTest, SymbolLookupFailed) {
  ProcessRunner runner(::testing::TempDir(), "FunctionInBinary",
                       FunctionInBinary);
  EXPECT_EQ(runner.Init(), kMinibenchmarkSuccess);
  EXPECT_EQ(runner.Run(nullptr, {}, &output, &exitcode, &signal),
            kMinibenchmarkCommandFailed)
      << output;
  EXPECT_EQ(exitcode, kMinibenchmarkRunnerMainSymbolLookupFailed) << output;
}

TEST_F(RunnerTest, ReadModelFromPipe) {
  ASSERT_NO_FATAL_FAILURE(Init("ReadFromPipe"));
  FlatBufferBuilder model = CreateTestModel();
  EXPECT_EQ(runner->Run(&model, {}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
  EXPECT_EQ(exitcode, kMinibenchmarkSuccess);
  EXPECT_EQ(signal, 0);
  EXPECT_EQ(output,
            std::string((char*)model.GetBufferPointer(), model.GetSize()));
}

#else  // __ANDROID__

TEST_F(RunnerTest, ReadModelFromPipeNonAndroid) {
  ASSERT_NO_FATAL_FAILURE(Init("ReadFromPipeInProcess"));
  FlatBufferBuilder model = CreateTestModel();
  EXPECT_EQ(runner->Run(&model, {}, &output, &exitcode, &signal),
            kMinibenchmarkSuccess);
}

#endif  // __ANDROID__
#endif  // !__aarch64__

}  // namespace
}  // namespace acceleration
}  // namespace tflite
