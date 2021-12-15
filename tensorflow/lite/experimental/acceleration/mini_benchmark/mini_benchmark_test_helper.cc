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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"

#include <fcntl.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // !_WIN32

#include <fstream>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/tools/logging.h"

#ifdef __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_validator_runner_entrypoint.h"
#endif  // __ANDROID__

namespace tflite {
namespace acceleration {
namespace {
#ifdef __ANDROID__
void* LoadEntryPointModule(const std::string& module_path) {
  void* module =
      dlopen(module_path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (module == nullptr) {
    TFLITE_LOG(FATAL) << dlerror();
  }
  return module;
}
#endif  // __ANDROID__

std::string JoinPath(const std::string& path1, const std::string& path2) {
  if (path1.empty()) return path2;
  if (path2.empty()) return path1;
  if (path1.back() == '/') {
    if (path2.front() == '/') return path1 + path2.substr(1);
  } else {
    if (path2.front() != '/') return path1 + "/" + path2;
  }
  return path1 + path2;
}
}  // namespace

MiniBenchmarkTestHelper::MiniBenchmarkTestHelper()
    : should_perform_test_(true) {
#ifdef __ANDROID__
  AndroidInfo android_info;
  const auto status = RequestAndroidInfo(&android_info);
  if (!status.ok() || android_info.is_emulator) {
    should_perform_test_ = false;
    return;
  }

  DumpToTempFile("librunner_main.so", g_tflite_acceleration_embedded_runner,
                 g_tflite_acceleration_embedded_runner_len);

  std::string validator_runner_so_path = DumpToTempFile(
      "libvalidator_runner_entrypoint.so",
      g_tflite_acceleration_embedded_validator_runner_entrypoint,
      g_tflite_acceleration_embedded_validator_runner_entrypoint_len);
  // Load this library here because it contains the validation entry point
  // "Java_org_tensorflow_lite_acceleration_validation_entrypoint" that is then
  // found using dlsym (using RTLD_DEFAULT hence not needing the handle) in the
  // mini-benchmark code.
  LoadEntryPointModule(validator_runner_so_path);
#endif  // __ANDROID__
}

std::string MiniBenchmarkTestHelper::DumpToTempFile(const std::string& filename,
                                                    const unsigned char* data,
                                                    size_t length) {
  std::string path = JoinPath(::testing::TempDir(), filename);
  (void)unlink(path.c_str());
  std::string contents(reinterpret_cast<const char*>(data), length);
  std::ofstream f(path, std::ios::binary);
  EXPECT_TRUE(f.is_open());
  f << contents;
  f.close();
  EXPECT_EQ(0, chmod(path.c_str(), 0500));
  return path;
}

}  // namespace acceleration
}  // namespace tflite
