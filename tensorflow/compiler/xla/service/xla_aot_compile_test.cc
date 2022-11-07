/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/resource_loader.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace xla_compile {
namespace {

TEST(XlaCompileTest, LoadBuiltExecutable) {
  std::string path = tsl::GetDataDependencyFilepath(
      "tensorflow/compiler/xla/service/xla_aot_compile_test_output");

  std::string serialized_aot_result;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));
  TF_ASSERT_OK_AND_ASSIGN(auto aot_result,
                          cpu::CpuXlaRuntimeAotCompilationResult::FromString(
                              serialized_aot_result));

  cpu::CpuCompiler cpu_compiler;
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          aot_result->LoadExecutable(&cpu_compiler, nullptr));
}

}  // namespace
}  // namespace xla_compile
}  // namespace xla
