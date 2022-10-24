/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

// This test is kept separate because it disables XLA compilation globaly.
TEST(XlaCompilationCacheTest, TestDisabledXlaCompilation) {
  NameAttrList fn;
  fn.set_name("afunction");

  // Create mock arguments so we see them in the VLOG when compilation fails.
  std::vector<XlaCompiler::Argument> args(2);
  for (int i = 0; i < 2; ++i) {
    args[i].kind = XlaCompiler::Argument::kParameter;
    args[i].type = DT_INT32;
    args[i].shape = TensorShape({2, i + 1});
    args[i].name = absl::StrCat("arg", i);
  }

  DisableXlaCompilation();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  const XlaCompiler::CompilationResult* compilation_result;
  xla::LocalExecutable* executable;

  auto cache = new XlaCompilationCache(XlaCompilationCache::Config(), client,
                                       device_type);
  core::ScopedUnref cache_ref(cache);

  // Check that strict compilation is disallowed.
  Status status = cache->Compile(XlaCompiler::Options{}, fn, args,
                                 XlaCompiler::CompileOptions{},
                                 XlaCompilationCache::CompileMode::kStrict,
                                 &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "XLA compilation disabled"));

  // Check that async compilation is disallowed.
  status = cache->Compile(XlaCompiler::Options{}, fn, args,
                          XlaCompiler::CompileOptions{},
                          XlaCompilationCache::CompileMode::kAsync,
                          &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "XLA compilation disabled"));

  // Check that lazy compilation is disallowed.
  status = cache->Compile(XlaCompiler::Options{}, fn, args,
                          XlaCompiler::CompileOptions{},
                          XlaCompilationCache::CompileMode::kLazy,
                          &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "XLA compilation disabled"));
}

}  // namespace
}  // namespace tensorflow
