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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

TEST(XlaCompilationCacheTest, SignatureEquality) {
  NameAttrList fn;
  fn.set_name("afunction");
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({4, 0});
  args[0].constant_value = Tensor(DT_INT32, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s1,
                          XlaCompilationCache::BuildSignature(fn, args));

  args[0].type = DT_FLOAT;
  args[0].constant_value = Tensor(DT_FLOAT, {4, 0});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s2,
                          XlaCompilationCache::BuildSignature(fn, args));

  args[0].shape = TensorShape({0, 4});
  args[0].constant_value = Tensor(DT_FLOAT, {0, 4});
  TF_ASSERT_OK_AND_ASSIGN(XlaCompilationCache::Signature s3,
                          XlaCompilationCache::BuildSignature(fn, args));

  std::vector<XlaCompilationCache::Signature> signatures = {s1, s2, s3};
  for (int i = 0; i < signatures.size(); ++i) {
    for (int j = 0; j < signatures.size(); ++j) {
      EXPECT_EQ(i == j, signatures[i] == signatures[j])
          << signatures[i].HumanString() << " " << signatures[j].HumanString();
    }
  }
}

TEST(XlaCompilationCacheTest, TestDisabledXlaCompilation) {
  NameAttrList fn;
  fn.set_name("afunction");

  DisableXlaCompilation();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  const XlaCompiler::CompilationResult* compilation_result;
  xla::LocalExecutable* executable;

  auto cache = new XlaCompilationCache(client, device_type);
  core::ScopedUnref cache_ref(cache);

  Status status = cache->Compile(XlaCompiler::Options{}, fn, {},
                                 XlaCompiler::CompileOptions{},
                                 XlaCompilationCache::CompileMode::kStrict,
                                 &compilation_result, &executable);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(), "XLA compilation disabled"));
}

void BM_BuildSignature(::testing::benchmark::State& state) {
  const int n_args = state.range(0);

  NameAttrList fn;
  fn.set_name("afunction");
  for (int i = 0; i < n_args; i++) {
    (*fn.mutable_attr())[absl::StrCat("T", i)].set_type(DT_FLOAT);
  }
  std::vector<XlaCompiler::Argument> args(n_args);
  for (int i = 0; i < n_args; i++) {
    args[i].kind = (((i % 3) == 0) ? XlaCompiler::Argument::kConstant
                                   : XlaCompiler::Argument::kParameter);
    args[i].type = DT_INT32;
    args[i].shape = TensorShape({4, 0});
    args[i].constant_value = Tensor(DT_INT32, {4, 0});
  }

  for (auto i : state) {
    StatusOr<XlaCompilationCache::Signature> s =
        XlaCompilationCache::BuildSignature(fn, args);
    CHECK(s.ok());
    XlaCompilationCache::Signature sig = std::move(s.ValueOrDie());
  }
}
BENCHMARK(BM_BuildSignature)->Arg(0)->Arg(1)->Arg(2)->Arg(5)->Arg(10);

}  // namespace
}  // namespace tensorflow
