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

#include "tensorflow/compiler/jit/device_compilation_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
struct FakeExecutable {
  std::string data;
  explicit FakeExecutable(const std::string& s) : data(s) {}
};

using Cache = DeviceCompilationCache<FakeExecutable>;
using Signature = DeviceCompilationClusterSignature;

absl::StatusOr<Signature> BuildSampleSignature(const std::string& fn_name) {
  NameAttrList fn;
  fn.set_name(fn_name);
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({4, 0});
  args[0].constant_value = Tensor(DT_INT32, {4, 0});
  return Signature::Build(fn, args);
}

TEST(DeviceCompilationCacheTest, LookupEntryDoesntExist) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));
  auto cache_value = cache->Lookup(key);

  EXPECT_FALSE(cache_value.has_value());
}

TEST(DeviceCompilationCacheTest, LookupOrCreateEntryDoesntExist) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));
  Cache::Value cache_value = cache->LookupOrCreate(key);

  EXPECT_EQ(cache_value.compile_state, DeviceCompileState::kUncompiled);
  EXPECT_EQ(cache_value.request_count, 1);
  EXPECT_EQ(cache_value.compilation_result, nullptr);
  EXPECT_EQ(cache_value.executable, nullptr);
}

TEST(DeviceCompilationCacheTest, IncrementRequestCountOnLookup) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));
  Cache::Value cache_value = cache->LookupOrCreate(key);
  EXPECT_EQ(cache_value.request_count, 1);

  cache_value = cache->LookupOrCreate(key);
  EXPECT_EQ(cache_value.request_count, 2);

  cache_value = cache->LookupOrCreate(key);
  EXPECT_EQ(cache_value.request_count, 3);
}

TEST(DeviceCompilationCacheTest, RequestCountUnchangedOnStore) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));
  Cache::Value cache_value = cache->LookupOrCreate(key);
  EXPECT_EQ(cache_value.request_count, 1);

  cache_value = cache->LookupOrCreate(key);
  EXPECT_EQ(cache_value.request_count, 2);

  cache_value = cache->LookupOrCreate(key);
  EXPECT_EQ(cache_value.request_count, 3);

  auto compilation_result = std::make_unique<XlaCompiler::CompilationResult>();
  cache->Store(key, DeviceCompileState::kCompiled, absl::OkStatus(),
               std::move(compilation_result), std::nullopt);
  cache_value = cache->LookupOrCreate(key);

  EXPECT_EQ(cache_value.request_count, 4);
}

TEST(DeviceCompilationCacheTest, StoreLookup) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));
  auto compilation_result = std::make_unique<XlaCompiler::CompilationResult>();
  auto executable = std::make_unique<FakeExecutable>("foo_exe");
  cache->Store(key, DeviceCompileState::kCompiled, absl::OkStatus(),
               std::move(compilation_result), std::move(executable));
  auto cache_value = cache->Lookup(key);

  EXPECT_EQ(cache_value->compile_state, DeviceCompileState::kCompiled);
  EXPECT_EQ(cache_value->request_count, 1);
  EXPECT_TRUE(cache_value->compilation_status.ok());
  EXPECT_TRUE(cache_value->compilation_result != nullptr);
  EXPECT_TRUE(cache_value->executable != nullptr);
  EXPECT_EQ(cache_value->executable->data, "foo_exe");
}

TEST(DeviceCompilationCacheTest, StoreLookupOrCreate) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));
  auto compilation_result = std::make_unique<XlaCompiler::CompilationResult>();
  auto executable = std::make_unique<FakeExecutable>("foo_exe");
  cache->Store(key, DeviceCompileState::kCompiled, absl::OkStatus(),
               std::move(compilation_result), std::move(executable));
  auto cache_value = cache->LookupOrCreate(key);

  EXPECT_EQ(cache_value.compile_state, DeviceCompileState::kCompiled);
  EXPECT_EQ(cache_value.request_count, 1);
  EXPECT_TRUE(cache_value.compilation_status.ok());
  EXPECT_TRUE(cache_value.compilation_result != nullptr);
  EXPECT_TRUE(cache_value.executable != nullptr);
  EXPECT_EQ(cache_value.executable->data, "foo_exe");
}

TEST(DeviceCompilationCacheTest, StoreOptionalArgs) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key, BuildSampleSignature("foo"));

  auto compilation_result = std::make_unique<XlaCompiler::CompilationResult>();
  auto executable = std::make_unique<FakeExecutable>("foo_exe");

  cache->Store(key, DeviceCompileState::kCompiled, std::nullopt, std::nullopt,
               std::nullopt);
  auto cache_value = cache->Lookup(key);

  EXPECT_EQ(cache_value->compile_state, DeviceCompileState::kCompiled);
  EXPECT_TRUE(cache_value->compilation_status.ok());
  EXPECT_TRUE(cache_value->compilation_result == nullptr);
  EXPECT_TRUE(cache_value->executable == nullptr);

  cache->Store(key, std::nullopt, errors::InvalidArgument("Couldn't compile."),
               std::nullopt, std::nullopt);
  cache_value = cache->Lookup(key);

  EXPECT_EQ(cache_value->compile_state, DeviceCompileState::kCompiled);
  EXPECT_EQ(cache_value->compilation_status.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(cache_value->compilation_result == nullptr);
  EXPECT_TRUE(cache_value->executable == nullptr);

  cache->Store(key, std::nullopt, std::nullopt, std::move(compilation_result),
               std::nullopt);
  cache_value = cache->Lookup(key);

  EXPECT_EQ(cache_value->compile_state, DeviceCompileState::kCompiled);
  EXPECT_EQ(cache_value->compilation_status.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(cache_value->compilation_result != nullptr);
  EXPECT_TRUE(cache_value->executable == nullptr);

  cache->Store(key, std::nullopt, std::nullopt, std::nullopt,
               std::move(executable));
  cache_value = cache->Lookup(key);

  EXPECT_EQ(cache_value->compile_state, DeviceCompileState::kCompiled);
  EXPECT_EQ(cache_value->compilation_status.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(cache_value->compilation_result != nullptr);
  EXPECT_TRUE(cache_value->executable != nullptr);
  EXPECT_EQ(cache_value->executable->data, "foo_exe");
}

TEST(DeviceCompilationCacheTest, StoreMultipleEntries) {
  auto cache = std::make_unique<Cache>();

  TF_ASSERT_OK_AND_ASSIGN(auto key1, BuildSampleSignature("foo"));
  TF_ASSERT_OK_AND_ASSIGN(auto key2, BuildSampleSignature("bar"));

  auto compilation_result1 = std::make_unique<XlaCompiler::CompilationResult>();
  auto compilation_result2 = std::make_unique<XlaCompiler::CompilationResult>();
  auto executable1 = std::make_unique<FakeExecutable>("foo_exe");
  auto executable2 = std::make_unique<FakeExecutable>("bar_exe");
  cache->Store(key1, DeviceCompileState::kCompiled,
               errors::InvalidArgument("Invalid argument."),
               std::move(compilation_result1), std::move(executable1));
  cache->Store(key2, DeviceCompileState::kCompiling, absl::OkStatus(),
               std::move(compilation_result2), std::move(executable2));
  auto cache_value_1 = cache->Lookup(key1);
  auto cache_value_2 = cache->Lookup(key2);

  EXPECT_EQ(cache_value_1->compile_state, DeviceCompileState::kCompiled);
  EXPECT_EQ(cache_value_1->compilation_status.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(cache_value_1->compilation_result != nullptr);
  EXPECT_TRUE(cache_value_1->executable != nullptr);
  EXPECT_EQ(cache_value_1->executable->data, "foo_exe");

  EXPECT_EQ(cache_value_2->compile_state, DeviceCompileState::kCompiling);
  EXPECT_TRUE(cache_value_2->compilation_status.ok());
  EXPECT_TRUE(cache_value_2->compilation_result != nullptr);
  EXPECT_TRUE(cache_value_2->executable != nullptr);
  EXPECT_EQ(cache_value_2->executable->data, "bar_exe");
}

}  // namespace
}  // namespace tensorflow
