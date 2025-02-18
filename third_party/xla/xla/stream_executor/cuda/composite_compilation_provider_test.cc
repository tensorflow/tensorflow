/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/composite_compilation_provider.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/mock_compilation_provider.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::cuda {

namespace {

using ::testing::HasSubstr;
using ::testing::Return;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(CompositeCompilationProviderTest, CreateFailsWithNoProviders) {
  EXPECT_THAT(CompositeCompilationProvider::Create({}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeCompilationProviderTest, Name) {
  auto provider0 = std::make_unique<MockCompilationProvider>();
  auto provider1 = std::make_unique<MockCompilationProvider>();
  auto provider2 = std::make_unique<MockCompilationProvider>();
  EXPECT_CALL(*provider0, name())
      .WillRepeatedly(Return("MockCompilationProvider0"));
  EXPECT_CALL(*provider1, name())
      .WillRepeatedly(Return("MockCompilationProvider1"));
  EXPECT_CALL(*provider2, name())
      .WillRepeatedly(Return("MockCompilationProvider2"));
  std::vector<std::unique_ptr<CompilationProvider>> providers;
  providers.push_back(std::move(provider0));
  providers.push_back(std::move(provider1));
  providers.push_back(std::move(provider2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto combining_provider,
      CompositeCompilationProvider::Create(std::move(providers)));

  EXPECT_THAT(combining_provider->name(),
              HasSubstr("CompositeCompilationProvider"));
  EXPECT_THAT(combining_provider->name(),
              HasSubstr("MockCompilationProvider0"));
  EXPECT_THAT(combining_provider->name(),
              HasSubstr("MockCompilationProvider1"));
  EXPECT_THAT(combining_provider->name(),
              HasSubstr("MockCompilationProvider2"));
}

TEST(CompositeCompilationProviderTest, SupportsCompileToRelocatableModule) {
  auto provider0 = std::make_unique<MockCompilationProvider>();
  auto provider1 = std::make_unique<MockCompilationProvider>();
  auto provider2 = std::make_unique<MockCompilationProvider>();
  EXPECT_CALL(*provider0, SupportsCompileToRelocatableModule())
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*provider1, SupportsCompileToRelocatableModule())
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*provider2, SupportsCompileToRelocatableModule())
      .WillRepeatedly(Return(false));
  std::vector<std::unique_ptr<CompilationProvider>> providers;
  providers.push_back(std::move(provider0));
  providers.push_back(std::move(provider1));
  providers.push_back(std::move(provider2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto combining_provider,
      CompositeCompilationProvider::Create(std::move(providers)));
  EXPECT_TRUE(combining_provider->SupportsCompileToRelocatableModule());
}

TEST(CompositeCompilationProviderTest, SupportsCompileAndLink) {
  auto provider0 = std::make_unique<MockCompilationProvider>();
  auto provider1 = std::make_unique<MockCompilationProvider>();
  auto provider2 = std::make_unique<MockCompilationProvider>();
  EXPECT_CALL(*provider0, SupportsCompileAndLink())
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*provider1, SupportsCompileAndLink())
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*provider2, SupportsCompileAndLink())
      .WillRepeatedly(Return(false));
  std::vector<std::unique_ptr<CompilationProvider>> providers;
  providers.push_back(std::move(provider0));
  providers.push_back(std::move(provider1));
  providers.push_back(std::move(provider2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto combining_provider,
      CompositeCompilationProvider::Create(std::move(providers)));
  EXPECT_TRUE(combining_provider->SupportsCompileAndLink());
}

TEST(CompositeCompilationProviderTest, Compile) {
  auto provider0 = std::make_unique<MockCompilationProvider>();
  auto provider1 = std::make_unique<MockCompilationProvider>();
  auto provider2 = std::make_unique<MockCompilationProvider>();
  EXPECT_CALL(*provider0, Compile).WillOnce(Return(Assembly{}));
  EXPECT_CALL(*provider1, Compile).Times(0);
  EXPECT_CALL(*provider2, Compile).Times(0);
  std::vector<std::unique_ptr<CompilationProvider>> providers;
  providers.push_back(std::move(provider0));
  providers.push_back(std::move(provider1));
  providers.push_back(std::move(provider2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto combining_provider,
      CompositeCompilationProvider::Create(std::move(providers)));
  EXPECT_THAT(combining_provider->Compile(CudaComputeCapability{10, 0}, "ptx",
                                          CompilationOptions{}),
              IsOk());
}

TEST(CompositeCompilationProviderTest, CompileToRelocatableModule) {
  auto provider0 = std::make_unique<MockCompilationProvider>();
  auto provider1 = std::make_unique<MockCompilationProvider>();
  auto provider2 = std::make_unique<MockCompilationProvider>();

  // Provider 0 gets asked whether it can produce relocatable modules, but
  // can't, so 'CompileToRelocatableModule' doesn't get called.
  EXPECT_CALL(*provider0, CompileToRelocatableModule).Times(0);
  EXPECT_CALL(*provider0, SupportsCompileToRelocatableModule)
      .WillOnce(Return(false));

  // Provider 1 can produce relocatable modules, so it gets asked to compile.
  EXPECT_CALL(*provider1, CompileToRelocatableModule)
      .WillOnce(Return(RelocatableModule{}));
  EXPECT_CALL(*provider1, SupportsCompileToRelocatableModule)
      .WillOnce(Return(true));

  // Provider 2 doesn't even get bothered because provider 1 can produce
  // relocatable modules.
  EXPECT_CALL(*provider2, CompileToRelocatableModule).Times(0);
  EXPECT_CALL(*provider2, SupportsCompileToRelocatableModule).Times(0);

  std::vector<std::unique_ptr<CompilationProvider>> providers;
  providers.push_back(std::move(provider0));
  providers.push_back(std::move(provider1));
  providers.push_back(std::move(provider2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto combining_provider,
      CompositeCompilationProvider::Create(std::move(providers)));
  EXPECT_THAT(combining_provider->CompileToRelocatableModule(
                  CudaComputeCapability{10, 0}, "ptx", CompilationOptions{}),
              IsOk());
}

TEST(CompositeCompilationProviderTest, CompileAndLink) {
  auto provider0 = std::make_unique<MockCompilationProvider>();
  auto provider1 = std::make_unique<MockCompilationProvider>();
  auto provider2 = std::make_unique<MockCompilationProvider>();

  // Provider 0 gets asked whether it can link, but
  // can't, so 'CompileAndLink' doesn't get called.
  EXPECT_CALL(*provider0, CompileAndLink).Times(0);
  EXPECT_CALL(*provider0, SupportsCompileAndLink).WillOnce(Return(false));

  // Provider 1 can link, so 'CompileAndLink' gets called.
  EXPECT_CALL(*provider1, CompileAndLink).WillOnce(Return(Assembly{}));
  EXPECT_CALL(*provider1, SupportsCompileAndLink).WillOnce(Return(true));

  // Provider 2 doesn't even get bothered because provider 1 can handle the
  // request.
  EXPECT_CALL(*provider2, CompileAndLink).Times(0);
  EXPECT_CALL(*provider2, SupportsCompileAndLink).Times(0);

  std::vector<std::unique_ptr<CompilationProvider>> providers;
  providers.push_back(std::move(provider0));
  providers.push_back(std::move(provider1));
  providers.push_back(std::move(provider2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto combining_provider,
      CompositeCompilationProvider::Create(std::move(providers)));
  EXPECT_THAT(
      combining_provider->CompileAndLink(CudaComputeCapability{10, 0},
                                         {Ptx{"ptx"}}, CompilationOptions{}),
      IsOk());
}

}  // namespace
}  // namespace stream_executor::cuda
