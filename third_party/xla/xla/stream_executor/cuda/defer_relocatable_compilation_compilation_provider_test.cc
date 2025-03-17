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

#include "xla/stream_executor/cuda/defer_relocatable_compilation_compilation_provider.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/mock_compilation_provider.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::cuda {
namespace {

using ::testing::Args;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::Return;
using ::testing::VariantWith;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

TEST(DeferRelocatableCompilationCompilationProviderTest,
     CreateFailsIfDelegateDoesNotSupportCompileAndLink) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  ON_CALL(*mock_compilation_provider, SupportsCompileAndLink())
      .WillByDefault(Return(false));
  ON_CALL(*mock_compilation_provider, SupportsCompileToRelocatableModule())
      .WillByDefault(Return(false));
  EXPECT_THAT(DeferRelocatableCompilationCompilationProvider::Create(
                  std::move(mock_compilation_provider)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DeferRelocatableCompilationCompilationProviderTest,
     CreateFailsIfDelegateSupportsCompileToRelocatableModule) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  ON_CALL(*mock_compilation_provider, SupportsCompileAndLink())
      .WillByDefault(Return(true));
  ON_CALL(*mock_compilation_provider, SupportsCompileToRelocatableModule())
      .WillByDefault(Return(true));
  EXPECT_THAT(DeferRelocatableCompilationCompilationProvider::Create(
                  std::move(mock_compilation_provider)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

constexpr absl::string_view kSomePtxString = "some ptx string";
constexpr absl::string_view kSomeOtherPtxString = "some other ptx string";
constexpr CudaComputeCapability kDefaultComputeCapability{10, 0};
constexpr CompilationOptions kDefaultCompilationOptions{};

TEST(DeferRelocatableCompilationCompilationProviderTest,
     CompileToRelocatableModuleNeverGetsCalledOnDelegate) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  ON_CALL(*mock_compilation_provider, SupportsCompileAndLink())
      .WillByDefault(Return(true));
  ON_CALL(*mock_compilation_provider, SupportsCompileToRelocatableModule())
      .WillByDefault(Return(false));
  EXPECT_CALL(*mock_compilation_provider, CompileToRelocatableModule).Times(0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto compilation_provider,
      DeferRelocatableCompilationCompilationProvider::Create(
          std::move(mock_compilation_provider)));

  EXPECT_THAT(compilation_provider->CompileToRelocatableModule(
                  kDefaultComputeCapability, kSomePtxString,
                  kDefaultCompilationOptions),
              IsOk());
}

TEST(DeferRelocatableCompilationCompilationProviderTest,
     DeferredPtxCompilationHappensInCompileAndLink) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  ON_CALL(*mock_compilation_provider, SupportsCompileAndLink())
      .WillByDefault(Return(true));
  ON_CALL(*mock_compilation_provider, SupportsCompileToRelocatableModule())
      .WillByDefault(Return(false));

  RelocatableModule some_actual_relocatable_module{{0x00, 0x01, 0x02}};

  // We expect to see the PTX string in the CompileAndLink call as the
  // compilation was deferred to the linking step.
  EXPECT_CALL(*mock_compilation_provider, CompileAndLink)
      .With(Args<1>(FieldsAre(ElementsAre(
          VariantWith<Ptx>(Field(&Ptx::ptx, kSomePtxString)),
          VariantWith<Ptx>(Field(&Ptx::ptx, kSomeOtherPtxString)),
          VariantWith<RelocatableModule>(some_actual_relocatable_module)))))
      .WillOnce(Return(Assembly{}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto compilation_provider,
      DeferRelocatableCompilationCompilationProvider::Create(
          std::move(mock_compilation_provider)));

  TF_ASSERT_OK_AND_ASSIGN(RelocatableModule opaque_relocatable_module,
                          compilation_provider->CompileToRelocatableModule(
                              kDefaultComputeCapability, kSomePtxString,
                              kDefaultCompilationOptions));

  // We pass in a RelocatableModule with deferred compilation (actually a PTX
  // string), a regular PTX string and an actual RelocatableModule. The latter
  // might have been produced by a different (compatible) compilation provider.
  // We expect that all three modules are supported and forwarded to the
  // delegate.
  EXPECT_THAT(compilation_provider->CompileAndLink(
                  kDefaultComputeCapability,
                  {std::move(opaque_relocatable_module),
                   Ptx{std::string(kSomeOtherPtxString)},
                   some_actual_relocatable_module},
                  kDefaultCompilationOptions),
              IsOk());
}

TEST(DeferRelocatableCompilationCompilationProviderTest,
     CompileGetsForwardedToDelegate) {
  auto mock_compilation_provider = std::make_unique<MockCompilationProvider>();
  ON_CALL(*mock_compilation_provider, SupportsCompileAndLink())
      .WillByDefault(Return(true));
  ON_CALL(*mock_compilation_provider, SupportsCompileToRelocatableModule())
      .WillByDefault(Return(false));

  // We expect to see the PTX string in the CompileAndLink call as the
  // compilation was deferred to the linking step.
  EXPECT_CALL(*mock_compilation_provider, Compile).WillOnce(Return(Assembly{}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto compilation_provider,
      DeferRelocatableCompilationCompilationProvider::Create(
          std::move(mock_compilation_provider)));

  EXPECT_THAT(
      compilation_provider->Compile(kDefaultComputeCapability, kSomePtxString,
                                    kDefaultCompilationOptions),
      IsOk());
}

}  // namespace
}  // namespace stream_executor::cuda
