/* Copyright 2026 The OpenXLA Authors.

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

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/mock_compilation_provider.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

struct PtxTarget {
  int ptx_isa_version;
  se::CudaComputeCapability compute_capability;
};

enum class Relocatable { kDisabled, kEnabled };

class NVPTXCompilerTargetTest
    : public ::testing::TestWithParam<std::tuple<PtxTarget, Relocatable>> {
 protected:
  void SetUp() override {
    module_config_.mutable_debug_options().set_xla_gpu_cuda_data_dir(
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service/gpu",
                          "nvptx_compiler_target_test_cuda"));
    device_description_.set_cuda_compute_capability(
        {10, 7,
         se::CudaComputeCapability::FeatureExtension::kAcceleratedFeatures});
    auto provider =
        std::make_unique<NiceMock<se::cuda::MockCompilationProvider>>();
    provider_ = provider.get();
    ON_CALL(*provider_, GetLatestPtxIsaVersion())
        .WillByDefault(Return(std::get<PtxTarget>(GetParam()).ptx_isa_version));
    compiler_ = std::make_unique<NVPTXCompiler>(
        [provider = std::move(provider)](
            const se::cuda::CompilationProviderOptions&) mutable
            -> absl::StatusOr<std::unique_ptr<se::cuda::CompilationProvider>> {
          return std::move(provider);
        });
  }

  std::unique_ptr<NVPTXCompiler> compiler_;
  HloModuleConfig module_config_;
  se::DeviceDescription device_description_;
  se::cuda::MockCompilationProvider* provider_;
};

TEST_P(NVPTXCompilerTargetTest, CompilationAndLinkingUseSameTarget) {
  se::CudaComputeCapability expected_cc =
      std::get<PtxTarget>(GetParam()).compute_capability;
  auto expected_ptx = HasSubstr(absl::StrCat(
      ".target ", expected_cc.GetPtxAsTargetName(
                      se::CudaComputeCapability::CompileMode::kSass)));

  // Linking must also work before the LLVM backend has compiled any PTX.
  EXPECT_CALL(*provider_, CompileAndLink(expected_cc, _, _))
      .WillOnce(Return(se::cuda::Assembly{}));
  EXPECT_OK(compiler_->LinkModules(device_description_, {{0}},
                                   module_config_.debug_options(), nullptr));

  llvm::LLVMContext context;
  llvm::Module module("test", context);
  new llvm::GlobalVariable(
      module, llvm::Type::getInt32Ty(context), true,
      llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0), "gv");
  Relocatable relocatable = std::get<Relocatable>(GetParam());
  if (relocatable == Relocatable::kEnabled) {
    EXPECT_CALL(*provider_,
                CompileToRelocatableModule(expected_cc, expected_ptx, _))
        .WillOnce(Return(se::cuda::RelocatableModule{}));
  } else {
    EXPECT_CALL(*provider_, Compile(expected_cc, expected_ptx, _))
        .WillOnce(Return(se::cuda::Assembly{}));
  }
  EXPECT_OK(compiler_->CompileTargetBinary(
      module_config_, &module, device_description_,
      relocatable == Relocatable::kEnabled, nullptr, std::nullopt));
}

using FeatureExtension = se::CudaComputeCapability::FeatureExtension;

INSTANTIATE_TEST_SUITE_P(
    ProviderPtxVersion, NVPTXCompilerTargetTest,
    ::testing::Combine(
        ::testing::Values(
            PtxTarget{90, {10, 3, FeatureExtension::kFamilyCompatibleFeatures}},
            PtxTarget{94, {10, 7, FeatureExtension::kAcceleratedFeatures}}),
        ::testing::Values(Relocatable::kDisabled, Relocatable::kEnabled)));

}  // namespace
}  // namespace xla::gpu
