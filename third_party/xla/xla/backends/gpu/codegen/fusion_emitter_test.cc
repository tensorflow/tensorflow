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

#include "xla/backends/gpu/codegen/fusion_emitter.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

llvm::Function* CreateTestKernel(llvm::Module* module,
                                 const std::string& triple) {
  module->setTargetTriple(llvm::Triple(triple));
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(module->getContext()),
                              /*isVarArg=*/false);
  return llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage,
                                "test_kernel", module);
}

TEST(AnnotateKernelLaunchDimensionsTest, AnnotatesNvptxWithoutClusterDims) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "nvptx64-nvidia-cuda");
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();

  LaunchDimensions dims(se::BlockDim(8, 4, 1), se::ThreadDim(128, 2, 1));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      IsOk());
  EXPECT_EQ(kernel->getFnAttribute("nvvm.reqntid").getValueAsString().str(),
            "128,2,1");
  EXPECT_FALSE(kernel->hasFnAttribute("nvvm.cluster_dim"));
}

TEST(AnnotateKernelLaunchDimensionsTest, AnnotatesNvptxWithValidClusterDims) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "nvptx64-nvidia-cuda");
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();

  LaunchDimensions dims(se::BlockDim(8, 4, 1), se::ThreadDim(128, 1, 1),
                        se::ClusterDim(2, 2, 1));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      IsOk());
  EXPECT_EQ(kernel->getFnAttribute("nvvm.reqntid").getValueAsString().str(),
            "128,1,1");
  EXPECT_EQ(kernel->getFnAttribute("nvvm.cluster_dim").getValueAsString().str(),
            "2,2,1");
}

TEST(AnnotateKernelLaunchDimensionsTest, AnnotatesAmdgpuFlatWorkGroupSize) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "amdgcn-amd-amdhsa");
  se::DeviceDescription device_info = TestGpuDeviceInfo::AMDMI300DeviceInfo();

  // The flat work group size is the total number of threads per block
  // (16 * 8 * 2), independent of the block counts.
  LaunchDimensions dims(se::BlockDim(8, 4, 1), se::ThreadDim(16, 8, 2));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      IsOk());
  // Our launch bounds are exact, so min and max are equal.
  EXPECT_EQ(kernel->getFnAttribute("amdgpu-flat-work-group-size")
                .getValueAsString()
                .str(),
            "256,256");
}

TEST(AnnotateKernelLaunchDimensionsTest, RejectsClusterDimsOnNonNvptxTarget) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "amdgcn-amd-amdhsa");
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();

  LaunchDimensions dims(se::BlockDim(8, 1, 1), se::ThreadDim(128, 1, 1),
                        se::ClusterDim(2, 1, 1));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("non-NVPTX")));
}

TEST(AnnotateKernelLaunchDimensionsTest,
     RejectsClusterDimsOnPreHopperCudaTarget) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "nvptx64-nvidia-cuda");
  se::DeviceDescription device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();

  LaunchDimensions dims(se::BlockDim(8, 1, 1), se::ThreadDim(128, 1, 1),
                        se::ClusterDim(2, 1, 1));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("pre-Hopper")));
}

TEST(AnnotateKernelLaunchDimensionsTest, RejectsNonPositiveClusterDims) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "nvptx64-nvidia-cuda");
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();

  LaunchDimensions dims(se::BlockDim(8, 1, 1), se::ThreadDim(128, 1, 1),
                        se::ClusterDim(0, 1, 1));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("non-positive")));
}

TEST(AnnotateKernelLaunchDimensionsTest,
     RejectsBlockCountsNotDivisibleByClusterDims) {
  llvm::LLVMContext context;
  llvm::Module module("test_module", context);
  llvm::Function* kernel = CreateTestKernel(&module, "nvptx64-nvidia-cuda");
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();

  LaunchDimensions dims(se::BlockDim(6, 1, 1), se::ThreadDim(128, 1, 1),
                        se::ClusterDim(4, 1, 1));
  EXPECT_THAT(
      AnnotateKernelLaunchDimensions(device_info, dims, kernel, &module),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("divisible")));
}

}  // namespace
}  // namespace xla::gpu
