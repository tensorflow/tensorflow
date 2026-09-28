/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

namespace xla {
namespace gpu {

using ::testing::Contains;

namespace {

std::vector<std::string> GetPassNames(const mlir::PassManager& pm) {
  std::vector<std::string> pass_names;
  for (const mlir::Pass& pass : pm.getPasses()) {
    pass_names.push_back(pass.getName().str());
  }
  return pass_names;
}

}  // namespace

TEST(CompilationPipelineTest, ContainsUnswitchLoopsCompositePass) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  CreateTritonXlaPipeline(&pm, stream_executor::CudaComputeCapability(),
                          /*rewrite_int4=*/false, /*allow_tma=*/true,
                          /*num_stages=*/1,
                          /*warp_specialization_allowed=*/true,
                          /*enable_pdl=*/false);

  std::vector<std::string> pass_names = GetPassNames(pm);
  ASSERT_THAT(pass_names, Contains("TritonXLAUnswitchLoopsComposite"));

  std::string pipeline_str;
  llvm::raw_string_ostream os(pipeline_str);
  pm.printAsTextualPipeline(os);
  EXPECT_THAT(pipeline_str, ::testing::HasSubstr("xla-simplify-arith"));
}

TEST(CompilationPipelineTest, OneApiPipelineDispatchesCorrectly) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  CreateTritonPipeline(&pm,
                       stream_executor::GpuComputeCapability(
                           stream_executor::OneAPIComputeCapability::BMG()),
                       /*num_warps=*/4, /*num_ctas=*/1, /*num_stages=*/2);

  std::vector<std::string> pass_names = GetPassNames(pm);
  if (absl::AsciiStrToUpper(
          PlatformUtil::CanonicalPlatformName("gpu").value()) == "SYCL") {
    ASSERT_THAT(pass_names, Contains("TritonIntelGPUPipeline"));
  } else {
    EXPECT_TRUE(pass_names.empty());
  }
}

}  // namespace gpu
}  // namespace xla
