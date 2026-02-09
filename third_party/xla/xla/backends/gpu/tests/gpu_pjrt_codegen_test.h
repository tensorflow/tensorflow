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

#ifndef XLA_BACKENDS_GPU_TESTS_GPU_PJRT_CODEGEN_TEST_H_
#define XLA_BACKENDS_GPU_TESTS_GPU_PJRT_CODEGEN_TEST_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {

// Tests that verify IR or PTX emitted by the GPU backend is as expected.
class GpuPjRtCodegenTest : public HloPjRtGpuTestBase {
 public:
  GpuPjRtCodegenTest() {
    is_built_with_rocm_ =
        device_description().gpu_compute_capability().IsRocm();
    compile_options_.gpu_topology = GetSingleDeviceGpuTopology(
        /*platform_version=*/"", gpu_target_config());
    compile_options_.early_exit_with_layouts = false;
  }

 protected:
  // Converts LLVM match to be platform-specific.
  std::string MakePlatformSpecificLlvm(absl::string_view input);

  // Like HloHardwareIndependentTestBase::CreateNewVerifiedModule(), with a flag
  // for configuring the ftz option.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModuleWithFTZ(bool ftz);

  // Compiles the given HLO module to PTX and verifies the PTX matches the given
  // FileCheck pattern.  (See http://llvm.org/docs/CommandGuide/FileCheck.html).
  // The "VerifyPtx" part only happens on the CUDA platform,
  // and hence the "Optionally" in function name.
  // For ROCm platform this routine will only do the "Compile" part.
  void CompileAndOptionallyVerifyPtx(
      std::unique_ptr<VerifiedHloModule> hlo_module, absl::string_view pattern,
      bool run_optimization_passes = true);

  // Compiles `hlo_module` with the GPU compiler. If `run_optimization_passes`
  // is true, also the HLO optimization pass pipeline is run.
  absl::StatusOr<std::unique_ptr<Executable>> CompileToExecutable(
      std::unique_ptr<HloModule> hlo_module, bool run_optimization_passes);

  // A thin wrapper around CompileAndVerifyIr that parses `hlo_text` to create
  // an HLO module.
  absl::Status CompileAndVerifyIr(absl::string_view hlo_text,
                                  absl::string_view expected_llvm_ir,
                                  bool match_optimized_ir = false,
                                  bool run_optimization_passes = true);

  bool is_built_with_rocm_{false};

 private:
  Compiler::CompileOptions compile_options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TESTS_GPU_PJRT_CODEGEN_TEST_H_
