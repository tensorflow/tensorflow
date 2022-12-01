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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_GPU_CODEGEN_TEST_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_GPU_CODEGEN_TEST_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

namespace xla {
namespace gpu {

// Tests that verify IR or PTX emitted by the GPU backend is as expected.
class GpuCodegenTest : public LlvmIrGenTestBase {
 public:
  GpuCodegenTest()
      : is_built_with_rocm_(
            se::MultiPlatformManager::PlatformWithName("ROCM").ok()) {}

 protected:
  // Converts LLVM match to be platform-specific.
  std::string MakePlatformSpecificLlvm(absl::string_view input);

  // Like HloTestBase::CreateNewVerifiedModule(), with a flag for configuring
  // the ftz option.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModuleWithFTZ(bool ftz);

  // Compiles the given HLO module to PTX and verifies the PTX matches the given
  // FileCheck pattern.  (See http://llvm.org/docs/CommandGuide/FileCheck.html).
  // The "VerifyPtx" part only happens on the CUDA platform,
  // and hence the "Optionally" in function name.
  // For ROCm platform this routine will only do the "Compile" part.
  void CompileAndOptionallyVerifyPtx(
      std::unique_ptr<VerifiedHloModule> hlo_module, absl::string_view pattern);

  bool is_built_with_rocm_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TESTS_GPU_CODEGEN_TEST_H_
