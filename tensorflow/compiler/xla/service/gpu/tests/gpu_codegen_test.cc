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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

std::unique_ptr<HloModule> GpuCodegenTest::CreateNewModuleWithFTZ(bool ftz) {
  HloModuleConfig config;
  auto debug_options = legacy_flags::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_ftz(ftz);
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  config.set_debug_options(debug_options);

  return MakeUnique<HloModule>(TestName(), config);
}

void GpuCodegenTest::CompileAndVerifyPtx(std::unique_ptr<HloModule> hlo_module,
                                         const string& pattern) {
  std::unique_ptr<Executable> executable =
      std::move(CompileToExecutable(std::move(hlo_module)).ValueOrDie());
  string ptx_str =
      std::string(static_cast<GpuExecutable*>(executable.get())->ptx());
  StatusOr<bool> filecheck_result = RunFileCheck(ptx_str, pattern);
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

}  // namespace gpu
}  // namespace xla
