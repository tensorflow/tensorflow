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

#ifndef XLA_BACKENDS_CPU_AUTOTUNER_LLVM_KERNEL_AUTOTUNER_H_
#define XLA_BACKENDS_CPU_AUTOTUNER_LLVM_KERNEL_AUTOTUNER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

inline constexpr absl::string_view kLlvmKernelAutotunerName =
    "llvm_kernel_autotuner";

// Llvm kernel autotuning pass. It tries to autotune the llvm kernel compilation
// provided by the LlvmKernelBackend.
class LlvmKernelAutotuner : public HloModulePass {
 public:
  LlvmKernelAutotuner() = default;

  absl::string_view name() const override { return kLlvmKernelAutotunerName; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_AUTOTUNER_LLVM_KERNEL_AUTOTUNER_H_
