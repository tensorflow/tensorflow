/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_CUBLAS_PAD_FOR_GEMMS_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CUBLAS_PAD_FOR_GEMMS_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Adds padding to dot operations to make them run faster on GPUs.
//
//
// This can be used to pad f16 dots on tensor cores, or s8 dots to multiples of
// four.
//
// This pass depends on xla::DotDecomposer pass,
// so it should go strictly later.
class CublasPadForGemms : public HloModulePass {
 public:
  CublasPadForGemms(const se::GpuComputeCapability gpu_compute_capability,
                    PrimitiveType datatype, int32_t pad_to_multiple_of)
      : gpu_compute_capability_(gpu_compute_capability),
        datatype_(datatype),
        pad_to_multiple_of_(pad_to_multiple_of) {}

  absl::string_view name() const override { return "cublas-pad-for-gemms"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::GpuComputeCapability gpu_compute_capability_;
  PrimitiveType datatype_;
  int32_t pad_to_multiple_of_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CUBLAS_PAD_FOR_GEMMS_H_
