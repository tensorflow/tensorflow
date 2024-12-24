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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_REWRITER_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Pattern matches (slice(s) + custom call) to custom address computation
// fusions and rewrites them into fusion instructions and fusion computations.
//
// Example:
//
//  ENTRY %main {
//    %p0 = bf16[2,8,8]{2,1,0} parameter(0)
//    %p1 = bf16[2,8,8]{2,1,0} parameter(1)
//    %slice_lhs = bf16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
//    %bitcast_lhs = bf16[8,8]{1,0} bitcast(%slice_lhs)
//    %slice_rhs = bf16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
//    %bitcast_rhs = bf16[8,8]{1,0} bitcast(%slice_rhs)
//    ROOT %dot = bf16[8,8]{1,0} custom-call(%bitcast_lhs, %bitcast_rhs),
//      custom_call_target="__cublas$gemm"
//  }
//
// After the pass:
//
//  %address_computation {
//    %p0 = bf16[2,8,8]{2,1,0} parameter(0)
//    %p1 = bf16[2,8,8]{2,1,0} parameter(1)
//    %slice_lhs = bf16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
//    %bitcast_lhs = bf16[8,8]{1,0} bitcast(%slice_lhs)
//    %slice_rhs = bf16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
//    %bitcast_rhs = bf16[8,8]{1,0} bitcast(%slice_rhs)
//    ROOT %dot = bf16[8,8]{1,0} custom-call(%bitcast_lhs, %bitcast_rhs),
//      custom_call_target="__cublas$gemm"
//  }
//
//  ENTRY %main {
//    %p0 = bf16[2,8,8]{2,1,0} parameter(0)
//    %p1 = bf16[2,8,8]{2,1,0} parameter(1)
//    ROOT %fusion.2 = bf16[8,8]{1,0} fusion(%p0, %p1),
//        kind=kCustom, calls=%address_computation,
//        backend_config={"fusion_backend_config":{
//            "kind":"__custom_fusion",
//            "custom_fusion_config":{"name":"address_computation"}
//        }}
//  }
//
class DynamicSliceFusionRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "dynamic-slice-fusion-rewriter";
  }

  explicit DynamicSliceFusionRewriter(std::string platform_name)
      : platform_name_(std::move(platform_name)) {}

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::string platform_name_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_REWRITER_H_
