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

#ifndef XLA_SERVICE_SPMD_SHARDY_SHARDY_CALL_INLINER_H_
#define XLA_SERVICE_SPMD_SHARDY_SHARDY_CALL_INLINER_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/call_inliner.h"

namespace xla {

// The same as CallInliner, except as part of
// go/jax-shmap -> `sdy.ManualComputationOp` importing, we require the pattern
// in MHLO:
// ```
// %shard_arg0_0 = custom_call @Sharding(%0)
// %shard_arg0_1 = custom_call @SPMDFullToShardShape(%shard_arg0_0)
// ...
// %shard_argN_0 = custom_call @Sharding(%N)
// %shard_argN_1 = custom_call @SPMDFullToShardShape(%shard_argN_0)
//
// %shard_result0, ..., %shard_resultN = func.call @shmap_body(%shard_arg0_1,
//                                                             ...,
//                                                             %shard_argN_1)
//
// %shard_result0_0 = custom_call @Sharding(%shard_result0)
// %shard_result0_1 = custom_call @SPMDShardToFullShape(%shard_result0_0)
// ...
// %shard_resultN_0 = custom_call @Sharding(%shard_resultN)
// %shard_resultN_1 = custom_call @SPMDShardToFullShape(%shard_resultN_0)
// ```
// We specifically match on the `func.call @shmap_body` since we want to inline
// the body of that function into the `ManualComputationOp` body. So this makes
// sure we inline all functions except for the shmap_body's when using
// Shardy. When Shardy is disabled, then we have the same behavior as
// CallInliner.
class ShardyCallInliner : public CallInliner {
 public:
  using CallInliner::CallInliner;
  absl::string_view name() const override { return "shardy-call-inliner"; }

  bool IsInlineableCallOp(HloInstruction* instruction) const override;
};

}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SHARDY_CALL_INLINER_H_
