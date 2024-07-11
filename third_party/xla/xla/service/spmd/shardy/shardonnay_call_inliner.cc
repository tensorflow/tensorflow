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

#include "xla/service/spmd/shardy/shardonnay_call_inliner.h"

#include <string>
#include <string_view>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/call_inliner.h"

namespace xla {

bool ShardonnayCallInliner::IsInlineableCallOp(
    HloInstruction* instruction) const {
  auto prefix_check = [](std::string_view name) {
    const std::string prefix = "shmap_body";
    return name.substr(0, prefix.size()) == prefix;
  };
  return CallInliner::IsInlineableCallOp(instruction) &&
         !instruction->has_backend_config() &&
         !(instruction->GetModule()
               ->config()
               .debug_options()
               .xla_use_shardonnay() &&
           prefix_check(instruction->to_apply()->name()));
}

}  // namespace xla
