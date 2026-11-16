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

#ifndef XLA_SERVICE_SPMD_SHARDY_SHARDY_MANUAL_COMPUTATION_INLINER_H_
#define XLA_SERVICE_SPMD_SHARDY_SHARDY_MANUAL_COMPUTATION_INLINER_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/call_inliner.h"

namespace xla::sdy {

// Inlines Shardy manual-computation body functions left outlined by
// `ShardMapExportPass` (`kInlineableManualComputationFuncName`) while leaving
// all other calls untouched.
class ShardyManualComputationInliner : public CallInliner {
 public:
  absl::string_view name() const override {
    return "shardy-manual-computation-inliner";
  }

 protected:
  bool IsInlineableCallOp(HloInstruction* instruction) const override;
};

}  // namespace xla::sdy

#endif  // XLA_SERVICE_SPMD_SHARDY_SHARDY_MANUAL_COMPUTATION_INLINER_H_
