/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_HLO_TRANSFORMS_MOVE_COPY_TO_USERS_H_
#define XLA_BACKENDS_GPU_HLO_TRANSFORMS_MOVE_COPY_TO_USERS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Sink kCopy operations as far down the graph as possible.
//
// A kCopy of an element of the initial state of a while loop is moved into the
// body of the loop if the body passes the element on unchanged, the element is
// only read by slices, in the body and after the loop, and the slices of all
// iterations and the slices after the loop are in total not larger than twice
// the element. The loop then carries the element in the layout it has before
// the copy, and only the slices are copied.
class MoveCopyToUsers : public HloModulePass {
 public:
  absl::string_view name() const override { return "move_copy_to_users"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // end namespace xla

#endif  // XLA_BACKENDS_GPU_HLO_TRANSFORMS_MOVE_COPY_TO_USERS_H_
