/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DOT_MERGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DOT_MERGER_H_

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Merges dots that share an operand.  Transforms
//
//   x = dot(a, b)
//   y = dot(a, c)
//
// into
//
//   z = dot(a, concat(b, c))
//   x = slice(z)
//   y = slice(z).
//
// This requires that x and y are independent -- that is, x does not
// transitively depend on y, and y does not transitively depend on x.
//
// This is a good transformation if the merged dot runs faster than the original
// dots.  On the other hand, merging the dots results in a single result buffer
// z whose live range is the union of x and y's live ranges, so can lead to
// increased memory pressure.  You probably only want to do this optimization on
// "small" dots which cannot saturate your device when run alone.
//
// We thus allow backends to set a max size above which an op will not be
// merged.  The input+output bytes of at least one dot must be below the
// threshold otherwise we won't merge.  (We don't require that both dots be
// below the threshold because backends likely want to allow merging a "small"
// dot into a "large" dot while preventing two large dots from being merged.)
//
// Assumes DotDecomposer has already canonicalized the gemms and will skip
// noncanonical gemms.
class DotMerger : public HloModulePass {
 public:
  explicit DotMerger(int64_t max_size_to_merge)
      : max_size_to_merge_(max_size_to_merge) {}

  absl::string_view name() const override { return "dot-merger"; }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  int64_t max_size_to_merge_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DOT_MERGER_H_
