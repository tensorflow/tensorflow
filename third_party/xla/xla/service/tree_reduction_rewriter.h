/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_TREE_REDUCTION_REWRITER_H_
#define XLA_SERVICE_TREE_REDUCTION_REWRITER_H_

#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// Increase precision for the reduction operation by applying the reduce-window
// first.
//
// E.g. suppose we want to reduce f32[1024] to a scalar. This pass first applies
// a reduce-window (with kSame padding) of size `reduce_window_size`, and then
// reduces the resulting array f32[32]. The rewrite is not applied if any of the
// reduced dimensions is smaller than the `reduce_window_size`.
//
// Applying this pass until a fixed point performs a variant of pairwise
// summation (https://en.wikipedia.org/wiki/Pairwise_summation), which is
// guaranteed to have an asymptotically smaller error bound provided that
// intermediate roundoff errors are random and have random sign.
//
// If this pass lowers the performance too much, the window size can always be
// increased to a larger value.
class TreeReductionRewriter : public HloModulePass {
 public:
  explicit TreeReductionRewriter(int64_t reduce_window_size = 32)
      : reduce_window_size_(reduce_window_size) {}
  ~TreeReductionRewriter() override = default;
  absl::string_view name() const override { return "tree_reduction_rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t reduce_window_size_;
};

}  // end namespace xla

#endif  // XLA_SERVICE_TREE_REDUCTION_REWRITER_H_
