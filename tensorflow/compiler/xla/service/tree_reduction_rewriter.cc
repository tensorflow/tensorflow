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

#include "tensorflow/compiler/xla/service/tree_reduction_rewriter.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(int64_t reduce_window_size)
      : reduce_window_size_(reduce_window_size) {}

  Status HandleReduce(HloInstruction *hlo) override {
    HloInstruction *reduced_op = hlo->mutable_operand(0);
    HloInstruction *initial_value = hlo->mutable_operand(1);
    const Shape &input_shape = reduced_op->shape();
    const Shape &reduce_shape = hlo->shape();

    if (!reduce_shape.IsArray()) {
      // TODO(b/210786051): Implement tree reduction rewrite for variadic
      // reductions on CPU as well.
      VLOG(1) << "Skipping rewrite for variadic reduction";
      return OkStatus();
    }

    // All of the reduced dimensions is smaller than the window size,
    // do not perform the rewrite.
    if (absl::c_all_of(hlo->dimensions(), [&](int64_t reduced_dim) {
          return input_shape.dimensions(reduced_dim) <= reduce_window_size_;
        })) {
      VLOG(1) << "Skipping tree reduction rewrite: all reduced dimensions are "
                 "smaller than "
              << reduce_window_size_;
      return OkStatus();
    }

    std::vector<int64_t> window_dimensions;
    std::vector<int64_t> window_strides;
    for (int64_t dim_idx = 0; dim_idx < input_shape.rank(); dim_idx++) {
      if (!absl::c_linear_search(hlo->dimensions(), dim_idx)) {
        window_dimensions.push_back(1);
        window_strides.push_back(1);
        continue;
      }

      int64_t window_size_for_dim =
          std::min(input_shape.dimensions(dim_idx), reduce_window_size_);

      window_dimensions.push_back(window_size_for_dim);
      window_strides.push_back(window_size_for_dim);
    }

    std::vector<std::pair<int64_t, int64_t>> padding =
        MakePadding(input_shape.dimensions(), window_dimensions, window_strides,
                    Padding::kSame);

    TF_ASSIGN_OR_RETURN(
        Window window, ShapeInference::InferWindowFromDimensions(
                           window_dimensions, window_strides, padding, {}, {}));

    TF_ASSIGN_OR_RETURN(Shape intermediate_shape,
                        ShapeInference::InferReduceWindowShape(
                            input_shape, initial_value->shape(), window));

    HloInstruction *reduce_window =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduceWindow(
            intermediate_shape, reduced_op, initial_value, window,
            hlo->to_apply()));

    std::unique_ptr<HloInstruction> new_output =
        HloInstruction::CreateReduce(reduce_shape, reduce_window, initial_value,
                                     hlo->dimensions(), hlo->to_apply());

    return ReplaceWithNewInstruction(hlo, std::move(new_output));
  }

 private:
  int64_t reduce_window_size_;
};

StatusOr<bool> TreeReductionRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  ReductionRewriterVisitor visitor(reduce_window_size_);
  bool changed = false;
  for (const auto &computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }

  return changed;
}

}  // end namespace xla
