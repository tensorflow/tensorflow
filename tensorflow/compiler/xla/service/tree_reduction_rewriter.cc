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

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(int64 reduce_window_size)
      : reduce_window_size_(reduce_window_size) {}

  Status HandleReduce(HloInstruction *hlo) override {
    HloInstruction *reduced_op = hlo->mutable_operand(0);
    HloInstruction *initial_value = hlo->mutable_operand(1);
    const Shape &input_shape = reduced_op->shape();
    const Shape &reduce_shape = hlo->shape();
    if (!reduce_shape.IsArray()) {
      return Status::OK();
    }
    auto reduced_dimensions = hlo->dimensions();
    std::vector<int64> window_dimensions;
    std::vector<int64> window_strides;
    for (int64 dim = 0; dim < input_shape.rank(); dim++) {
      if (!absl::c_linear_search(hlo->dimensions(), dim)) {
        window_dimensions.push_back(1);
        window_strides.push_back(1);
        continue;
      }
      // One of the reduced dimensions is smaller than the window size,
      // do not perform the rewrite.
      if (input_shape.dimensions(dim) < reduce_window_size_) {
        return Status::OK();
      }

      window_dimensions.push_back(reduce_window_size_);
      window_strides.push_back(reduce_window_size_);
    }

    std::vector<std::pair<int64, int64>> padding =
        MakePadding(AsInt64Slice(input_shape.dimensions()), window_dimensions,
                    window_strides, Padding::kSame);

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
  int64 reduce_window_size_;
};

StatusOr<bool> TreeReductionRewriter::Run(HloModule *module) {
  ReductionRewriterVisitor visitor(reduce_window_size_);
  bool changed = false;
  for (const auto &computation : module->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }

  return changed;
}

}  // end namespace xla
