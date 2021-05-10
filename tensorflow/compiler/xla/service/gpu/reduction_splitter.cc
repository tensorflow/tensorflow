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

#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"

#include <algorithm>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

class ReductionSplitterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReduce(HloInstruction *reduce) override {
    VLOG(4) << "Input: " << reduce->ToString();

    // Reductions with contiguous dimensions are lowered to efficient code. No
    // need to split such ops.
    if (IsReductionFromOrToContiguousDimensions(*reduce)) {
      return Status::OK();
    }
    if (reduce->dimensions().size() < 2) {
      return Status::OK();
    }
    if (!reduce->shape().IsArray()) {
      // TODO(cheshire): Handle variadic reduction.
      return Status::OK();
    }

    HloInstruction *operand = reduce->mutable_operand(0);
    const Shape &shape = operand->shape();
    CHECK(shape == LayoutUtil::GetWithDefaultLayout(shape))
        << "Default layout should be enforced on reduction operand";
    // Verify that contiguous dimensions have been grouped by the
    // ReductionDimensionGrouper pass.
    for (int64 i = 0; i < reduce->dimensions().size(); ++i) {
      for (int64 j = i + 1; j < reduce->dimensions().size(); ++j) {
        CHECK(abs(reduce->dimensions(i) - reduce->dimensions(j)) > 1)
            << "Reduction dimensions must not be consecutive";
      }
    }

    // The reduce op has non-contiguous dimensions. Look for the dimension with
    // the largest shape dimension. Reducing along this dimension first will
    // reduce the output size most effectively.
    int64 max_shape_dim = 0;
    int64 max_reduce_dim = 0;
    const auto &input_shape = reduce->operand(0)->shape();
    for (int64 i = 0; i < reduce->dimensions().size(); ++i) {
      if (input_shape.dimensions(reduce->dimensions(i)) > max_shape_dim) {
        max_reduce_dim = reduce->dimensions(i);
        max_shape_dim = input_shape.dimensions(max_reduce_dim);
      }
    }
    // TODO(tjoerg): Run microbenchmarks to tune this threshold.
    if (max_shape_dim < 128) {
      return Status::OK();
    }

    // Split the reduction into a pre-reduction and a final reduction.
    VLOG(3) << "Splitting reduction " << reduce->name() << " at dimension "
            << max_reduce_dim;
    std::vector<int64> pre_reduce_dims;
    pre_reduce_dims.push_back(max_reduce_dim);
    std::vector<int64> pre_reduce_shape_dims(input_shape.dimensions().begin(),
                                             input_shape.dimensions().end());
    pre_reduce_shape_dims.erase(pre_reduce_shape_dims.begin() + max_reduce_dim);
    Shape pre_reduce_shape = ShapeUtil::MakeShape(
        reduce->shape().element_type(), pre_reduce_shape_dims);
    std::unique_ptr<HloInstruction> pre_reduce = HloInstruction::CreateReduce(
        pre_reduce_shape, reduce->mutable_operand(0),
        reduce->mutable_operand(1), pre_reduce_dims, reduce->to_apply());
    pre_reduce->set_metadata(reduce->metadata());

    std::vector<int64> final_reduce_dims(reduce->dimensions().begin(),
                                         reduce->dimensions().end());
    final_reduce_dims.erase(
        std::remove(final_reduce_dims.begin(), final_reduce_dims.end(),
                    max_reduce_dim),
        final_reduce_dims.end());
    for (int64 i = 0; i < final_reduce_dims.size(); ++i) {
      if (final_reduce_dims[i] > max_reduce_dim) {
        final_reduce_dims[i]--;
      }
    }
    std::unique_ptr<HloInstruction> final_reduce = HloInstruction::CreateReduce(
        reduce->shape(),
        reduce->parent()->AddInstruction(std::move(pre_reduce)),
        reduce->mutable_operand(1), final_reduce_dims, reduce->to_apply());
    return ReplaceWithNewInstruction(reduce, std::move(final_reduce));
  }
};

StatusOr<bool> ReductionSplitter::Run(HloModule *module) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionSplitterVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
