
/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reduce_decomposer.h"

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

namespace {

class ReduceDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReduceDecomposerVisitor(HloPredicate custom_layout_allowed)
      : custom_layout_allowed_(std::move(custom_layout_allowed)) {}

  Status HandleReduce(HloInstruction* hlo) override {
    auto reduce = Cast<HloReduceInstruction>(hlo);
    auto shape = reduce->shape();
    if (custom_layout_allowed_ && custom_layout_allowed_(reduce)) {
      return OkStatus();
    }

    std::vector<Shape> expected_shapes(reduce->input_count());
    for (int i = 0; i < reduce->input_count(); i++) {
      expected_shapes[i] = ExpectedOutputShape(reduce, i);
      TF_RET_CHECK(reduce->inputs()[i]->shape().layout() ==
                   reduce->inputs()[0]->shape().layout());
    }

    std::vector<Shape> output_shapes;
    if (shape.IsTuple()) {
      for (int i = 0; i < shape.tuple_shapes_size(); i++) {
        output_shapes.push_back(ShapeUtil::GetTupleElementShape(shape, i));
        TF_RET_CHECK(output_shapes[i].layout() == output_shapes[0].layout());
      }
    } else {
      output_shapes.push_back(shape);
    }

    TF_RET_CHECK(!output_shapes.empty());
    if (ShapeUtil::MakeMaybeTupleShape(expected_shapes) !=
        ShapeUtil::MakeMaybeTupleShape(output_shapes)) {
      TF_ASSIGN_OR_RETURN(auto r_prime,
                          MakeReduceHlo(reduce->inputs(), reduce->init_values(),
                                        reduce->dimensions(),
                                        reduce->called_computations()[0]));
      TF_RET_CHECK(r_prime->shape() ==
                   ShapeUtil::MakeMaybeTupleShape(expected_shapes));

      if (!shape.IsTuple()) {
        auto copy = MakeCopyHlo(r_prime, shape);
        TF_RETURN_IF_ERROR(ReplaceInstruction(reduce, copy));
        return OkStatus();
      }

      std::vector<HloInstruction*> copies;
      for (int i = 0; i < reduce->input_count(); i++) {
        TF_ASSIGN_OR_RETURN(auto from, GetOutput(r_prime, i));
        auto copy = MakeCopyHlo(from, output_shapes[i]);
        copies.push_back(copy);
      }
      auto out = MaybeMakeTuple(copies);
      TF_RETURN_IF_ERROR(ReplaceInstruction(reduce, out));
    }
    return OkStatus();
  }

 private:
  StatusOr<HloInstruction*> GetOutput(HloInstruction* instr, int idx) {
    if (instr->shape().IsTuple()) {
      return MakeGetTupleElementHlo(instr, idx);
    } else {
      TF_RET_CHECK(idx == 0);
      return instr;
    }
  }

  Shape ExpectedOutputShape(HloReduceInstruction* reduce, int input_idx) {
    Shape reduce_shape = reduce->shape();
    auto output_shape = reduce_shape.IsTuple()
                            ? reduce_shape.tuple_shapes(input_idx)
                            : reduce_shape;
    auto* operand = reduce->inputs()[input_idx];
    auto operand_shape = operand->shape();
    return ShapeUtil::DeleteDimensions(reduce->dimensions(), operand_shape);
  }

  HloPredicate custom_layout_allowed_;
};

}  // namespace

StatusOr<bool> ReduceDecomposer::Run(HloModule* module) {
  return ReduceDecomposerVisitor{custom_layout_allowed_}.RunOnModule(module);
}

}  // namespace xla
