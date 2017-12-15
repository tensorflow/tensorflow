/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

// Find the index when embedding a shape into a tuple. The tuple_index is the
// index of the shape in the new tuple, and the original_index is the index
// of the tensor in the original shape.
int64 InsertIntoTuple(const Shape& tuple, int64 tuple_index,
                      int64 original_index) {
  // Count up the base tensors inside all tuple element preceeding the
  // tuple_index one.
  int64 tensor_count = 0;
  for (int64 i = 0; i < tuple_index; i++) {
    tensor_count += CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  return tensor_count + original_index;
}

// Find the index of a tensor after extracting it (or a tuple containing it)
// from a tuple. tuple_index is the index of one of the elements of the tuple,
// and original_index is the tensor position within the original tuple.
int64 ExtractFromTuple(const Shape& tuple, int64 tuple_index,
                       int64 original_index) {
  int64 index = original_index;
  for (int64 i = 0; i < tuple_index; i++) {
    index -= CountShapes(ShapeUtil::GetTupleElementShape(tuple, i));
  }
  int64 n = CountShapes(ShapeUtil::GetTupleElementShape(tuple, tuple_index));
  if (index < 0 || index >= n) {
    return -1;
  }
  return index;
}

class FindAllocatingInstructions : public DfsHloVisitorWithDefault {
public:
  FindAllocatingInstructions() {}

  ~FindAllocatingInstructions() override = default;

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Status::OK();
  }

  Status HandleConstant(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleRng(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleParameter(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  std::vector<HloInstruction*> allocating_instructions;
};

}

bool
AllocationFinder::CompareConvolutionTargets(const TensorTarget& a,
                                            const TensorTarget& b) {
  return IsForwardConvolution(a.first) && !IsForwardConvolution(b.first);
}

bool
AllocationFinder::CompareDotTargets(const TensorTarget& a,
                                    const TensorTarget& b) {
  return IsForwardMatMul(a.first) && !IsForwardMatMul(b.first);
}

void
AllocationFinder::FindConsumers(HloInstruction* src, HloInstruction* tgt,
                                int64 index) {
  for (auto user : tgt->users()) {
    if (visited.count(user) == 0) {
      visited.insert(user);
      int64 op_index = user->operand_index(tgt);
      switch (user->opcode()) {
        case HloOpcode::kConvolution:
        {
          auto t = std::make_pair(user, op_index);
          auto i = tensor_allocation_map.find(src);
          if (i != tensor_allocation_map.end() &&
              CompareConvolutionTargets(t, i->second)) {
            tensor_allocation_map.erase(src);
          }
          tensor_allocation_map.insert(std::make_pair(src, t));
          return;
        }
        case HloOpcode::kDot:
        {
          auto t = std::make_pair(user, op_index);
          auto i = tensor_allocation_map.find(src);
          if (i != tensor_allocation_map.end() &&
              CompareDotTargets(t, i->second)) {
            tensor_allocation_map.erase(src);
          }
          tensor_allocation_map.insert(std::make_pair(src, t));
          return;
        }
        case HloOpcode::kCall:
        {
          HloComputation* comp = user->to_apply();
          HloInstruction* param = comp->parameter_instruction(op_index);
          FindConsumers(src, param, index);
          break;
        }
        case HloOpcode::kWhile:
        {
          HloComputation* comp = user->while_body();
          HloInstruction* param = comp->parameter_instruction(op_index);
          FindConsumers(src, param, index);
          break;
        }
        case HloOpcode::kTuple:
        {
          int64 new_index = InsertIntoTuple(user->shape(), op_index, index);
          FindConsumers(src, user, new_index);
          break;
        }
        case HloOpcode::kGetTupleElement:
        {
          int64 tuple_index = user->tuple_index();
          int64 new_index = ExtractFromTuple(tgt->shape(), tuple_index, index);
          if (new_index != -1) {
            FindConsumers(src, user, new_index);
          }
          break;
        }
        default:
          if (ShapeUtil::Equal(src->shape(), user->shape())) {
            FindConsumers(src, user, index);
          }
          break;
      }
    }
  }
  return;
}

Status AllocationFinder::CreateAllocationMap(HloModule* module) {
  FindAllocatingInstructions finder;

  for (const auto& comp : module->computations()) {
    TF_RETURN_IF_ERROR(comp->Accept(&finder));
  }

  for (auto inst : finder.allocating_instructions) {
    visited.clear();
    FindConsumers(inst, inst, 0);
  }

  return Status::OK();
}

}
}
