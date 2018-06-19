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
#include "tensorflow/compiler/plugin/poplar/driver/classification_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
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
    allocating_instructions.push_back(std::make_pair(inst, 0));
    return Status::OK();
  }

  Status HandleRng(HloInstruction* inst) override {
    allocating_instructions.push_back(std::make_pair(inst, 0));
    return Status::OK();
  }

  Status HandleParameter(HloInstruction* inst) override {
    auto shapes = FlattenedXlaShape(inst->shape());
    for (int i = 0; i < shapes.size(); i++) {
      allocating_instructions.push_back(std::make_pair(inst, i));
    }
    return Status::OK();
  }

  Status HandleCall(HloInstruction* inst) override {
    if (inst->to_apply()->name().substr(0, 17) == "_pop_op_wide_const") {
      allocating_instructions.push_back(std::make_pair(inst, 0));
    }
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* inst) override {
    allocating_instructions.push_back(std::make_pair(inst, 0));
    return Status::OK();
  }

  std::vector<TensorSource> allocating_instructions;
};

}  // namespace

bool AllocationFinder::CompareTargets(const TensorTarget& a,
                                      const TensorTarget& b) {
  return IsForward(a.tgt, annotations) && !IsForward(b.tgt, annotations);
}

void AllocationFinder::FindConsumers(const TensorSource& src,
                                     const HloInstruction* tgt, int64 index) {
  path.emplace_back(tgt);
  for (auto user : tgt->users()) {
    if (visited.count(user) == 0) {
      visited.insert(user);
      int64 op_index = user->operand_index(tgt);
      switch (user->opcode()) {
        case HloOpcode::kConvolution: {
          auto t = TensorTarget(user, op_index, path);
          auto i = tensor_allocation_map.find(src);
          if (i != tensor_allocation_map.end() &&
              CompareTargets(t, i->second)) {
            tensor_allocation_map.erase(src);
          }
          tensor_allocation_map.insert(std::make_pair(src, t));
          break;
        }
        case HloOpcode::kDot: {
          auto t = TensorTarget(user, op_index, path);
          auto i = tensor_allocation_map.find(src);
          if (i != tensor_allocation_map.end() &&
              CompareTargets(t, i->second)) {
            tensor_allocation_map.erase(src);
          }
          tensor_allocation_map.insert(std::make_pair(src, t));
          break;
        }
        case HloOpcode::kDynamicSlice: {
          if (op_index == 0) {
            auto t = TensorTarget(user, op_index, path);
            auto i = tensor_allocation_map.find(src);
            if (i != tensor_allocation_map.end()) {
              tensor_allocation_map.erase(src);
            }
            tensor_allocation_map.insert(std::make_pair(src, t));
          }
          break;
        }
        case HloOpcode::kDynamicUpdateSlice: {
          if (op_index == 0 || op_index == 1) {
            auto t = TensorTarget(user, op_index, path);
            auto i = tensor_allocation_map.find(src);
            if (i != tensor_allocation_map.end()) {
              tensor_allocation_map.erase(src);
            }
            tensor_allocation_map.insert(std::make_pair(src, t));
          }
          break;
        }
        case HloOpcode::kCall: {
          HloComputation* comp = user->to_apply();
          if (comp->name().substr(0, 8) != "_pop_op_") {
            HloInstruction* param = comp->parameter_instruction(op_index);
            FindConsumers(src, param, index);
          } else {
            auto end = comp->name().find('.');
            std::string name = comp->name().substr(8, end - 8);
            if (name == "depthwise_conv") {
              auto t = TensorTarget(user, op_index, path);
              auto i = tensor_allocation_map.find(src);
              if (i != tensor_allocation_map.end()) {
                tensor_allocation_map.erase(src);
              }
              tensor_allocation_map.insert(std::make_pair(src, t));
            }
          }
          break;
        }
        case HloOpcode::kWhile: {
          HloComputation* comp = user->while_body();
          HloInstruction* param = comp->parameter_instruction(op_index);
          FindConsumers(src, param, index);
          break;
        }
        case HloOpcode::kTuple: {
          int64 new_index = InsertIntoTuple(user->shape(), op_index, index);
          FindConsumers(src, user, new_index);
          break;
        }
        case HloOpcode::kGetTupleElement: {
          int64 tuple_index = user->tuple_index();
          int64 new_index = ExtractFromTuple(tgt->shape(), tuple_index, index);
          if (new_index != -1) {
            FindConsumers(src, user, new_index);
          }
          break;
        }
        case HloOpcode::kReshape: {
          FindConsumers(src, user, index);
          break;
        }
        case HloOpcode::kTranspose: {
          FindConsumers(src, user, index);
          break;
        }
        default: {
          auto shapes = FlattenedXlaShape(src.first->shape());
          if (ShapeUtil::Equal(shapes[src.second], user->shape())) {
            FindConsumers(src, user, index);
          }
          break;
        }
      }
    }
  }
  path.pop_back();
  return;
}

StatusOr<bool> AllocationFinder::Run(HloModule* module) {
  FindAllocatingInstructions finder;

  for (const auto& comp : module->computations()) {
    TF_RETURN_IF_ERROR(comp->Accept(&finder));
  }

  for (auto inst : finder.allocating_instructions) {
    visited.clear();
    FindConsumers(inst, inst.first, inst.second);
  }

  return true;
}

AllocationFinder::AllocationFinder(CompilerAnnotations& annotations)
    : tensor_allocation_map(annotations.tensor_allocation_map),
      annotations(annotations) {}

}  // namespace poplarplugin
}  // namespace xla
