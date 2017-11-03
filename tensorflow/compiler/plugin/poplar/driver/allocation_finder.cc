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

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

class FindAllocatingInstructions : public DfsHloVisitorWithDefault {
public:
  FindAllocatingInstructions() {}

  ~FindAllocatingInstructions() override = default;

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Status::OK();
  }

  Status HandleConstant(HloInstruction* inst, const Literal& literal) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleRng(HloInstruction* inst, RandomDistribution) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleParameter(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* inst,
                      HloInstruction* /*arg*/,
                      HloInstruction* /*init_value*/,
                      tensorflow::gtl::ArraySlice<int64> /*dimensions*/,
                      HloComputation* /*function*/) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* inst,
                            HloInstruction* /*operand*/,
                            const Window& /*window*/,
                            HloComputation* /*function*/) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleSelectAndScatter(HloInstruction* inst) override {
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
AllocationFinder::FindConsumers(HloInstruction* src, HloInstruction* tgt) {
  for (auto user : tgt->users()) {
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
        FindConsumers(src, param);
        break;
      }
      default:
        if (ShapeUtil::Equal(src->shape(), user->shape())) {
          FindConsumers(src, user);
        }
        break;
    }
  }
  return;
}

Status AllocationFinder::CreateAllocationMap(HloModule* module) {
  for (const auto& comp : module->computations()) {

    FindAllocatingInstructions finder;
    TF_RETURN_IF_ERROR(comp->Accept(&finder));

    for (auto inst : finder.allocating_instructions) {
      FindConsumers(inst, inst);
    }
  }

  return Status::OK();
}

}
}
