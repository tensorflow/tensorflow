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

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_all_reduce.h"

#include "tensorflow/core/lib/core/errors.h"

#include <algorithm>

namespace xla {
namespace poplarplugin {

namespace {
// Check that a region of instruction can be combined. This means checking
// that they are not sequenced due to a data dependency.
template <typename Iter>
bool CanCombine(Iter begin, Iter end) {
  for (Iter itr1 = begin; itr1 != end; ++itr1) {
    auto instr1 = *itr1;
    const auto& operands = instr1->operands();
    for (Iter itr2 = begin; itr2 != itr1; ++itr2) {
      auto instr2 = *itr2;

      if (std::find(operands.begin(), operands.end(), instr2) !=
          operands.end()) {
        // Found a data dependency :(
        return false;
      }
    }
  }

  return true;
}

// Partition the all-reduce ops into regions where the computation and sharding
// information matches
template <typename Iter>
std::vector<std::pair<Iter, Iter>> Partition(Iter begin, Iter end) {
  std::vector<std::pair<Iter, Iter>> result;

  while (begin != end) {
    auto pred = [&](const HloInstruction* inst) {
      auto comp1 = (*begin)->to_apply();
      auto comp2 = inst->to_apply();

      return *comp1 == *comp2 &&
             (*begin)->has_sharding() == inst->has_sharding() &&
             (!inst->has_sharding() ||
              (*begin)->sharding() == inst->sharding());
    };

    auto itr = std::stable_partition(begin, end, pred);

    result.push_back({begin, itr});
    begin = itr;
  }

  return result;
}

template <typename Iter>
HloInstruction* Combine(HloComputation* comp, Iter begin, Iter end) {
  auto dist = std::distance(begin, end);

  // We only have a single all reduce, so nothing to combine
  if (dist == 1) {
    return *begin;
  }

  std::vector<Shape> shapes(dist);
  std::transform(begin, end, shapes.begin(),
                 [](HloInstruction* inst) { return inst->shape(); });

  // The new output shape
  auto shape = ShapeUtil::MakeTupleShape(shapes);

  // The new list of operands
  auto operands = std::accumulate(
      begin, end, std::vector<HloInstruction*>{},
      [](std::vector<HloInstruction*>& accum, HloInstruction* inst) {
        accum.insert(accum.end(), inst->operands().begin(),
                     inst->operands().end());

        return accum;
      });

  // Add the new all reduce to the computation
  return comp->AddInstruction(
      HloInstruction::CreateAllReduce(shape, operands, (*begin)->to_apply(), {},
                                      (*begin)->all_reduce_barrier(), {}));
}

template <typename Iter>
StatusOr<std::vector<HloInstruction*>> Replace(HloComputation* comp, Iter begin,
                                               Iter end,
                                               HloInstruction* all_reduce) {
  auto dist = std::distance(begin, end);

  if (dist == 1) {
    return std::vector<HloInstruction*>{};
  }

  std::vector<HloInstruction*> result;
  result.reserve(dist);

  for (auto itr = begin; itr != end; ++itr) {
    // Add a get tuple to unpack the all-reduce result
    result.push_back(comp->AddInstruction(HloInstruction::CreateGetTupleElement(
        (*itr)->shape(), all_reduce, std::distance(begin, itr))));

    // Replace the op
    TF_RETURN_IF_ERROR((*itr)->ReplaceAllUsesWith(result.back()));
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(*itr));
  }

  return result;
}

StatusOr<HloInstructionSequence> CombineAllReduces(
    HloComputation* comp, const HloInstructionSequence& sequence) {
  auto instructions = sequence.instructions();

  std::vector<const HloInstruction*> result;
  result.reserve(instructions.size());

  const auto is_all_reduce_pred = [](HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };

  const auto is_not_all_reduce_pred = [&](HloInstruction* inst) {
    return !is_all_reduce_pred(inst);
  };

  // Find the first region of consecutive all reduce instructions
  //       v beg
  // [a,b,c|r,r,r,r,r|d,e,f,g,r,r,r]
  auto region_begin = std::find_if(instructions.begin(), instructions.end(),
                                   is_all_reduce_pred);
  //       v beg     v end
  // [a,b,c|r,r,r,r,r|d,e,f,g,r,r,r]
  auto region_end =
      std::find_if(region_begin, instructions.end(), is_not_all_reduce_pred);

  // While we have a region to process
  while (region_begin != instructions.end()) {
    // If all of the all reduce instructions can be combined
    if (CanCombine(region_begin, region_end)) {
      // Partition the all reduce instructions into combinable groups
      auto subregions = Partition(region_begin, region_end);

      std::vector<HloInstruction*> replacements;

      // Create the combined all reduce instructions
      for (auto& subregion : subregions) {
        replacements.push_back(
            Combine(comp, subregion.first, subregion.second));

        TF_ASSIGN_OR_RETURN(auto ops,
                            Replace(comp, subregion.first, subregion.second,
                                    replacements.back()));

        replacements.insert(replacements.end(), ops.begin(), ops.end());
      }

      // Replace the previous all reduce instruction in the schedule
      //       v beg     v end
      // [a,b,c|r,r,r,r,r|d,e,f,g,r,r,r]
      // becomes
      //       v itr
      // [a,b,c|d,e,f,g,r,r,r]
      auto insert_itr = instructions.erase(region_begin, region_end);
      //       v itr   v end
      // [a,b,c|r,t,t,t|d,e,f,g,r,r,r]
      region_end = instructions.insert(insert_itr, replacements.begin(),
                                       replacements.end()) +
                   replacements.size();
    }

    // Find the next region of consecutive all reduce instructions
    //               v end   v beg
    // [a,b,c,r,t,t,t|d,e,f,g|r,r,r]
    region_begin =
        std::find_if(region_end, instructions.end(), is_all_reduce_pred);
    //                       v beg v end
    // [a,b,c,r,t,t,t,d,e,f,g|r,r,r|]
    region_end =
        std::find_if(region_begin, instructions.end(), is_not_all_reduce_pred);
  }

  // Return the new schedule
  return HloInstructionSequence(instructions);
}
}  // namespace

StatusOr<bool> CombineAllReduce::Run(HloModule* module) {
  if (!module->has_schedule()) {
    return tensorflow::errors::FailedPrecondition(
        "CombineAllReduce: module doesn't have a schedule");
  }

  const auto& schedule = module->schedule();
  const auto& sequences = schedule.sequences();

  absl::flat_hash_map<int64, HloComputation*> computations;
  for (int i = 0; i < module->computation_count(); ++i) {
    auto comp = module->mutable_computation(i);
    computations[comp->unique_id()] = comp;
  }

  HloSchedule new_schedule(module);
  for (auto& pair : sequences) {
    auto comp = computations[pair.first];
    TF_ASSIGN_OR_RETURN(auto sched, CombineAllReduces(comp, pair.second));
    new_schedule.set_sequence(comp, sched);
  }

  TF_RETURN_IF_ERROR(new_schedule.Verify());
  module->set_schedule(new_schedule);

  VLOG(1) << "Combined all reduce schedule " << new_schedule.ToString();

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
