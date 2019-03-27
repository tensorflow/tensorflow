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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_allocation_visitor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {

Status DeferredAllocationVisitor::AllocateInput(const HloInstruction* inst,
                                                int64 flat_tuple_index,
                                                const Shape& shape) {
  poplar::Graph& graph =
      GetGraphWithOutputIndex(resources_, inst, flat_tuple_index);

  auto source = std::make_pair(inst, flat_tuple_index);

  // Do the allocation
  VLOG(2) << "Allocating input tensor for " << inst->name() << ":"
          << flat_tuple_index << " shape " << shape.ToString() << " on shard "
          << GetShardForOutputIndex(inst, flat_tuple_index);
  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      AddTensor(graph, source, shape, resources_, tensor_map));

  // Each visitor might need to post process the allocation, for example change
  // the layout of the tensor - this depends on the original input op type.
  const HloInstruction* original_input = inst;
  int64 original_flat_tuple_index = flat_tuple_index;
  // For deferred allocation the original input is the back of the
  // deferred_allocations_path
  if (IsDeferredAllocation(inst, flat_tuple_index)) {
    auto& deferred_allocations_path =
        resources_.annotations.tensor_allocation_map.at(source)
            .deferred_allocations_path;
    original_input = deferred_allocations_path.back().first;
    original_flat_tuple_index = deferred_allocations_path.back().second;
  }

  switch (original_input->opcode()) {
    case HloOpcode::kInfeed: {
      TF_ASSIGN_OR_RETURN(
          out, PostProcessInfeedAllocation(original_input,
                                           original_flat_tuple_index, out));
      break;
    }
    case HloOpcode::kParameter: {
      TF_ASSIGN_OR_RETURN(
          out, PostProcessParameterAllocation(original_input,
                                              original_flat_tuple_index, out));
      break;
    }
    default: {
      return xla::FailedPrecondition(
          "Unsupported input allocation for opcode %s.",
          HloOpcodeString(original_input->opcode()).c_str());
    }
  }

  // Add the post processed tensor.
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, flat_tuple_index, out));

  // If this was a deferred allocation, we need to make sure that all the ops
  // which we skipped the allocation for have their tensor set to this
  // allocation.
  if (IsDeferredAllocation(inst, flat_tuple_index)) {
    auto& deferred_allocations_path =
        resources_.annotations.tensor_allocation_map.at(source)
            .deferred_allocations_path;
    for (TensorSource deferred : deferred_allocations_path) {
      TF_RETURN_IF_ERROR(
          AddOutputTensor(tensor_map, deferred.first, deferred.second, out));
      original_input = deferred.first;
      original_flat_tuple_index = deferred.second;
    }
  }
  return Status::OK();
}

Status DeferredAllocationVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // Go through all the shapes for inst, don't allocate any tensors which are
  // marked as deferred.
  std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());
  bool defer_any_allocations = false;
  bool allocate = false;
  // First check if there are any deferred allocation - if not we can call the
  // parent class to deal with it.
  for (int64 i = 0; i < shapes.size(); i++) {
    defer_any_allocations |= DeferAllocation(inst, i);
    allocate |= IsDeferredAllocation(inst, i);
  }

  if (allocate) {
    if (shapes.size() != 1) {
      return xla::FailedPrecondition(
          "Trying to allocate multiple deferred tensors.");
    }
    return AllocateInput(inst, 0, shapes[0]);
  } else if (defer_any_allocations) {
    // Note that the forward allocation finder makes sure that this is inplace -
    // we therefore don't need to worry about copies.
    for (int64 i = 0; i < shapes.size(); i++) {
      if (!DeferAllocation(inst, i)) {
        // Get the tensor for this shape and assign it as an output.
        auto range = std::make_pair(i, i + 1);
        auto outputs = FindInstructionInputsInRange(
            tensor_map, resources_, inst, 0, range, sequence, false);
        CHECK_EQ(outputs.size(), 1);
        TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, outputs[0]));
      }
    }
    return Status::OK();
  } else {
    return FullVisitor::HandleGetTupleElement(inst);
  }
}

Status DeferredAllocationVisitor::HandleInfeed(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // We currently have no way of ordering infeeds in the same compuation - this
  // can result in unexpected results.
  if (has_infeed_) {
    return xla::FailedPrecondition(
        "Currently calling `get_next()` multiple times on the same "
        "IPUInfeedQueue in the same computation block is not supported.");
  }

  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);
  // We allow the same infeed queue to be dequeued multiple times, however
  // we don't support multiple infeed queues in the same program.
  if (absl::c_any_of(resources_.annotations.infeed_infos,
                     [&](const HloInfeedInstruction* inst) {
                       return inst->infeed_config() != infeed->infeed_config();
                     })) {
    return xla::FailedPrecondition(
        "Currently multiple IPUInfeedQueue in the same program are not "
        "supported.");
  }
  std::vector<Shape> shapes = FlattenedXlaShape(infeed->infeed_shape());
  for (int64 i = 0; i < shapes.size(); i++) {
    if (!DeferAllocation(inst, i)) {
      TF_RETURN_IF_ERROR(AllocateInput(inst, i, shapes[i]));
    } else {
      VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
              << i << ".";
    }
  }
  has_infeed_ = true;
  resources_.annotations.infeed_infos.insert(infeed);

  return Status::OK();
}

StatusOr<poplar::Tensor> DeferredAllocationVisitor::PostProcessInfeedAllocation(
    const HloInstruction* inst, int64 flat_tuple_index, poplar::Tensor tensor) {
  poplar::Graph& master_graph = GetMasterGraph(resources_);
  if (!UseSyntheticData()) {
    poplar::Tensor master_tensor = tensor;

    if (HasReplicatedGraph(resources_)) {
      master_tensor = master_graph.getNonReplicatedTensor(master_tensor);
    }

    auto fifo = master_graph.addHostToDeviceFIFO(
        GetInfeedCopyHandle(inst->name(), flat_tuple_index),
        master_tensor.elementType(), master_tensor.numElements());

    sequence.add(poplar::program::Copy(fifo, master_tensor, false));
  }
  return tensor;
}

bool DeferredAllocationVisitor::DeferAllocation(const HloInstruction* inst,
                                                int64 flat_tuple_index) {
  return resources_.annotations.deferred_allocations.contains(
      std::make_pair(inst, flat_tuple_index));
}

bool DeferredAllocationVisitor::IsDeferredAllocation(const HloInstruction* inst,
                                                     int64 flat_tuple_index) {
  auto& tensor_allocation_map = resources_.annotations.tensor_allocation_map;
  auto itr = tensor_allocation_map.find(std::make_pair(inst, flat_tuple_index));
  if (itr != tensor_allocation_map.end()) {
    return itr->second.deferred_allocations_path.size() > 0;
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
