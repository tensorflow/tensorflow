/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>

using ::tensorflow::str_util::Join;

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

FullVisitor::FullVisitor(CompilerResources& res) : BaseVisitor(res) {}

Status FullVisitor::HandleConcatenate(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  int64 dimension(inst->concatenate_dimension());
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), inst->operand_count());
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  for (int i = 1; i < inst->operand_count(); i++) {
    CHECK_EQ(inputs[i].size(), 1);
    poplar::Tensor t = inputs[i][0];
    out = poplar::concat(out, t, dimension);
  }
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleDot(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog,
      CreateMatMulForDotOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleConvolution(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateConv2D(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleReverse(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(
      t, FindInstructionInput(tensor_map, resources_, inst, 0, sequence));
  TF_ASSIGN_OR_RETURN(t, ReverseTensor(t, inst->dimensions()));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status FullVisitor::HandleReduce(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (IsReducableArtithmetic(inst->to_apply())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateSimpleReduction(resources_, inst, GetOutputShape(inst),
                                    tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleBitcast(HloInstruction* inst) {
  return HandleReshape(inst);
}

Status FullVisitor::HandleBroadcast(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  TF_ASSIGN_OR_RETURN(
      out, BroadcastTensor(out, GetOutputShape(inst), inst->dimensions()));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleReshape(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleTranspose(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];
  auto optional_permutation =
      convert_array<std::vector<unsigned>>(inst->dimensions());
  if (!optional_permutation) {
    return xla::FailedPrecondition(
        "HandleTranspose - cannot cast permutation.");
  }
  std::vector<unsigned> permutation = *optional_permutation;
  out = out.dimShuffle(permutation);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), 1);
  poplar::Tensor out = inputs[0][0];

  auto optional_begin =
      convert_array<std::vector<size_t>>(inst->slice_starts());
  if (!optional_begin) {
    return xla::FailedPrecondition("HandleSlice - cannot cast slice starts.");
  }
  std::vector<size_t> begin = *optional_begin;

  auto optional_end = convert_array<std::vector<size_t>>(inst->slice_limits());
  if (!optional_end) {
    return xla::FailedPrecondition("HandleSlice - cannot cast slice limits.");
  }
  std::vector<size_t> end = *optional_end;

  std::vector<int64> strides(inst->slice_strides());
  bool simple(true);
  for (std::size_t s : strides) {
    simple &= (s == 1);
  }
  if (simple) {
    out = out.slice(begin, end);
  } else {
    for (size_t d = 0; d < strides.size(); d++) {
      int64 s = strides[d];
      if (s > 0) {
        out = out.slice(begin[d], end[d], d);
        out = out.subSample(strides[d], d);
      } else {
        out = out.slice(end[d] + 1, begin[d] + 1, d);
        out = out.reverse(d);
        out = out.subSample(-strides[d], d);
      }
    }
  }
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleDynamicSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog,
      CreateDynamicSliceOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleDynamicUpdateSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateDynamicSliceUpdateOp(resources_, inst, GetOutputShape(inst),
                                       tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleReduceWindow(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (IsPoplibsPool(inst, inst->to_apply())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreatePoplibsWindowReduction(resources_, inst,
                                           GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  if (IsReducableArtithmetic(inst->to_apply())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateSimpleWindowReduction(resources_, inst,
                                          GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  if (IsSimpleSelection(inst->select()) &&
      IsReducableArtithmetic(inst->scatter())) {
    VLOG(1) << "Processing " << inst->name();
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateSimpleSelectAndScatter(resources_, inst,
                                           GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleWhile(HloInstruction* inst) {
  poplar::program::Program prog;
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      prog, CreateWhileOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandlePad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(inputs[0].size(), 1);
  CHECK_EQ(inputs[1].size(), 1);
  poplar::Tensor out = inputs[0][0];
  poplar::Tensor pad = inputs[1][0];
  TF_ASSIGN_OR_RETURN(out, PadTensor(inst->padding_config(), out, pad));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleIota(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(resources_, inst);

  auto* iota = Cast<HloIotaInstruction>(inst);
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(
      t, AddIotaTensor(graph, std::make_pair(inst, 0), GetOutputShape(inst),
                       iota->iota_dimension(), resources_, tensor_map));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status FullVisitor::HandleSort(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog, CreateSort(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleBatchNormInference(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateBatchNormInf(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleBatchNormTraining(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateBatchNormTraining(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::HandleBatchNormGrad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  TF_ASSIGN_OR_RETURN(auto prog,
                      CreateBatchNormGrad(resources_, inst, tensor_map));

  sequence.add(prog);

  return Status::OK();
}

Status FullVisitor::Postprocess(HloInstruction* inst) {
  if (!inst->shape().IsTuple()) {
    auto outs = FindInstructionOutputs(tensor_map, inst);
    if (outs.size() == 1) {
      if (!PoplarShapeMatchesXLAShape(outs[0], inst->shape())) {
        return xla::InternalError(
            "Instruction %s has mismatched Poplar (%s) and XLA (%s) shapes",
            inst->name().c_str(), Join(outs[0].shape(), ",").c_str(),
            Join(inst->shape().dimensions(), ",").c_str());
      }
      TF_ASSIGN_OR_RETURN(poplar::Type expected_type,
                          PoplarDataType(inst->shape()));
      if (expected_type != outs[0].elementType()) {
        return xla::InternalError(
            "Instruction %s has mismatched Poplar (%s) and XLA (%s) type",
            inst->name().c_str(),
            expected_type.toString().cloneAsString().c_str(),
            outs[0].elementType().toString().cloneAsString().c_str());
      }
    }
  }
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
