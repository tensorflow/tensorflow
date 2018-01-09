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

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_base.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Tensor.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Engine.hpp>
#include <poplar/exceptions.hpp>

namespace se = ::perftools::gputools;

namespace xla {
namespace poplarplugin {

BaseVisitor::BaseVisitor(poplar::Graph* graph,
                         CompilerResources& res)
        : graph_(graph),
          resources_(res) {}

const Shape&
BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return port::Status(port::error::UNIMPLEMENTED,
                      port::StrCat(inst->name(),
                                   " not implemented"));
}

Status BaseVisitor::HandleElementwiseUnary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateUnaryElementwiseOp(*graph_,
                                               resources_,
                                               inst,
                                               GetOutputShape(inst),
                                               tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleElementwiseBinary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateBinaryElementwiseOp(*graph_,
                                                resources_,
                                                inst,
                                                GetOutputShape(inst),
                                                tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateCastOp(*graph_,
                                   resources_,
                                   inst,
                                   GetOutputShape(inst),
                                   tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleCopy(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor in;
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  sequence.add(poplar::program::Copy(in, out));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status BaseVisitor::HandleClamp(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateClampOp(*graph_,
                                    resources_,
                                    inst,
                                    GetOutputShape(inst),
                                    tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateSelectOp(*graph_,
                                     resources_,
                                     inst,
                                     GetOutputShape(inst),
                                     tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConcatenate(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBitcastConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(inst->shape()));
  out = out.reinterpret(type);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status BaseVisitor::HandleDot(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConvolution(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleCrossReplicaSum(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleRng(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReverse(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSort(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConstant(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, AddConstantTensor(*graph_,
                                           std::make_pair(inst, 0),
                                           GetOutputShape(inst),
                                           inst->literal(),
                                           resources_));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status BaseVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  ArgVector inputs =
          FindTupleInInstructionInput(tensor_map, inst, 0, inst->tuple_index());
  for (unsigned int i=0; i<inputs.size(); i++) {
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, inputs[i]));
  }
  return Status::OK();
}

Status BaseVisitor::HandleReduce(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBitcast(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBroadcast(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReshape(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleTranspose(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleFusion(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateFusionOp(*graph_,
                                     resources_,
                                     inst,
                                     GetOutputShape(inst),
                                     tensor_map));
  sequence.add(prog);
  return Status::OK();
};


Status BaseVisitor::HandleCall(HloInstruction* inst) {
  HloComputation* comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : " << comp->name();

  // If is is a special fusion-type op
  if (comp->name().substr(0, 8) == "_pop_op_") {
    auto end = comp->name().find('.');
    std::string name = comp->name().substr(8, end-8);

    if (name == "const_slice_update") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateSliceUpdateOp(*graph_,
                                              resources_,
                                              inst,
                                              GetOutputShape(inst),
                                              tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "const_slice") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateSliceOp(*graph_,
                                        resources_,
                                        inst,
                                        GetOutputShape(inst),
                                        tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "relu") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateReluOp(*graph_,
                                       resources_,
                                       inst,
                                       GetOutputShape(inst),
                                       tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "relugrad") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateReluGradOp(*graph_,
                                           resources_,
                                           inst,
                                           GetOutputShape(inst),
                                           tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "sigmoid") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateSigmoidOp(*graph_,
                                          resources_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "biasadd") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateBiasAddOp(*graph_,
                                          resources_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "biasadd_broadcast") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateBiasAddBcastOp(*graph_,
                                               resources_,
                                               inst,
                                               GetOutputShape(inst),
                                               tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "zero_pad") {
      const HloInstruction* root = inst->to_apply()->root_instruction();
      const PaddingConfig& cfg(root->padding_config());
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
      TF_ASSIGN_OR_RETURN(out, PadWithConstantZero(*graph_, cfg, out));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
      return Status::OK();
    } else if (name == "trunc_norm_scale_add") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
        TruncatedNormalScale(*graph_, resources_, inst, GetOutputShape(inst), tensor_map));
      sequence.add(prog);
    }
    else if (name == "trunc_norm") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          TruncatedNormal(*graph_, resources_, inst, GetOutputShape(inst), tensor_map));
      sequence.add(prog);
    }
    else if (name == "norm_scale_add") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          RandomNormalScale(*graph_, resources_, inst, GetOutputShape(inst), tensor_map));
      sequence.add(prog);
    }
    else if (name == "uniform_scale_add") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          RandomUniformScale(*graph_, resources_, inst, GetOutputShape(inst), tensor_map));
      sequence.add(prog);
    }
    else if (name == "norm") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          RandomNormal(*graph_, resources_, inst, GetOutputShape(inst), tensor_map));
      sequence.add(prog);
    }
    else if (name == "uniform") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          RandomUniform(*graph_, resources_, inst, GetOutputShape(inst), tensor_map));
      sequence.add(prog);
    }
    else if (name == "avgpool") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreatePoplibsWindowReduction(*graph_,
                                                       resources_,
                                                       inst,
                                                       GetOutputShape(inst),
                                                       tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "wide_const") {
      const HloInstruction* root = inst->to_apply()->root_instruction();
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, AddConstantTensor(*graph_,
                                                 std::make_pair(inst, 0),
                                                 inst->shape(),
                                                 root->operand(0)->literal(),
                                                 resources_));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
      return Status::OK();
    }
    else if (name == "depthwise_conv") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateDepthwiseConvolutionOp(*graph_,
                                                       resources_,
                                                       inst,
                                                       GetOutputShape(inst),
                                                       tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "conv_with_reverse") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          Create2DConvWithReverse(*graph_,
                                                  resources_,
                                                  inst,
                                                  GetOutputShape(inst),
                                                  tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else if (name == "bias_apply") {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          ConvBiasApply(*graph_,
                                        resources_,
                                        inst,
                                        GetOutputShape(inst),
                                        tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    else {
      return port::Status(port::error::FAILED_PRECONDITION,
                          port::StrCat("Unrecognized special call op ",
                                       inst->name(), ": ", name));
    }
  } else {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateCallOp(*graph_,
                                     resources_,
                                     inst,
                                     GetOutputShape(inst),
                                     tensor_map));
    sequence.add(prog);
  }
  return Status::OK();
}

Status BaseVisitor::HandleCustomCall(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSlice(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleDynamicSlice(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleDynamicUpdateSlice(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleTuple(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  uint64 operand_count(inst->operand_count());
  int64 n=0;
  for (uint64 i=0; i<operand_count; i++) {
    ArgVector inputs = FindInstructionInputs(tensor_map, inst, i);
    for(poplar::Tensor t : inputs) {
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, n, t));
      n++;
    }
  }
  return Status::OK();
}

Status BaseVisitor::HandleReduceWindow(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleMap(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  bool simple_parallel;
  TF_ASSIGN_OR_RETURN(simple_parallel,
                      IsParallelMap(inst, inst->to_apply()));
  if (simple_parallel) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateParallelMap(*graph_,
                                          resources_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleWhile(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConditional(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandlePad(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReducePrecision(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleInfeed(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleOutfeed(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSend(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSendDone(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleRecv(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleRecvDone(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBatchNormInference(HloInstruction* inst) {
  return Unimplemented(inst);
}


Status BaseVisitor::HandleBatchNormTraining(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBatchNormGrad(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleFft(HloInstruction* inst) {
  return Unimplemented(inst);
}

}  // namespace poplarplugin
}  // namespace xla
