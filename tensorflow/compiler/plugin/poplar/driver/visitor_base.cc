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

#include "tensorflow/compiler/plugin/poplar/driver/visitor_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

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

namespace xla {
namespace poplarplugin {

typedef StatusOr<poplar::program::Program> (*CustomCallFn)(
    CompilerResources&, const HloInstruction*, const xla::Shape&, TensorMap&);

static std::map<std::string, CustomCallFn> custom_call_map = {
    {"relu", CreateReluOp},
    {"relugrad", CreateReluGradOp},
    {"sigmoid", CreateSigmoidOp},
    {"sigmoidgrad", CreateSigmoidGradOp},
    {"conv_biasadd", CreateConvBiasAddOp},
    {"matmul_biasadd", CreateMatMulBiasAddOp},
    {"trunc_norm", TruncatedNormal},
    {"norm_scale_add", RandomNormalScale},
    {"uniform_scale_add", RandomUniformScale},
    {"avg_pool", CreatePoplibsWindowReduction},
    {"wide_const", CreateWideConstant},
    {"depthwise_conv", CreateConv2D},
    {"conv_with_reverse", Create2DConvWithReverse},
    {"bias_apply", ConvBiasApply},
    {"zero_pad", CreateZeroPadOp},
    {"depthwise_filter", CreateDepthwiseBackpropFilter},
    {"max_pool", CreatePoplibsWindowReduction},
    {"max_pool_grad", CreateBwdMaxPool},
    {"scaled_inplace", CreateScaledInplace},
    {"conv_scaled_inplace", CreateConvScaledInplace},
    {"padding_reduce_window", CreatePaddingReduceWindow},
};

BaseVisitor::BaseVisitor(CompilerResources& res) : resources_(res) {}

const Shape& BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return xla::Unimplemented("%s not implemented", inst->name().c_str());
}

Status BaseVisitor::HandleElementwiseUnary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateUnaryElementwiseOp(resources_, inst, GetOutputShape(inst),
                                     tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleElementwiseBinary(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateBinaryElementwiseOp(resources_, inst, GetOutputShape(inst),
                                      tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateCastOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleCopy(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor in;
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      in, FindInstructionInput(tensor_map, resources_, inst, 0, sequence));

  out = GetGraph(resources_, inst).clone(in);
  sequence.add(poplar::program::Copy(in, out));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return Status::OK();
}

Status BaseVisitor::HandleClamp(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateClampOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateSelectOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleTupleSelect(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateSelectOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConcatenate(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBitcastConvert(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      out, FindInstructionInput(tensor_map, resources_, inst, 0, sequence));
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(inst->shape()));
  out = out.reinterpret(type);
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status BaseVisitor::HandleDot(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConvolution(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleAllReduce(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleRng(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (inst->operand_count() != 2) {
    LOG(FATAL) << "RNG must have two operands.";
  }
  for (auto* op : inst->operands()) {
    if (op->opcode() != HloOpcode::kConstant) {
      LOG(FATAL) << "Operand " << op->ToString() << " is not a constant.";
    }
  }
  poplar::program::Program prog;
  switch (inst->random_distribution()) {
    case RandomDistribution::RNG_NORMAL: {
      TF_ASSIGN_OR_RETURN(prog, RandomNormal(resources_, inst,
                                             GetOutputShape(inst), tensor_map));
      break;
    }
    case RandomDistribution::RNG_UNIFORM: {
      TF_ASSIGN_OR_RETURN(
          prog,
          RandomUniform(resources_, inst, GetOutputShape(inst), tensor_map));
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported random distribution type "
                 << inst->random_distribution() << ".";
    }
  }
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleReverse(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSort(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConstant(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Graph& graph = GetGraph(resources_, inst);

  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(
      t, AddConstantTensor(graph, std::make_pair(inst, 0), GetOutputShape(inst),
                           inst->literal(), resources_, tensor_map));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status BaseVisitor::HandleGetTupleElement(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0].size(), CountShapes(inst->shape()));
  for (int64 i = 0; i < inputs[0].size(); i++) {
    poplar::Tensor out;
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, inputs[0][i]));
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
  HloComputation* comp = inst->fused_instructions_computation();

  // If is is a special fusion-type op
  if (IsPopOpsFusion(comp)) {
    VLOG(1) << "Processing " << inst->name()
            << " as Poplibs fusion: " << comp->name();
    auto end = comp->name().find('.');
    std::string name = comp->name().substr(8, end - 8);

    if (custom_call_map.count(name) == 1) {
      TF_ASSIGN_OR_RETURN(
          prog, custom_call_map.at(name)(resources_, inst, GetOutputShape(inst),
                                         tensor_map));
    } else {
      return xla::FailedPrecondition("Unrecognized special call op %s: %s",
                                     inst->name().c_str(), name.c_str());
    }
  } else {
    TF_ASSIGN_OR_RETURN(prog, CreateFusionOp(resources_, inst,
                                             GetOutputShape(inst), tensor_map));
  }
  sequence.add(prog);
  return Status::OK();
};

Status BaseVisitor::HandleCall(HloInstruction* inst) {
  HloComputation* comp = inst->to_apply();
  VLOG(1) << "Processing " << inst->name() << " : " << comp->name();

  if (IsRepeatCall(comp)) {
    TF_ASSIGN_OR_RETURN(
        poplar::program::Program prog,
        CreateRepeatOp(resources_, inst, GetOutputShape(inst), tensor_map));
    sequence.add(prog);
  } else {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateCallOp(resources_, inst, GetOutputShape(inst), tensor_map));
    sequence.add(prog);
  }
  return Status::OK();
}

Status BaseVisitor::HandleCustomCall(HloInstruction* inst) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog,
      CreateCustomCallOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);

  return Status::OK();
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
  TF_ASSIGN_OR_RETURN(
      ArgVectors inputs,
      GetInplaceOutputTensors(tensor_map, resources_, inst, sequence));
  CHECK_EQ(inputs.size(), inst->operand_count());
  uint64 n = 0;
  for (uint64 i = 0; i < inputs.size(); i++) {
    CHECK_EQ(inputs[i].size(), CountShapes(inst->operand(i)->shape()));
    for (uint64 j = 0; j < inputs[i].size(); j++) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, n, inputs[i][j]));
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
  TF_ASSIGN_OR_RETURN(simple_parallel, IsParallelMap(inst, inst->to_apply()));
  if (simple_parallel) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog,
        CreateParallelMap(resources_, inst, GetOutputShape(inst), tensor_map));
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
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateIfOp(resources_, inst, GetOutputShape(inst), tensor_map));
  sequence.add(prog);

  return Status::OK();
}

Status BaseVisitor::HandleReal(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor in;
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(
      in, FindInstructionInput(tensor_map, resources_, inst, 0, sequence));

  out = GetGraph(resources_, inst).clone(in);
  sequence.add(poplar::program::Copy(in, out));
  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));

  return Status::OK();
}

Status BaseVisitor::HandlePad(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReducePrecision(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleInfeed(HloInstruction* inst) {
  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(inst);
  // We allow the same infeed queue to be dequeued multiple times, however
  // we don't support multiple infeed queues in the same program.
  if (absl::c_any_of(resources_.annotations.infeed_infos,
                     [&](const HloInfeedInstruction* inst) {
                       return inst->infeed_config() != infeed->infeed_config();
                     })) {
    LOG(FATAL) << "Currently multiple infeed queues in the same program are "
                  "not supported.";
  }
  std::vector<Shape> shapes = FlattenedXlaShape(infeed->infeed_shape());

  poplar::Graph& graph = GetGraph(resources_, infeed);
  poplar::program::Sequence& seq = sequence;

  for (unsigned i = 0; i < shapes.size(); i++) {
    const auto& shape = shapes[i];
    TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                        AddTensor(graph, std::make_pair(infeed, i), shape,
                                  resources_, tensor_map));

    if (!UseSyntheticData()) {
      auto fifo =
          graph.addHostToDeviceFIFO(GetInfeedCopyHandle(infeed->name(), i),
                                    out.elementType(), out.numElements());

      seq.add(poplar::program::Copy(fifo, out, false));
    }

    TF_CHECK_OK(AddOutputTensor(tensor_map, infeed, i, out));
  }
  resources_.annotations.infeed_infos.insert(infeed);

  return Status::OK();
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

Status BaseVisitor::HandleGather(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleAfterAll(HloInstruction* inst) {
  // TODO(shauryas) : figure out how to use this for something useful
  return Status::OK();
}

Status BaseVisitor::HandleIota(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleScatter(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleAllToAll(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleCollectivePermute(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleGetDimensionSize(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleAddDependency(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReplicaId(HloInstruction* inst) {
  return Unimplemented(inst);
}

}  // namespace poplarplugin
}  // namespace xla
