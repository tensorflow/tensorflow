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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"

#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/Util.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {
namespace {

void ZeroTensor(poplar::Graph& graph, poplar::Tensor& tensor,
                poplar::program::Sequence& seq) {
  auto zero =
      graph.addConstant(tensor.elementType(), tensor.shape(), 0, "Zero");
  graph.setTileMapping(zero, 0);
  poputil::broadcastToMatch(tensor, zero);
  seq.add(poplar::program::Copy(zero, tensor));
}

class StatefulGradientAccumulateOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::Graph& master_graph = GetMasterGraph(res);
    poplar::program::Sequence seq;

    const HloStatefulGradientAccumulate* grad_inst =
        Cast<HloStatefulGradientAccumulate>(inst);

    TF_ASSIGN_OR_RETURN(
        ArgVectors inputs,
        FindInplaceOutputTensors(tensor_map, res, inst, seq, false));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    poplar::Tensor input = inputs[0][0];
    poplar::Tensor counter = master_graph.addVariable(
        poplar::UNSIGNED_INT, {}, GetDebugName(inst) + "/Counter");
    graph.setTileMapping(counter, 0);

    poplar::Tensor accumulator =
        graph.clone(input, GetDebugName(inst) + "/Accumulator");
    // Accumulate the input into the buffer.
    popops::addInPlace(graph, accumulator, input, seq,
                       GetDebugName(inst) + "/Accumulate");

    // Output the accumulated gradients if counter == MiniBatchesToAccumulate -
    // 1 otherwise output all zeros.
    poplar::Tensor output_grads = popops::map(
        master_graph,
        pe::Equal(pe::_1, pe::Const(grad_inst->MiniBatchesToAccumulate() - 1)),
        {counter}, seq, GetDebugName(inst) + "/CheckOutputGradients");
    poplar::program::Sequence if_true;
    {
      // Copy accumulator into input.
      if_true.add(poplar::program::Copy(accumulator, input));
      // Zero the accumulator.
      ZeroTensor(graph, accumulator, if_true);
      // Zero the counter.
      ZeroTensor(master_graph, counter, if_true);
    }
    poplar::program::Sequence if_false;
    {
      // Set input to all zeros.
      ZeroTensor(graph, input, if_false);
      // Increase counter.
      popops::mapInPlace(master_graph, pe::Add(pe::_1, pe::Const(1)), {counter},
                         if_false, GetDebugName(inst) + "/IncreaseCounter");
    }
    seq.add(poplar::program::If(output_grads, if_true, if_false));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, input));

    return seq;
  }
};
REGISTER_POPLIBS_OP(Poputil, StatefulGradientAccumulate,
                    StatefulGradientAccumulateOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
