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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/arg_min_max.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <popnn/Loss.hpp>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {
namespace {

class ArgMinMaxOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // Create the control program.
    poplar::program::Sequence seq;

    // Get the input.
    poplar::Tensor input =
        FindInstructionInputs(tensor_map, res, inst, 0, seq, false)[0];

    bool is_max = DynCast<HloArgMax>(inst) != nullptr;

    if (!is_max && DynCast<HloArgMin>(inst) == nullptr) {
      return xla::FailedPrecondition(
          "Expected HLO instruction to be one of HloArgMax or HloArgMin!");
    }

    int64 axis = DynCast<HloArgMinMax>(inst)->Axis();

    std::vector<std::size_t> index_shape;

    if (axis != 0) {
      // Roll the axis dim to the end.
      input = input.dimRoll(axis, input.rank() - 1);

      // Use the remaining dims as the dims of the output.
      index_shape = input.shape();

      // Remove the last element.
      index_shape.pop_back();

      std::size_t sum = std::accumulate(index_shape.begin(), index_shape.end(),
                                        1, std::multiplies<std::size_t>());

      // Flatten the remaining dims as popnn expects a 2d input.
      input = input.reshapePartial(0, input.rank() - 1, {sum});
    } else {
      input = input.reshape({1, input.numElements()});
      index_shape = {1};
    }

    // Call into the
    poplar::Tensor output;
    if (is_max) {
      output = popnn::argMax(graph, input, seq, "ArgMax");
    } else {
      output = popnn::argMin(graph, input, seq, "ArgMin");
    }
    output = output.reinterpret(poplar::INT);

    // Reshape the output to be the actual arangement of the index, it will be a
    // 1D vector.
    output = output.reshape(index_shape);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};

REGISTER_POPLIBS_OP(Popnn, ArgMax, ArgMinMaxOp);
REGISTER_POPLIBS_OP(Popnn, ArgMin, ArgMinMaxOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
