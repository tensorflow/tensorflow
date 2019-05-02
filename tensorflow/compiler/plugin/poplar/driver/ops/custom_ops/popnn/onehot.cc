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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/onehot.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <popops/Encoding.hpp>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {
namespace {

class OneHotOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    // Create the control program.
    poplar::program::Sequence seq;

    // We expect only three arguments. Other two, depth and axis, are expected
    // to be compile time constants.
    ArgVector indices =
        FindInstructionInputs(tensor_map, res, inst, 0, seq, false);

    ArgVector on = FindInstructionInputs(tensor_map, res, inst, 1, seq, false);

    ArgVector off = FindInstructionInputs(tensor_map, res, inst, 2, seq, false);

    const poplarplugin::HloOneHotInstruction* one_hot_op =
        dynamic_cast<const HloOneHotInstruction*>(inst);
    if (!one_hot_op) {
      return xla::FailedPrecondition(
          "Expected HLO instruction to be HloOneHotInstruction!");
    }
    int64 depth = one_hot_op->Depth();

    // flatten all but one-hot axis
    poplar::Tensor indices_tensor = indices[0].flatten();

    xla::Shape tmp_output_shape = XlaShapeFromPoplarShape(
        output_shape.element_type(), indices_tensor.shape());
    tmp_output_shape.add_dimensions(depth);

    // Create the output tensor to store the result in (as popops takes this by
    // reference rather than returning the output).
    TF_ASSIGN_OR_RETURN(poplar::Tensor output,
                        AddTensor(graph, std::make_pair(inst, 0),
                                  tmp_output_shape, res, tensor_map));

    poplar::Tensor on_tensor = on[0];
    poplar::Tensor off_tensor = off[0];

    bool is_input_floating_point = on_tensor.elementType() == poplar::FLOAT ||
                                   on_tensor.elementType() == poplar::HALF;

    // Encode one hot returns void but stores output in "output".
    popops::encodeOneHot(graph, indices_tensor, output, seq, on_tensor,
                         off_tensor, "OneHot");

    // Reshape to the input size
    output = output.reshapePartial(0, 1, indices[0].shape());

    int64 axis = one_hot_op->Axis();
    output = output.dimRoll(output.rank() - 1, axis);

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};

REGISTER_POPLIBS_OP(Popnn, OneHot, OneHotOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
