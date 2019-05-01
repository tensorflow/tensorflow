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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include <popops/ElementWise.hpp>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include <poputil/TileMapping.hpp>
#include <random>

namespace xla {
namespace poplarplugin {
namespace {
class DropoutOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape&,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;
    // Get the "x" tensor, aka the input.
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, false));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in_seed,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    const HloDropoutInstruction* dropout_instruction =
        dynamic_cast<const HloDropoutInstruction*>(inst);
    assert(op && "Expected operation to be an xla::poplarplugin::DropoutOp");

    // The probabilty that any given element of "x" will be disgarded.
    double rate = dropout_instruction->Rate();

    // The value to scale the non-droped elements by.
    double scale = dropout_instruction->Scale();

    // If false, pass in user_seed, else pass in global_seed.
    bool is_user_seed = dropout_instruction->IsUserSeed();
    int32_t seed_modifier = dropout_instruction->SeedModifier();

    // The global seed value we use.
    std::srand(std::time(nullptr));
    int32_t global_seed = std::rand();

    // Create an empty tensor for the dropout. This is internal to the poprand
    // implementation but is exposed anyway so we need to provide it.
    TF_ASSIGN_OR_RETURN(poplar::Tensor reference,
                        AddTensor(graph, std::make_pair(inst, 0),
                                  XlaShapeFromPoplarShape(
                                      xla::PrimitiveType::F32, input.shape()),
                                  res, tensor_map));

    xla::Shape seed_shape =
        XlaShapeFromPoplarShape(xla::PrimitiveType::U32, {2});

    // By default we will use any seed provided by the user.
    poplar::Tensor* seed_to_use = &in_seed;

    poplar::Tensor global_seed_tensor;
    // If we aren't using a user provided seed we need to create a temp seed and
    // use that.
    if (!is_user_seed) {
      // Create the variable to hold the seed state.s
      global_seed_tensor = graph.addVariable(poplar::INT, {2});
      poputil::mapTensorLinearly(graph, global_seed_tensor);

      // Create the literal value to add onto the seed each iteration.
      int32_t increment_literal[] = {0, 1};
      poplar::Tensor increment_tensor =
          graph.addConstant(poplar::INT, {2}, increment_literal);
      poputil::mapTensorLinearly(graph, increment_tensor);

      // Add one to the seed state so we get a different number on each call.
      popops::addInPlace(graph, global_seed_tensor, increment_tensor, seq);
      seed_to_use = &global_seed_tensor;
    }

    // Dropout expects an unsigned int but tensorflow takes in int32 when
    // targeting IPU.
    poplar::Tensor as_unsgined = seed_to_use->reinterpret(poplar::UNSIGNED_INT);

    // Perform the actual dropout by calling into the poprand function.
    poplar::Tensor final_output =
        poprand::dropout(graph, &as_unsgined, seed_modifier, input, reference,
                         rate, scale, seq, "Dropout");

    // Mark that tensor as our output.
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, final_output));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 1, *seed_to_use));

    return seq;
  }
};

REGISTER_POPLIBS_OP(Poprand, Dropout, DropoutOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
