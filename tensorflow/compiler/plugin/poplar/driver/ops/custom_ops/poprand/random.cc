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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateless_random.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include "tensorflow/stream_executor/lib/statusor.h"

#include "absl/container/flat_hash_map.h"

#include <string>

namespace xla {
namespace poplarplugin {
namespace {

class TruncatedNormalOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    return TruncatedNormal(res, inst, output_shape, tensor_map);
  }
};
REGISTER_POPLIBS_OP(Poprand, TruncatedNormal, TruncatedNormalOp);

class StatelessRandomUniformOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor seed,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddTensor(graph, std::make_pair(inst, 0), output_shape,
                                  res, tensor_map));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::uniform(graph, &seed, 0, ref, dtype, 0.0, 1.0, seq,
                                GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poprand, StatelessRandomUniform, StatelessRandomUniformOp);

class StatelessRandomUniformIntOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor seed,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    const HloInstruction* lower = inst->operand(1);
    CHECK_EQ(lower->opcode(), HloOpcode::kConstant);
    const HloInstruction* upper = inst->operand(2);
    CHECK_EQ(upper->opcode(), HloOpcode::kConstant);

    TF_ASSIGN_OR_RETURN(int lower_val,
                        LiteralScalarToNativeType<int>(lower->literal()));
    TF_ASSIGN_OR_RETURN(int upper_val,
                        LiteralScalarToNativeType<int>(upper->literal()));

    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddTensor(graph, std::make_pair(inst, 0), output_shape,
                                  res, tensor_map));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::uniform(graph, &seed, 0, ref, dtype, lower_val,
                                upper_val, seq, GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poprand, StatelessRandomUniformInt,
                    StatelessRandomUniformIntOp);

class StatelessRandomNormalOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    poplar::Tensor seed;
    TF_ASSIGN_OR_RETURN(seed,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddTensor(graph, std::make_pair(inst, 0), output_shape,
                                  res, tensor_map));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::normal(graph, &seed, 0, ref, dtype, 0.0, 1.0, seq,
                               GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poprand, StatelessRandomNormal, StatelessRandomNormalOp);

class StatelessTruncatedNormalOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor seed,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));
    seed = seed.reinterpret(poplar::UNSIGNED_INT);

    TF_ASSIGN_OR_RETURN(poplar::Tensor ref,
                        AddTensor(graph, std::make_pair(inst, 0), output_shape,
                                  res, tensor_map));

    TF_ASSIGN_OR_RETURN(poplar::Type dtype, PoplarDataType(output_shape));

    auto out = poprand::truncatedNormal(graph, &seed, 0, ref, dtype, 0.0, 1.0,
                                        1.0, seq, GetDebugName(inst));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};
REGISTER_POPLIBS_OP(Poprand, StatelessTruncatedNormal,
                    StatelessTruncatedNormalOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
