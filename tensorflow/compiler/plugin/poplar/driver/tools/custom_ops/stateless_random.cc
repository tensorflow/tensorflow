/* Copyright 2019 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateless_random.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloStatelessRandom::HloStatelessRandom(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::string& op_string)
    : HloPoplarInstruction(shape, operands, op_string, {}) {}

HloStatelessRandomUniform::HloStatelessRandomUniform(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(
          shape, operands,
          GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                         PoplibsOp::StatelessRandomUniform)) {}

HloStatelessRandomUniformInt::HloStatelessRandomUniformInt(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(
          shape, operands,
          GetPoplibsCustomOpTargetString(
              PoplibsOp::Poprand, PoplibsOp::StatelessRandomUniformInt)) {}

HloStatelessRandomNormal::HloStatelessRandomNormal(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(
          shape, operands,
          GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                         PoplibsOp::StatelessRandomNormal)) {}

HloStatelessTruncatedNormal::HloStatelessTruncatedNormal(
    const Shape& shape, absl::Span<HloInstruction* const> operands)
    : HloStatelessRandom(
          shape, operands,
          GetPoplibsCustomOpTargetString(
              PoplibsOp::Poprand, PoplibsOp::StatelessTruncatedNormal)) {}

namespace {

static HloPoplarInstructionFactory stateless_random_uniform_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                   PoplibsOp::StatelessRandomUniform),
    [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessRandomUniform>(call->shape(),
                                                       call->operands());
      return inst;
    });

static HloPoplarInstructionFactory stateless_random_uniform_int_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                   PoplibsOp::StatelessRandomUniformInt),
    [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessRandomUniformInt>(call->shape(),
                                                          call->operands());
      return inst;
    });

static HloPoplarInstructionFactory stateless_random_normal_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                   PoplibsOp::StatelessRandomNormal),
    [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessRandomNormal>(call->shape(),
                                                      call->operands());
      return inst;
    });

static HloPoplarInstructionFactory stateless_truncated_normal_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                   PoplibsOp::StatelessTruncatedNormal),
    [](HloCustomCallInstruction* call) {
      std::unique_ptr<HloInstruction> inst =
          absl::make_unique<HloStatelessTruncatedNormal>(call->shape(),
                                                         call->operands());
      return inst;
    });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
