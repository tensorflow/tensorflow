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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloPoolingInstruction::HloPoolingInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, xla::Window window)
    : HloPoplarInstruction(shape, operands, custom_call_target, {}),
      window_(window) {}

const xla::Window& HloPoolingInstruction::window() const { return window_; }

absl::flat_hash_set<int64> HloPoolingInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloPoolingInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloPoolingInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloPoolingInstruction::IsPopOpsElementwise() const { return false; }

std::vector<std::string> HloPoolingInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {};
}

HloMaxPoolInstruction::HloMaxPoolInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             xla::Window window)
    : HloPoolingInstruction(
          shape, {operand},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::MaxPool),
          window) {}

const HloInstruction* HloMaxPoolInstruction::to_reduce() const {
  return operand(0);
}

HloInstruction* HloMaxPoolInstruction::mutable_to_reduce() {
  return mutable_operand(0);
}

std::unique_ptr<HloInstruction> HloMaxPoolInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateMaxPool(shape, operands[0], window());
}

std::unique_ptr<HloInstruction> CreateMaxPool(const Shape& shape,
                                              HloInstruction* operand,
                                              xla::Window window) {
  return absl::make_unique<HloMaxPoolInstruction>(shape, operand, window);
}

HloAvgPoolInstruction::HloAvgPoolInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             xla::Window window)
    : HloPoolingInstruction(
          shape, {operand},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::AvgPool),
          window) {}

const HloInstruction* HloAvgPoolInstruction::to_reduce() const {
  return operand(0);
}

HloInstruction* HloAvgPoolInstruction::mutable_to_reduce() {
  return mutable_operand(0);
}

std::unique_ptr<HloInstruction> HloAvgPoolInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateAvgPool(shape, operands[0], window());
}

std::unique_ptr<HloInstruction> CreateAvgPool(const Shape& shape,
                                              HloInstruction* operand,
                                              xla::Window window) {
  return absl::make_unique<HloAvgPoolInstruction>(shape, operand, window);
}

HloMaxPoolGradInstruction::HloMaxPoolGradInstruction(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    HloInstruction* output_grad, xla::Window window)
    : HloPoolingInstruction(shape, {input, output, output_grad},
                            GetPoplibsCustomOpTargetString(
                                PoplibsOp::Popnn, PoplibsOp::MaxPoolGrad),
                            window) {}

const HloInstruction* HloMaxPoolGradInstruction::input() const {
  return operand(0);
}
const HloInstruction* HloMaxPoolGradInstruction::output() const {
  return operand(1);
}
const HloInstruction* HloMaxPoolGradInstruction::output_grad() const {
  return operand(2);
}

HloInstruction* HloMaxPoolGradInstruction::mutable_input() {
  return mutable_operand(0);
}
HloInstruction* HloMaxPoolGradInstruction::mutable_output() {
  return mutable_operand(1);
}
HloInstruction* HloMaxPoolGradInstruction::mutable_output_grad() {
  return mutable_operand(2);
}

std::unique_ptr<HloInstruction>
HloMaxPoolGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateMaxPoolGrad(shape, operands[0], operands[1], operands[2],
                           window());
}

std::unique_ptr<HloInstruction> CreateMaxPoolGrad(const Shape& shape,
                                                  HloInstruction* input,
                                                  HloInstruction* output,
                                                  HloInstruction* output_grad,
                                                  xla::Window window) {
  return absl::make_unique<HloMaxPoolGradInstruction>(shape, input, output,
                                                      output_grad, window);
}

HloAvgPoolGradInstruction::HloAvgPoolGradInstruction(
    const Shape& shape, HloInstruction* output_grad, xla::Window window)
    : HloPoolingInstruction(shape, {output_grad},
                            GetPoplibsCustomOpTargetString(
                                PoplibsOp::Popnn, PoplibsOp::AvgPoolGrad),
                            window) {}

const HloInstruction* HloAvgPoolGradInstruction::output_grad() const {
  return operand(0);
}
HloInstruction* HloAvgPoolGradInstruction::mutable_output_grad() {
  return mutable_operand(0);
}

std::unique_ptr<HloInstruction>
HloAvgPoolGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateAvgPoolGrad(shape, operands[0], window());
}

std::unique_ptr<HloInstruction> CreateAvgPoolGrad(const Shape& shape,
                                                  HloInstruction* output_grad,
                                                  xla::Window window) {
  return absl::make_unique<HloAvgPoolGradInstruction>(shape, output_grad,
                                                      window);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> MaxPoolFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateMaxPool(call->shape(), call->mutable_operand(0), window);
}

static HloPoplarInstructionFactory max_pool_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::MaxPool),
    MaxPoolFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> AvgPoolFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateAvgPool(call->shape(), call->mutable_operand(0), window);
}

static HloPoplarInstructionFactory avg_pool_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::AvgPool),
    AvgPoolFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> MaxPoolGradFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateMaxPoolGrad(call->shape(), call->mutable_operand(0),
                           call->mutable_operand(1), call->mutable_operand(2),
                           window);
}

static HloPoplarInstructionFactory max_pool_grad_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::MaxPoolGrad),
    MaxPoolGradFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> AvgPoolGradFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(xla::Window window,
                      attribute_map.GetAttributeAsWindow("window"));

  return CreateAvgPoolGrad(call->shape(), call->mutable_operand(0), window);
}

static HloPoplarInstructionFactory avg_pool_grad_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::AvgPoolGrad),
    AvgPoolGradFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
