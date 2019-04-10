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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/lstm.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloLSTMInstruction::HloLSTMInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::string& custom_call_target, bool is_training, int32 num_channels,
    xla::PrimitiveType partials_type)
    : HloPoplarInstruction(shape, operands, custom_call_target, {}),
      is_training_(is_training),
      num_channels_(num_channels),
      partials_type_(partials_type) {}

bool HloLSTMInstruction::is_training() const { return is_training_; }
int32 HloLSTMInstruction::num_channels() const { return num_channels_; }
xla::PrimitiveType HloLSTMInstruction::partials_type() const {
  return partials_type_;
}

std::vector<std::string> HloLSTMInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("is_training=" + std::to_string(is_training_));
  attributes.push_back("num_channels=" + std::to_string(num_channels_));
  attributes.push_back("partials_type=" +
                       xla::PrimitiveType_Name(partials_type_));

  return attributes;
}

HloLSTMFwdInstruction::HloLSTMFwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type)
    : HloLSTMInstruction(shape, operands,
                         GetPoplibsCustomOpTargetString(
                             PoplibsOp::Popnn, PoplibsOp::LstmLayerFwd),
                         is_training, num_channels, partials_type) {}

absl::flat_hash_set<int64> HloLSTMFwdInstruction::AllocatingIndices() const {
  return {0, 1, 2, 3, 4};
}

absl::flat_hash_map<int64, int64> HloLSTMFwdInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloLSTMFwdInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloLSTMFwdInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloLSTMFwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateLSTMFwd(shape, operands, is_training(), num_channels(),
                       partials_type());
}

std::unique_ptr<HloInstruction> CreateLSTMFwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type) {
  return absl::make_unique<HloLSTMFwdInstruction>(shape, operands, is_training,
                                                  num_channels, partials_type);
}

HloLSTMBwdInstruction::HloLSTMBwdInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type)
    : HloLSTMInstruction(shape, operands,
                         GetPoplibsCustomOpTargetString(
                             PoplibsOp::Popnn, PoplibsOp::LstmLayerBwd),
                         is_training, num_channels, partials_type) {}

absl::flat_hash_set<int64> HloLSTMBwdInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloLSTMBwdInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloLSTMBwdInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloLSTMBwdInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction> HloLSTMBwdInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* ctx) const {
  return CreateLSTMBwd(shape, operands, is_training(), num_channels(),
                       partials_type());
}

std::unique_ptr<HloInstruction> CreateLSTMBwd(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    bool is_training, int32 num_channels, xla::PrimitiveType partials_type) {
  return absl::make_unique<HloLSTMBwdInstruction>(shape, operands, is_training,
                                                  num_channels, partials_type);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloLSTMFwdFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(int32 num_channels,
                      attribute_map.GetAttributeAsInt("num_channels"));

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));

  TF_ASSIGN_OR_RETURN(tensorflow::DataType partials_dtype,
                      attribute_map.GetAttributeAsTFDataType("partials_dtype"));

  xla::PrimitiveType partials_xla_type;
  TF_CHECK_OK(DataTypeToPrimitiveType(partials_dtype, &partials_xla_type));

  return CreateLSTMFwd(call->shape(), call->operands(), is_training,
                       num_channels, partials_xla_type);
}

static HloPoplarInstructionFactory lstm_fwd_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::LstmLayerFwd),
    HloLSTMFwdFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloLSTMBwdFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(int32 num_channels,
                      attribute_map.GetAttributeAsInt("num_channels"));

  TF_ASSIGN_OR_RETURN(bool is_training,
                      attribute_map.GetAttributeAsBool("is_training"));

  TF_ASSIGN_OR_RETURN(tensorflow::DataType partials_dtype,
                      attribute_map.GetAttributeAsTFDataType("partials_dtype"));

  xla::PrimitiveType partials_xla_type;
  TF_CHECK_OK(DataTypeToPrimitiveType(partials_dtype, &partials_xla_type));

  return CreateLSTMBwd(call->shape(), call->operands(), is_training,
                       num_channels, partials_xla_type);
}

static HloPoplarInstructionFactory lstm_bwd_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::LstmLayerBwd),
    HloLSTMBwdFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
