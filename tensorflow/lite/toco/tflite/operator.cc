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
#include "tensorflow/lite/toco/tflite/operator.h"

#include <map>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/util/ptr_util.h"

// TODO(ycling): Consider refactoring to extract the LSTM definition out of
// graph_transformation module.
#include "tensorflow/lite/delegates/flex/whitelisted_flex_ops.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/graph_transformations/lstm_utils.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/builtin_operator.h"
#include "tensorflow/lite/toco/tflite/custom_operator.h"
#include "tensorflow/lite/toco/tflite/simple_operator.h"
#include "tensorflow/lite/toco/tflite/types.h"
#include "tensorflow/lite/tools/versioning/op_version.h"

namespace toco {

namespace tflite {

// LINT.IfChange

::tflite::TensorType GetTensorType(const ArrayDataType type) {
  const std::map<ArrayDataType, ::tflite::TensorType> tensor_type_map = {
      {ArrayDataType::kBool, ::tflite::TensorType_BOOL},
      {ArrayDataType::kFloat, ::tflite::TensorType_FLOAT32},
      {ArrayDataType::kInt8, ::tflite::TensorType_INT8},
      {ArrayDataType::kUint8, ::tflite::TensorType_UINT8},
      {ArrayDataType::kInt16, ::tflite::TensorType_INT16},
      {ArrayDataType::kInt32, ::tflite::TensorType_INT32},
      {ArrayDataType::kInt64, ::tflite::TensorType_INT64},
      {ArrayDataType::kString, ::tflite::TensorType_STRING},
      {ArrayDataType::kComplex64, ::tflite::TensorType_COMPLEX64},
      {ArrayDataType::kFloat16, ::tflite::TensorType_FLOAT16}};

  auto it = tensor_type_map.find(type);
  if (it != tensor_type_map.end()) {
    return it->second;
  }
  return static_cast<::tflite::TensorType>(-1);
}

::tflite::OpSignature GetVersioningOpSig(
    const ::tflite::BuiltinOperator op, const OperatorSignature& op_signature) {
  std::vector<::tflite::TensorType> input_types, output_types;
  for (auto input_name : op_signature.op->inputs) {
    ::tflite::TensorType input_type = static_cast<::tflite::TensorType>(-1);
    if (op_signature.model->HasArray(input_name)) {
      const Array& input_array = op_signature.model->GetArray(input_name);
      input_type = GetTensorType(input_array.data_type);
    }
    input_types.push_back(input_type);
  }
  for (auto output_name : op_signature.op->outputs) {
    ::tflite::TensorType output_type = static_cast<::tflite::TensorType>(-1);
    if (op_signature.model->HasArray(output_name)) {
      const Array& output_array = op_signature.model->GetArray(output_name);
      output_type = GetTensorType(output_array.data_type);
    }
    output_types.push_back(output_type);
  }
  return ::tflite::OpSignature{op, input_types, output_types};
}

class AveragePool
    : public BuiltinOperator<AveragePoolOperator, ::tflite::Pool2DOptions,
                             ::tflite::BuiltinOptions_Pool2DOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreatePool2DOptions(*builder, padding, op.stride_width,
                                         op.stride_height, op.kwidth,
                                         op.kheight, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->kwidth = options.filter_width();
    op->kheight = options.filter_height();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class Convolution
    : public BuiltinOperator<ConvOperator, ::tflite::Conv2DOptions,
                             ::tflite::BuiltinOptions_Conv2DOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateConv2DOptions(*builder, padding, op.stride_width,
                                         op.stride_height, activation_function,
                                         op.dilation_width_factor,
                                         op.dilation_height_factor);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->dilation_width_factor = options.dilation_w_factor();
    op->dilation_height_factor = options.dilation_h_factor();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class DepthwiseConvolution
    : public BuiltinOperator<DepthwiseConvOperator,
                             ::tflite::DepthwiseConv2DOptions,
                             ::tflite::BuiltinOptions_DepthwiseConv2DOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateDepthwiseConv2DOptions(
        *builder, padding, op.stride_width, op.stride_height,
        op.depth_multiplier, activation_function, op.dilation_width_factor,
        op.dilation_height_factor);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->depth_multiplier = options.depth_multiplier();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
    op->dilation_width_factor = options.dilation_w_factor();
    op->dilation_height_factor = options.dilation_h_factor();
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    const auto& conv_op =
        static_cast<const DepthwiseConvOperator&>(*op_signature.op);
    ::tflite::OpSignature op_sig =
        GetVersioningOpSig(builtin_op(), op_signature);
    op_sig.options.depthwise_conv_2d.dilation_w_factor =
        conv_op.dilation_width_factor;
    op_sig.options.depthwise_conv_2d.dilation_h_factor =
        conv_op.dilation_height_factor;
    return ::tflite::GetBuiltinOperatorVersion(op_sig);
  }
};

class Add : public BuiltinOperator<AddOperator, ::tflite::AddOptions,
                                   ::tflite::BuiltinOptions_AddOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateAddOptions(*builder, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class AddN : public BuiltinOperator<AddNOperator, ::tflite::AddNOptions,
                                    ::tflite::BuiltinOptions_AddNOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateAddNOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class SpaceToBatchND
    : public BuiltinOperator<SpaceToBatchNDOperator,
                             ::tflite::SpaceToBatchNDOptions,
                             ::tflite::BuiltinOptions_SpaceToBatchNDOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSpaceToBatchNDOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Sub : public BuiltinOperator<SubOperator, ::tflite::SubOptions,
                                   ::tflite::BuiltinOptions_SubOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateSubOptions(*builder, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class Div : public BuiltinOperator<DivOperator, ::tflite::DivOptions,
                                   ::tflite::BuiltinOptions_DivOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateDivOptions(*builder, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class BatchToSpaceND
    : public BuiltinOperator<BatchToSpaceNDOperator,
                             ::tflite::BatchToSpaceNDOptions,
                             ::tflite::BuiltinOptions_BatchToSpaceNDOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateBatchToSpaceNDOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Cast : public BuiltinOperator<CastOperator, ::tflite::CastOptions,
                                    ::tflite::BuiltinOptions_CastOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateCastOptions(*builder,
                                       DataType::Serialize(op.src_data_type),
                                       DataType::Serialize(op.dst_data_type));
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->src_data_type = DataType::Deserialize(options.in_data_type());
    op->dst_data_type = DataType::Deserialize(options.out_data_type());
  }
};

class Concatenation
    : public BuiltinOperator<ConcatenationOperator,
                             ::tflite::ConcatenationOptions,
                             ::tflite::BuiltinOptions_ConcatenationOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateConcatenationOptions(*builder, op.axis);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->axis = options.axis();
  }
};

class DepthToSpace
    : public BuiltinOperator<DepthToSpaceOperator,
                             ::tflite::DepthToSpaceOptions,
                             ::tflite::BuiltinOptions_DepthToSpaceOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateDepthToSpaceOptions(*builder, op.block_size);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->block_size = options.block_size();
  }
};

class FakeQuant
    : public BuiltinOperator<FakeQuantOperator, ::tflite::FakeQuantOptions,
                             ::tflite::BuiltinOptions_FakeQuantOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateFakeQuantOptions(
        *builder, op.minmax->min, op.minmax->max, op.num_bits, op.narrow_range);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    auto* minmax = new MinMax;
    minmax->min = options.min();
    minmax->max = options.max();
    op->minmax.reset(minmax);
    op->num_bits = options.num_bits();
    op->narrow_range = options.narrow_range();
  }
  int GetVersion(const OperatorSignature& op_signature) const override {
    const auto& fq_op = static_cast<const FakeQuantOperator&>(*op_signature.op);
    ::tflite::OpSignature op_sig =
        GetVersioningOpSig(builtin_op(), op_signature);
    op_sig.options.fakequant.narrow_range = fq_op.narrow_range;
    return ::tflite::GetBuiltinOperatorVersion(op_sig);
  }
};

class FullyConnected
    : public BuiltinOperator<FullyConnectedOperator,
                             ::tflite::FullyConnectedOptions,
                             ::tflite::BuiltinOptions_FullyConnectedOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  ::tflite::FullyConnectedOptionsWeightsFormat GetWeightFormat(
      FullyConnectedWeightsFormat fmt) const {
    switch (fmt) {
      case FullyConnectedWeightsFormat::kDefault:
        return ::tflite::FullyConnectedOptionsWeightsFormat_DEFAULT;
      case FullyConnectedWeightsFormat::kShuffled4x16Int8:
        return ::tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8;
      default:
        LOG(ERROR) << "Unhandled FC weights format";
        return ::tflite::FullyConnectedOptionsWeightsFormat_DEFAULT;
    }
  }

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateFullyConnectedOptions(
        *builder, activation_function, GetWeightFormat(op.weights_format));
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
    switch (options.weights_format()) {
      case ::tflite::FullyConnectedOptionsWeightsFormat_DEFAULT:
        op->weights_format = FullyConnectedWeightsFormat::kDefault;
        break;
      case ::tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
        op->weights_format = FullyConnectedWeightsFormat::kShuffled4x16Int8;
        break;
      default:
        LOG(ERROR) << "Unhandled FC weights format";
        op->weights_format = FullyConnectedWeightsFormat::kDefault;
    }
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    const auto& fc_op =
        static_cast<const FullyConnectedOperator&>(*op_signature.op);
    ::tflite::OpSignature op_sig =
        GetVersioningOpSig(builtin_op(), op_signature);
    op_sig.options.fully_connected.keep_num_dims = fc_op.keep_num_dims;
    op_sig.options.fully_connected.weights_format =
        GetWeightFormat(fc_op.weights_format);
    return ::tflite::GetBuiltinOperatorVersion(op_sig);
  }
};

class Gather : public BuiltinOperator<GatherOperator, ::tflite::GatherOptions,
                                      ::tflite::BuiltinOptions_GatherOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    int axis = op.axis ? op.axis.value() : 0;
    return ::tflite::CreateGatherOptions(*builder, axis);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->axis = {options.axis()};
  }
};

class GatherNd
    : public BuiltinOperator<GatherNdOperator, ::tflite::GatherNdOptions,
                             ::tflite::BuiltinOptions_GatherNdOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateGatherNdOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Svdf : public BuiltinOperator<SvdfOperator, ::tflite::SVDFOptions,
                                    ::tflite::BuiltinOptions_SVDFOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateSVDFOptions(*builder, op.rank, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
    op->rank = options.rank();
  }
};

class L2Normalization
    : public BuiltinOperator<L2NormalizationOperator, ::tflite::L2NormOptions,
                             ::tflite::BuiltinOptions_L2NormOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateL2NormOptions(*builder, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class L2Pool : public BuiltinOperator<L2PoolOperator, ::tflite::Pool2DOptions,
                                      ::tflite::BuiltinOptions_Pool2DOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreatePool2DOptions(*builder, padding, op.stride_width,
                                         op.stride_height, op.kwidth,
                                         op.kheight, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->kwidth = options.filter_width();
    op->kheight = options.filter_height();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class LocalResponseNormalization
    : public BuiltinOperator<
          LocalResponseNormalizationOperator,
          ::tflite::LocalResponseNormalizationOptions,
          ::tflite::BuiltinOptions_LocalResponseNormalizationOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateLocalResponseNormalizationOptions(
        *builder, op.range, op.bias, op.alpha, op.beta);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->range = options.radius();
    op->bias = options.bias();
    op->alpha = options.alpha();
    op->beta = options.beta();
  }
};

class MaxPool : public BuiltinOperator<MaxPoolOperator, ::tflite::Pool2DOptions,
                                       ::tflite::BuiltinOptions_Pool2DOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreatePool2DOptions(*builder, padding, op.stride_width,
                                         op.stride_height, op.kwidth,
                                         op.kheight, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->kwidth = options.filter_width();
    op->kheight = options.filter_height();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }
};

class Mul : public BuiltinOperator<MulOperator, ::tflite::MulOptions,
                                   ::tflite::BuiltinOptions_MulOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateMulOptions(*builder, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    const string& input1_name = op_signature.op->inputs[0];
    const string& input2_name = op_signature.op->inputs[1];
    const string& output_name = op_signature.op->outputs[0];
    const Array& input1_array = op_signature.model->GetArray(input1_name);
    const Array& input2_array = op_signature.model->GetArray(input2_name);
    const Array& output_array = op_signature.model->GetArray(output_name);
    const auto& input1_quant = input1_array.quantization_params;
    const auto& input2_quant = input2_array.quantization_params;
    const auto& output_quant = output_array.quantization_params;
    const float input1_scale = input1_quant ? input1_quant->scale : 0.0f;
    const float input2_scale = input2_quant ? input2_quant->scale : 0.0f;
    const float output_scale = output_quant ? output_quant->scale : 0.0f;
    ::tflite::OpSignature op_sig =
        GetVersioningOpSig(builtin_op(), op_signature);
    op_sig.options.mul.input1_scale = input1_scale;
    op_sig.options.mul.input2_scale = input2_scale;
    op_sig.options.mul.output_scale = output_scale;
    return ::tflite::GetBuiltinOperatorVersion(op_sig);
  }
};

class Pad : public BuiltinOperator<PadOperator, ::tflite::PadOptions,
                                   ::tflite::BuiltinOptions_PadOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreatePadOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Tile
    : public BuiltinOperator<TensorFlowTileOperator, ::tflite::TileOptions,
                             ::tflite::BuiltinOptions_TileOptions> {
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateTileOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class PadV2 : public BuiltinOperator<PadV2Operator, ::tflite::PadV2Options,
                                     ::tflite::BuiltinOptions_PadV2Options> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreatePadV2Options(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Reshape
    : public BuiltinOperator<TensorFlowReshapeOperator,
                             ::tflite::ReshapeOptions,
                             ::tflite::BuiltinOptions_ReshapeOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReshapeOptions(*builder,
                                          builder->CreateVector(op.shape));
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->shape.insert(op->shape.end(), options.new_shape()->begin(),
                     options.new_shape()->end());
  }
};

class Softmax
    : public BuiltinOperator<SoftmaxOperator, ::tflite::SoftmaxOptions,
                             ::tflite::BuiltinOptions_SoftmaxOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSoftmaxOptions(*builder, op.beta);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->beta = options.beta();
  }
};

class SpaceToDepth
    : public BuiltinOperator<SpaceToDepthOperator,
                             ::tflite::SpaceToDepthOptions,
                             ::tflite::BuiltinOptions_SpaceToDepthOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSpaceToDepthOptions(*builder, op.block_size);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->block_size = options.block_size();
  }
};

class Transpose
    : public BuiltinOperator<TransposeOperator, ::tflite::TransposeOptions,
                             ::tflite::BuiltinOptions_TransposeOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateTransposeOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Lstm : public BuiltinOperator<LstmCellOperator, ::tflite::LSTMOptions,
                                    ::tflite::BuiltinOptions_LSTMOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  ::tflite::LSTMKernelType GetKernelType(
      LstmCellOperator::KernelType type) const {
    switch (type) {
      case LstmCellOperator::KERNEL_BASIC:
        return ::tflite::LSTMKernelType_BASIC;
        break;
      case LstmCellOperator::KERNEL_FULL:
        return ::tflite::LSTMKernelType_FULL;
        break;
      default:
        LOG(ERROR) << "Unhandled Kernel Type";
        return static_cast<::tflite::LSTMKernelType>(-1);
    }
  }

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    ::tflite::LSTMKernelType kernel_type = GetKernelType(op.kernel_type);

    // Current toco converter only supports tanh, no clip.
    return ::tflite::CreateLSTMOptions(*builder, /*fused_activation_function=*/
                                       ::tflite::ActivationFunctionType_TANH,
                                       /*cell_clip=*/0.0,
                                       /*proj_clip=*/0.0, kernel_type);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    // Only support tanh activation, so check that tflite type is tanh.
    CHECK(options.fused_activation_function() ==
          ::tflite::ActivationFunctionType_TANH);

    switch (options.kernel_type()) {
      case ::tflite::LSTMKernelType_BASIC:
        op->kernel_type = LstmCellOperator::KERNEL_BASIC;
        break;
      case ::tflite::LSTMKernelType_FULL:
        op->kernel_type = LstmCellOperator::KERNEL_FULL;
        break;
    }
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    const auto& lstm_op =
        static_cast<const LstmCellOperator&>(*op_signature.op);
    ::tflite::OpSignature op_sig =
        GetVersioningOpSig(builtin_op(), op_signature);
    op_sig.options.lstm.kernel_type = GetKernelType(lstm_op.kernel_type);
    return ::tflite::GetBuiltinOperatorVersion(op_sig);
  }

  std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const override {
    const auto& lstm_op = static_cast<const LstmCellOperator&>(op);

    std::vector<bool> mutating_input_variables(op.inputs.size(), false);
    switch (lstm_op.kernel_type) {
      case LstmCellOperator::KERNEL_FULL: {
        mutating_input_variables[kInputActivationStateTensor] = true;
        mutating_input_variables[kInputCellStateTensor] = true;
        break;
      }
      case LstmCellOperator::KERNEL_BASIC: {
        mutating_input_variables[LstmCellOperator::PREV_ACTIV_INPUT] = true;
        mutating_input_variables[LstmCellOperator::PREV_STATE_INPUT] = true;
        break;
      }
    }
    return mutating_input_variables;
  }
};

class UnidirectionalSequenceLstm
    : public BuiltinOperator<
          UnidirectionalSequenceLstmOperator,
          ::tflite::UnidirectionalSequenceLSTMOptions,
          ::tflite::BuiltinOptions_UnidirectionalSequenceLSTMOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    // Current toco converter only supports tanh, no clip.
    return ::tflite::CreateUnidirectionalSequenceLSTMOptions(
        *builder, /*fused_activation_function=*/
        ::tflite::ActivationFunctionType_TANH,
        /*cell_clip=*/0.0,
        /*proj_clip=*/0.0,
        /*time_major=*/true);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    // Only support tanh activation, so check that tflite type is tanh.
    DCHECK(options.fused_activation_function() ==
           ::tflite::ActivationFunctionType_TANH);
  }

  std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const override {
    std::vector<bool> mutating_input_variables(op.inputs.size(), false);
    mutating_input_variables[kInputActivationStateTensor] = true;
    mutating_input_variables[kInputCellStateTensor] = true;
    return mutating_input_variables;
  }
};

class BidirectionalSequenceLstm
    : public BuiltinOperator<
          BidirectionalSequenceLstmOperator,
          ::tflite::BidirectionalSequenceLSTMOptions,
          ::tflite::BuiltinOptions_BidirectionalSequenceLSTMOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    // Current toco converter only supports tanh, no clip.
    return ::tflite::CreateBidirectionalSequenceLSTMOptions(
        *builder, /*fused_activation_function=*/
        ::tflite::ActivationFunctionType_TANH,
        /*cell_clip=*/0.0,
        /*proj_clip=*/0.0,
        /*merge_outputs=*/op.merge_outputs,
        /*time_major=*/true);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    // Only support tanh activation, so check that tflite type is tanh.
    DCHECK(options.fused_activation_function() ==
           ::tflite::ActivationFunctionType_TANH);
    op->merge_outputs = options.merge_outputs();
  }

  std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const override {
    std::vector<bool> mutating_input_variables(op.inputs.size(), false);
    // Forward input activation state.
    mutating_input_variables[35] = true;
    // Forward input cell state.
    mutating_input_variables[36] = true;
    // Backward input activation state.
    mutating_input_variables[37] = true;
    // Backward input cell state.
    mutating_input_variables[38] = true;
    return mutating_input_variables;
  }
};

class BidirectionalSequenceRnn
    : public BuiltinOperator<
          BidirectionalSequenceRnnOperator,
          ::tflite::BidirectionalSequenceRNNOptions,
          ::tflite::BuiltinOptions_BidirectionalSequenceRNNOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    // Current toco converter only supports tanh, no clip.
    return ::tflite::CreateBidirectionalSequenceRNNOptions(
        *builder, /*time_major=*/true,
        /*fused_activation_function=*/
        ::tflite::ActivationFunctionType_TANH,
        /*merge_outputs=*/op.merge_outputs);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    // Only support tanh activation, so check that tflite type is tanh.
    DCHECK(options.fused_activation_function() ==
           ::tflite::ActivationFunctionType_TANH);
    op->merge_outputs = options.merge_outputs();
  }

  std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const override {
    std::vector<bool> mutating_input_variables(op.inputs.size(), false);
    // Forward hidden state.
    mutating_input_variables[4] = true;
    // Backward hidden state.
    mutating_input_variables[8] = true;
    return mutating_input_variables;
  }
};

class Mean : public BuiltinOperator<MeanOperator, ::tflite::ReducerOptions,
                                    ::tflite::BuiltinOptions_ReducerOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReducerOptions(*builder, op.keep_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->keep_dims = options.keep_dims();
  }
};

class Sum
    : public BuiltinOperator<TensorFlowSumOperator, ::tflite::ReducerOptions,
                             ::tflite::BuiltinOptions_ReducerOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReducerOptions(*builder, op.keep_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->keep_dims = options.keep_dims();
  }
};

class ReduceMax
    : public BuiltinOperator<TensorFlowMaxOperator, ::tflite::ReducerOptions,
                             ::tflite::BuiltinOptions_ReducerOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReducerOptions(*builder, op.keep_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->keep_dims = options.keep_dims();
  }
};

class ReduceMin
    : public BuiltinOperator<TensorFlowMinOperator, ::tflite::ReducerOptions,
                             ::tflite::BuiltinOptions_ReducerOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReducerOptions(*builder, op.keep_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->keep_dims = options.keep_dims();
  }
};

class ReduceProd
    : public BuiltinOperator<TensorFlowProdOperator, ::tflite::ReducerOptions,
                             ::tflite::BuiltinOptions_ReducerOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReducerOptions(*builder, op.keep_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->keep_dims = options.keep_dims();
  }
};

class ReduceAny
    : public BuiltinOperator<TensorFlowAnyOperator, ::tflite::ReducerOptions,
                             ::tflite::BuiltinOptions_ReducerOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReducerOptions(*builder, op.keep_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->keep_dims = options.keep_dims();
  }
};

class ResizeBilinear
    : public BuiltinOperator<ResizeBilinearOperator,
                             ::tflite::ResizeBilinearOptions,
                             ::tflite::BuiltinOptions_ResizeBilinearOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateResizeBilinearOptions(*builder, op.align_corners);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->align_corners = options.align_corners();
  }
};

class ResizeNearestNeighbor
    : public BuiltinOperator<
          ResizeNearestNeighborOperator, ::tflite::ResizeNearestNeighborOptions,
          ::tflite::BuiltinOptions_ResizeNearestNeighborOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateResizeNearestNeighborOptions(*builder,
                                                        op.align_corners);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->align_corners = options.align_corners();
  }
};

class Squeeze
    : public BuiltinOperator<SqueezeOperator, ::tflite::SqueezeOptions,
                             ::tflite::BuiltinOptions_SqueezeOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto squeeze_dims = builder->CreateVector(op.squeeze_dims);
    return ::tflite::CreateSqueezeOptions(*builder, squeeze_dims);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->squeeze_dims.insert(op->squeeze_dims.end(),
                            options.squeeze_dims()->begin(),
                            options.squeeze_dims()->end());
  }
};

class Split
    : public BuiltinOperator<TensorFlowSplitOperator, ::tflite::SplitOptions,
                             ::tflite::BuiltinOptions_SplitOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSplitOptions(*builder, op.num_split);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->num_split = options.num_splits();
  }
};

class SplitV
    : public BuiltinOperator<TensorFlowSplitVOperator, ::tflite::SplitVOptions,
                             ::tflite::BuiltinOptions_SplitVOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSplitVOptions(*builder, op.num_split);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->num_split = options.num_splits();
  }
};

class StridedSlice
    : public BuiltinOperator<StridedSliceOperator,
                             ::tflite::StridedSliceOptions,
                             ::tflite::BuiltinOptions_StridedSliceOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateStridedSliceOptions(
        *builder, op.begin_mask, op.end_mask, op.ellipsis_mask,
        op.new_axis_mask, op.shrink_axis_mask);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->begin_mask = options.begin_mask();
    op->end_mask = options.end_mask();
    op->ellipsis_mask = options.ellipsis_mask();
    op->new_axis_mask = options.new_axis_mask();
    op->shrink_axis_mask = options.shrink_axis_mask();
  }
};

class TopK_V2 : public BuiltinOperator<TopKV2Operator, ::tflite::TopKV2Options,
                                       ::tflite::BuiltinOptions_TopKV2Options> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateTopKV2Options(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class ArgMax : public BuiltinOperator<ArgMaxOperator, ::tflite::ArgMaxOptions,
                                      ::tflite::BuiltinOptions_ArgMaxOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateArgMaxOptions(
        *builder, DataType::Serialize(op.output_data_type));
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->output_data_type = DataType::Deserialize(options.output_type());
  }
};

class ArgMin : public BuiltinOperator<ArgMinOperator, ::tflite::ArgMinOptions,
                                      ::tflite::BuiltinOptions_ArgMinOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateArgMinOptions(
        *builder, DataType::Serialize(op.output_data_type));
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->output_data_type = DataType::Deserialize(options.output_type());
  }
};

class TransposeConv
    : public BuiltinOperator<TransposeConvOperator,
                             ::tflite::TransposeConvOptions,
                             ::tflite::BuiltinOptions_TransposeConvOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto padding = Padding::Serialize(op.padding.type);
    return ::tflite::CreateTransposeConvOptions(
        *builder, padding, op.stride_width, op.stride_height);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
  }
};

class SparseToDense
    : public BuiltinOperator<SparseToDenseOperator,
                             ::tflite::SparseToDenseOptions,
                             ::tflite::BuiltinOptions_SparseToDenseOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSparseToDenseOptions(*builder, op.validate_indices);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->validate_indices = options.validate_indices();
  }
};

class ExpandDims
    : public BuiltinOperator<ExpandDimsOperator, ::tflite::ExpandDimsOptions,
                             ::tflite::BuiltinOptions_ExpandDimsOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateExpandDimsOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class Pack : public BuiltinOperator<PackOperator, ::tflite::PackOptions,
                                    ::tflite::BuiltinOptions_PackOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreatePackOptions(*builder, op.values_count, op.axis);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->values_count = options.values_count();
    op->axis = options.axis();
  }
};

class Shape
    : public BuiltinOperator<TensorFlowShapeOperator, ::tflite::ShapeOptions,
                             ::tflite::BuiltinOptions_ShapeOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateShapeOptions(
        *builder, DataType::Serialize(op.output_data_type));
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->output_data_type = DataType::Deserialize(options.out_type());
  }
};

class OneHot : public BuiltinOperator<OneHotOperator, ::tflite::OneHotOptions,
                                      ::tflite::BuiltinOptions_OneHotOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateOneHotOptions(*builder, op.axis);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->axis = options.axis();
  }
};

class CTCBeamSearchDecoder
    : public CustomOperator<CTCBeamSearchDecoderOperator> {
 public:
  using CustomOperator::CustomOperator;

  void WriteOptions(const TocoOperator& op,
                    flexbuffers::Builder* fbb) const override {
    fbb->Int("beam_width", op.beam_width);
    fbb->Int("top_paths", op.top_paths);
    fbb->Bool("merge_repeated", op.merge_repeated);
  }

  void ReadOptions(const flexbuffers::Map& m, TocoOperator* op) const override {
    op->beam_width = m["beam_width"].AsInt32();
    op->top_paths = m["top_paths"].AsInt32();
    op->merge_repeated = m["merge_repeated"].AsBool();
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    return 1;
  }
};

class Unpack : public BuiltinOperator<UnpackOperator, ::tflite::UnpackOptions,
                                      ::tflite::BuiltinOptions_UnpackOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateUnpackOptions(*builder, op.num, op.axis);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->num = options.num();
    op->axis = options.axis();
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    const string& input_name = op_signature.op->inputs[0];
    const Array& input_array = op_signature.model->GetArray(input_name);
    // If the op take int8/uint8 input, it is version 2.
    if (input_array.data_type == ArrayDataType::kInt8 ||
        input_array.data_type == ArrayDataType::kUint8) {
      return 2;
    }
    // If the op take bool input, it is version 3.
    if (input_array.data_type == ArrayDataType::kBool) {
      return 3;
    }
    return 1;
  }
};

class LeakyRelu
    : public BuiltinOperator<LeakyReluOperator, ::tflite::LeakyReluOptions,
                             ::tflite::BuiltinOptions_LeakyReluOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateLeakyReluOptions(*builder, op.alpha);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->alpha = options.alpha();
  }
};

class SquaredDifference
    : public BuiltinOperator<
          SquaredDifferenceOperator, ::tflite::SquaredDifferenceOptions,
          ::tflite::BuiltinOptions_SquaredDifferenceOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSquaredDifferenceOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class MirrorPad
    : public BuiltinOperator<MirrorPadOperator, ::tflite::MirrorPadOptions,
                             ::tflite::BuiltinOptions_MirrorPadOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateMirrorPadOptions(
        *builder, op.mode == MirrorPadMode::kReflect
                      ? ::tflite::MirrorPadMode::MirrorPadMode_REFLECT
                      : ::tflite::MirrorPadMode::MirrorPadMode_SYMMETRIC);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->mode = options.mode() == ::tflite::MirrorPadMode::MirrorPadMode_REFLECT
                   ? MirrorPadMode::kReflect
                   : MirrorPadMode::kSymmetric;
  }
};

class Unique : public BuiltinOperator<UniqueOperator, ::tflite::UniqueOptions,
                                      ::tflite::BuiltinOptions_UniqueOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    const UniqueOperator& unique_op = static_cast<const UniqueOperator&>(op);
    return ::tflite::CreateUniqueOptions(
        *builder, unique_op.idx_out_type == toco::ArrayDataType::kInt64
                      ? ::tflite::TensorType::TensorType_INT64
                      : ::tflite::TensorType_INT32);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    UniqueOperator* unique_op = static_cast<UniqueOperator*>(op);
    unique_op->idx_out_type =
        options.idx_out_type() == ::tflite::TensorType_INT64
            ? toco::ArrayDataType::kInt64
            : toco::ArrayDataType::kInt32;
  }
};

class UnidirectionalSequenceRnn
    : public BuiltinOperator<UnidirectionalSequenceRnnOperator,
                             ::tflite::SequenceRNNOptions,
                             ::tflite::BuiltinOptions_SequenceRNNOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateSequenceRNNOptions(
        *builder, /*time_major=*/true,
        /*fused_activation_function=*/
        ::tflite::ActivationFunctionType_TANH);
  }
  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    // Only support tanh activation, so check that tflite type is tanh.
    DCHECK(options.fused_activation_function() ==
           ::tflite::ActivationFunctionType_TANH);
  }

  std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const override {
    std::vector<bool> mutating_input_variables(op.inputs.size(), false);
    mutating_input_variables[4] = true;
    return mutating_input_variables;
  }
};

class Where : public BuiltinOperator<WhereOperator, ::tflite::WhereOptions,
                                     ::tflite::BuiltinOptions_WhereOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateWhereOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

std::unique_ptr<flexbuffers::Builder> WriteFlexOpOptions(
    const string& tensorflow_node_def) {
  auto fbb = absl::make_unique<flexbuffers::Builder>();

  ::tensorflow::NodeDef node_def;
  if (!node_def.ParseFromString(tensorflow_node_def)) {
    LOG(ERROR) << "Failed to parse TensorFlow NodeDef";
    return {};
  }

  fbb->Vector([&]() {
    fbb->String(node_def.op());
    fbb->String(tensorflow_node_def);
  });
  fbb->Finish();
  LOG(INFO) << "Writing flex op: " << node_def.op();
  return std::unique_ptr<flexbuffers::Builder>(fbb.release());
}

class TensorFlowUnsupported : public BaseOperator {
 public:
  TensorFlowUnsupported(const string& name, OperatorType type,
                        bool enable_select_tf_ops)
      : BaseOperator(name, type), enable_select_tf_ops_(enable_select_tf_ops) {}

  Options Serialize(const Operator& op,
                    flatbuffers::FlatBufferBuilder* builder) const override {
    auto fbb =
        WriteOptions(static_cast<const TensorFlowUnsupportedOperator&>(op));
    if (fbb) {
      return Options::Custom(builder->CreateVector(fbb->GetBuffer()));
    } else {
      return Options::Custom(0);
    }
  }

  std::unique_ptr<Operator> Deserialize(
      const BuiltinOptions* builtin_options,
      const CustomOptions* custom_options) const override {
    // Deserializing Flex ops doesn't work now.
    // TODO(ycling): Revisit and decide if we should fix the flow for importing
    // TFLite models with Flex ops.
    auto op = absl::make_unique<TensorFlowUnsupportedOperator>();
    if (custom_options) {
      auto flexbuffer_map =
          flexbuffers::GetRoot(custom_options->data(), custom_options->size())
              .AsMap();
      ReadOptions(flexbuffer_map, op.get());
    }
    return std::unique_ptr<Operator>(op.release());
  }

  std::unique_ptr<flexbuffers::Builder> WriteOptions(
      const TensorFlowUnsupportedOperator& op) const {
    if (enable_select_tf_ops_) {
      return WriteFlexOpOptions(op.tensorflow_node_def);
    }
    auto fbb = absl::make_unique<flexbuffers::Builder>();

    ::tensorflow::NodeDef node_def;
    if (!node_def.ParseFromString(op.tensorflow_node_def)) {
      LOG(ERROR) << "Failed to parse TensorFlow NodeDef";
      return std::unique_ptr<flexbuffers::Builder>();
    }

    if (ShouldExportAsFlexOp(enable_select_tf_ops_, node_def.op())) {
      fbb->Vector([&]() {
        fbb->String(node_def.op());
        fbb->String(op.tensorflow_node_def);
      });
      fbb->Finish();
      LOG(INFO) << "Writing flex op: " << node_def.op();
      return std::unique_ptr<flexbuffers::Builder>(fbb.release());
    }

    bool has_valid_attr = false;
    size_t map_start = fbb->StartMap();
    for (const auto& pair : node_def.attr()) {
      const char* key = pair.first.c_str();
      const auto& attr = pair.second;
      switch (attr.value_case()) {
        case ::tensorflow::AttrValue::kS:
          fbb->String(key, attr.s());
          has_valid_attr = true;
          break;
        case ::tensorflow::AttrValue::kI:
          fbb->Int(key, attr.i());
          has_valid_attr = true;
          break;
        case ::tensorflow::AttrValue::kF:
          fbb->Float(key, attr.f());
          has_valid_attr = true;
          break;
        case ::tensorflow::AttrValue::kB:
          fbb->Bool(key, attr.b());
          has_valid_attr = true;
          break;
        case tensorflow::AttrValue::kList:
          if (attr.list().s_size() > 0) {
            auto start = fbb->StartVector(key);
            for (const string& v : attr.list().s()) {
              fbb->Add(v);
            }
            fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
            has_valid_attr = true;
          } else if (attr.list().i_size() > 0) {
            auto start = fbb->StartVector(key);
            for (const int64_t v : attr.list().i()) {
              fbb->Add(v);
            }
            fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
            has_valid_attr = true;
          } else if (attr.list().f_size() > 0) {
            auto start = fbb->StartVector(key);
            for (const float v : attr.list().f()) {
              fbb->Add(v);
            }
            fbb->EndVector(start, /*typed=*/true, /*fixed=*/false);
            has_valid_attr = true;
          } else {
            LOG(WARNING)
                << "Ignoring unsupported type in list attribute with key '"
                << key << "'";
          }
          break;
        default:
          LOG(WARNING) << "Ignoring unsupported attribute type with key '"
                       << key << "'";
          break;
      }
    }
    if (!has_valid_attr) {
      return std::unique_ptr<flexbuffers::Builder>();
    }
    fbb->EndMap(map_start);
    fbb->Finish();
    return std::unique_ptr<flexbuffers::Builder>(fbb.release());
  }

  void ReadOptions(const flexbuffers::Map& m,
                   TensorFlowUnsupportedOperator* op) const {
    ::tensorflow::NodeDef node_def;
    auto attr = node_def.mutable_attr();

    const auto& keys = m.Keys();
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto key = keys[i].AsKey();
      const auto& value = m[key];
      // TODO(wvo): hack to make this code compile with 2 different API
      // versions.
      // Please remove once OS/internal versions are in sync.
      // See hardcoded values in the switch below.
      switch (value.GetType()) {
        case 5:  // flexbuffers::FBT_STRING:
          (*attr)[key].set_s(value.AsString().c_str());
          break;
        case 1:  // flexbuffers::FBT_INT:
          (*attr)[key].set_i(value.AsInt64());
          break;
        case 3:  // flexbuffers::FBT_FLOAT:
          (*attr)[key].set_f(value.AsFloat());
          break;
        case 26:  // flexbuffers::FBT_BOOL:
          (*attr)[key].set_b(value.AsBool());
          if (string(key) == "_output_quantized") {
            op->quantized = value.AsBool();
          }
          if (string(key) == "_support_output_type_float_in_quantized_op") {
            op->support_output_type_float_in_quantized_op = value.AsBool();
          }
          break;
        case 11: {  // flexbuffers::FBT_VECTOR_INT: {
          auto* list = (*attr)[key].mutable_list();
          const auto& vector = value.AsTypedVector();
          for (size_t i = 0; i < vector.size(); i++) {
            list->add_i(vector[i].AsInt64());
          }
          break;
        }
        case 13: {  // flexbuffers::FBT_VECTOR_FLOAT: {
          auto* list = (*attr)[key].mutable_list();
          const auto& vector = value.AsTypedVector();
          for (size_t i = 0; i < vector.size(); i++) {
            list->add_f(vector[i].AsFloat());
          }
          break;
        }
        case 15: {  // flexbuffers::FBT_VECTOR_STRING: {
          auto* list = (*attr)[key].mutable_list();
          const auto& vector = value.AsTypedVector();
          for (size_t i = 0; i < vector.size(); i++) {
            list->add_s(vector[i].AsString().str());
          }
          break;
        }
        default:
          LOG(WARNING) << "Ignoring unsupported attribute type with key '"
                       << key << "'";
          break;
      }
    }
    node_def.SerializeToString(&op->tensorflow_node_def);
  }

  int GetVersion(const OperatorSignature& op_signature) const override {
    // TODO(ycling): Design and implement a way to plumb the version of
    // custom ops.
    return 1;
  }

 private:
  const bool enable_select_tf_ops_;
};

class Dequantize
    : public BuiltinOperator<DequantizeOperator, ::tflite::DequantizeOptions,
                             ::tflite::BuiltinOptions_DequantizeOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateDequantizeOptions(*builder);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {}
};

class ReverseSequence
    : public BuiltinOperator<ReverseSequenceOperator,
                             ::tflite::ReverseSequenceOptions,
                             ::tflite::BuiltinOptions_ReverseSequenceOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;

  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateReverseSequenceOptions(*builder, op.seq_dim,
                                                  op.batch_dim);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->seq_dim = options.seq_dim();
    op->batch_dim = options.batch_dim();
  }
};

namespace {
// Build a vector containing all the known operators.
std::vector<std::unique_ptr<BaseOperator>> BuildOperatorList(
    bool enable_select_tf_ops = false) {
  std::vector<std::unique_ptr<BaseOperator>> ops;
  using tensorflow::MakeUnique;
  // Builtin Operators.
  ops.push_back(
      MakeUnique<Add>(::tflite::BuiltinOperator_ADD, OperatorType::kAdd));
  ops.push_back(
      MakeUnique<AddN>(::tflite::BuiltinOperator_ADD_N, OperatorType::kAddN));
  ops.push_back(
      MakeUnique<Div>(::tflite::BuiltinOperator_DIV, OperatorType::kDiv));
  ops.push_back(
      MakeUnique<Sub>(::tflite::BuiltinOperator_SUB, OperatorType::kSub));
  ops.push_back(MakeUnique<AveragePool>(
      ::tflite::BuiltinOperator_AVERAGE_POOL_2D, OperatorType::kAveragePool));
  ops.push_back(
      MakeUnique<SpaceToBatchND>(::tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                                 OperatorType::kSpaceToBatchND));
  ops.push_back(
      MakeUnique<BatchToSpaceND>(::tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                                 OperatorType::kBatchToSpaceND));
  ops.push_back(MakeUnique<Concatenation>(
      ::tflite::BuiltinOperator_CONCATENATION, OperatorType::kConcatenation));
  ops.push_back(MakeUnique<Convolution>(::tflite::BuiltinOperator_CONV_2D,
                                        OperatorType::kConv));
  ops.push_back(MakeUnique<DepthwiseConvolution>(
      ::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      OperatorType::kDepthwiseConv));
  ops.push_back(MakeUnique<Dequantize>(::tflite::BuiltinOperator_DEQUANTIZE,
                                       OperatorType::kDequantize));
  ops.push_back(
      MakeUnique<FullyConnected>(::tflite::BuiltinOperator_FULLY_CONNECTED,
                                 OperatorType::kFullyConnected));
  ops.push_back(MakeUnique<Gather>(::tflite::BuiltinOperator_GATHER,
                                   OperatorType::kGather));
  ops.push_back(MakeUnique<GatherNd>(::tflite::BuiltinOperator_GATHER_ND,
                                     OperatorType::kGatherNd));
  ops.push_back(
      MakeUnique<L2Normalization>(::tflite::BuiltinOperator_L2_NORMALIZATION,
                                  OperatorType::kL2Normalization));
  ops.push_back(MakeUnique<L2Pool>(::tflite::BuiltinOperator_L2_POOL_2D,
                                   OperatorType::kL2Pool));
  ops.push_back(MakeUnique<LocalResponseNormalization>(
      ::tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
      OperatorType::kLocalResponseNormalization));
  ops.push_back(MakeUnique<MaxPool>(::tflite::BuiltinOperator_MAX_POOL_2D,
                                    OperatorType::kMaxPool));
  ops.push_back(
      MakeUnique<Mul>(::tflite::BuiltinOperator_MUL, OperatorType::kMul));

  ops.push_back(
      MakeUnique<Pad>(::tflite::BuiltinOperator_PAD, OperatorType::kPad));
  ops.push_back(
      MakeUnique<PadV2>(::tflite::BuiltinOperator_PADV2, OperatorType::kPadV2));
  ops.push_back(MakeUnique<Reshape>(::tflite::BuiltinOperator_RESHAPE,
                                    OperatorType::kReshape));
  ops.push_back(MakeUnique<Softmax>(::tflite::BuiltinOperator_SOFTMAX,
                                    OperatorType::kSoftmax));
  ops.push_back(MakeUnique<SpaceToDepth>(
      ::tflite::BuiltinOperator_SPACE_TO_DEPTH, OperatorType::kSpaceToDepth));
  ops.push_back(MakeUnique<DepthToSpace>(
      ::tflite::BuiltinOperator_DEPTH_TO_SPACE, OperatorType::kDepthToSpace));
  ops.push_back(
      MakeUnique<Svdf>(::tflite::BuiltinOperator_SVDF, OperatorType::kSvdf));
  ops.push_back(MakeUnique<Transpose>(::tflite::BuiltinOperator_TRANSPOSE,
                                      OperatorType::kTranspose));
  ops.push_back(
      MakeUnique<Mean>(::tflite::BuiltinOperator_MEAN, OperatorType::kMean));
  ops.push_back(
      MakeUnique<Sum>(::tflite::BuiltinOperator_SUM, OperatorType::kSum));
  ops.push_back(MakeUnique<ReduceProd>(::tflite::BuiltinOperator_REDUCE_PROD,
                                       OperatorType::kReduceProd));
  ops.push_back(MakeUnique<ReduceMax>(::tflite::BuiltinOperator_REDUCE_MAX,
                                      OperatorType::kReduceMax));
  ops.push_back(MakeUnique<ReduceMin>(::tflite::BuiltinOperator_REDUCE_MIN,
                                      OperatorType::kReduceMin));
  ops.push_back(MakeUnique<ReduceAny>(::tflite::BuiltinOperator_REDUCE_ANY,
                                      OperatorType::kAny));
  ops.push_back(
      MakeUnique<ResizeBilinear>(::tflite::BuiltinOperator_RESIZE_BILINEAR,
                                 OperatorType::kResizeBilinear));
  ops.push_back(MakeUnique<ResizeNearestNeighbor>(
      ::tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      OperatorType::kResizeNearestNeighbor));
  ops.push_back(MakeUnique<Squeeze>(::tflite::BuiltinOperator_SQUEEZE,
                                    OperatorType::kSqueeze));
  ops.push_back(
      MakeUnique<Split>(::tflite::BuiltinOperator_SPLIT, OperatorType::kSplit));
  ops.push_back(MakeUnique<SplitV>(::tflite::BuiltinOperator_SPLIT_V,
                                   OperatorType::kSplitV));
  ops.push_back(MakeUnique<StridedSlice>(
      ::tflite::BuiltinOperator_STRIDED_SLICE, OperatorType::kStridedSlice));
  ops.push_back(MakeUnique<TopK_V2>(::tflite::BuiltinOperator_TOPK_V2,
                                    OperatorType::kTopK_V2));
  ops.push_back(MakeUnique<Lstm>(::tflite::BuiltinOperator_LSTM,
                                 OperatorType::kLstmCell));
  ops.push_back(
      MakeUnique<Cast>(::tflite::BuiltinOperator_CAST, OperatorType::kCast));
  ops.push_back(MakeUnique<ArgMax>(::tflite::BuiltinOperator_ARG_MAX,
                                   OperatorType::kArgMax));
  ops.push_back(MakeUnique<ArgMin>(::tflite::BuiltinOperator_ARG_MIN,
                                   OperatorType::kArgMin));
  ops.push_back(
      MakeUnique<Tile>(::tflite::BuiltinOperator_TILE, OperatorType::kTile));
  ops.push_back(MakeUnique<ExpandDims>(::tflite::BuiltinOperator_EXPAND_DIMS,
                                       OperatorType::kExpandDims));
  ops.push_back(MakeUnique<TransposeConv>(
      ::tflite::BuiltinOperator_TRANSPOSE_CONV, OperatorType::kTransposeConv));
  ops.push_back(MakeUnique<SparseToDense>(
      ::tflite::BuiltinOperator_SPARSE_TO_DENSE, OperatorType::kSparseToDense));
  ops.push_back(
      MakeUnique<Shape>(::tflite::BuiltinOperator_SHAPE, OperatorType::kShape));
  ops.push_back(MakeUnique<FakeQuant>(::tflite::BuiltinOperator_FAKE_QUANT,
                                      OperatorType::kFakeQuant));
  ops.push_back(
      MakeUnique<Pack>(::tflite::BuiltinOperator_PACK, OperatorType::kPack));
  ops.emplace_back(MakeUnique<UnidirectionalSequenceLstm>(
      ::tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
      OperatorType::kUnidirectionalSequenceLstm));
  ops.emplace_back(MakeUnique<BidirectionalSequenceLstm>(
      ::tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
      OperatorType::kBidirectionalSequenceLstm));
  ops.emplace_back(MakeUnique<BidirectionalSequenceRnn>(
      ::tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
      OperatorType::kBidirectionalSequenceRnn));
  ops.push_back(MakeUnique<OneHot>(::tflite::BuiltinOperator_ONE_HOT,
                                   OperatorType::kOneHot));
  ops.push_back(MakeUnique<Unpack>(::tflite::BuiltinOperator_UNPACK,
                                   OperatorType::kUnpack));
  ops.push_back(MakeUnique<LeakyRelu>(::tflite::BuiltinOperator_LEAKY_RELU,
                                      OperatorType::kLeakyRelu));
  ops.push_back(MakeUnique<SquaredDifference>(
      ::tflite::BuiltinOperator_SQUARED_DIFFERENCE,
      OperatorType::kSquaredDifference));
  ops.push_back(MakeUnique<MirrorPad>(::tflite::BuiltinOperator_MIRROR_PAD,
                                      OperatorType::kMirrorPad));
  ops.push_back(MakeUnique<Unique>(::tflite::BuiltinOperator_UNIQUE,
                                   OperatorType::kUnique));
  ops.push_back(MakeUnique<UnidirectionalSequenceRnn>(
      ::tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
      OperatorType::kUnidirectionalSequenceRnn));
  ops.push_back(
      MakeUnique<Where>(::tflite::BuiltinOperator_WHERE, OperatorType::kWhere));
  ops.push_back(
      MakeUnique<ReverseSequence>(::tflite::BuiltinOperator_REVERSE_SEQUENCE,
                                  OperatorType::kReverseSequence));
  ops.push_back(MakeUnique<SimpleOperator<MatrixDiagOperator>>(
      ::tflite::BuiltinOperator_MATRIX_DIAG, OperatorType::kMatrixDiag));
  ops.push_back(MakeUnique<SimpleOperator<MatrixSetDiagOperator>>(
      ::tflite::BuiltinOperator_MATRIX_SET_DIAG, OperatorType::kMatrixSetDiag));
  // Custom Operators.
  ops.push_back(MakeUnique<CTCBeamSearchDecoder>(
      "CTC_BEAM_SEARCH_DECODER", OperatorType::kCTCBeamSearchDecoder));
  ops.push_back(MakeUnique<TensorFlowUnsupported>("TENSORFLOW_UNSUPPORTED",
                                                  OperatorType::kUnsupported,
                                                  enable_select_tf_ops));

  // SimpleOperator was designed to export CUSTOM TF Lite ops, but has since
  // been modified to also export builtins. As TOCO evolved we added warnings
  // when custom ops are exported but SimpleOperator bypasses thoses. To
  // prevent user confusion we are settling on using SimpleOperator only for
  // builtins.
  ops.push_back(MakeUnique<SimpleOperator<FloorOperator>>(
      ::tflite::BuiltinOperator_FLOOR, OperatorType::kFloor));
  ops.push_back(MakeUnique<SimpleOperator<CeilOperator>>(
      ::tflite::BuiltinOperator_CEIL, OperatorType::kCeil));
  ops.push_back(MakeUnique<SimpleOperator<EluOperator>>(
      ::tflite::BuiltinOperator_ELU, OperatorType::kElu));
  ops.push_back(MakeUnique<SimpleOperator<RoundOperator>>(
      ::tflite::BuiltinOperator_ROUND, OperatorType::kRound));
  ops.push_back(MakeUnique<SimpleOperator<ReluOperator>>(
      ::tflite::BuiltinOperator_RELU, OperatorType::kRelu));
  ops.push_back(MakeUnique<SimpleOperator<Relu1Operator>>(
      ::tflite::BuiltinOperator_RELU_N1_TO_1, OperatorType::kRelu1));
  ops.push_back(MakeUnique<SimpleOperator<Relu6Operator>>(
      ::tflite::BuiltinOperator_RELU6, OperatorType::kRelu6));
  ops.push_back(MakeUnique<SimpleOperator<PReluOperator>>(
      ::tflite::BuiltinOperator_PRELU, OperatorType::kPRelu));
  ops.push_back(MakeUnique<SimpleOperator<LogisticOperator>>(
      ::tflite::BuiltinOperator_LOGISTIC, OperatorType::kLogistic));
  ops.push_back(MakeUnique<SimpleOperator<TanhOperator>>(
      ::tflite::BuiltinOperator_TANH, OperatorType::kTanh));
  ops.push_back(MakeUnique<SimpleOperator<ExpOperator>>(
      ::tflite::BuiltinOperator_EXP, OperatorType::kExp));
  ops.push_back(MakeUnique<SimpleOperator<CosOperator>>(
      ::tflite::BuiltinOperator_COS, OperatorType::kCos));
  ops.push_back(MakeUnique<SimpleOperator<LogSoftmaxOperator>>(
      ::tflite::BuiltinOperator_LOG_SOFTMAX, OperatorType::kLogSoftmax));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowMaximumOperator>>(
      ::tflite::BuiltinOperator_MAXIMUM, OperatorType::kMaximum));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowMinimumOperator>>(
      ::tflite::BuiltinOperator_MINIMUM, OperatorType::kMinimum));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowGreaterOperator>>(
      ::tflite::BuiltinOperator_GREATER, OperatorType::kGreater));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowGreaterEqualOperator>>(
      ::tflite::BuiltinOperator_GREATER_EQUAL, OperatorType::kGreaterEqual));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowLessOperator>>(
      ::tflite::BuiltinOperator_LESS, OperatorType::kLess));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowLessEqualOperator>>(
      ::tflite::BuiltinOperator_LESS_EQUAL, OperatorType::kLessEqual));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowEqualOperator>>(
      ::tflite::BuiltinOperator_EQUAL, OperatorType::kEqual));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowNotEqualOperator>>(
      ::tflite::BuiltinOperator_NOT_EQUAL, OperatorType::kNotEqual));
  ops.push_back(MakeUnique<SimpleOperator<NegOperator>>(
      ::tflite::BuiltinOperator_NEG, OperatorType::kNeg));
  ops.push_back(MakeUnique<SimpleOperator<SelectOperator>>(
      ::tflite::BuiltinOperator_SELECT, OperatorType::kSelect));
  ops.push_back(MakeUnique<SimpleOperator<SliceOperator>>(
      ::tflite::BuiltinOperator_SLICE, OperatorType::kSlice));
  ops.push_back(MakeUnique<SimpleOperator<PowOperator>>(
      ::tflite::BuiltinOperator_POW, OperatorType::kPow));
  ops.push_back(MakeUnique<SimpleOperator<LogicalOrOperator>>(
      ::tflite::BuiltinOperator_LOGICAL_OR, OperatorType::kLogicalOr));
  ops.emplace_back(new SimpleOperator<LogicalAndOperator>(
      ::tflite::BuiltinOperator_LOGICAL_AND, OperatorType::kLogicalAnd));
  ops.emplace_back(new SimpleOperator<LogicalNotOperator>(
      ::tflite::BuiltinOperator_LOGICAL_NOT, OperatorType::kLogicalNot));
  ops.emplace_back(new SimpleOperator<FloorDivOperator>(
      ::tflite::BuiltinOperator_FLOOR_DIV, OperatorType::kFloorDiv));
  ops.emplace_back(new SimpleOperator<FloorModOperator>(
      ::tflite::BuiltinOperator_FLOOR_MOD, OperatorType::kFloorMod));
  ops.emplace_back(new SimpleOperator<RangeOperator>(
      ::tflite::BuiltinOperator_RANGE, OperatorType::kRange));
  // Element-wise operator
  ops.push_back(MakeUnique<SimpleOperator<SinOperator>>(
      ::tflite::BuiltinOperator_SIN, OperatorType::kSin));
  ops.push_back(MakeUnique<SimpleOperator<LogOperator>>(
      ::tflite::BuiltinOperator_LOG, OperatorType::kLog));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowSqrtOperator>>(
      ::tflite::BuiltinOperator_SQRT, OperatorType::kSqrt));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowRsqrtOperator>>(
      ::tflite::BuiltinOperator_RSQRT, OperatorType::kRsqrt));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowSquareOperator>>(
      ::tflite::BuiltinOperator_SQUARE, OperatorType::kSquare));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowZerosLikeOperator>>(
      ::tflite::BuiltinOperator_ZEROS_LIKE, OperatorType::kZerosLike));
  ops.push_back(MakeUnique<SimpleOperator<AbsOperator>>(
      ::tflite::BuiltinOperator_ABS, OperatorType::kAbs));
  ops.push_back(MakeUnique<SimpleOperator<HardSwishOperator>>(
      ::tflite::BuiltinOperator_HARD_SWISH, OperatorType::kHardSwish));
  ops.push_back(MakeUnique<SimpleOperator<FillOperator>>(
      ::tflite::BuiltinOperator_FILL, OperatorType::kFill));
  ops.push_back(MakeUnique<SimpleOperator<ReverseV2Operator>>(
      ::tflite::BuiltinOperator_REVERSE_V2, OperatorType::kReverseV2));
  ops.push_back(MakeUnique<SimpleOperator<TensorFlowRankOperator>>(
      ::tflite::BuiltinOperator_RANK, OperatorType::kRank));
  ops.emplace_back(new SimpleOperator<SegmentSumOperator>(
      ::tflite::BuiltinOperator_SEGMENT_SUM, OperatorType::kSegmentSum));
  return ops;
}
}  // namespace

// LINT.ThenChange(//tensorflow/lite/tools/versioning/op_version.cc)

std::map<OperatorType, std::unique_ptr<BaseOperator>> BuildOperatorByTypeMap(
    bool enable_select_tf_ops) {
  std::map<OperatorType, std::unique_ptr<BaseOperator>> result;

  std::vector<std::unique_ptr<BaseOperator>> ops =
      BuildOperatorList(enable_select_tf_ops);
  for (auto& op : ops) {
    result[op->type()] = std::move(op);
  }

  return result;
}

std::map<string, std::unique_ptr<BaseOperator>> BuildOperatorByNameMap(
    bool enable_select_tf_ops) {
  std::map<string, std::unique_ptr<BaseOperator>> result;

  std::vector<std::unique_ptr<BaseOperator>> ops =
      BuildOperatorList(enable_select_tf_ops);
  for (auto& op : ops) {
    result[op->name()] = std::move(op);
  }

  return result;
}

bool ShouldExportAsFlexOp(bool enable_select_tf_ops,
                          const string& tensorflow_op_name) {
  // If Flex ops aren't allow at all, simply return false.
  if (!enable_select_tf_ops) {
    return false;
  }
  // Check if we can find the `OpDef` for the TensorFlow op. If we can find
  // it and it has been whitelisted, export the op as an Flex op. Otherwise,
  // export it as a regular custom op.
  const tensorflow::OpDef* op_def = nullptr;
  if (!tensorflow::OpRegistry::Global()
           ->LookUpOpDef(tensorflow_op_name, &op_def)
           .ok()) {
    return false;
  }

  if (!::tflite::flex::IsWhitelistedFlexOp(tensorflow_op_name)) {
    LOG(WARNING) << "Op " << tensorflow_op_name
                 << " is a valid TensorFlow op but has not been whitelisted for"
                    " the TensorFlow Lite flex op set.";
    return false;
  }

  return true;
}

}  // namespace tflite

}  // namespace toco
