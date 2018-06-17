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
#include "tensorflow/contrib/lite/toco/tflite/operator.h"

#include "tensorflow/contrib/lite/toco/tflite/builtin_operator.h"
#include "tensorflow/contrib/lite/toco/tflite/custom_operator.h"
#include "tensorflow/contrib/lite/toco/tflite/simple_operator.h"
#include "tensorflow/contrib/lite/toco/tflite/types.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace toco {

namespace tflite {

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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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
        op.depth_multiplier, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->padding.type = Padding::Deserialize(options.padding());
    op->stride_width = options.stride_w();
    op->stride_height = options.stride_h();
    op->depth_multiplier = options.depth_multiplier();
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
};

class DepthToSpace : public CustomOperator<DepthToSpaceOperator> {
 public:
  using CustomOperator::CustomOperator;
  void WriteOptions(const TocoOperator& op,
                    flexbuffers::Builder* fbb) const override {
    fbb->Int("block_size", op.block_size);
  }
  void ReadOptions(const flexbuffers::Map& m, TocoOperator* op) const override {
    op->block_size = m["block_size"].AsInt64();
  }

  int GetVersion(const Operator& op) const override { return 1; }
};

class FakeQuant : public CustomOperator<FakeQuantOperator> {
 public:
  using CustomOperator::CustomOperator;
  void WriteOptions(const TocoOperator& op,
                    flexbuffers::Builder* fbb) const override {
    fbb->Float("min", op.minmax->min);
    fbb->Float("max", op.minmax->max);
    fbb->Int("num_bits", op.num_bits);
  }
  void ReadOptions(const flexbuffers::Map& m, TocoOperator* op) const override {
    auto* minmax = new MinMax;
    minmax->min = m["min"].AsFloat();
    minmax->max = m["max"].AsFloat();
    op->minmax.reset(minmax);
    const auto& num_bits = m["num_bits"];
    op->num_bits = num_bits.IsInt() ? num_bits.AsInt32() : 8;
  }

  int GetVersion(const Operator& op) const override { return 1; }
};

class FullyConnected
    : public BuiltinOperator<FullyConnectedOperator,
                             ::tflite::FullyConnectedOptions,
                             ::tflite::BuiltinOptions_FullyConnectedOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    auto activation_function =
        ActivationFunction::Serialize(op.fused_activation_function);
    return ::tflite::CreateFullyConnectedOptions(*builder, activation_function);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->fused_activation_function =
        ActivationFunction::Deserialize(options.fused_activation_function());
  }

  int GetVersion(const Operator& op) const override { return 1; }
};

class Gather : public BuiltinOperator<GatherOperator, ::tflite::GatherOptions,
                                      ::tflite::BuiltinOptions_GatherOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    return ::tflite::CreateGatherOptions(*builder, op.axis);
  }

  void ReadOptions(const TfLiteOptions& options,
                   TocoOperator* op) const override {
    op->axis = options.axis();
  }

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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
  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
};

class Lstm : public BuiltinOperator<LstmCellOperator, ::tflite::LSTMOptions,
                                    ::tflite::BuiltinOptions_LSTMOptions> {
 public:
  using BuiltinOperator::BuiltinOperator;
  flatbuffers::Offset<TfLiteOptions> WriteOptions(
      const TocoOperator& op,
      flatbuffers::FlatBufferBuilder* builder) const override {
    ::tflite::LSTMKernelType kernel_type;
    switch (op.kernel_type) {
      case LstmCellOperator::KERNEL_BASIC:
        kernel_type = ::tflite::LSTMKernelType_BASIC;
        break;
      case LstmCellOperator::KERNEL_FULL:
        kernel_type = ::tflite::LSTMKernelType_FULL;
        break;
    }

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

  int GetVersion(const Operator& op) const override {
    const auto& lstm_op = static_cast<const LstmCellOperator&>(op);
    switch (lstm_op.kernel_type) {
      case LstmCellOperator::KERNEL_FULL:
        return 1;
      case LstmCellOperator::KERNEL_BASIC:
        return 2;
    }
  }

  std::vector<bool> GetMutatingInputVariables(
      const Operator& op) const override {
    const auto& lstm_op = static_cast<const LstmCellOperator&>(op);

    switch (lstm_op.kernel_type) {
      case LstmCellOperator::KERNEL_FULL:
        // TODO(ycling): Change the full kernel to use the new variable tensor
        // design. This requires moving the state tensors from output to input.
        return std::vector<bool>();
      case LstmCellOperator::KERNEL_BASIC: {
        std::vector<bool> mutating_input_variables(op.inputs.size(), false);
        mutating_input_variables[LstmCellOperator::PREV_ACTIV_INPUT] = true;
        mutating_input_variables[LstmCellOperator::PREV_STATE_INPUT] = true;
        return mutating_input_variables;
      }
    }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
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

  int GetVersion(const Operator& op) const override { return 1; }
};

class TensorFlowUnsupported : public BaseOperator {
 public:
  using BaseOperator::BaseOperator;

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
    auto fbb = absl::make_unique<flexbuffers::Builder>();

    ::tensorflow::NodeDef node_def;
    if (!node_def.ParseFromString(op.tensorflow_node_def)) {
      LOG(ERROR) << "Failed to parse TensorFlow NodeDef";
      return std::unique_ptr<flexbuffers::Builder>();
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
      switch (value.GetType()) {
        case flexbuffers::TYPE_STRING:
          (*attr)[key].set_s(value.AsString().c_str());
          break;
        case flexbuffers::TYPE_INT:
          (*attr)[key].set_i(value.AsInt64());
          break;
        case flexbuffers::TYPE_FLOAT:
          (*attr)[key].set_f(value.AsFloat());
          break;
        case flexbuffers::TYPE_BOOL:
          (*attr)[key].set_b(value.AsBool());
          break;
        default:
          LOG(WARNING) << "Ignoring unsupported attribute type with key '"
                       << key << "'";
          break;
      }
    }
    node_def.SerializeToString(&op->tensorflow_node_def);
  }

  int GetVersion(const Operator& op) const override {
    // TODO(ycling): Deisng and implement a way to plumb the version of
    // custom ops.
    return 1;
  }
};

namespace {
// Build a vector containing all the known operators.
std::vector<std::unique_ptr<BaseOperator>> BuildOperatorList() {
  std::vector<std::unique_ptr<BaseOperator>> ops;

  // Builtin Operators.
  ops.emplace_back(new Add(::tflite::BuiltinOperator_ADD, OperatorType::kAdd));
  ops.emplace_back(new Div(::tflite::BuiltinOperator_DIV, OperatorType::kDiv));
  ops.emplace_back(new Sub(::tflite::BuiltinOperator_SUB, OperatorType::kSub));
  ops.emplace_back(new AveragePool(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
                                   OperatorType::kAveragePool));
  ops.emplace_back(
      new SpaceToBatchND(::tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                         OperatorType::kSpaceToBatchND));
  ops.emplace_back(
      new BatchToSpaceND(::tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                         OperatorType::kBatchToSpaceND));
  ops.emplace_back(new Concatenation(::tflite::BuiltinOperator_CONCATENATION,
                                     OperatorType::kConcatenation));
  ops.emplace_back(
      new Convolution(::tflite::BuiltinOperator_CONV_2D, OperatorType::kConv));
  ops.emplace_back(
      new DepthwiseConvolution(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                               OperatorType::kDepthwiseConv));
  ops.emplace_back(new FullyConnected(::tflite::BuiltinOperator_FULLY_CONNECTED,
                                      OperatorType::kFullyConnected));
  ops.emplace_back(
      new Gather(::tflite::BuiltinOperator_GATHER, OperatorType::kGather));
  ops.emplace_back(
      new L2Normalization(::tflite::BuiltinOperator_L2_NORMALIZATION,
                          OperatorType::kL2Normalization));
  ops.emplace_back(
      new L2Pool(::tflite::BuiltinOperator_L2_POOL_2D, OperatorType::kL2Pool));
  ops.emplace_back(new LocalResponseNormalization(
      ::tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
      OperatorType::kLocalResponseNormalization));
  ops.emplace_back(new MaxPool(::tflite::BuiltinOperator_MAX_POOL_2D,
                               OperatorType::kMaxPool));
  ops.emplace_back(new Mul(::tflite::BuiltinOperator_MUL, OperatorType::kMul));
  ops.emplace_back(new Pad(::tflite::BuiltinOperator_PAD, OperatorType::kPad));
  ops.emplace_back(
      new PadV2(::tflite::BuiltinOperator_PADV2, OperatorType::kPadV2));
  ops.emplace_back(new Reshape(::tflite::BuiltinOperator_RESHAPE,
                               OperatorType::kTensorFlowReshape));
  ops.emplace_back(
      new Softmax(::tflite::BuiltinOperator_SOFTMAX, OperatorType::kSoftmax));
  ops.emplace_back(new SpaceToDepth(::tflite::BuiltinOperator_SPACE_TO_DEPTH,
                                    OperatorType::kSpaceToDepth));
  ops.emplace_back(
      new Svdf(::tflite::BuiltinOperator_SVDF, OperatorType::kSvdf));
  ops.emplace_back(new Transpose(::tflite::BuiltinOperator_TRANSPOSE,
                                 OperatorType::kTranspose));
  ops.emplace_back(
      new Mean(::tflite::BuiltinOperator_MEAN, OperatorType::kMean));
  ops.emplace_back(
      new Sum(::tflite::BuiltinOperator_SUM, OperatorType::kTensorFlowSum));
  ops.emplace_back(new ResizeBilinear(::tflite::BuiltinOperator_RESIZE_BILINEAR,
                                      OperatorType::kResizeBilinear));
  ops.emplace_back(
      new Squeeze(::tflite::BuiltinOperator_SQUEEZE, OperatorType::kSqueeze));
  ops.emplace_back(new Split(::tflite::BuiltinOperator_SPLIT,
                             OperatorType::kTensorFlowSplit));
  ops.emplace_back(new StridedSlice(::tflite::BuiltinOperator_STRIDED_SLICE,
                                    OperatorType::kStridedSlice));
  ops.emplace_back(
      new TopK_V2(::tflite::BuiltinOperator_TOPK_V2, OperatorType::kTopK_V2));
  ops.emplace_back(
      new Lstm(::tflite::BuiltinOperator_LSTM, OperatorType::kLstmCell));
  ops.emplace_back(
      new Cast(::tflite::BuiltinOperator_CAST, OperatorType::kCast));
  ops.emplace_back(
      new ArgMax(::tflite::BuiltinOperator_ARG_MAX, OperatorType::kArgMax));
  ops.emplace_back(
      new Tile(::tflite::BuiltinOperator_TILE, OperatorType::kTensorFlowTile));
  ops.emplace_back(new ExpandDims(::tflite::BuiltinOperator_EXPAND_DIMS,
                                  OperatorType::kExpandDims));
  ops.emplace_back(new TransposeConv(::tflite::BuiltinOperator_TRANSPOSE_CONV,
                                     OperatorType::kTransposeConv));
  ops.emplace_back(new SparseToDense(::tflite::BuiltinOperator_SPARSE_TO_DENSE,
                                     OperatorType::kSparseToDense));

  // Custom Operators.
  ops.emplace_back(
      new DepthToSpace("DEPTH_TO_SPACE", OperatorType::kDepthToSpace));
  ops.emplace_back(new FakeQuant("FAKE_QUANT", OperatorType::kFakeQuant));
  ops.emplace_back(new TensorFlowUnsupported(
      "TENSORFLOW_UNSUPPORTED", OperatorType::kTensorFlowUnsupported));

  // There operators are supported by Toco, but not by TF Lite, and has no
  // attributes.
  ops.emplace_back(
      new SimpleOperator<AddNOperator>("ADDN", OperatorType::kAddN));
  ops.emplace_back(new SimpleOperator<TensorFlowRsqrtOperator>(
      "RSQRT", OperatorType::kTensorFlowRsqrt));
  // Simple Operators.
  ops.emplace_back(new SimpleOperator<DequantizeOperator>(
      "DEQUANTIZE", OperatorType::kDequantize));
  ops.emplace_back(
      new SimpleOperator<FloorOperator>("FLOOR", OperatorType::kFloor));
  ops.emplace_back(
      new SimpleOperator<ReluOperator>("RELU", OperatorType::kRelu));
  ops.emplace_back(
      new SimpleOperator<Relu1Operator>("RELU_N1_TO_1", OperatorType::kRelu1));
  ops.emplace_back(
      new SimpleOperator<Relu6Operator>("RELU6", OperatorType::kRelu6));
  ops.emplace_back(
      new SimpleOperator<PReluOperator>("PRELU", OperatorType::kPRelu));
  ops.emplace_back(new SimpleOperator<LogisticOperator>(
      "LOGISTIC", OperatorType::kLogistic));
  ops.emplace_back(
      new SimpleOperator<TanhOperator>("TANH", OperatorType::kTanh));
  ops.emplace_back(new SimpleOperator<ExpOperator>("EXP", OperatorType::kExp));
  ops.emplace_back(new SimpleOperator<LogSoftmaxOperator>(
      "LOG_SOFTMAX", OperatorType::kLogSoftmax));
  ops.emplace_back(new SimpleOperator<TensorFlowMaximumOperator>(
      "MAXIMUM", OperatorType::kTensorFlowMaximum));
  ops.emplace_back(new SimpleOperator<TensorFlowMinimumOperator>(
      "MINIMUM", OperatorType::kTensorFlowMinimum));
  ops.emplace_back(new SimpleOperator<TensorFlowGreaterOperator>(
      "GREATER", OperatorType::kTensorFlowGreater));
  ops.emplace_back(new SimpleOperator<TensorFlowGreaterEqualOperator>(
      "GREATER_EQUAL", OperatorType::kTensorFlowGreaterEqual));
  ops.emplace_back(new SimpleOperator<TensorFlowLessOperator>(
      "LESS", OperatorType::kTensorFlowLess));
  ops.emplace_back(new SimpleOperator<TensorFlowLessEqualOperator>(
      "LESS_EQUAL", OperatorType::kTensorFlowLessEqual));
  ops.emplace_back(new SimpleOperator<TensorFlowEqualOperator>(
      "EQUAL", OperatorType::kTensorFlowEqual));
  ops.emplace_back(new SimpleOperator<TensorFlowNotEqualOperator>(
      "NOT_EQUAL", OperatorType::kTensorFlowNotEqual));
  ops.emplace_back(new SimpleOperator<NegOperator>("NEG", OperatorType::kNeg));
  ops.emplace_back(
      new SimpleOperator<SelectOperator>("SELECT", OperatorType::kSelect));
  ops.emplace_back(
      new SimpleOperator<SliceOperator>("SLICE", OperatorType::kSlice));
  // Element-wise operator
  ops.emplace_back(new SimpleOperator<SinOperator>("SIN", OperatorType::kSin));
  ops.emplace_back(new SimpleOperator<LogOperator>("LOG", OperatorType::kLog));

  return ops;
}
}  // namespace

std::map<OperatorType, std::unique_ptr<BaseOperator>> BuildOperatorByTypeMap() {
  std::map<OperatorType, std::unique_ptr<BaseOperator>> result;

  std::vector<std::unique_ptr<BaseOperator>> ops = BuildOperatorList();
  for (auto& op : ops) {
    result[op->type()] = std::move(op);
  }

  return result;
}

std::map<string, std::unique_ptr<BaseOperator>> BuildOperatorByNameMap() {
  std::map<string, std::unique_ptr<BaseOperator>> result;

  std::vector<std::unique_ptr<BaseOperator>> ops = BuildOperatorList();
  for (auto& op : ops) {
    result[op->name()] = std::move(op);
  }

  return result;
}

}  // namespace tflite

}  // namespace toco
