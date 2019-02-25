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

#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {
namespace {
void GetAndSetNormOpts(OpKernelConstruction* ctx,
                       IPUCustomKernelsUtil::AttributeMap& attribute_map,
                       int32& num_groups, TensorFormat& data_format) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_groups", &num_groups));
  attribute_map.AddAttribute("num_groups", num_groups);
  float epsilon;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
  attribute_map.AddAttribute("epsilon", epsilon);
  std::string data_format_str;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
  OP_REQUIRES(
      ctx, FormatFromString(data_format_str, &data_format),
      errors::InvalidArgument("Invalid data format: ", data_format_str));
};
}  // namespace

class PopnnLstmLayerOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnLstmLayerOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    AddRequiredAttributesToMap();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_channels", &num_channels_));
    attribute_map_.AddAttribute("num_channels", num_channels_);
    bool is_training;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
    attribute_map_.AddAttribute("is_training", is_training);
    tensorflow::DataType partials_dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partials_dtype", &partials_dtype));
    attribute_map_.AddAttribute("partials_dtype", partials_dtype);
  }

 public:
  ~PopnnLstmLayerOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    const TensorShape input_shape = ctx->InputShape(0);
    const auto time_steps = input_shape.dim_size(0);
    const auto batch_size = input_shape.dim_size(1);
    const auto input_size = input_shape.dim_size(2);

    // Validate shapes.
    TensorShape expected_input_h_state_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({batch_size, num_channels_}),
                                &expected_input_h_state_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(1) == expected_input_h_state_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The initial hidden state tensor needs to be of shape [%u, %u].",
            batch_size, num_channels_)));

    TensorShape expected_input_c_state_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({batch_size, num_channels_}),
                                &expected_input_c_state_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(2) == expected_input_c_state_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The initial cell state tensor needs to be of shape [%u, %u].",
            batch_size, num_channels_)));

    TensorShape expected_kernel_shape;
    TensorShapeUtils::MakeShape(
        std::vector<int64>({input_size + num_channels_, 4 * num_channels_}),
        &expected_kernel_shape);
    OP_REQUIRES(ctx, ctx->InputShape(3) == expected_kernel_shape,
                errors::InvalidArgument(absl::StrFormat(
                    "The input kernel tensor needs to be of shape [%u, %u].",
                    input_size + num_channels_, 4 * num_channels_)));

    TensorShape expected_biases_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({4, num_channels_}),
                                &expected_biases_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(4) == expected_biases_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The biases tensor needs to be of shape [4, %u].", num_channels_)));

    xla::Shape output_seq_shape = xla::ShapeUtil::MakeShape(
        input_type, {time_steps, batch_size, num_channels_});
    xla::Shape output_h_state_shape =
        xla::ShapeUtil::MakeShape(input_type, {batch_size, num_channels_});
    xla::Shape output_c_state_shape =
        xla::ShapeUtil::MakeShape(input_type, {batch_size, num_channels_});
    // The 6 in intermidate shape represents the number of gates (4) + tanh
    // output (1) + cell state (1)
    xla::Shape intermidates_shape = xla::ShapeUtil::MakeShape(
        input_type, {time_steps, 6, batch_size, num_channels_});
    xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
        {output_seq_shape, output_h_state_shape, output_c_state_shape,
         intermidates_shape});

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (unsigned idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    xla::XlaOp output_tuple =
        xla::CustomCall(&b,
                        GetPoplibsCustomOpTargetString(PoplibsLib::Popnn,
                                                       PoplibsOp::LstmLayerFwd),
                        args, output_tuple_shape, attribute_map_.Serialise());

    xla::XlaOp output_seq = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp output_h_state = xla::GetTupleElement(output_tuple, 1);
    xla::XlaOp output_c_state = xla::GetTupleElement(output_tuple, 2);
    xla::XlaOp intermediates = xla::GetTupleElement(output_tuple, 3);
    ctx->SetOutput(0, output_seq);
    ctx->SetOutput(1, output_h_state);
    ctx->SetOutput(2, output_c_state);
    ctx->SetOutput(3, intermediates);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override {
    return {0, 1, 2, 3, 4};
  }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

 private:
  int32 num_channels_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnLstmLayerOp);
};
REGISTER_IPU_OP("PopnnLstmLayer", PopnnLstmLayerOp);

class PopnnLstmLayerBackpropOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnLstmLayerBackpropOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    AddRequiredAttributesToMap();
    int32 num_channels;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_channels", &num_channels));
    attribute_map_.AddAttribute("num_channels", num_channels);
    bool is_training;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
    attribute_map_.AddAttribute("is_training", is_training);
    tensorflow::DataType partials_dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partials_dtype", &partials_dtype));
    attribute_map_.AddAttribute("partials_dtype", partials_dtype);
  }

 public:
  ~PopnnLstmLayerBackpropOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    // Don't need to validate shapes as this is a grad op.
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    xla::Shape input_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(0));
    xla::Shape input_h_state_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(1));
    xla::Shape input_c_state_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(2));
    xla::Shape kernel_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(3));
    xla::Shape biases_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(4));

    xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
        {input_backprop_shape, input_h_state_backprop_shape,
         input_c_state_backprop_shape, kernel_backprop_shape,
         biases_backprop_shape});

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (unsigned idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    xla::XlaOp output_tuple =
        xla::CustomCall(&b,
                        GetPoplibsCustomOpTargetString(PoplibsLib::Popnn,
                                                       PoplibsOp::LstmLayerBwd),
                        args, output_tuple_shape, attribute_map_.Serialise());
    xla::XlaOp input_backprop = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp input_h_state_backprop = xla::GetTupleElement(output_tuple, 1);
    xla::XlaOp input_c_state_backprop = xla::GetTupleElement(output_tuple, 2);
    xla::XlaOp kernel_backprop = xla::GetTupleElement(output_tuple, 3);
    xla::XlaOp biases_backprop = xla::GetTupleElement(output_tuple, 4);

    ctx->SetOutput(0, input_backprop);
    ctx->SetOutput(1, input_h_state_backprop);
    ctx->SetOutput(2, input_c_state_backprop);
    ctx->SetOutput(3, kernel_backprop);
    ctx->SetOutput(4, biases_backprop);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopnnLstmLayerBackpropOp);
};
REGISTER_IPU_OP("PopnnLstmLayerBackprop", PopnnLstmLayerBackpropOp);

class PopnnGroupNorm : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGroupNorm(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    AddRequiredAttributesToMap();
    GetAndSetNormOpts(ctx, attribute_map_, num_groups_, data_format_);
  }

 public:
  ~PopnnGroupNorm() override{};

  virtual void Compile(XlaOpKernelContext* ctx) {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    TensorShape input_shape = ctx->InputShape(0);

    const int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);
    attribute_map_.AddAttribute("feature_index", feature_index);
    const int batch_index =
        GetTensorBatchDimIndex(input_shape.dims(), data_format_);

    const auto num_batches = input_shape.dim_size(batch_index);
    const auto num_channels = input_shape.dim_size(feature_index);

    xla::XlaBuilder& b = *ctx->builder();

    // All the inputs are arguments.
    std::vector<xla::XlaOp> args;
    for (unsigned idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    // Validate scale and offset shapes are per channel.
    TensorShape expected_scale_offset_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({num_channels}),
                                &expected_scale_offset_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(1) == expected_scale_offset_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The scale tensor needs to be of shape [%u].", num_channels)));
    OP_REQUIRES(
        ctx, ctx->InputShape(2) == expected_scale_offset_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The offset tensor needs to be of shape [%u].", num_channels)));

    if (ctx->num_inputs() == 5) {
      // Inference
      // Validate mean/inv_std_dev shape is per group
      TensorShape expected_mean_inv_std_dev_shape;
      TensorShapeUtils::MakeShape(
          std::vector<int64>({num_groups_ * num_batches}),
          &expected_mean_inv_std_dev_shape);
      OP_REQUIRES(ctx, ctx->InputShape(3) == expected_mean_inv_std_dev_shape,
                  errors::InvalidArgument(absl::StrFormat(
                      "The mean tensor needs to be of shape [%u].",
                      num_groups_ * num_batches)));
      OP_REQUIRES(ctx, ctx->InputShape(4) == expected_mean_inv_std_dev_shape,
                  errors::InvalidArgument(absl::StrFormat(
                      "The inv_std_dev tensor needs to be of shape [%u].",
                      num_groups_ * num_batches)));
      xla::Shape output_shape = TensorShapeToXLAShape(input_type, input_shape);
      xla::XlaOp call_output =
          xla::CustomCall(&b,
                          GetPoplibsCustomOpTargetString(
                              PoplibsLib::Popnn, PoplibsOp::GroupNormInference),
                          args, output_shape, attribute_map_.Serialise());
      ctx->SetOutput(0, call_output);
    } else if (ctx->num_inputs() == 3) {
      // Training
      xla::Shape output_shape = TensorShapeToXLAShape(input_type, input_shape);
      xla::Shape mean_inv_std_dev_shape =
          xla::ShapeUtil::MakeShape(input_type, {num_groups_ * num_batches});

      xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
          {output_shape, mean_inv_std_dev_shape, mean_inv_std_dev_shape});
      xla::XlaOp call_output =
          xla::CustomCall(&b,
                          GetPoplibsCustomOpTargetString(
                              PoplibsLib::Popnn, PoplibsOp::GroupNormTraining),
                          args, output_tuple_shape, attribute_map_.Serialise());
      xla::XlaOp output = xla::GetTupleElement(call_output, 0);
      xla::XlaOp mean = xla::GetTupleElement(call_output, 1);
      xla::XlaOp inv_std_dev = xla::GetTupleElement(call_output, 2);

      ctx->SetOutput(0, output);
      ctx->SetOutput(1, mean);
      ctx->SetOutput(2, inv_std_dev);
    } else {
      LOG(FATAL) << "Unsupported use of PopnnGroupNorm.";
    }
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    // Scale and offset layouts depend on the passed in activations.
    return {{1, 0}, {2, 0}};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

 private:
  int32 num_groups_;
  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGroupNorm);
};
REGISTER_IPU_OP("PopnnGroupNormInference", PopnnGroupNorm);
REGISTER_IPU_OP("PopnnGroupNormTraining", PopnnGroupNorm);

class PopnnGroupNormGrad : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGroupNormGrad(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    AddRequiredAttributesToMap();
    GetAndSetNormOpts(ctx, attribute_map_, num_groups_, data_format_);
  }

 public:
  void Compile(XlaOpKernelContext* ctx) override {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    TensorShape input_shape = ctx->InputShape(0);
    xla::Shape input_backprop_shape =
        TensorShapeToXLAShape(input_type, input_shape);
    TensorShape gamma_beta_shape = ctx->InputShape(1);
    xla::Shape gamma_beta_backprop_shape =
        TensorShapeToXLAShape(input_type, gamma_beta_shape);
    // Don't need to validate shapes as they are coming from the training op.

    const int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);
    attribute_map_.AddAttribute("feature_index", feature_index);

    xla::XlaBuilder& b = *ctx->builder();

    // All the inputs are arguments.
    std::vector<xla::XlaOp> args;
    for (unsigned idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
        {input_backprop_shape, gamma_beta_backprop_shape,
         gamma_beta_backprop_shape});
    xla::XlaOp call_output =
        xla::CustomCall(&b,
                        GetPoplibsCustomOpTargetString(
                            PoplibsLib::Popnn, PoplibsOp::GroupNormGrad),
                        args, output_tuple_shape, attribute_map_.Serialise());
    xla::XlaOp input_backprop = xla::GetTupleElement(call_output, 0);
    xla::XlaOp gamma_backprop = xla::GetTupleElement(call_output, 1);
    xla::XlaOp beta_backprop = xla::GetTupleElement(call_output, 2);

    ctx->SetOutput(0, input_backprop);
    ctx->SetOutput(1, gamma_backprop);
    ctx->SetOutput(2, beta_backprop);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

 private:
  int32 num_groups_;
  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGroupNormGrad);
};
REGISTER_IPU_OP("PopnnGroupNormGrad", PopnnGroupNormGrad);

class PopnnGroupNormStatistics : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGroupNormStatistics(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    AddRequiredAttributesToMap();
    GetAndSetNormOpts(ctx, attribute_map_, num_groups_, data_format_);
  }

 public:
  virtual void Compile(XlaOpKernelContext* ctx) {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    TensorShape input_shape = ctx->InputShape(0);

    const int feature_index =
        GetTensorFeatureDimIndex(input_shape.dims(), data_format_);
    attribute_map_.AddAttribute("feature_index", feature_index);
    const int batch_index =
        GetTensorBatchDimIndex(input_shape.dims(), data_format_);

    const auto num_batches = input_shape.dim_size(batch_index);

    xla::XlaBuilder& b = *ctx->builder();

    // All the inputs are arguments.
    std::vector<xla::XlaOp> args;
    for (unsigned idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    xla::Shape mean_inv_std_dev_shape =
        xla::ShapeUtil::MakeShape(input_type, {num_groups_ * num_batches});

    xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
        {mean_inv_std_dev_shape, mean_inv_std_dev_shape});
    xla::XlaOp call_output =
        xla::CustomCall(&b,
                        GetPoplibsCustomOpTargetString(
                            PoplibsLib::Popnn, PoplibsOp::GroupNormStatistics),
                        args, output_tuple_shape, attribute_map_.Serialise());
    xla::XlaOp mean = xla::GetTupleElement(call_output, 0);
    xla::XlaOp inv_std_dev = xla::GetTupleElement(call_output, 1);

    ctx->SetOutput(0, mean);
    ctx->SetOutput(1, inv_std_dev);
  }

 protected:
  const absl::flat_hash_set<int64> AllocatingIndexes() override { return {}; }

  const absl::flat_hash_map<int64, int64> LayoutDependencies() override {
    return {};
  };

  const uint64 NumberOfInplaceOperands() override { return 0; }

 private:
  int32 num_groups_;
  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGroupNormStatistics);
};
REGISTER_IPU_OP("PopnnGroupNormStatistics", PopnnGroupNormStatistics);

}  // namespace tensorflow
