/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/ops/tpu_embedding_ops.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/proto_helper.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/tpu_embedding_spmd_sharding_utils.h"

namespace tensorflow {

using xla::LiteralUtil;

namespace {

void CompileRecvTPUEmbeddingActivations(
    XlaOpKernelContext* ctx, const std::string& config_string,
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config,
    const std::string& embedding_partitions_string,
    const std::string& hbm_buffers_config_string,
    const std::string& tpu_topology_string) {
  xla::XlaOp deduplication_data = ctx->Input("deduplication_data");
  TpuEmbeddingEngine_RecvActivationsComputation_Params params;
  params.tpu_embedding_config.bytes = config_string.c_str();
  params.tpu_embedding_config.size = config_string.size();
  params.embedding_partitions.bytes = embedding_partitions_string.c_str();
  params.embedding_partitions.size = embedding_partitions_string.size();
  params.hbm_buffers_config.bytes = hbm_buffers_config_string.c_str();
  params.hbm_buffers_config.size = hbm_buffers_config_string.size();
  params.tpu_topology.bytes = tpu_topology_string.c_str();
  params.tpu_topology.size = tpu_topology_string.size();
  StatusHelper status;
  params.status = status.c_status;
  auto builder = ctx->builder();
  OP_REQUIRES_VALUE(auto shape, ctx, builder->GetShape(deduplication_data));
  TpuSerializedProto xla_computation_serialized;
  auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
    StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
  });
  params.xla_computation = &xla_computation_serialized;
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  auto c_shape_cleanup =
      absl::MakeCleanup([&c_shape] { ApiConverter::Destroy(&c_shape); });
  params.deduplication_data_shape = &c_shape;

  TpuSerializedProto op_sharding_proto_serialized;
  if (ctx->builder()->sharding().has_value()) {
    stream_executor::tpu::SerializeProto(ctx->builder()->sharding().value(),
                                         &op_sharding_proto_serialized);
    params.op_sharding = &op_sharding_proto_serialized;
  } else {
    params.op_sharding = nullptr;
  }
  auto op_sharding_cleanup = absl::MakeCleanup([&] {
    if (params.op_sharding) {
      StreamExecutor_Tpu_FreeSerializedProto(&op_sharding_proto_serialized);
    }
  });

  stream_executor::tpu::OpsApiFn()
      ->TpuEmbeddingEngine_RecvActivationsComputationFn(&params);
  OP_REQUIRES_OK(ctx, status.status());
  auto xla_computation =
      stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
          xla_computation_serialized);
  auto final_activations =
      xla::Call(builder, xla_computation, {deduplication_data});

  // Ensure that the number of outputs is the same as the number of user
  // tables.
  const int32_t output_count =
      (tpu_embedding_config.feature_descriptor_size() == 0)
          ? tpu_embedding_config.table_descriptor_size()
          : tpu_embedding_config.feature_descriptor_size();
  OP_REQUIRES(ctx, ctx->num_outputs() == output_count,
              errors::InvalidArgument(
                  "Kernel has %d outputs but configuration expects %d outputs.",
                  ctx->num_outputs(), output_count));

  for (int32_t output_id = 0; output_id < output_count; ++output_id) {
    ctx->SetOutput(output_id,
                   xla::GetTupleElement(final_activations, output_id));
  }
}

void CompileRecvTPUEmbeddingDeduplicationData(
    XlaOpKernelContext* ctx, const std::string& config_string,
    const std::string& embedding_partitions_string,
    const std::string& hbm_buffers_config_string,
    const std::string& tpu_topology_string) {
  TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params params;

  params.tpu_embedding_config.bytes = config_string.c_str();
  params.tpu_embedding_config.size = config_string.size();
  params.embedding_partitions.bytes = embedding_partitions_string.c_str();
  params.embedding_partitions.size = embedding_partitions_string.size();
  params.hbm_buffers_config.bytes = hbm_buffers_config_string.c_str();
  params.hbm_buffers_config.size = hbm_buffers_config_string.size();
  params.tpu_topology.bytes = tpu_topology_string.c_str();
  params.tpu_topology.size = tpu_topology_string.size();
  TpuSerializedProto xla_computation_serialized;
  auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
    StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
  });
  params.xla_computation = &xla_computation_serialized;
  StatusHelper status;
  params.status = status.c_status;

  TpuSerializedProto op_sharding_proto_serialized;
  if (ctx->builder()->sharding().has_value()) {
    stream_executor::tpu::SerializeProto(ctx->builder()->sharding().value(),
                                         &op_sharding_proto_serialized);
    params.op_sharding = &op_sharding_proto_serialized;
  } else {
    params.op_sharding = nullptr;
  }
  auto op_sharding_cleanup = absl::MakeCleanup([&] {
    if (params.op_sharding) {
      StreamExecutor_Tpu_FreeSerializedProto(&op_sharding_proto_serialized);
    }
  });

  stream_executor::tpu::OpsApiFn()
      ->TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputationFn(
          &params);
  OP_REQUIRES_OK(ctx, status.status());

  auto xla_computation =
      stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
          xla_computation_serialized);

  const xla::XlaOp deduplication_data =
      xla::Call(ctx->builder(), xla_computation, {});

  // Ensure that the number of outputs is equal to 1 (for deduplication data).
  OP_REQUIRES(ctx, ctx->num_outputs() == 1,
              errors::InvalidArgument(
                  "Kernel has %d outputs but configuration expects 1 output.",
                  ctx->num_outputs()));

  ctx->SetOutput(0, deduplication_data);
}

void CompileSendTPUEmbeddingGradients(
    XlaOpKernelContext* ctx, const std::string& config_string,
    const std::string& embedding_partitions_string,
    const std::string& hbm_buffers_config_string,
    const std::string& tpu_topology_string) {
  std::vector<xla::XlaOp> gradients;
  std::vector<TensorShape> tf_gradient_shapes;
  OP_REQUIRES_OK(ctx,
                 ctx->InputList("gradients", &gradients, &tf_gradient_shapes));
  std::vector<xla::Shape> gradient_shapes;
  auto builder = ctx->builder();
  gradient_shapes.reserve(gradients.size());
  for (int i = 0; i < gradients.size(); ++i) {
    DataType dtype = ctx->input_type(i);
    xla::Shape gradient_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, tf_gradient_shapes[i],
                                              &gradient_shape));
    gradient_shapes.push_back(gradient_shape);
  }

  std::vector<xla::XlaOp> learning_rates;
  std::vector<TensorShape> tf_learning_rate_shapes;
  OP_REQUIRES_OK(ctx, ctx->InputList("learning_rates", &learning_rates,
                                     &tf_learning_rate_shapes));
  std::vector<xla::Shape> learning_rate_shapes;
  learning_rate_shapes.reserve(learning_rates.size());
  for (xla::XlaOp op : learning_rates) {
    learning_rate_shapes.push_back(builder->GetShape(op).value());
  }

  xla::XlaOp deduplication_data = ctx->Input("deduplication_data");

  TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params params;
  params.tpu_embedding_config.bytes = config_string.c_str();
  params.tpu_embedding_config.size = config_string.size();
  params.embedding_partitions.bytes = embedding_partitions_string.c_str();
  params.embedding_partitions.size = embedding_partitions_string.size();
  params.hbm_buffers_config.bytes = hbm_buffers_config_string.c_str();
  params.hbm_buffers_config.size = hbm_buffers_config_string.size();
  params.tpu_topology.bytes = tpu_topology_string.c_str();
  params.tpu_topology.size = tpu_topology_string.size();
  TpuSerializedProto xla_computation_serialized;
  auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
    StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
  });
  params.xla_computation = &xla_computation_serialized;
  StatusHelper status;
  params.status = status.c_status;
  OP_REQUIRES_VALUE(auto deduplication_shape, ctx,
                    builder->GetShape(deduplication_data));
  XLA_Shape gradient_tuple_c_shape;
  params.gradient_tuple_shape = &gradient_tuple_c_shape;
  ApiConverter::ToC(xla::ShapeUtil::MakeTupleShape(gradient_shapes),
                    &gradient_tuple_c_shape);
  XLA_Shape learning_rate_tuple_c_shape;
  params.learning_rate_tuple_shape = &learning_rate_tuple_c_shape;
  ApiConverter::ToC(xla::ShapeUtil::MakeTupleShape(learning_rate_shapes),
                    &learning_rate_tuple_c_shape);
  XLA_Shape deduplication_c_shape;
  params.deduplication_data_shape = &deduplication_c_shape;
  ApiConverter::ToC(deduplication_shape, &deduplication_c_shape);

  auto c_shape_cleanup =
      absl::MakeCleanup([&gradient_tuple_c_shape, &learning_rate_tuple_c_shape,
                         &deduplication_c_shape] {
        ApiConverter::Destroy(&gradient_tuple_c_shape);
        ApiConverter::Destroy(&learning_rate_tuple_c_shape);
        ApiConverter::Destroy(&deduplication_c_shape);
      });
  params.num_inputs = ctx->num_inputs();

  TpuSerializedProto op_sharding_proto_serialized;
  if (ctx->builder()->sharding().has_value()) {
    stream_executor::tpu::SerializeProto(ctx->builder()->sharding().value(),
                                         &op_sharding_proto_serialized);
    params.op_sharding = &op_sharding_proto_serialized;
  } else {
    params.op_sharding = nullptr;
  }
  auto op_sharding_cleanup = absl::MakeCleanup([&] {
    if (params.op_sharding) {
      StreamExecutor_Tpu_FreeSerializedProto(&op_sharding_proto_serialized);
    }
  });

  stream_executor::tpu::OpsApiFn()
      ->TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputationFn(&params);
  OP_REQUIRES_OK(ctx, status.status());

  auto xla_computation =
      stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
          xla_computation_serialized);

  xla::Call(builder, xla_computation,
            {xla::Tuple(builder, gradients),
             xla::Tuple(builder, learning_rates), deduplication_data});
}

void CompileComputeDedupDataSize(XlaOpKernelContext* ctx,
                                 const std::string& config_string,
                                 const std::string& embedding_partitions_string,
                                 const std::string& hbm_buffers_config_string,
                                 const std::string& tpu_topology_string) {
  TpuEmbeddingEngine_DedupDataSizeComputation_Params params;
  params.tpu_embedding_config.bytes = config_string.c_str();
  params.tpu_embedding_config.size = config_string.size();
  params.embedding_partitions.bytes = embedding_partitions_string.c_str();
  params.embedding_partitions.size = embedding_partitions_string.size();
  params.hbm_buffers_config.bytes = hbm_buffers_config_string.c_str();
  params.hbm_buffers_config.size = hbm_buffers_config_string.size();
  params.tpu_topology.bytes = tpu_topology_string.c_str();
  params.tpu_topology.size = tpu_topology_string.size();
  int num_elements = -1;
  params.num_elements = &num_elements;
  StatusHelper status;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()
      ->TpuEmbeddingEngine_DedupDataSizeComputationFn(&params);
  OP_REQUIRES_OK(ctx, status.status());

  auto output = xla::ConstantLiteral(
      ctx->builder(), LiteralUtil::CreateR0<int32_t>(num_elements));
  ctx->SetOutput(0, output);
}

void CompileComputeDedupDataTupleMask(
    XlaOpKernelContext* ctx, const std::string& config_string,
    const std::string& embedding_partitions_string,
    const std::string& hbm_buffers_config_string,
    const std::string& tpu_topology_string) {
  TpuEmbeddingEngine_DedupDataTupleMaskComputation_Params params;
  params.tpu_embedding_config.bytes = config_string.c_str();
  params.tpu_embedding_config.size = config_string.size();
  params.embedding_partitions.bytes = embedding_partitions_string.c_str();
  params.embedding_partitions.size = embedding_partitions_string.size();
  params.hbm_buffers_config.bytes = hbm_buffers_config_string.c_str();
  params.hbm_buffers_config.size = hbm_buffers_config_string.size();
  params.tpu_topology.bytes = tpu_topology_string.c_str();
  params.tpu_topology.size = tpu_topology_string.size();

  TpuSerializedProto xla_computation_serialized;
  auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
    StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
  });

  params.xla_computation = &xla_computation_serialized;
  StatusHelper status;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()
      ->TpuEmbeddingEngine_DedupDataTupleMaskComputationFn(&params);
  OP_REQUIRES_OK(ctx, status.status());

  auto xla_computation =
      stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
          xla_computation_serialized);
  const xla::XlaOp deduplication_data_tuple_mask =
      xla::Call(ctx->builder(), xla_computation, {});
  ctx->SetOutput(0, deduplication_data_tuple_mask);
}

// This TensorFlow op receives a batch of activations from the
// TpuEmbeddingEngine.
class RecvTPUEmbeddingActivationsOp : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingActivationsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));

    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  ~RecvTPUEmbeddingActivationsOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 1,
        errors::Internal("Kernel has ", ctx->num_inputs(),
                         " inputs but configuration expects one input"));
    CompileRecvTPUEmbeddingActivations(ctx, config_string_,
                                       tpu_embedding_config_, "", "", "");
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;
  std::string config_string_;

  RecvTPUEmbeddingActivationsOp(const RecvTPUEmbeddingActivationsOp&) = delete;
  void operator=(const RecvTPUEmbeddingActivationsOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaRecvTPUEmbeddingActivations").AllowVariantTypes(),
                RecvTPUEmbeddingActivationsOp);

// This TensorFlow op receives a batch of deduplication data from the
// TPUEmbeddingEngine. The output is a list of R1 tensors containing the weights
// and indices for gathering the embedding vectors.
class RecvTPUEmbeddingDeduplicationDataOp : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingDeduplicationDataOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  ~RecvTPUEmbeddingDeduplicationDataOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile RecvTPUEmbeddingDeduplicationDataOp";

    CompileRecvTPUEmbeddingDeduplicationData(ctx, config_string_, "", "", "");

    VLOG(1) << "Compile RecvTPUDeduplicationDataOp done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;

  RecvTPUEmbeddingDeduplicationDataOp(
      const RecvTPUEmbeddingDeduplicationDataOp&) = delete;
  void operator=(const RecvTPUEmbeddingDeduplicationDataOp&) = delete;
};

REGISTER_XLA_OP(
    Name("XlaRecvTPUEmbeddingDeduplicationData").AllowVariantTypes(),
    RecvTPUEmbeddingDeduplicationDataOp);

// This TensorFlow op sends a batch of gradient and learning rate updates to the
// TpuEmbeddingEngine.
class SendTPUEmbeddingGradientsOp : public XlaOpKernel {
 public:
  explicit SendTPUEmbeddingGradientsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  ~SendTPUEmbeddingGradientsOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile SendTPUEmbeddingGradientsOp";

    CompileSendTPUEmbeddingGradients(ctx, config_string_, "", "", "");

    VLOG(1) << "Compile SendTPUEmbeddingGradientsOp done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;

  SendTPUEmbeddingGradientsOp(const SendTPUEmbeddingGradientsOp&) = delete;
  void operator=(const SendTPUEmbeddingGradientsOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSendTPUEmbeddingGradients").AllowVariantTypes(),
                SendTPUEmbeddingGradientsOp);

// `XLARecvTPUEmbeddingDeduplicationDataOp` gives an XLA Tuple as results, which
// can not be returned as static shape results. `SplitDedupDataOp` is to split
// this XLA tuple into integer and float tensors to return.
class SplitDedupDataOp : public XlaOpKernel {
 public:
  explicit SplitDedupDataOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tuple_mask", &tuple_mask_string_));
    OP_REQUIRES(ctx, tuple_mask_tensor_.ParseFromString(tuple_mask_string_),
                errors::InvalidArgument(
                    "Malformed `tuple_mask` attr in SplitDedupData Op."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    if (!config_string_.empty()) {
      OP_REQUIRES(
          ctx, tpu_embedding_config_.ParseFromString(config_string_),
          errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                  "proto from config attr"));
      spmd_enabled_ = tpu_embedding_config_.spmd_sharding().enabled();
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 1,
        errors::InvalidArgument("SplitDedupDataOp must have 1 input but gets ",
                                ctx->num_inputs()));
    const xla::XlaOp& input_tuple = ctx->Input(0);
    xla::XlaBuilder* builder = ctx->builder();

    absl::StatusOr<xla::Shape> tuple_shape = builder->GetShape(input_tuple);
    OP_REQUIRES_OK(ctx, tuple_shape.status());

    const int num_tuple_elements = tuple_shape->tuple_shapes_size();
    OP_REQUIRES(
        ctx,
        tuple_mask_tensor_.tensor_shape().dim(0).size() == num_tuple_elements,
        errors::InvalidArgument(
            "Number of elements in `input` tuple does not match with "
            "`tuple_mask`."));

    if (num_tuple_elements == 0) {
      // Returns empty tensors when tuple is empty.
      ctx->SetOutput(
          0, xla::ConstantLiteral(
                 builder, LiteralUtil::CreateFromDimensions(xla::U32, {0})));
      ctx->SetOutput(
          1, xla::ConstantLiteral(
                 builder, LiteralUtil::CreateFromDimensions(xla::F32, {0})));
      return;
    }

    // Split tuple elements in `input_tuple` into two vectors: integers_vec and
    // floats_vec, corresponding to their mask.
    std::vector<xla::XlaOp> integers_vec, floats_vec;
    int integer_offset = 0;  // Offset of integer elements in tuple.
    int float_offset = 0;    // Offset of floating elements in tuple.
    for (int i = 0; i < num_tuple_elements; ++i) {
      const xla::XlaOp& element = xla::GetTupleElement(input_tuple, i);
      const int element_type = tuple_mask_tensor_.int_val(2 * i);
      const int span_size = tuple_mask_tensor_.int_val(2 * i + 1);
      OP_REQUIRES(
          ctx,
          element_type == DedupTupleElementType::kInteger ||
              element_type == DedupTupleElementType::kFloat,
          errors::InvalidArgument(
              "Elements in first column of tuple_mask_tensor are enums of ",
              "DedupTupleElementType, which can only be 0 or 1. Where 0 ",
              "represents integer and 1 represents float. But gets unexpected ",
              "enum = ", element_type));
      OP_REQUIRES_VALUE(auto element_shape, ctx, builder->GetShape(element));
      OP_REQUIRES(
          ctx, element_shape.dimensions().size() == 1,
          errors::InvalidArgument("Elements of input tuple should be 1-D."));

      if (element_type == DedupTupleElementType::kInteger) {
        integers_vec.push_back(element);
        integer_offset += span_size;
      } else {
        floats_vec.push_back(element);
        float_offset += span_size;
      }
    }
    // Concatenate elements of integer and floating as return tensors.
    xla::XlaOp integer_tensor = xla::ConcatInDim(builder,
                                                 /*operands=*/integers_vec,
                                                 /*dimension=*/0);
    xla::XlaOp float_tensor = xla::ConcatInDim(builder,
                                               /*operands=*/floats_vec,
                                               /*dimension=*/0);

    if (config_string_.empty() || !spmd_enabled_) {
      ctx->SetOutput(0, integer_tensor);
      ctx->SetOutput(1, float_tensor);
      return;
    }

    const int num_cores_per_replica =
        tpu_embedding_config_.spmd_sharding().num_cores_per_replica();
    // Creating full shape of integer tensor based on accumulated
    // `integer_offset` from original tuple mask proto. Similarly, make full
    // shape of float tensor in following.
    xla::PrimitiveType int_elements_type = xla::U32;
    if (!integers_vec.empty()) {
      int_elements_type = builder->GetShape(integers_vec[0])->element_type();
    }
    xla::Shape integer_tensor_full_shape =
        xla::ShapeUtil::MakeShape(int_elements_type, {integer_offset});

    // Compute SPMD sharding if TPUEmbeddingConfig SPMD is enabled.
    // When using TPUEmbedding SPMD, we need manually convert integer tensor
    // and floating tensor to full shape, and convert them to local shards
    // in `MergeDedupDataOp`.
    OP_REQUIRES_VALUE(
        const xla::OpSharding integer_tensor_spmd, ctx,
        tensorflow::tpu::SpmdShardingAnnotationOnFirstDim(
            integer_tensor_full_shape, num_cores_per_replica, builder));

    OP_REQUIRES_VALUE(xla::XlaOp full_shaped_integer_tensor, ctx,
                      xla::ConvertSpmdShardToFullShape(
                          builder,
                          /*input=*/integer_tensor,
                          /*output_shape=*/integer_tensor_full_shape,
                          /*single_dim=*/0,
                          /*manual_sharding=*/integer_tensor_spmd,
                          /*unspecified_dims=*/absl::Span<const int64_t>{}));

    xla::PrimitiveType float_elements_type = xla::F32;
    if (!floats_vec.empty()) {
      float_elements_type = builder->GetShape(floats_vec[0])->element_type();
    }
    xla::Shape float_tensor_full_shape =
        xla::ShapeUtil::MakeShape(float_elements_type, {float_offset});
    OP_REQUIRES_VALUE(
        const xla::OpSharding float_tensor_spmd, ctx,
        tensorflow::tpu::SpmdShardingAnnotationOnFirstDim(
            float_tensor_full_shape, num_cores_per_replica, builder));
    OP_REQUIRES_VALUE(xla::XlaOp full_shaped_float_tensor, ctx,
                      xla::ConvertSpmdShardToFullShape(
                          builder,
                          /*input=*/float_tensor,
                          /*output_shape=*/float_tensor_full_shape,
                          /*single_dim=*/0,
                          /*manual_sharding=*/float_tensor_spmd,
                          /*unspecified_dims=*/absl::Span<const int64_t>{}));

    ctx->SetOutput(0, full_shaped_integer_tensor);
    ctx->SetOutput(1, full_shaped_float_tensor);
    VLOG(1) << "Compile SplitDedupDataOp done";
  }

 private:
  // TPU Embedding config.
  std::string config_string_;
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;
  bool spmd_enabled_ = false;

  // Deduplication data tuple mask string.
  std::string tuple_mask_string_;
  tensorflow::TensorProto tuple_mask_tensor_;

  SplitDedupDataOp(const SplitDedupDataOp&) = delete;
  void operator=(const SplitDedupDataOp&) = delete;
};

REGISTER_XLA_OP(Name("SplitDedupData").AllowVariantTypes(), SplitDedupDataOp);

// MergeDedupDataOp merges integer and floating point tensors back to xla tuple.
class MergeDedupDataOp : public XlaOpKernel {
 public:
  explicit MergeDedupDataOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tuple_mask", &tuple_mask_string_));
    OP_REQUIRES(ctx, tuple_mask_tensor_.ParseFromString(tuple_mask_string_),
                errors::InvalidArgument(
                    "Malformed `tuple_mask` attr in MergeDedupData Op"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    if (!config_string_.empty()) {
      OP_REQUIRES(
          ctx, tpu_embedding_config_.ParseFromString(config_string_),
          errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                  "proto from config attr"));
      spmd_enabled_ = tpu_embedding_config_.spmd_sharding().enabled();
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        errors::InvalidArgument("MergeDedupDataOp expects 2 inputs, but ",
                                "gets ", ctx->num_inputs()));

    auto builder = ctx->builder();
    xla::XlaOp integer_input = ctx->Input(0);
    xla::XlaOp float_input = ctx->Input(1);

    xla::XlaOp integer_tensor, float_tensor;
    if (spmd_enabled_) {
      // Compute SPMD sharding of integer tensor and float tensor.
      const int num_cores_per_replica =
          tpu_embedding_config_.spmd_sharding().num_cores_per_replica();

      OP_REQUIRES_VALUE(const xla::Shape integer_input_shape, ctx,
                        builder->GetShape(integer_input));
      OP_REQUIRES_VALUE(
          const xla::OpSharding integer_tensor_spmd, ctx,
          tensorflow::tpu::SpmdShardingAnnotationOnFirstDim(
              integer_input_shape, num_cores_per_replica, builder));
      OP_REQUIRES_VALUE(integer_tensor, ctx,
                        xla::ConvertSpmdFullToShardShape(
                            builder,
                            /*input=*/integer_input,
                            /*single_dim=*/0,
                            /*manual_sharding=*/integer_tensor_spmd,
                            /*unspecified_dims=*/absl::Span<const int64_t>{}));

      OP_REQUIRES_VALUE(const xla::Shape float_input_shape, ctx,
                        builder->GetShape(float_input));
      OP_REQUIRES_VALUE(const xla::OpSharding float_tensor_spmd, ctx,
                        tensorflow::tpu::SpmdShardingAnnotationOnFirstDim(
                            float_input_shape, num_cores_per_replica, builder));
      OP_REQUIRES_VALUE(float_tensor, ctx,
                        xla::ConvertSpmdFullToShardShape(
                            builder,
                            /*input=*/float_input,
                            /*single_dim=*/0,
                            /*manual_sharding=*/float_tensor_spmd,
                            /*unspecified_dims=*/absl::Span<const int64_t>{}));
    } else {
      integer_tensor = integer_input;
      float_tensor = float_input;
    }

    // `integer_tensor` should be a 1-D tensor.
    absl::StatusOr<xla::Shape> integer_tensor_shape =
        ctx->builder()->GetShape(integer_tensor);
    OP_REQUIRES_OK(ctx, integer_tensor_shape.status());
    OP_REQUIRES(ctx, integer_tensor_shape->rank() == 1,
                errors::InvalidArgument(
                    "Expected rank of integer_vals is 1, but gets, ",
                    integer_tensor_shape->rank()));
    const int64_t num_integers = integer_tensor_shape->dimensions(0);

    // `float_tensor` should be a 1-D tensor.
    absl::StatusOr<xla::Shape> float_tensor_shape =
        ctx->builder()->GetShape(float_tensor);
    OP_REQUIRES_OK(ctx, float_tensor_shape.status());
    OP_REQUIRES(ctx, float_tensor_shape->rank() == 1,
                errors::InvalidArgument("Expects rank of value is 1, but gets ",
                                        float_tensor_shape->rank()));
    const int64_t num_floats = float_tensor_shape->dimensions(0);

    // Get total number of elements in deduplication data tuple.
    const tensorflow::TensorShapeProto& tuple_tensor_shape =
        tuple_mask_tensor_.tensor_shape();
    const int64_t num_tuple_elements = tuple_tensor_shape.dim(0).size();
    if (num_tuple_elements == 0) {
      OP_REQUIRES(
          ctx, num_integers == 0 && num_floats == 0,
          errors::InvalidArgument(
              "Tuple mask indicates empty tuple, but integer_tensor ",
              "shape is ", integer_tensor_shape->DebugString(),
              " float_tensor shape is ", float_tensor_shape->DebugString()));
      ctx->SetOutput(0, xla::Tuple(builder, {}));
      return;
    }
    OP_REQUIRES(
        ctx, tuple_tensor_shape.dim_size() == 2,
        errors::InvalidArgument("Expects rank of tuple mask is 1, but gets ",
                                tuple_tensor_shape.dim_size()));

    std::vector<xla::XlaOp> output_vec;
    output_vec.reserve(num_tuple_elements);

    const int num_cores_per_replica =
        tpu_embedding_config_.spmd_sharding().num_cores_per_replica();
    // Merge elements of integer and float tensor into a tuple.
    int integer_offset = 0;
    int float_offset = 0;
    for (int i = 0; i < num_tuple_elements; ++i) {
      const int element_type = tuple_mask_tensor_.int_val(2 * i);
      int span_size = tuple_mask_tensor_.int_val(2 * i + 1);
      if (spmd_enabled_) {
        // When TPUEmbedding SPMD is enabled, the `span_size` got from
        // `tuple_mask` is full size of this span, need to be divided by
        // `num_cores_per_replica` for local span size.
        OP_REQUIRES(
            ctx, span_size % num_cores_per_replica == 0,
            errors::InvalidArgument(
                "Expects all `span_size` in tuple mask are divisible by ",
                "`num_cores_per_replica`. But get span_size = ", span_size,
                "while num_cores_per_replica = ", num_cores_per_replica));
        span_size /= num_cores_per_replica;
      }
      OP_REQUIRES(
          ctx,
          element_type == DedupTupleElementType::kInteger ||
              element_type == DedupTupleElementType::kFloat,
          errors::InvalidArgument(
              "Elements in first column of tuple_mask_tensor are enums of ",
              "DedupTupleElementType, which can only be 0 or 1. Where 0 ",
              "represents integer and 1 represents float. But gets unexpected ",
              "enum = ", element_type));

      if (element_type == DedupTupleElementType::kInteger) {
        OP_REQUIRES(ctx, integer_offset < num_integers,
                    errors::InvalidArgument(
                        "Offset of integers = ", integer_offset,
                        " exceeds total number of integers = ", num_integers));
        xla::XlaOp integer_slice =
            xla::SliceInDim(integer_tensor,
                            /*start_index=*/integer_offset,
                            /*limit_index*/ integer_offset + span_size,
                            /*stride=*/1, /*dimno=*/0);
        output_vec.push_back(integer_slice);
        integer_offset += span_size;
      } else {
        OP_REQUIRES(ctx, float_offset < num_floats,
                    errors::InvalidArgument(
                        "Offset of integers = ", float_offset,
                        " exceeds total number of floats = ", num_floats));
        xla::XlaOp float_slice =
            xla::SliceInDim(float_tensor,
                            /*start_index=*/float_offset,
                            /*limit_index*/ float_offset + span_size,
                            /*stride=*/1, /*dimno=*/0);
        output_vec.push_back(float_slice);
        float_offset += span_size;
      }
    }
    OP_REQUIRES(ctx, integer_offset == num_integers,
                errors::InvalidArgument(
                    "Number of integers does not match, expect num_integers = ",
                    num_integers, " but actually get = ", integer_offset));
    OP_REQUIRES(ctx, float_offset == num_floats,
                errors::InvalidArgument(
                    "Number of floats does not match, expect num_floats = ",
                    num_floats, " but actually get = ", float_offset));

    xla::XlaOp output_tuple = xla::Tuple(builder, output_vec);
    ctx->SetOutput(0, output_tuple);
  }

 private:
  // TPU Embedding config.
  std::string config_string_;
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;
  bool spmd_enabled_ = false;

  // Deduplication data tuple mask string.
  std::string tuple_mask_string_;
  tensorflow::TensorProto tuple_mask_tensor_;

  MergeDedupDataOp(const MergeDedupDataOp&) = delete;
  void operator=(const MergeDedupDataOp&) = delete;
};

REGISTER_XLA_OP(Name("MergeDedupData").AllowVariantTypes(), MergeDedupDataOp);

// This op computes the size of the deduplication data from infeed.
class ComputeDedupDataSizeOp : public XlaOpKernel {
 public:
  explicit ComputeDedupDataSizeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        absl::InvalidArgumentError("Failed to parse TPUEmbeddingConfiguration "
                                   "proto from config attr."));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile ComputeDedupDataSizeOp";

    CompileComputeDedupDataSize(ctx, config_string_, "", "", "");

    VLOG(1) << "Compile ComputeDedupDataSizeOp done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;

  ComputeDedupDataSizeOp(const ComputeDedupDataSizeOp&) = delete;
  void operator=(const ComputeDedupDataSizeOp&) = delete;
};

REGISTER_XLA_OP(Name("ComputeDedupDataSize"), ComputeDedupDataSizeOp);

// This op computes deduplication data tuple mask.
class ComputeDedupDataTupleMaskOp : public XlaOpKernel {
 public:
  explicit ComputeDedupDataTupleMaskOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile ComputeDedupDataTupleMaskOp";

    CompileComputeDedupDataTupleMask(ctx, config_string_, "", "", "");

    VLOG(1) << "Compile ComputeDedupDataTupleMaskOp done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;

  ComputeDedupDataTupleMaskOp(const ComputeDedupDataTupleMaskOp&) = delete;
  void operator=(const ComputeDedupDataTupleMaskOp&) = delete;
};

REGISTER_XLA_OP(Name("ComputeDedupDataTupleMask").AllowVariantTypes(),
                ComputeDedupDataTupleMaskOp);

// This Op has the same functionality as `XlaRecvTPUEmbeddingActivations`, but
// it accepts `embedding_partitions` and `hbm_buffers_config` (which can be
// obtained from `FinalizeTPUEmbeddingV2`). This is meaningful for use cases
// where the kernel runs in a different address space from where
// `embedding_partitions` and `hbm_buffers_config` are stored.
// The same principle applies to all the other V2 Ops here.
class RecvTPUEmbeddingActivationsV2Op : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingActivationsV2Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_partitions",
                                     &embedding_partitions_string_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("hbm_buffers_config", &hbm_buffers_config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tpu_topology", &tpu_topology_string_));

    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  ~RecvTPUEmbeddingActivationsV2Op() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 1,
        errors::Internal("Kernel has ", ctx->num_inputs(),
                         " inputs but configuration expects one input"));

    CompileRecvTPUEmbeddingActivations(
        ctx, config_string_, tpu_embedding_config_,
        embedding_partitions_string_, hbm_buffers_config_string_,
        tpu_topology_string_);
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;
  std::string config_string_;
  std::string embedding_partitions_string_;
  std::string hbm_buffers_config_string_;
  std::string tpu_topology_string_;

  RecvTPUEmbeddingActivationsV2Op(const RecvTPUEmbeddingActivationsV2Op&) =
      delete;
  void operator=(const RecvTPUEmbeddingActivationsV2Op&) = delete;
};

REGISTER_XLA_OP(Name("XlaRecvTPUEmbeddingActivationsV2").AllowVariantTypes(),
                RecvTPUEmbeddingActivationsV2Op);

class RecvTPUEmbeddingDeduplicationDataV2Op : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingDeduplicationDataV2Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_partitions",
                                     &embedding_partitions_string_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("hbm_buffers_config", &hbm_buffers_config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tpu_topology", &tpu_topology_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  ~RecvTPUEmbeddingDeduplicationDataV2Op() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile RecvTPUEmbeddingDeduplicationDataV2Op";

    CompileRecvTPUEmbeddingDeduplicationData(
        ctx, config_string_, embedding_partitions_string_,
        hbm_buffers_config_string_, tpu_topology_string_);

    VLOG(1) << "Compile RecvTPUDeduplicationDataV2Op done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;
  std::string embedding_partitions_string_;
  std::string hbm_buffers_config_string_;
  std::string tpu_topology_string_;

  RecvTPUEmbeddingDeduplicationDataV2Op(
      const RecvTPUEmbeddingDeduplicationDataV2Op&) = delete;
  void operator=(const RecvTPUEmbeddingDeduplicationDataV2Op&) = delete;
};

REGISTER_XLA_OP(
    Name("XlaRecvTPUEmbeddingDeduplicationDataV2").AllowVariantTypes(),
    RecvTPUEmbeddingDeduplicationDataV2Op);

class SendTPUEmbeddingGradientsV2Op : public XlaOpKernel {
 public:
  explicit SendTPUEmbeddingGradientsV2Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_partitions",
                                     &embedding_partitions_string_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("hbm_buffers_config", &hbm_buffers_config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tpu_topology", &tpu_topology_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  ~SendTPUEmbeddingGradientsV2Op() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile SendTPUEmbeddingGradientsV2Op";

    CompileSendTPUEmbeddingGradients(
        ctx, config_string_, embedding_partitions_string_,
        hbm_buffers_config_string_, tpu_topology_string_);

    VLOG(1) << "Compile SendTPUEmbeddingGradientsV2Op done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;
  std::string embedding_partitions_string_;
  std::string hbm_buffers_config_string_;
  std::string tpu_topology_string_;

  SendTPUEmbeddingGradientsV2Op(const SendTPUEmbeddingGradientsV2Op&) = delete;
  void operator=(const SendTPUEmbeddingGradientsV2Op&) = delete;
};

REGISTER_XLA_OP(Name("XlaSendTPUEmbeddingGradientsV2").AllowVariantTypes(),
                SendTPUEmbeddingGradientsV2Op);

class ComputeDedupDataSizeV2Op : public XlaOpKernel {
 public:
  explicit ComputeDedupDataSizeV2Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_partitions",
                                     &embedding_partitions_string_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("hbm_buffers_config", &hbm_buffers_config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tpu_topology", &tpu_topology_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        absl::InvalidArgumentError("Failed to parse TPUEmbeddingConfiguration "
                                   "proto from config attr."));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile ComputeDedupDataSizeV2Op";

    CompileComputeDedupDataSize(
        ctx, config_string_, embedding_partitions_string_,
        hbm_buffers_config_string_, tpu_topology_string_);

    VLOG(1) << "Compile ComputeDedupDataSizeV2Op done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;
  std::string embedding_partitions_string_;
  std::string hbm_buffers_config_string_;
  std::string tpu_topology_string_;

  ComputeDedupDataSizeV2Op(const ComputeDedupDataSizeV2Op&) = delete;
  void operator=(const ComputeDedupDataSizeV2Op&) = delete;
};

REGISTER_XLA_OP(Name("ComputeDedupDataSizeV2"), ComputeDedupDataSizeV2Op);

class ComputeDedupDataTupleMaskV2Op : public XlaOpKernel {
 public:
  explicit ComputeDedupDataTupleMaskV2Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("embedding_partitions",
                                     &embedding_partitions_string_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("hbm_buffers_config", &hbm_buffers_config_string_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tpu_topology", &tpu_topology_string_));
    OP_REQUIRES(
        ctx,
        tensorflow::tpu::TPUEmbeddingConfiguration().ParseFromString(
            config_string_),
        errors::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                                "proto from config attr"));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile ComputeDedupDataTupleMaskV2Op";

    CompileComputeDedupDataTupleMask(
        ctx, config_string_, embedding_partitions_string_,
        hbm_buffers_config_string_, tpu_topology_string_);

    VLOG(1) << "Compile ComputeDedupDataTupleMaskV2Op done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;
  std::string embedding_partitions_string_;
  std::string hbm_buffers_config_string_;
  std::string tpu_topology_string_;

  ComputeDedupDataTupleMaskV2Op(const ComputeDedupDataTupleMaskV2Op&) = delete;
  void operator=(const ComputeDedupDataTupleMaskV2Op&) = delete;
};

REGISTER_XLA_OP(Name("ComputeDedupDataTupleMaskV2").AllowVariantTypes(),
                ComputeDedupDataTupleMaskV2Op);

}  // anonymous namespace
}  // namespace tensorflow
