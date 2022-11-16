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

#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/proto_helper.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/status_helper.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_api.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_configuration.h"

namespace tensorflow {

namespace {

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
    ResourceMgr* rm = GetTPUConfigResourceMgr();
    OP_REQUIRES(ctx, rm, errors::Internal("No TPUConfigResourceMgr."));

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 1,
        errors::Internal("Kernel has ", ctx->num_inputs(),
                         " inputs but configuration expects one input"));

    xla::XlaOp deduplication_data = ctx->Input("deduplication_data");

    TpuEmbeddingEngine_RecvActivationsComputation_Params params;
    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();
    StatusHelper status;
    params.status = status.c_status;
    params.tpu_mesh_state = mesh_state->data();
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
    const int32 output_count =
        (tpu_embedding_config_.feature_descriptor_size() == 0)
            ? tpu_embedding_config_.table_descriptor_size()
            : tpu_embedding_config_.feature_descriptor_size();
    OP_REQUIRES(
        ctx, ctx->num_outputs() == output_count,
        errors::InvalidArgument(
            "Kernel has %d outputs but configuration expects %d outputs.",
            ctx->num_outputs(), output_count));

    for (int32 output_id = 0; output_id < output_count; ++output_id) {
      ctx->SetOutput(output_id,
                     xla::GetTupleElement(final_activations, output_id));
    }
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvTPUEmbeddingActivationsOp);
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

    ResourceMgr* rm = GetTPUConfigResourceMgr();
    OP_REQUIRES(ctx, rm, errors::Internal("No TPUConfigResourceMgr."));

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);

    TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params
        params;

    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    params.status = status.c_status;
    params.tpu_mesh_state = mesh_state->data();

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
    VLOG(1) << "Compile RecvTPUDeduplicationDataOp done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvTPUEmbeddingDeduplicationDataOp);
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

    ResourceMgr* rm = GetTPUConfigResourceMgr();
    OP_REQUIRES(ctx, rm, errors::Internal("No TPUConfigResourceMgr."));

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);

    std::vector<xla::XlaOp> gradients;
    std::vector<TensorShape> tf_gradient_shapes;
    OP_REQUIRES_OK(
        ctx, ctx->InputList("gradients", &gradients, &tf_gradient_shapes));
    std::vector<xla::Shape> gradient_shapes;
    auto builder = ctx->builder();
    gradient_shapes.reserve(gradients.size());
    for (xla::XlaOp op : gradients) {
      gradient_shapes.push_back(builder->GetShape(op).value());
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
    params.tpu_embedding_config.bytes = config_string_.c_str();
    params.tpu_embedding_config.size = config_string_.size();
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    params.status = status.c_status;
    params.tpu_mesh_state = mesh_state->data();
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

    auto c_shape_cleanup = absl::MakeCleanup([&gradient_tuple_c_shape,
                                              &learning_rate_tuple_c_shape,
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

    VLOG(1) << "Compile SendTPUEmbeddingGradientsOp done";
  }

 private:
  // TPU Embedding config string.
  std::string config_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendTPUEmbeddingGradientsOp);
};

REGISTER_XLA_OP(Name("XlaSendTPUEmbeddingGradients").AllowVariantTypes(),
                SendTPUEmbeddingGradientsOp);

// This TensorFlow op splits xla tuple to index and value tensors
class SplitXLATupleToTensorsOp : public XlaOpKernel {
 public:
  explicit SplitXLATupleToTensorsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() == 1,
                errors::InvalidArgument(
                    "SplitXLATupleToTensorsOp must have 1 input but gets ",
                    ctx->num_inputs()));
    const xla::XlaOp& input_tuple = ctx->Input(0);
    xla::XlaBuilder* builder = ctx->builder();

    StatusOr<xla::Shape> tuple_shape = builder->GetShape(input_tuple);
    OP_REQUIRES_OK(ctx, tuple_shape.status());

    int tuple_shapes_size = tuple_shape->tuple_shapes_size();
    OP_REQUIRES(ctx, tuple_shapes_size % 2 == 0,
                errors::InvalidArgument("Input tuple size must be even."));
    int output_size = tuple_shapes_size / 2;

    std::vector<xla::XlaOp> indices_vec, values_vec;
    indices_vec.reserve(output_size);
    values_vec.reserve(output_size);

    for (int i = 0; i < tuple_shapes_size - 1; i = i + 2) {
      indices_vec.push_back(xla::GetTupleElement(input_tuple, i));
      values_vec.push_back(xla::GetTupleElement(input_tuple, i + 1));
    }
    // Convert std::vector to XlaOp
    xla::XlaOp indices = xla::ConcatInDim(builder, indices_vec, 0);
    xla::XlaOp values = xla::ConcatInDim(builder, values_vec, 0);
    // output the indices array and value array.
    ctx->SetOutput(0, indices);
    ctx->SetOutput(1, values);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SplitXLATupleToTensorsOp);
};

REGISTER_XLA_OP(Name("SplitXLATupleToTensors").AllowVariantTypes(),
                SplitXLATupleToTensorsOp);

// This TensorFlow op merges index and value tensors into xla tuple
class InterleaveTensorsToXLATupleOp : public XlaOpKernel {
 public:
  explicit InterleaveTensorsToXLATupleOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() == 2,
                errors::InvalidArgument(
                    "InterleaveTensorsToXLATupleOp expects 2 inputs, but gets ",
                    ctx->num_inputs()));

    xla::XlaOp indices = ctx->Input(0);
    StatusOr<xla::Shape> indices_shape = ctx->builder()->GetShape(indices);
    OP_REQUIRES_OK(ctx, indices_shape.status());
    OP_REQUIRES(ctx, indices_shape->rank() == 1,
                errors::InvalidArgument("Rank of indices must be 1, but gets",
                                        indices_shape->rank()));

    xla::XlaOp values = ctx->Input(1);
    StatusOr<xla::Shape> values_shape = ctx->builder()->GetShape(values);
    OP_REQUIRES_OK(ctx, values_shape.status());
    OP_REQUIRES(ctx, indices_shape->rank() == 1,
                errors::InvalidArgument("Expects rank of value is 1, but gets ",
                                        indices_shape->rank()));
    OP_REQUIRES(ctx,
                indices_shape->dimensions(0) == values_shape->dimensions(0),
                errors::InvalidArgument(
                    "Lengths of indices and values must be same, but length of "
                    "indices  = ",
                    indices_shape->dimensions(0),
                    "length of values = ", values_shape->dimensions(0)));

    std::vector<xla::XlaOp> output_vec;
    output_vec.reserve(2 * indices_shape->dimensions(0));
    // Interleave elements of indices and values as output tuple.
    for (int i = 0; i < indices_shape->dimensions(0); ++i) {
      output_vec.push_back(xla::Slice(indices, {i}, {i + 1}, {1}));
      output_vec.push_back(xla::Slice(values, {i}, {i + 1}, {1}));
    }

    xla::XlaOp output_tuple = xla::Tuple(ctx->builder(), output_vec);
    ctx->SetOutput(0, output_tuple);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(InterleaveTensorsToXLATupleOp);
};

REGISTER_XLA_OP(Name("InterleaveTensorsToXLATuple").AllowVariantTypes(),
                InterleaveTensorsToXLATupleOp);

}  // anonymous namespace
}  // namespace tensorflow
