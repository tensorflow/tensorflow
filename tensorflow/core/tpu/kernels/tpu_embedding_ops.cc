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

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/status_helper.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"

namespace tensorflow {

namespace {

// This TensorFlow op receives a batch of activations from the
// TpuEmbeddingEngine.
class RecvTPUEmbeddingActivationsOp : public XlaOpKernel {
 public:
  explicit RecvTPUEmbeddingActivationsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    string config_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string),
        xla::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                             "proto from config attr"));
  }

  ~RecvTPUEmbeddingActivationsOp() override {}

  void Compile(XlaOpKernelContext* ctx) override {
    ResourceMgr* rm = GetTPUConfigResourceMgr();

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

    TpuEmbeddingEngine_RecvActivationsComputation_Params recv_activation_params;
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    recv_activation_params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    recv_activation_params.status = status.c_status;
    recv_activation_params.tpu_mesh_state = mesh_state->data();
    auto builder = ctx->builder();
    OP_REQUIRES_VALUE(auto shape, ctx, builder->GetShape(deduplication_data));
    XLA_Shape c_shape;
    ApiConverter::ToC(shape, &c_shape);
    auto c_shape_cleanup =
        absl::MakeCleanup([&c_shape] { ApiConverter::Destroy(&c_shape); });
    recv_activation_params.deduplication_data_shape = &c_shape;
    tpu::OpsApiFn()->TpuEmbeddingEngine_RecvActivationsComputationFn(
        &recv_activation_params);
    OP_REQUIRES_OK(ctx, status.status());
    auto xla_computation =
        stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
            xla_computation_serialized);
    auto final_activations =
        xla::Call(builder, xla_computation, {deduplication_data});

    int32 output_count = tpu_embedding_config_.feature_descriptor_size();
    OP_REQUIRES(
        ctx, ctx->num_outputs() == output_count,
        xla::InvalidArgument(
            "Kernel has %d outputs but configuration expects %d outputs.",
            ctx->num_outputs(), output_count));

    for (int32 output_id = 0; output_id < output_count; ++output_id) {
      ctx->SetOutput(output_id,
                     xla::GetTupleElement(final_activations, output_id));
    }
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;

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
    std::string config_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));
    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string),
        xla::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                             "proto from config attr"));
  }

  ~RecvTPUEmbeddingDeduplicationDataOp() override {}

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile RecvTPUDeduplicationDataOp";

    ResourceMgr* rm = GetTPUConfigResourceMgr();

    tensorflow::tpu::TpuMeshStateInterface* mesh_state;
    OP_REQUIRES_OK(
        ctx, rm->Lookup(rm->default_container(),
                        tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                        &mesh_state));
    core::ScopedUnref mesh_state_unref(mesh_state);

    TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputation_Params
        recv_deduplication_params;
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    recv_deduplication_params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    recv_deduplication_params.status = status.c_status;
    recv_deduplication_params.tpu_mesh_state = mesh_state->data();

    tpu::OpsApiFn()
        ->TpuEmbeddingEngine_RecvTPUEmbeddingDeduplicationDataComputationFn(
            &recv_deduplication_params);
    OP_REQUIRES_OK(ctx, status.status());

    auto xla_computation =
        stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
            xla_computation_serialized);

    const xla::XlaOp deduplication_data =
        xla::Call(ctx->builder(), xla_computation, {});

    // Ensure that the number of outputs is equal to 1 (for deduplication data).
    OP_REQUIRES(ctx, ctx->num_outputs() == 1,
                xla::InvalidArgument(
                    "Kernel has %d outputs but configuration expects 1 output.",
                    ctx->num_outputs()));

    ctx->SetOutput(0, deduplication_data);
    VLOG(1) << "Compile RecvTPUDeduplicationDataOp done";
  }

 private:
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;

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
    string config_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &config_string));

    OP_REQUIRES(
        ctx, tpu_embedding_config_.ParseFromString(config_string),
        xla::InvalidArgument("Failed to parse TPUEmbeddingConfiguration "
                             "proto from config attr"));
  }

  ~SendTPUEmbeddingGradientsOp() override {}

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Compile SendTPUEmbeddingGradientsOp";

    ResourceMgr* rm = GetTPUConfigResourceMgr();

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
      gradient_shapes.push_back(builder->GetShape(op).ValueOrDie());
    }

    std::vector<xla::XlaOp> learning_rates;
    std::vector<TensorShape> tf_learning_rate_shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("learning_rates", &learning_rates,
                                       &tf_learning_rate_shapes));
    std::vector<xla::Shape> learning_rate_shapes;
    learning_rate_shapes.reserve(learning_rates.size());
    for (xla::XlaOp op : learning_rates) {
      learning_rate_shapes.push_back(builder->GetShape(op).ValueOrDie());
    }

    xla::XlaOp deduplication_data = ctx->Input("deduplication_data");

    TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputation_Params
        send_gradients_params;
    TpuSerializedProto xla_computation_serialized;
    auto proto_cleanup = absl::MakeCleanup([&xla_computation_serialized] {
      StreamExecutor_Tpu_FreeSerializedProto(&xla_computation_serialized);
    });
    send_gradients_params.xla_computation = &xla_computation_serialized;
    StatusHelper status;
    send_gradients_params.status = status.c_status;
    send_gradients_params.tpu_mesh_state = mesh_state->data();
    OP_REQUIRES_VALUE(auto deduplication_shape, ctx,
                      builder->GetShape(deduplication_data));
    XLA_Shape gradient_tuple_c_shape;
    ApiConverter::ToC(xla::ShapeUtil::MakeTupleShape(gradient_shapes),
                      &gradient_tuple_c_shape);
    XLA_Shape learning_rate_tuple_c_shape;
    ApiConverter::ToC(xla::ShapeUtil::MakeTupleShape(learning_rate_shapes),
                      &learning_rate_tuple_c_shape);
    XLA_Shape deduplication_c_shape;
    ApiConverter::ToC(deduplication_shape, &deduplication_c_shape);

    auto c_shape_cleanup = absl::MakeCleanup([&gradient_tuple_c_shape,
                                              &learning_rate_tuple_c_shape,
                                              &deduplication_c_shape] {
      ApiConverter::Destroy(&gradient_tuple_c_shape);
      ApiConverter::Destroy(&learning_rate_tuple_c_shape);
      ApiConverter::Destroy(&deduplication_c_shape);
    });
    send_gradients_params.num_inputs = ctx->num_inputs();

    tpu::OpsApiFn()->TpuEmbeddingEngine_SendTPUEmbeddingGradientsComputationFn(
        &send_gradients_params);
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
  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendTPUEmbeddingGradientsOp);
};

REGISTER_XLA_OP(Name("XlaSendTPUEmbeddingGradients").AllowVariantTypes(),
                SendTPUEmbeddingGradientsOp);

}  // anonymous namespace
}  // namespace tensorflow
