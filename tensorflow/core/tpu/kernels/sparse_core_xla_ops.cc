/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/sparse_core_xla_ops.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/macros.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

typedef tensorflow::monitoring::Gauge<int64_t, 2> TFGaugeMetric;
static TFGaugeMetric* max_ids_per_partition_gauge_ = TFGaugeMetric::New(
    "/tensorflow/tpu/embedding/maximum_ids_per_partition",
    "Max ids_per_partition limit for each table", "device", "table");
static TFGaugeMetric* max_unique_ids_per_partition_gauge_ = TFGaugeMetric::New(
    "/tensorflow/tpu/embedding/maximum_unique_ids_per_partition",
    "Max unique_ids_per_partition limit for each table", "device", "table");

constexpr char kUnknownProgramKey[] = "";

namespace tensorflow {
namespace {

// Get the SparseCore logical replica count.
absl::StatusOr<int64_t> GetSparseCoresPerChip() {
  return stream_executor::tpu::OpsApiFn()->TpuTopology_AvailableCoresPerChipFn(
      /*tpu_core_type=*/TpuCoreTypeEnum::kEmbeddingV2);
}

// This TensorFlow op performs the embedding lookup on SparseCore. It takes the
// embedding table and input sparse tensor represented by the `row_ids`,
// `col_ids` and `values`(gains). It produces the embedding look up result and
// the preserved result which will be used in the gradient calculation op.
class XlaSparseDenseMatmulOp : public XlaOpKernel {
 public:
  explicit XlaSparseDenseMatmulOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_size", &input_size_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("max_ids_per_partition", &max_ids_per_partition_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_id_per_partition",
                                     &max_unique_ids_per_partition_));
  }

  ~XlaSparseDenseMatmulOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    const int32 num_physical_replica =
        stream_executor::tpu::OpsApiFn()->TpuTopology_AvailableCoreCountFn(
            /*mesh_state=*/nullptr,
            /*tpu_core_type=*/TpuCoreTypeEnum::kEmbeddingV2);

    OP_REQUIRES(ctx, num_physical_replica > 0,
                errors::InvalidArgument(
                    "No SparseCore is available in the tpu system."));

    // TODO(pineapplejuice233): Add error checking logic.
    xla::XlaOp row_ids = ctx->Input("row_ids");
    xla::XlaOp col_ids = ctx->Input("col_ids");
    xla::XlaOp values = ctx->Input("values");
    // TODO(pineapplejuice233): Right now we are passing this argument as
    // non_zero_element_num, switch to actual 'offsets' once the decomposer
    // supports it.
    xla::XlaOp offsets = ctx->Input("offsets");
    xla::XlaOp embedding_table = ctx->Input("embedding_table");

    // Construct the shape and a const 0 input for the activations
    xla::XlaOp zero = xla::ConstantLiteral(
        builder, xla::LiteralUtil::Zero(ctx->InputXlaType("embedding_table")));
    OP_REQUIRES_VALUE(xla::Shape activation_shape, ctx,
                      ctx->InputXlaShape("embedding_table"));
    activation_shape.set_dimensions(0, input_size_);
    xla::Shape row_pointers_shape =
        xla::ShapeUtil::MakeShapeWithType<int32_t>({num_physical_replica});

    // Get the number of ids from row_ids.
    OP_REQUIRES_VALUE(xla::Shape row_ids_shape, ctx,
                      ctx->InputXlaShape("row_ids"));
    int64_t token_count = row_ids_shape.dimensions(0);

    // TODO(pineapplejuice233): Change this to include padding once minibatching is done.
    xla::Shape sorted_ids_shape =
        xla::ShapeUtil::MakeShapeWithType<int32_t>({token_count});
    xla::Shape sorted_gains_shape =
        xla::ShapeUtil::MakeShapeWithType<float>({token_count});

    xla::XlaOp activation_init =
        xla::Broadcast(zero, activation_shape.dimensions());

    xla::FrontendAttributes original_frontend_attributes =
        builder->frontend_attributes();

    xla::FrontendAttributes new_frontend_attributes;

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    builder->SetFrontendAttributes(new_frontend_attributes);

    // Pack the input tensors as a tuple. This is a intermediate stage before
    // switching to SparseTensor type.
    xla::XlaOp coo_tensor_input =
        xla::Tuple(builder, {row_ids, col_ids, values});

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_sharding_strategy", "mod"});

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_max_ids_per_partition", absl::StrCat(max_ids_per_partition_)});

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_max_unique_ids_per_partition",
         absl::StrCat(max_unique_ids_per_partition_)});

    builder->SetFrontendAttributes(new_frontend_attributes);

    xla::XlaOp result = xla::CustomCall(
        builder, "SparseDenseMatmulOp",
        {coo_tensor_input, embedding_table, offsets, activation_init},
        xla::ShapeUtil::MakeTupleShape({activation_shape, row_pointers_shape,
                                        sorted_ids_shape, sorted_ids_shape,
                                        sorted_gains_shape}));

    builder->SetFrontendAttributes(original_frontend_attributes);

    // Embedding activation.
    ctx->SetOutput(0, xla::GetTupleElement(result, 0));
    // CSR pointer corresponding to logical sharding replicas.
    ctx->SetOutput(1, xla::GetTupleElement(result, 1));
    // CSR values of embedding ids.
    ctx->SetOutput(2, xla::GetTupleElement(result, 2));
    // CSR values of sample ids.
    ctx->SetOutput(3, xla::GetTupleElement(result, 3));
    // CSR values of gains.
    ctx->SetOutput(4, xla::GetTupleElement(result, 4));
  }

 private:
  int input_size_;
  int max_ids_per_partition_;
  int max_unique_ids_per_partition_;

  XlaSparseDenseMatmulOp(const XlaSparseDenseMatmulOp&) = delete;
  void operator=(const XlaSparseDenseMatmulOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmul"), XlaSparseDenseMatmulOp);

// This TensorFlow op performs the embedding lookup on SparseCore. It has
// different input and output format comparing to the XlaSparseDenseMatmulOp.
// It takes the embedding table and input csr tensor represented by the
// `row_pointers`, `sorted_sample_ids`, `sorted_token_ids` and `sorted_gains`.
// It only produces the embedding look up result.
class XlaSparseDenseMatmulWithCsrInputOp : public XlaOpKernel {
 public:
  explicit XlaSparseDenseMatmulWithCsrInputOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_size", &input_size_));
    OP_REQUIRES_VALUE(num_sparsecores_per_chip_, ctx, GetSparseCoresPerChip());
    OP_REQUIRES(ctx, input_size_ % num_sparsecores_per_chip_ == 0,
                errors::InvalidArgument("input_size_ ", input_size_,
                                        " not divisible by the number "
                                        "of sparsecores per chip ",
                                        num_sparsecores_per_chip_));

    // Get and save quantization config params, if they were configured.
    // num_buckets == 0 indicate no quantization configs were provided.
    int check_num_buckets;
    absl::Status status =
        ctx->GetAttr("quantization_config_num_buckets", &check_num_buckets);
    if (status.ok() && check_num_buckets > 0) {
      quantization_config_num_buckets_ = check_num_buckets;
      float quant_clipping_float;
      status = ctx->GetAttr("quantization_config_low", &quant_clipping_float);
      if (status.ok()) {
        quantization_config_low_ = quant_clipping_float;
      }
      status = ctx->GetAttr("quantization_config_high", &quant_clipping_float);
      if (status.ok()) {
        quantization_config_high_ = quant_clipping_float;
      }
    }
    device_name_ = ctx->device()->name();
    // Check for incomplete quantization config.
    OP_REQUIRES(ctx,
                quantization_config_low_.has_value() ==
                        quantization_config_high_.has_value() &&
                    quantization_config_low_.has_value() ==
                        quantization_config_num_buckets_.has_value(),
                errors::InvalidArgument("Quantization config is incomplete."));
  }

  ~XlaSparseDenseMatmulWithCsrInputOp() override = default;

  virtual absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition, int64_t* max_unique_ids_per_partition) {
    return GetMaxIdsAndUniquesExternal(
        kUnknownProgramKey, table_name_, num_samples_per_sparse_core,
        feature_width, max_ids_per_partition, max_unique_ids_per_partition);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64_t per_sparse_core_batch_size =
        input_size_ / num_sparsecores_per_chip_;
    int64_t max_ids_per_partition = 0;
    int64_t max_unique_ids_per_partition = 0;

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp row_pointers = ctx->Input("row_pointers");
    xla::XlaOp sorted_sample_ids = ctx->Input("sorted_sample_ids");
    xla::XlaOp sorted_token_ids = ctx->Input("sorted_token_ids");
    xla::XlaOp sorted_gains = ctx->Input("sorted_gains");
    xla::XlaOp embedding_table = ctx->Input("embedding_table");

    OP_REQUIRES_VALUE(xla::Shape embedding_table_shape, ctx,
                      ctx->InputXlaShape("embedding_table"));
    const int32_t feature_width = embedding_table_shape.dimensions(1);

    OP_REQUIRES_OK(
        ctx, GetMaxIdsAndUniques(per_sparse_core_batch_size, feature_width,
                                 &max_ids_per_partition,
                                 &max_unique_ids_per_partition));
    // Log max_ids and max_uniques for offline analysis. We do this here since
    // these values are fixed at TPU compile time and remain fixed during
    // training.
    max_ids_per_partition_gauge_->GetCell(device_name_, table_name_)
        ->Set(max_ids_per_partition);
    max_unique_ids_per_partition_gauge_->GetCell(device_name_, table_name_)
        ->Set(max_unique_ids_per_partition);
    LOG(INFO) << "Lowering XlaSparseDenseMatmulWithCsrInputOp to HLO: "
              << "table_name = '" << table_name_
              << "', max_ids = " << max_ids_per_partition
              << ", max_uniques = " << max_unique_ids_per_partition;
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(ctx->InputShape(
                    "num_minibatches_per_physical_sparse_core")),
                errors::InvalidArgument(
                    "num_minibatches_per_physical_sparse_core must be scalar"));

    xla::XlaOp num_minibatches_per_physical_sparse_core =
        ctx->Input("num_minibatches_per_physical_sparse_core");

    // Construct the shape and a const 0 input for the activations
    xla::XlaOp zero = xla::ConstantLiteral(
        builder, xla::LiteralUtil::Zero(ctx->InputXlaType("embedding_table")));
    OP_REQUIRES_VALUE(xla::Shape activation_shape, ctx,
                      ctx->InputXlaShape("embedding_table"));
    activation_shape.set_dimensions(0, input_size_);

    xla::XlaOp activation_init =
        xla::Broadcast(zero, activation_shape.dimensions());

    xla::FrontendAttributes new_frontend_attributes;

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    builder->SetFrontendAttributes(new_frontend_attributes);

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_sharding_strategy", "mod"});

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_pad_value", absl::StrCat(kXlaPadValue)});

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_max_ids_per_partition", absl::StrCat(max_ids_per_partition)});

    new_frontend_attributes.mutable_map()->insert(
        {"_xla_max_unique_ids_per_partition",
         absl::StrCat(max_unique_ids_per_partition)});

    if (quantization_config_low_.has_value()) {
      new_frontend_attributes.mutable_map()->insert(
          {"_xla_quantization_high_value",
           absl::StrCat(quantization_config_high_.value())});
      new_frontend_attributes.mutable_map()->insert(
          {"_xla_quantization_low_value",
           absl::StrCat(quantization_config_low_.value())});
      new_frontend_attributes.mutable_map()->insert(
          {"_xla_quantization_num_buckets_value",
           absl::StrCat(quantization_config_num_buckets_.value())});
    }
    builder->SetFrontendAttributes(new_frontend_attributes);

    xla::XlaOp result =
        xla::CustomCall(builder, "SparseDenseMatmulWithMinibatchingOp",
                        {row_pointers, sorted_token_ids, sorted_sample_ids,
                         sorted_gains, num_minibatches_per_physical_sparse_core,
                         embedding_table, activation_init},
                        activation_shape);

    // Embedding activation.
    ctx->SetOutput(0, result);
  }

 protected:
  int input_size_;
  int64_t num_sparsecores_per_chip_;
  std::optional<float> quantization_config_low_;
  std::optional<float> quantization_config_high_;
  std::optional<int> quantization_config_num_buckets_;
  std::string device_name_;
  std::string table_name_;

  XlaSparseDenseMatmulWithCsrInputOp(
      const XlaSparseDenseMatmulWithCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulWithCsrInputOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulWithCsrInput"),
                XlaSparseDenseMatmulWithCsrInputOp);

// Similar to XlaSparseDenseMatmulWithCsrInputOp, but with an additional field
// `sorted_pos_ids` in the input Csr, `weights` which is a tensor of shape
// [num_weights] to be used by the `combiner_computation`. It produces the same
// embedding look up result as `XlaSparseDenseMatmulWithCsrInputOp`.
class XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp
    : public XlaSparseDenseMatmulWithCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulWithCsrInputOp(ctx) {
    const NameAttrList* name_attr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_valency", &max_valency_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_weights", &num_weights_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner_computation", &name_attr));
    combiner_computation_ = *name_attr;
  }

  ~XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp() override = default;

  absl::StatusOr<xla::XlaComputation> BuildTcCustomCombinerComputation(
      XlaOpKernelContext* ctx, const int32_t feature_width) {
    XlaCompiler::CompileOptions options;
    options.use_tuple_arg = false;
    options.always_return_tuple = false;
    options.is_entry_computation = false;

    XlaCompiler* compiler = ctx->compiler();
    XlaCompiler::CompilationResult custom_combiner_computation_result;

    XlaCompiler::Argument valencies_arg;
    XlaCompiler::Argument vectors_arg;

    valencies_arg.kind = XlaCompiler::Argument::kParameter;
    valencies_arg.type = DT_INT32;
    valencies_arg.shape = xla::ShapeUtil::MakeShape(xla::S32, {input_size_});
    valencies_arg.name = "valencies";
    vectors_arg.kind = XlaCompiler::Argument::kParameter;
    vectors_arg.type = DT_FLOAT;
    vectors_arg.shape = xla::ShapeUtil::MakeShape(
        xla::F32, {input_size_, max_valency_, feature_width});
    vectors_arg.name = "vectors";

    std::vector<XlaCompiler::Argument> arguments = {valencies_arg, vectors_arg};

    // Don't add the weights argument if it's not needed. This helps avoid
    // issues of passing around zero-sized tensors and Xla values.
    if (num_weights_ > 0) {
      XlaCompiler::Argument weights_arg;
      weights_arg.kind = XlaCompiler::Argument::kParameter;
      weights_arg.type = DT_FLOAT;
      weights_arg.shape =
          xla::ShapeUtil::MakeShape(xla::F32, {input_size_, num_weights_});
      weights_arg.name = "weights";
      arguments.push_back(weights_arg);
    }

    TF_RETURN_IF_ERROR(
        compiler->CompileFunction(options, combiner_computation_, arguments,
                                  &custom_combiner_computation_result));
    return std::move(*custom_combiner_computation_result.computation);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64_t per_sparse_core_batch_size =
        input_size_ / num_sparsecores_per_chip_;
    int64_t max_ids_per_partition = 0;
    int64_t max_unique_ids_per_partition = 0;

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp row_pointers = ctx->Input("row_pointers");
    xla::XlaOp sorted_sample_ids = ctx->Input("sorted_sample_ids");
    xla::XlaOp sorted_token_ids = ctx->Input("sorted_token_ids");
    xla::XlaOp sorted_pos_ids = ctx->Input("sorted_pos_ids");
    xla::XlaOp sorted_gains = ctx->Input("sorted_gains");
    xla::XlaOp embedding_table = ctx->Input("embedding_table");

    OP_REQUIRES_VALUE(xla::Shape embedding_table_shape, ctx,
                      ctx->InputXlaShape("embedding_table"));
    const int32_t feature_width = embedding_table_shape.dimensions(1);

    OP_REQUIRES_OK(
        ctx, GetMaxIdsAndUniques(per_sparse_core_batch_size, feature_width,
                                 &max_ids_per_partition,
                                 &max_unique_ids_per_partition));
    // Log max_ids and max_uniques for offline analysis. We do this here since
    // these values are fixed at TPU compile time and remain fixed during
    // training.
    max_ids_per_partition_gauge_->GetCell(device_name_, table_name_)
        ->Set(max_ids_per_partition);
    max_unique_ids_per_partition_gauge_->GetCell(device_name_, table_name_)
        ->Set(max_unique_ids_per_partition);
    LOG(INFO) << "Lowering "
                 "XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp to HLO: "
              << "table_name = '" << table_name_
              << "', max_ids = " << max_ids_per_partition
              << ", max_uniques = " << max_unique_ids_per_partition;

    xla::FrontendAttributes tc_frontend_attributes;
    xla::FrontendAttributes sc_frontend_attributes;

    sc_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    sc_frontend_attributes.mutable_map()->insert(
        {"_xla_sharding_strategy", "mod"});

    sc_frontend_attributes.mutable_map()->insert(
        {"_xla_pad_value", absl::StrCat(kXlaPadValue)});

    sc_frontend_attributes.mutable_map()->insert(
        {"_xla_max_ids_per_partition", absl::StrCat(max_ids_per_partition)});

    sc_frontend_attributes.mutable_map()->insert(
        {"_xla_max_unique_ids_per_partition",
         absl::StrCat(max_unique_ids_per_partition)});

    sc_frontend_attributes.mutable_map()->insert(
        {"_xla_max_valency", absl::StrCat(max_valency_)});

    if (quantization_config_low_.has_value()) {
      sc_frontend_attributes.mutable_map()->insert(
          {"_xla_quantization_high_value",
           absl::StrCat(quantization_config_high_.value())});
      sc_frontend_attributes.mutable_map()->insert(
          {"_xla_quantization_low_value",
           absl::StrCat(quantization_config_low_.value())});
      sc_frontend_attributes.mutable_map()->insert(
          {"_xla_quantization_num_buckets_value",
           absl::StrCat(quantization_config_num_buckets_.value())});
    }

    tc_frontend_attributes =
        builder->SwapFrontendAttributes(sc_frontend_attributes);

    // Emit the custom call that performs the SC embedding lookup.
    xla::Shape valencies_shape =
        xla::ShapeUtil::MakeShape(xla::S32, {input_size_});
    xla::Shape vectors_shape = xla::ShapeUtil::MakeShape(
        xla::F32, {input_size_, max_valency_, feature_width});
    xla::Shape gains_shape =
        xla::ShapeUtil::MakeShape(xla::F32, {input_size_, max_valency_});
    xla::XlaOp sc_lookup_result_tuple = xla::CustomCall(
        builder, "SparseDenseMatmulCustomCombinerTcCombinerMegachipOp",
        {row_pointers, sorted_token_ids, sorted_sample_ids, sorted_pos_ids,
         sorted_gains, embedding_table},
        xla::ShapeUtil::MakeTupleShape(
            {valencies_shape, vectors_shape, gains_shape}));

    // Emit the custom combiner computation into an HLO computation.
    OP_REQUIRES_VALUE(xla::XlaComputation custom_combiner_tc_computation, ctx,
                      BuildTcCustomCombinerComputation(ctx, feature_width));

    builder->SetFrontendAttributes(tc_frontend_attributes);

    xla::XlaOp valencies = xla::GetTupleElement(sc_lookup_result_tuple, 0);
    xla::XlaOp vectors = xla::GetTupleElement(sc_lookup_result_tuple, 1);

    std::vector<xla::XlaOp> tc_combiner_args = {valencies, vectors};
    if (num_weights_ > 0) {
      xla::XlaOp weights = ctx->Input("weights");
      tc_combiner_args.push_back(xla::Broadcast(weights, {input_size_}));
    }

    xla::XlaOp tc_activations =
        xla::Call(builder, custom_combiner_tc_computation, tc_combiner_args);

    ctx->SetOutput(0, tc_activations);
    ctx->SetOutput(1, valencies);
    ctx->SetOutput(2, vectors);
  }

 private:
  int max_valency_;
  int num_weights_;
  NameAttrList combiner_computation_;

  XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp(
      const XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp&) =
      delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInput"),
                XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInputOp);

// Base class for all the minibatch with CSR input optimizer kernel.
class XlaSparseDenseMatmulGradWithCsrInputBase : public XlaOpKernel {
 public:
  explicit XlaSparseDenseMatmulGradWithCsrInputBase(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("clip_weight_min", &clip_weight_min_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("clip_weight_max", &clip_weight_max_));

    OP_REQUIRES(ctx, clip_weight_min_ <= clip_weight_max_,
                absl::InvalidArgumentError(
                    absl::StrCat("clip_weight_min must be smaller or equal to "
                                 "clip_weight_max but got clip_weight_min as ",
                                 clip_weight_min_, " and clip_weight_max as ",
                                 clip_weight_max_, ".")));
  }

  ~XlaSparseDenseMatmulGradWithCsrInputBase() override = default;

  virtual xla::XlaComputation build_optimizer_computation(
      int32_t feature_width) = 0;

  virtual xla::XlaOp get_tables_input(XlaOpKernelContext* ctx) = 0;

  virtual xla::XlaOp get_hyperparameters_input(XlaOpKernelContext* ctx) = 0;

  virtual xla::Shape get_tables_shape(xla::Shape embedding_table_shape) = 0;

  xla::XlaOp apply_weight_clipping_to_table(xla::XlaBuilder* builder,
                                            xla::XlaOp table) {
    xla::XlaOp clip_weight_min = xla::ConstantR0(builder, clip_weight_min_);
    xla::XlaOp clip_weight_max = xla::ConstantR0(builder, clip_weight_max_);
    xla::XlaOp clipped_table =
        xla::Clamp(clip_weight_min, table, clip_weight_max);
    return clipped_table;
  }

  virtual absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition, int64_t* max_unique_ids_per_partition) {
    return GetMaxIdsAndUniquesExternal(
        kUnknownProgramKey, table_name_, num_samples_per_sparse_core,
        feature_width, max_ids_per_partition, max_unique_ids_per_partition);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    // TODO(pineapplejuice233): Add error checking logic.
    xla::XlaOp row_pointers = ctx->Input("row_pointers");
    xla::XlaOp sorted_sample_ids = ctx->Input("sorted_sample_ids");
    xla::XlaOp sorted_token_ids = ctx->Input("sorted_token_ids");
    xla::XlaOp sorted_gains = ctx->Input("sorted_gains");
    xla::XlaOp activation_gradients = ctx->Input("activation_gradients");
    xla::XlaOp num_minibatches_per_physical_sparse_core =
        ctx->Input("num_minibatches_per_physical_sparse_core");

    // Get the shape of the gradient.
    OP_REQUIRES_VALUE(xla::Shape activation_shape, ctx,
                      ctx->InputXlaShape("activation_gradients"));
    OP_REQUIRES(ctx,
                activation_shape.is_static() &&
                    activation_shape.dimensions().size() == 2,
                errors::InvalidArgument(
                    "activations input has non static or non-rank 2 shape: ",
                    activation_shape.ToString()));
    OP_REQUIRES_VALUE(int64_t num_sparsecores_per_chip, ctx,
                      GetSparseCoresPerChip());
    int64 num_samples_per_chip = activation_shape.dimensions(0);
    OP_REQUIRES(ctx, num_samples_per_chip % num_sparsecores_per_chip == 0,
                errors::InvalidArgument(
                    "num_samples_per_chip ", num_samples_per_chip,
                    " not divisible by the number of sparsecores per chip ",
                    num_sparsecores_per_chip));
    int64_t per_sparse_core_batch_size =
        num_samples_per_chip / num_sparsecores_per_chip;
    int64_t max_ids_per_partition = 0;
    int64_t max_unique_ids_per_partition = 0;
    OP_REQUIRES_VALUE(xla::Shape embedding_table_shape, ctx,
                      ctx->InputXlaShape("embedding_table"));

    const int32_t feature_width = embedding_table_shape.dimensions(1);
    OP_REQUIRES_OK(
        ctx, GetMaxIdsAndUniques(per_sparse_core_batch_size, feature_width,
                                 &max_ids_per_partition,
                                 &max_unique_ids_per_partition));
    LOG(INFO) << "Lowering XlaSparseDenseMatmulGradWithCsrInputOp to HLO: "
              << "table_name = '" << table_name_
              << "', max_ids = " << max_ids_per_partition
              << ", max_uniques = " << max_unique_ids_per_partition;

    xla::XlaComputation optimizer = build_optimizer_computation(feature_width);

    xla::FrontendAttributes original_frontend_attributes =
        builder->frontend_attributes();

    xla::FrontendAttributes tuple_frontend_attributes;

    tuple_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    builder->SetFrontendAttributes(tuple_frontend_attributes);

    xla::XlaOp tables = get_tables_input(ctx);

    xla::XlaOp hyperparameters = get_hyperparameters_input(ctx);

    xla::Shape tables_shape = get_tables_shape(embedding_table_shape);

    xla::FrontendAttributes custom_call_frontend_attributes;

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_sharding_strategy", "mod"});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_pad_value", absl::StrCat(kXlaPadValue)});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_max_ids_per_partition", absl::StrCat(max_ids_per_partition)});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_max_unique_ids_per_partition",
         absl::StrCat(max_unique_ids_per_partition)});

    builder->SetFrontendAttributes(custom_call_frontend_attributes);

    xla::XlaOp updated_tables = xla::CustomCallWithComputation(
        builder, "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp",
        {row_pointers, sorted_token_ids, sorted_sample_ids, sorted_gains,
         num_minibatches_per_physical_sparse_core, tables, activation_gradients,
         hyperparameters},
        optimizer, tables_shape);

    builder->SetFrontendAttributes(tuple_frontend_attributes);

    // Updated embedding table.
    for (int i = 0; i < tables_shape.tuple_shapes_size(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(updated_tables, i));
    }

    builder->SetFrontendAttributes(original_frontend_attributes);
  }

 protected:
  float clip_weight_min_;
  float clip_weight_max_;

 private:
  std::string table_name_;

  XlaSparseDenseMatmulGradWithCsrInputBase(
      const XlaSparseDenseMatmulGradWithCsrInputBase&) = delete;
  void operator=(const XlaSparseDenseMatmulGradWithCsrInputBase&) = delete;
};

class XlaSparseDenseMatmulGradWithCsrInputOp : public XlaOpKernel {
 public:
  explicit XlaSparseDenseMatmulGradWithCsrInputOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    const NameAttrList* name_attr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("custom_computation", &name_attr));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
    custom_computation_ = *name_attr;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp row_pointers = ctx->Input("row_pointers");
    xla::XlaOp sorted_sample_ids = ctx->Input("sorted_sample_ids");
    xla::XlaOp sorted_token_ids = ctx->Input("sorted_token_ids");
    xla::XlaOp sorted_gains = ctx->Input("sorted_gains");
    xla::XlaOp activation_gradients = ctx->Input("activation_gradients");
    xla::XlaOp num_minibatches_per_physical_sparse_core =
        ctx->Input("num_minibatches_per_physical_sparse_core");

    std::vector<xla::XlaOp> tables_inputs;
    std::vector<TensorShape> tables_shapes;
    OP_REQUIRES_OK(ctx,
                   ctx->InputList("tables", &tables_inputs, &tables_shapes));

    std::vector<xla::XlaOp> hyperparameters_inputs;
    std::vector<TensorShape> hyperparameters_shapes;
    OP_REQUIRES_OK(ctx,
                   ctx->InputList("hyperparameters", &hyperparameters_inputs,
                                  &hyperparameters_shapes));

    // Get the shape of the gradient.
    OP_REQUIRES_VALUE(xla::Shape activation_shape, ctx,
                      ctx->InputXlaShape("activation_gradients"));
    OP_REQUIRES(ctx,
                activation_shape.is_static() &&
                    activation_shape.dimensions().size() == 2,
                absl::InvalidArgumentError(absl::StrCat(
                    "activations input has non static or non-rank 2 shape: ",
                    activation_shape.ToString())));
    OP_REQUIRES_VALUE(int64_t num_sparsecores_per_chip, ctx,
                      GetSparseCoresPerChip());
    int64_t num_samples_per_chip = activation_shape.dimensions(0);
    OP_REQUIRES(ctx, num_samples_per_chip % num_sparsecores_per_chip == 0,
                absl::InvalidArgumentError(absl::StrCat(
                    "num_samples_per_chip ", num_samples_per_chip,
                    " not divisible by the number of sparsecores per chip ",
                    num_sparsecores_per_chip)));

    int64_t per_sparse_core_batch_size =
        num_samples_per_chip / num_sparsecores_per_chip;
    int64_t max_ids_per_partition = 0;
    int64_t max_unique_ids_per_partition = 0;

    const int32_t feature_width = tables_shapes[0].dim_size(1);
    OP_REQUIRES_OK(
        ctx, GetMaxIdsAndUniquesExternal(kUnknownProgramKey, table_name_,
                                         per_sparse_core_batch_size,
                                         feature_width, &max_ids_per_partition,
                                         &max_unique_ids_per_partition));
    LOG(INFO) << "Lowering XlaSparseDenseMatmulGradWithCsrInputOp to HLO: "
              << "table_name = '" << table_name_
              << "', max_ids = " << max_ids_per_partition
              << ", max_uniques = " << max_unique_ids_per_partition;

    // Build the optimizer computation.
    XlaCompiler::CompileOptions options;

    // We don't use tuple args and always return tuple for this computation.
    options.use_tuple_arg = false;
    options.always_return_tuple = true;
    options.is_entry_computation = false;

    XlaCompiler* compiler = ctx->compiler();

    XlaCompiler::CompilationResult custom_computation_result;

    // The number of arguments is the number of tables + the number of
    // hyperparameters + 1 for the activation gradients.
    int32_t num_arguments =
        1 + tables_inputs.size() + hyperparameters_inputs.size();

    std::vector<XlaCompiler::Argument> arguments(num_arguments);

    // For all the arguments, we use the float type and the shape is
    // {1, feature_width}.
    for (int32_t i = 0; i < num_arguments; ++i) {
      arguments[i].kind = XlaCompiler::Argument::kParameter;
      arguments[i].type = DT_FLOAT;
      arguments[i].shape =
          xla::ShapeUtil::MakeShape(xla::F32, {1, feature_width});
    }

    CHECK_OK(compiler->CompileFunction(options, custom_computation_, arguments,
                                       &custom_computation_result));

    xla::XlaComputation optimizer =
        std::move(*custom_computation_result.computation);

    xla::FrontendAttributes original_frontend_attributes =
        builder->frontend_attributes();

    xla::FrontendAttributes tuple_frontend_attributes;

    tuple_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    builder->SetFrontendAttributes(tuple_frontend_attributes);

    xla::XlaOp tables = xla::Tuple(ctx->builder(), tables_inputs);

    xla::XlaOp hyperparameters =
        xla::Tuple(ctx->builder(), hyperparameters_inputs);

    std::vector<xla::Shape> xla_tables_shapes;

    xla_tables_shapes.reserve(tables_shapes.size());
    for (const auto& table_shape : tables_shapes) {
      xla_tables_shapes.push_back(xla::ShapeUtil::MakeShape(
          xla::F32, {table_shape.dim_size(0), table_shape.dim_size(1)}));
    }

    xla::Shape tables_shape = xla::ShapeUtil::MakeTupleShape(xla_tables_shapes);

    xla::FrontendAttributes custom_call_frontend_attributes;

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_compute_type", "sparse"});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_sharding_strategy", "mod"});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_pad_value", absl::StrCat(kXlaPadValue)});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_max_ids_per_partition", absl::StrCat(max_ids_per_partition)});

    custom_call_frontend_attributes.mutable_map()->insert(
        {"_xla_max_unique_ids_per_partition",
         absl::StrCat(max_unique_ids_per_partition)});

    builder->SetFrontendAttributes(custom_call_frontend_attributes);

    xla::XlaOp updated_tables = xla::CustomCallWithComputation(
        builder, "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp",
        {row_pointers, sorted_token_ids, sorted_sample_ids, sorted_gains,
         num_minibatches_per_physical_sparse_core, tables, activation_gradients,
         hyperparameters},
        optimizer, tables_shape);

    builder->SetFrontendAttributes(tuple_frontend_attributes);

    // Updated embedding table.
    for (int i = 0; i < tables_shape.tuple_shapes_size(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(updated_tables, i));
    }

    builder->SetFrontendAttributes(original_frontend_attributes);
  }

 private:
  std::string table_name_;
  NameAttrList custom_computation_;
  XlaSparseDenseMatmulGradWithCsrInputOp(
      const XlaSparseDenseMatmulGradWithCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulGradWithCsrInputOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithCsrInput"),
                XlaSparseDenseMatmulGradWithCsrInputOp);

// This TensorFlow op calculates the gradients and performs SGD update on the
// embedding table on SparseCore. It takes the activation gradients, input
// sparse tensor represented by the `row_pointers`, `sorted_embedding_ids`,
// `sorted_sample_ids` and 'learning_rate'. It produces the updated embedding
// table. It also supports minibatching.
class XlaSparseDenseMatmulGradWithSgdAndCsrInputOp
    : public XlaSparseDenseMatmulGradWithCsrInputBase {
 public:
  explicit XlaSparseDenseMatmulGradWithSgdAndCsrInputOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithCsrInputBase(ctx) {}

  ~XlaSparseDenseMatmulGradWithSgdAndCsrInputOp() override = default;

  xla::XlaComputation build_optimizer_computation(
      const int32_t feature_width) override {
    xla::XlaComputation sgd_optimizer = [&] {
      auto sgd_optimizer_builder =
          std::make_unique<xla::XlaBuilder>("sgd_optimizer_builder");

      xla::Shape per_row_shape =
          xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

      xla::XlaOp gradient = xla::Parameter(sgd_optimizer_builder.get(), 0,
                                           per_row_shape, "gradient");

      xla::XlaOp embedding_table = xla::Parameter(
          sgd_optimizer_builder.get(), 1, per_row_shape, "embedding_table");

      xla::XlaOp learning_rate = xla::Parameter(sgd_optimizer_builder.get(), 2,
                                                per_row_shape, "learning_rate");

      xla::XlaOp updated_embedding_table =
          embedding_table - learning_rate * gradient;

      // Apply the weight clipping.
      xla::XlaOp clipped_embedding_table = apply_weight_clipping_to_table(
          sgd_optimizer_builder.get(), updated_embedding_table);

      xla::XlaOp updated_tables =
          xla::Tuple(sgd_optimizer_builder.get(), {clipped_embedding_table});

      return sgd_optimizer_builder->Build(updated_tables).value();
    }();

    return sgd_optimizer;
  }

  xla::XlaOp get_tables_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("embedding_table")});
  }

  xla::XlaOp get_hyperparameters_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("learning_rate")});
  }

  xla::Shape get_tables_shape(xla::Shape embedding_table_shape) override {
    return xla::ShapeUtil::MakeTupleShape({embedding_table_shape});
  }

 private:
  XlaSparseDenseMatmulGradWithSgdAndCsrInputOp(
      const XlaSparseDenseMatmulGradWithSgdAndCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulGradWithSgdAndCsrInputOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithSgdAndCsrInput"),
                XlaSparseDenseMatmulGradWithSgdAndCsrInputOp);

// This TensorFlow op calculates the gradients and performs Adagrad update on
// the embedding table on SparseCore. It takes the activation gradients, input
// sparse tensor represented by the `row_pointers`, `sorted_embedding_ids`,
// `sorted_sample_ids` and 'learning_rate'. It produces the updated embedding
// table. It also supports minibatching.
class XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp
    : public XlaSparseDenseMatmulGradWithCsrInputBase {
 public:
  explicit XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithCsrInputBase(ctx) {}

  ~XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp() override = default;

  xla::XlaComputation build_optimizer_computation(
      const int32_t feature_width) override {
    xla::XlaComputation adagrad_optimizer = [&] {
      auto adagrad_optimizer_builder =
          std::make_unique<xla::XlaBuilder>("adagrad_optimizer_builder");

      xla::Shape per_row_shape =
          xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

      xla::XlaOp gradient = xla::Parameter(adagrad_optimizer_builder.get(), 0,
                                           per_row_shape, "gradient");

      xla::XlaOp embedding_table = xla::Parameter(
          adagrad_optimizer_builder.get(), 1, per_row_shape, "embedding_table");

      xla::XlaOp accumulator = xla::Parameter(adagrad_optimizer_builder.get(),
                                              2, per_row_shape, "accumulator");

      xla::XlaOp learning_rate = xla::Parameter(
          adagrad_optimizer_builder.get(), 3, per_row_shape, "learning_rate");

      xla::XlaOp new_accumulator = accumulator + gradient * gradient;

      xla::XlaOp updated_embedding_table =
          embedding_table -
          learning_rate * gradient / xla::Sqrt(new_accumulator);

      // Apply the weight clipping.
      xla::XlaOp clipped_embedding_table = apply_weight_clipping_to_table(
          adagrad_optimizer_builder.get(), updated_embedding_table);

      xla::XlaOp updated_tables =
          xla::Tuple(adagrad_optimizer_builder.get(),
                     {clipped_embedding_table, new_accumulator});
      return adagrad_optimizer_builder->Build(updated_tables).value();
    }();

    return adagrad_optimizer;
  }

  xla::XlaOp get_tables_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("embedding_table"),
                                       ctx->Input("accumulator")});
  }

  xla::XlaOp get_hyperparameters_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("learning_rate")});
  }

  xla::Shape get_tables_shape(xla::Shape embedding_table_shape) override {
    return xla::ShapeUtil::MakeTupleShape(
        {embedding_table_shape, embedding_table_shape});
  }

 private:
  XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp(
      const XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp&) =
      delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithAdagradAndCsrInput"),
                XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp);

// This TensorFlow op calculates the gradients and performs Adagrad with
// momentum update on the embedding table on SparseCore. It takes the activation
// gradients, input sparse tensor represented by the `row_pointers`,
// `sorted_embedding_ids`, `sorted_sample_ids` and 'learning_rate'. It produces
// the updated embedding table. It also supports minibatching.
class XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp
    : public XlaSparseDenseMatmulGradWithCsrInputBase {
 public:
  explicit XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithCsrInputBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exponent", &exponent_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta1", &beta1_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta2", &beta2_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  ~XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp() override =
      default;

  xla::XlaComputation build_optimizer_computation(
      const int32_t feature_width) override {
    xla::XlaComputation adagrad_momentum_optimizer = [&] {
      auto adagrad_momentum_optimizer_builder =
          std::make_unique<xla::XlaBuilder>(
              "adagrad_momentum_optimizer_builder");

      xla::Shape per_row_shape =
          xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

      xla::XlaOp gradient =
          xla::Parameter(adagrad_momentum_optimizer_builder.get(), 0,
                         per_row_shape, "gradient");
      xla::XlaOp embedding_table =
          xla::Parameter(adagrad_momentum_optimizer_builder.get(), 1,
                         per_row_shape, "embedding_table");
      xla::XlaOp accumulator =
          xla::Parameter(adagrad_momentum_optimizer_builder.get(), 2,
                         per_row_shape, "accumulator");
      xla::XlaOp momenta =
          xla::Parameter(adagrad_momentum_optimizer_builder.get(), 3,
                         per_row_shape, "momenta");
      xla::XlaOp learning_rate =
          xla::Parameter(adagrad_momentum_optimizer_builder.get(), 4,
                         per_row_shape, "learning_rate");

      xla::XlaOp beta1 =
          xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), beta1_);
      xla::XlaOp beta2 =
          xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), beta2_);
      xla::XlaOp epsilon =
          xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), epsilon_);

      // If beta_2 == 1:
      //    accumulator(t) = accumulator(t-1) + gradient(t) ^ 2
      // Else:
      //    accumulator(t) = beta_2 * accumulator(t-1) +
      //                    (1-beta_2) * gradient(t) ^ 2
      xla::XlaOp exponent = xla::ConstantR0(
          adagrad_momentum_optimizer_builder.get(), 1.0f / exponent_);
      xla::XlaOp one =
          xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), 1.0f);

      xla::XlaOp new_accumulator = xla::Select(
          xla::Eq(beta2, one), accumulator + gradient * gradient,
          beta2 * accumulator + (one - beta2) * gradient * gradient);

      // scaled_gradient = (accumulator + epsilon)^(-1/k) * gradient
      xla::XlaOp scaled_gradients =
          Pow(new_accumulator + epsilon, xla::Neg(exponent)) * gradient;

      // momenta(t) = beta1 * momenta(t-1) + scaled_gradient(t)
      xla::XlaOp new_momenta = beta1 * momenta + scaled_gradients;

      // Table update:
      // non-nesterov: update = momenta_t
      // nesterov:     update = beta_1 * momenta_t + scaled_gradient
      // weights(t) = weights(t-1) - lr * update
      xla::XlaOp updated_embedding_table;
      if (use_nesterov_) {
        updated_embedding_table =
            embedding_table -
            learning_rate * (beta1 * new_momenta + scaled_gradients);
      } else {
        updated_embedding_table = embedding_table - learning_rate * new_momenta;
      }

      // Apply the weight clipping.
      xla::XlaOp clipped_embedding_table = apply_weight_clipping_to_table(
          adagrad_momentum_optimizer_builder.get(), updated_embedding_table);

      xla::XlaOp updated_tables =
          xla::Tuple(adagrad_momentum_optimizer_builder.get(),
                     {clipped_embedding_table, new_accumulator, new_momenta});
      return adagrad_momentum_optimizer_builder->Build(updated_tables).value();
    }();

    return adagrad_momentum_optimizer;
  }

  xla::XlaOp get_tables_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(),
                      {ctx->Input("embedding_table"), ctx->Input("accumulator"),
                       ctx->Input("momenta")});
  }

  xla::XlaOp get_hyperparameters_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("learning_rate")});
  }

  xla::Shape get_tables_shape(xla::Shape embedding_table_shape) override {
    return xla::ShapeUtil::MakeTupleShape(
        {embedding_table_shape, embedding_table_shape, embedding_table_shape});
  }

 private:
  bool use_nesterov_;
  float exponent_;
  float beta1_;
  float beta2_;
  float epsilon_;

  TF_DISALLOW_COPY_AND_ASSIGN(
      XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp);
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput"),
                XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp);

// This TensorFlow op calculates the gradients and performs Adam
// update on the embedding table on SparseCore. It takes the activation
// gradients, input sparse tensor represented by the `row_pointers`,
// `sorted_embedding_ids`, `sorted_sample_ids` and 'learning_rate'. It produces
// the updated embedding table. It also supports minibatching.
class XlaSparseDenseMatmulGradWithAdamAndCsrInputOp
    : public XlaSparseDenseMatmulGradWithCsrInputBase {
 public:
  explicit XlaSparseDenseMatmulGradWithAdamAndCsrInputOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithCsrInputBase(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_sum_inside_sqrt", &use_sum_inside_sqrt_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta1", &beta1_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta2", &beta2_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  ~XlaSparseDenseMatmulGradWithAdamAndCsrInputOp() override = default;

  xla::XlaComputation build_optimizer_computation(
      const int32_t feature_width) override {
    xla::XlaComputation adam_optimizer = [&] {
      auto adam_optimizer_builder =
          std::make_unique<xla::XlaBuilder>("adam_optimizer_builder");

      xla::Shape per_row_shape =
          xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

      xla::XlaOp gradient = xla::Parameter(adam_optimizer_builder.get(), 0,
                                           per_row_shape, "gradient");
      xla::XlaOp embedding_table = xla::Parameter(
          adam_optimizer_builder.get(), 1, per_row_shape, "embedding_table");
      xla::XlaOp momenta = xla::Parameter(adam_optimizer_builder.get(), 2,
                                          per_row_shape, "momenta");
      xla::XlaOp velocity = xla::Parameter(adam_optimizer_builder.get(), 3,
                                           per_row_shape, "velocity");
      xla::XlaOp learning_rate = xla::Parameter(adam_optimizer_builder.get(), 4,
                                                per_row_shape, "learning_rate");

      xla::XlaOp beta1 = xla::ConstantR0(adam_optimizer_builder.get(), beta1_);
      xla::XlaOp beta2 = xla::ConstantR0(adam_optimizer_builder.get(), beta2_);
      xla::XlaOp epsilon =
          xla::ConstantR0(adam_optimizer_builder.get(), epsilon_);

      // Depending on sum_inside_sqrt, the denominator is either:
      //     sum_inside_sqrt==true: sqrt(v + eps^2)
      //     sum_inside_sqrt==false: sqrt(v) + eps
      // To simplify the for loop below, write the sqrt denominator as:
      //     sqrt(v + e1) + e2
      // and set e1 and e2 appropriately:
      xla::XlaOp zero = xla::ConstantR0(adam_optimizer_builder.get(), 0.0f);
      xla::XlaOp one = xla::ConstantR0(adam_optimizer_builder.get(), 1.0f);
      xla::XlaOp e1 = use_sum_inside_sqrt_ ? epsilon * epsilon : zero;
      xla::XlaOp e2 = use_sum_inside_sqrt_ ? zero : epsilon;

      // momentum(t) = beta_1 * momentum(t-1)
      //                      + (1-beta_1)*gradient(t)
      xla::XlaOp new_momenta = beta1 * momenta + (one - beta1) * gradient;

      // velocity(t) = beta_2 * velocity(t-1)
      //                      + (1-beta_2)*gradient(t)*gradient(t)
      xla::XlaOp new_velocity =
          beta2 * velocity + (one - beta2) * gradient * gradient;

      xla::XlaOp updated_embedding_table =
          embedding_table -
          learning_rate * new_momenta / (xla::Sqrt(new_velocity + e1) + e2);

      // Apply the weight clipping.
      xla::XlaOp clipped_embedding_table = apply_weight_clipping_to_table(
          adam_optimizer_builder.get(), updated_embedding_table);

      xla::XlaOp updated_tables =
          xla::Tuple(adam_optimizer_builder.get(),
                     {clipped_embedding_table, new_momenta, new_velocity});
      return adam_optimizer_builder->Build(updated_tables).value();
    }();

    return adam_optimizer;
  }

  xla::XlaOp get_tables_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(),
                      {ctx->Input("embedding_table"), ctx->Input("momenta"),
                       ctx->Input("velocity")});
  }

  xla::XlaOp get_hyperparameters_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("learning_rate")});
  }

  xla::Shape get_tables_shape(xla::Shape embedding_table_shape) override {
    return xla::ShapeUtil::MakeTupleShape(
        {embedding_table_shape, embedding_table_shape, embedding_table_shape});
  }

 private:
  bool use_sum_inside_sqrt_;
  float beta1_;
  float beta2_;
  float epsilon_;

  XlaSparseDenseMatmulGradWithAdamAndCsrInputOp(
      const XlaSparseDenseMatmulGradWithAdamAndCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulGradWithAdamAndCsrInputOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithAdamAndCsrInput"),
                XlaSparseDenseMatmulGradWithAdamAndCsrInputOp);

// This TensorFlow op calculates the gradients and performs FTRL
// update on the embedding table on SparseCore. It takes the activation
// gradients, input sparse tensor represented by the `row_pointers`,
// `sorted_embedding_ids`, `sorted_sample_ids` and 'learning_rate'. It produces
// the updated embedding table. It also supports minibatching.
class XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp
    : public XlaSparseDenseMatmulGradWithCsrInputBase {
 public:
  explicit XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithCsrInputBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("multiply_linear_by_learning_rate",
                                     &multiply_linear_by_learning_rate_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("learning_rate_power", &learning_rate_power_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("l1_regularization_strength",
                                     &l1_regularization_strength_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("l2_regularization_strength",
                                     &l2_regularization_strength_));
  }

  ~XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp() override = default;

  xla::XlaComputation build_optimizer_computation(
      const int32_t feature_width) override {
    xla::XlaComputation ftrl_optimizer = [&] {
      auto ftrl_optimizer_builder =
          std::make_unique<xla::XlaBuilder>("ftrl_optimizer_builder");

      xla::Shape per_row_shape =
          xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

      xla::XlaOp gradient = xla::Parameter(ftrl_optimizer_builder.get(), 0,
                                           per_row_shape, "gradient");

      xla::XlaOp embedding_table = xla::Parameter(
          ftrl_optimizer_builder.get(), 1, per_row_shape, "embedding_table");
      xla::XlaOp accumulator = xla::Parameter(ftrl_optimizer_builder.get(), 2,
                                              per_row_shape, "accumulator");
      xla::XlaOp linear = xla::Parameter(ftrl_optimizer_builder.get(), 3,
                                         per_row_shape, "linear");
      xla::XlaOp learning_rate = xla::Parameter(ftrl_optimizer_builder.get(), 4,
                                                per_row_shape, "learning_rate");

      // accumulator(t) = accumulator(t-1) + gradient(t) ^ 2
      xla::XlaOp new_accumulator = accumulator + gradient * gradient;

      xla::XlaOp learning_rate_power =
          xla::ConstantR0(ftrl_optimizer_builder.get(), learning_rate_power_);

      xla::XlaOp power_old = Pow(accumulator, xla::Neg(learning_rate_power));
      xla::XlaOp power_new =
          Pow(new_accumulator, xla::Neg(learning_rate_power));
      xla::XlaOp delta_p = power_new - power_old;

      xla::XlaOp zero = xla::ConstantR0(ftrl_optimizer_builder.get(), 0.0f);

      xla::XlaOp two = xla::ConstantR0(ftrl_optimizer_builder.get(), 2.0f);

      xla::XlaOp l1_regularization_strength = xla::ConstantR0(
          ftrl_optimizer_builder.get(), l1_regularization_strength_);

      xla::XlaOp l2_regularization_strength = xla::ConstantR0(
          ftrl_optimizer_builder.get(), l2_regularization_strength_);

      xla::XlaOp beta = xla::ConstantR0(ftrl_optimizer_builder.get(), beta_);

      // Note:
      //    min(|linear(t)|, lr*l1)*sgn(linear(t))
      // can be written as
      //    clamp( -lr*l1, linear(t), lr*l1)
      // assuming lr>0 and l1>0.
      xla::XlaOp new_linear;
      xla::XlaOp numer;
      xla::XlaOp denom;
      if (multiply_linear_by_learning_rate_) {
        new_linear =
            linear + learning_rate * gradient - delta_p * embedding_table;
        // if multiply_linear:
        //   linear(t) = linear(t-1) + lr*g - delta_p * table(t-1)
        //   Update numerator:
        //      N = min(|linear(t)|, lr*l1)*sgn(linear(t)) - linear(t)
        //   Update denomninator:
        //      D = power(t) + 2*lr*l2 + beta
        //   table(t) = N / D
        numer = xla::Select(
            xla::Eq(l1_regularization_strength, zero), xla::Neg(new_linear),
            xla::Clamp(xla::Neg(learning_rate * l1_regularization_strength),
                       new_linear, learning_rate * l1_regularization_strength) -
                new_linear);
        denom =
            power_new + two * learning_rate * l2_regularization_strength + beta;
      } else {
        new_linear =
            linear + gradient - delta_p * embedding_table / learning_rate;
        // if NOT multiply_linear:
        //   linear(t) = linear(t-1) + g - (1/lr) * delta_p * table(t-1)
        //   Update numerator:
        //     N = min(|linear(t)|, l1)*sgn(linear(t)) - linear(t)
        //   Update denomninator:
        //     D = (1/lr) * (power(t) + beta) + 2*l2
        //   table(t) = N / D
        numer = xla::Select(xla::Eq(l1_regularization_strength, zero),
                            xla::Neg(new_linear),
                            xla::Clamp(xla::Neg(l1_regularization_strength),
                                       new_linear, l1_regularization_strength) -
                                new_linear);
        denom = (power_new + beta) / learning_rate +
                two * l2_regularization_strength;
      }
      xla::XlaOp updated_embedding_table = numer / denom;

      // Apply the weight clipping.
      xla::XlaOp clipped_embedding_table = apply_weight_clipping_to_table(
          ftrl_optimizer_builder.get(), updated_embedding_table);

      xla::XlaOp updated_tables =
          xla::Tuple(ftrl_optimizer_builder.get(),
                     {clipped_embedding_table, new_accumulator, new_linear});
      return ftrl_optimizer_builder->Build(updated_tables).value();
    }();

    return ftrl_optimizer;
  }

  xla::XlaOp get_tables_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(),
                      {ctx->Input("embedding_table"), ctx->Input("accumulator"),
                       ctx->Input("linear")});
  }

  xla::XlaOp get_hyperparameters_input(XlaOpKernelContext* ctx) override {
    return xla::Tuple(ctx->builder(), {ctx->Input("learning_rate")});
  }

  xla::Shape get_tables_shape(xla::Shape embedding_table_shape) override {
    return xla::ShapeUtil::MakeTupleShape(
        {embedding_table_shape, embedding_table_shape, embedding_table_shape});
  }

 private:
  bool multiply_linear_by_learning_rate_;
  float beta_;
  float learning_rate_power_;
  float l1_regularization_strength_;
  float l2_regularization_strength_;

  XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp(
      const XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp&) = delete;
  void operator=(const XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithFtrlAndCsrInput"),
                XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp);

class XlaSparseCoreOptimizerOpBase : public XlaOpKernel {
 public:
  explicit XlaSparseCoreOptimizerOpBase(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_width", &feature_width_));
  }

  ~XlaSparseCoreOptimizerOpBase() override = default;

  // Gather table rows into a dense working buffer. This helper assumes the
  // table is structured like a typical embedding table.
  xla::XlaOp GatherVectors(xla::XlaOp table, xla::XlaOp indices) {
    xla::GatherDimensionNumbers gather_dimension_numbers;
    gather_dimension_numbers.add_offset_dims(1);
    gather_dimension_numbers.add_collapsed_slice_dims(0);
    gather_dimension_numbers.add_start_index_map(0);
    gather_dimension_numbers.set_index_vector_dim(1);
    const std::vector<int64_t> slice_sizes = {1, feature_width_};
    return xla::Gather(table, indices, gather_dimension_numbers, slice_sizes);
  }

  xla::XlaOp ScatterReplace(xla::XlaOp table, xla::XlaOp indices,
                            xla::XlaOp updates) {
    // Scatter the updated table rows from the dense buffer to the
    // full (local shard) table.
    xla::ScatterDimensionNumbers scatter_dimension_numbers;
    scatter_dimension_numbers.add_update_window_dims(1);
    scatter_dimension_numbers.add_inserted_window_dims(0);
    scatter_dimension_numbers.add_scatter_dims_to_operand_dims(0);
    scatter_dimension_numbers.set_index_vector_dim(1);

    // Determines how updates are applied to the destination data.
    xla::XlaComputation scatter_update_fcn = [&] {
      auto sb = std::make_unique<xla::XlaBuilder>("scatter_builder");
      auto scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
      // Param 0 ("orig") is unused but still needs to be defined so we can
      // access Param 1 ("update").
      xla::Parameter(sb.get(), 0, scalar_shape, "orig");
      auto update = xla::Parameter(sb.get(), 1, scalar_shape, "update");
      // Update becomes the new table value.
      return sb->Build(update).value();
    }();

    // Do we have sorted or unique indices?
    return xla::Scatter(table, indices, updates, scatter_update_fcn,
                        scatter_dimension_numbers);
  }

 protected:
  int feature_width_;

  XlaSparseCoreOptimizerOpBase(const XlaSparseCoreOptimizerOpBase&) = delete;
  void operator=(const XlaSparseCoreOptimizerOpBase&) = delete;
};

// This class uses the SGD optimizer to update the embedding table weights.
class XlaSparseCoreSgdOp : public XlaSparseCoreOptimizerOpBase {
 public:
  explicit XlaSparseCoreSgdOp(OpKernelConstruction* ctx)
      : XlaSparseCoreOptimizerOpBase(ctx) {}

  ~XlaSparseCoreSgdOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    UseSparseCoreFrontendAttributes fe_attributes(builder);

    // TODO(patn): Add error checking
    xla::XlaOp gradient = ctx->Input("gradient");
    xla::XlaOp indices = ctx->Input("indices");
    xla::XlaOp learning_rate = ctx->Input("learning_rate");
    xla::XlaOp table = ctx->Input("embedding_table");

    // The result will be scatter-added into the original embedding table so
    // this is just the "delta".
    xla::XlaOp table_old = GatherVectors(table, indices);
    xla::XlaOp updates = table_old - learning_rate * gradient;
    xla::XlaOp result = ScatterReplace(table, indices, updates);

    ctx->SetOutput(0, result);
  }

  XlaSparseCoreSgdOp(const XlaSparseCoreSgdOp&) = delete;
  void operator=(const XlaSparseCoreSgdOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseCoreSgd"), XlaSparseCoreSgdOp);

// This class uses the Adagrad optimizer to update the embedding table weights.
class XlaSparseCoreAdagradOp : public XlaSparseCoreOptimizerOpBase {
 public:
  explicit XlaSparseCoreAdagradOp(OpKernelConstruction* ctx)
      : XlaSparseCoreOptimizerOpBase(ctx) {}

  ~XlaSparseCoreAdagradOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    UseSparseCoreFrontendAttributes fe_attributes(builder);

    // TODO(patn): Add error checking
    xla::XlaOp gradient = ctx->Input("gradient");
    xla::XlaOp indices = ctx->Input("indices");
    xla::XlaOp learning_rate = ctx->Input("learning_rate");
    xla::XlaOp table = ctx->Input("embedding_table");
    xla::XlaOp accum = ctx->Input("accumulator");

    xla::XlaOp accum_old = GatherVectors(accum, indices);
    xla::XlaOp accum_new = accum_old + gradient * gradient;

    xla::XlaOp table_old = GatherVectors(table, indices);
    xla::XlaOp updates =
        table_old - learning_rate * gradient / xla::Sqrt(accum_new);

    xla::XlaOp accum_result = ScatterReplace(accum, indices, accum_new);

    xla::XlaOp table_result = ScatterReplace(table, indices, updates);

    ctx->SetOutput(0, table_result);
    ctx->SetOutput(1, accum_result);
  }

  XlaSparseCoreAdagradOp(const XlaSparseCoreAdagradOp&) = delete;
  void operator=(const XlaSparseCoreAdagradOp&) = delete;
};

REGISTER_XLA_OP(Name("XlaSparseCoreAdagrad"), XlaSparseCoreAdagradOp);

// This class uses the AdagradMomentum optimizer to update the embedding table
// weights.
class XlaSparseCoreAdagradMomentumOp : public XlaSparseCoreOptimizerOpBase {
 public:
  explicit XlaSparseCoreAdagradMomentumOp(OpKernelConstruction* ctx)
      : XlaSparseCoreOptimizerOpBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta_2", &beta_2_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exponent", &exponent_));
  }

  ~XlaSparseCoreAdagradMomentumOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    UseSparseCoreFrontendAttributes fe_attributes(builder);

    // TODO(patn): Add error checking
    xla::XlaOp gradient = ctx->Input("gradient");
    xla::XlaOp indices = ctx->Input("indices");
    xla::XlaOp learning_rate = ctx->Input("learning_rate");
    xla::XlaOp beta_1 = ctx->Input("beta_1");
    xla::XlaOp epsilon = ctx->Input("epsilon");
    xla::XlaOp table = ctx->Input("embedding_table");
    xla::XlaOp accum = ctx->Input("accumulator");
    xla::XlaOp momen = ctx->Input("momentum");

    xla::XlaOp accum_old = GatherVectors(accum, indices);
    xla::XlaOp momen_old = GatherVectors(momen, indices);

    // If beta_2 == 1:
    //    accumulator(t) = accumulator(t-1) + gradient(t) ^ 2
    // Else:
    //    accumulator(t) = beta_2 * accumulator(t-1) +
    //                    (1-beta_2) * gradient(t) ^ 2
    xla::XlaOp exponent = xla::ConstantR0(builder, 1.0f / exponent_);
    xla::XlaOp accum_new;

    if (beta_2_ == 1.0f) {
      accum_new = accum_old + gradient * gradient;
    } else {
      xla::XlaOp beta_2 = xla::ConstantR0(builder, beta_2_);
      xla::XlaOp one_m_beta_2 = xla::ConstantR0(builder, 1.0f - beta_2_);
      accum_new = beta_2 * accum_old + one_m_beta_2 * gradient * gradient;
    }
    // scaled_gradient = (accumulator + epsilon)^(-1/k) * gradient
    xla::XlaOp scaled_gradients =
        Pow(accum_new + epsilon, xla::Neg(exponent)) * gradient;

    // momentum(t) = beta_1 * momentum(t-1) + scaled_gradient(t)
    xla::XlaOp momen_new = beta_1 * momen_old + scaled_gradients;

    // Table update:
    // non-nesterov: update = momentum_t
    // nesterov:     update = beta_1 * momentum_t + scaled_gradient
    // weights(t) = weights(t-1) - lr * update
    xla::XlaOp table_old = GatherVectors(table, indices);
    xla::XlaOp updates;
    if (use_nesterov_) {
      updates =
          table_old - learning_rate * (beta_1 * momen_new + scaled_gradients);
    } else {
      updates = table_old - learning_rate * momen_new;
    }

    xla::XlaOp accum_result = ScatterReplace(accum, indices, accum_new);

    xla::XlaOp momen_result = ScatterReplace(momen, indices, momen_new);

    xla::XlaOp table_result = ScatterReplace(table, indices, updates);

    ctx->SetOutput(0, table_result);
    ctx->SetOutput(1, accum_result);
    ctx->SetOutput(2, momen_result);
  }

  XlaSparseCoreAdagradMomentumOp(const XlaSparseCoreAdagradMomentumOp&) =
      delete;
  void operator=(const XlaSparseCoreAdagradMomentumOp&) = delete;

 private:
  bool use_nesterov_;
  float beta_2_;
  float exponent_;
};

REGISTER_XLA_OP(Name("XlaSparseCoreAdagradMomentum"),
                XlaSparseCoreAdagradMomentumOp);

// This class uses the Adam optimizer to update the embedding table weights.
class XlaSparseCoreAdamOp : public XlaSparseCoreOptimizerOpBase {
 public:
  explicit XlaSparseCoreAdamOp(OpKernelConstruction* ctx)
      : XlaSparseCoreOptimizerOpBase(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("use_sum_inside_sqrt", &use_sum_inside_sqrt_));
  }

  ~XlaSparseCoreAdamOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    UseSparseCoreFrontendAttributes fe_attributes(builder);

    // TODO(patn): Add error checking
    xla::XlaOp gradient = ctx->Input("gradient");
    xla::XlaOp indices = ctx->Input("indices");
    xla::XlaOp learning_rate = ctx->Input("learning_rate");
    xla::XlaOp beta_1 = ctx->Input("beta_1");
    xla::XlaOp beta_2 = ctx->Input("beta_2");
    xla::XlaOp epsilon = ctx->Input("epsilon");
    xla::XlaOp table = ctx->Input("embedding_table");
    xla::XlaOp momen = ctx->Input("momentum");
    xla::XlaOp veloc = ctx->Input("velocity");
    // xla::XlaOp use_non_lazy_adam = ctx->Input("use_non_lazy_adam");

    xla::XlaOp momen_old = GatherVectors(momen, indices);
    xla::XlaOp veloc_old = GatherVectors(veloc, indices);

    // Depending on sum_inside_sqrt, the denominator is either:
    //     sum_inside_sqrt==true: sqrt(v + eps^2)
    //     sum_inside_sqrt==false: sqrt(v) + eps
    // To simplify the for loop below, write the sqrt denominator as:
    //     sqrt(v + e1) + e2
    // and set e1 and e2 appropriately:
    xla::XlaOp zero = xla::ConstantR0(builder, 0.0f);
    xla::XlaOp one = xla::ConstantR0(builder, 1.0f);
    xla::XlaOp e1 = use_sum_inside_sqrt_ ? epsilon * epsilon : zero;
    xla::XlaOp e2 = use_sum_inside_sqrt_ ? zero : epsilon;

    // momentum(t) = beta_1 * momentum(t-1)
    //                      + (1-beta_1)*gradient(t)
    xla::XlaOp momen_new = beta_1 * momen_old + (one - beta_1) * gradient;

    // velocity(t) = beta_2 * velocity(t-1)
    //                      + (1-beta_2)*gradient(t)*gradient(t)
    xla::XlaOp veloc_new =
        beta_2 * veloc_old + (one - beta_2) * gradient * gradient;

    // table(t) = table(t-1) - lr * (m(t) / (sqrt(v(t) + e1) + e2))
    xla::XlaOp table_old = GatherVectors(table, indices);
    xla::XlaOp updates = table_old - learning_rate * momen_new /
                                         (xla::Sqrt(veloc_new + e1) + e2);

    xla::XlaOp momen_result = ScatterReplace(momen, indices, momen_new);

    xla::XlaOp veloc_result = ScatterReplace(veloc, indices, veloc_new);

    xla::XlaOp table_result = ScatterReplace(table, indices, updates);

    ctx->SetOutput(0, table_result);
    ctx->SetOutput(1, veloc_result);
    ctx->SetOutput(2, momen_result);
  }

  XlaSparseCoreAdamOp(const XlaSparseCoreAdamOp&) = delete;
  void operator=(const XlaSparseCoreAdamOp&) = delete;

 private:
  bool use_sum_inside_sqrt_;
};

REGISTER_XLA_OP(Name("XlaSparseCoreAdam"), XlaSparseCoreAdamOp);

// This class uses the Ftrl optimizer to update the embedding table weights.
class XlaSparseCoreFtrlOp : public XlaSparseCoreOptimizerOpBase {
 public:
  explicit XlaSparseCoreFtrlOp(OpKernelConstruction* ctx)
      : XlaSparseCoreOptimizerOpBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("multiply_linear_by_learning_rate",
                                     &multiply_linear_by_learning_rate_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("l1_regularization_strength",
                                     &l1_regularization_strength_));
  }

  ~XlaSparseCoreFtrlOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();
    UseSparseCoreFrontendAttributes fe_attributes(builder);

    xla::XlaOp gradient = ctx->Input("gradient");
    xla::XlaOp indices = ctx->Input("indices");
    xla::XlaOp learning_rate = ctx->Input("learning_rate");
    xla::XlaOp learning_rate_power = ctx->Input("learning_rate_power");
    xla::XlaOp l2_regularization_strength =
        ctx->Input("l2_regularization_strength");
    xla::XlaOp beta = ctx->Input("beta");
    xla::XlaOp table = ctx->Input("embedding_table");
    xla::XlaOp accum = ctx->Input("accumulator");
    xla::XlaOp linear = ctx->Input("linear");

    xla::XlaOp accum_old = GatherVectors(accum, indices);
    xla::XlaOp linear_old = GatherVectors(linear, indices);

    // accumulator(t) = accumulator(t-1) + gradient(t) ^ 2
    xla::XlaOp accum_new = accum_old + gradient * gradient;

    xla::XlaOp power_old = Pow(accum_old, -learning_rate_power);
    xla::XlaOp power_new = Pow(accum_new, -learning_rate_power);
    xla::XlaOp delta_p = power_new - power_old;

    xla::XlaOp two = xla::ConstantR0(builder, 2.0f);
    xla::XlaOp table_old = GatherVectors(table, indices);

    // Note:
    //    min(|linear(t)|, lr*l1)*sgn(linear(t))
    // can be written as
    //    clamp( -lr*l1, linear(t), lr*l1)
    // assuming lr>0 and l1>0.
    xla::XlaOp linear_new;
    xla::XlaOp numer;
    xla::XlaOp denom;
    if (multiply_linear_by_learning_rate_) {
      linear_new = linear_old + learning_rate * gradient - delta_p * table_old;
      // if multiply_linear:
      //   linear(t) = linear(t-1) + lr*g - delta_p * table(t-1)
      //   Update numerator:
      //      N = min(|linear(t)|, lr*l1)*sgn(linear(t)) - linear(t)
      //   Update denomninator:
      //      D = power(t) + 2*lr*l2 + beta
      //   table(t) = N / D
      if (l1_regularization_strength_ == 0) {
        numer = -linear_new;
      } else {
        xla::XlaOp l1_regularization_strength =
            xla::ConstantR0(builder, l1_regularization_strength_);
        numer =
            xla::Clamp(-learning_rate * l1_regularization_strength, linear_new,
                       learning_rate * l1_regularization_strength) -
            linear_new;
      }
      denom =
          power_new + two * learning_rate * l2_regularization_strength + beta;
    } else {
      linear_new = linear_old + gradient - delta_p * table_old / learning_rate;
      // if NOT multiply_linear:
      //   linear(t) = linear(t-1) + g - (1/lr) * delta_p * table(t-1)
      //   Update numerator:
      //     N = min(|linear(t)|, l1)*sgn(linear(t)) - linear(t)
      //   Update denomninator:
      //     D = (1/lr) * (power(t) + beta) + 2*l2
      //   table(t) = N / D
      if (l1_regularization_strength_ == 0) {
        numer = -linear_new;
      } else {
        xla::XlaOp l1_regularization_strength =
            xla::ConstantR0(builder, l1_regularization_strength_);
        numer = xla::Clamp(-l1_regularization_strength, linear_new,
                           l1_regularization_strength) -
                linear_new;
      }
      denom =
          (power_new + beta) / learning_rate + two * l2_regularization_strength;
    }
    xla::XlaOp updates = numer / denom;

    xla::XlaOp accum_result = ScatterReplace(accum, indices, accum_new);

    xla::XlaOp linear_result = ScatterReplace(linear, indices, linear_new);

    xla::XlaOp table_result = ScatterReplace(table, indices, updates);

    ctx->SetOutput(0, table_result);
    ctx->SetOutput(1, accum_result);
    ctx->SetOutput(2, linear_result);
  }

  XlaSparseCoreFtrlOp(const XlaSparseCoreFtrlOp&) = delete;
  void operator=(const XlaSparseCoreFtrlOp&) = delete;

 private:
  bool multiply_linear_by_learning_rate_;
  float l1_regularization_strength_;
};

REGISTER_XLA_OP(Name("XlaSparseCoreFtrl"), XlaSparseCoreFtrlOp);

//***************************************************************************
// Below is the SparseCore ops with static buffer size. They are the same as
// the above ops except that they take the max_ids/uniques as input attributes.
//***************************************************************************
class XlaSparseDenseMatmulWithStaticBufferSizeOp
    : public XlaSparseDenseMatmulWithCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulWithStaticBufferSizeOp(OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulWithCsrInputOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_sparse_core",
                                     &max_ids_per_sparse_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_ids_per_sparse_core",
                                     &max_unique_ids_per_sparse_core_));

    OP_REQUIRES(
        ctx, max_ids_per_sparse_core_ > 0,
        absl::InvalidArgumentError("max_ids_per_sparse_core must be > 0"));
    OP_REQUIRES(ctx, max_unique_ids_per_sparse_core_ > 0,
                absl::InvalidArgumentError(
                    "max_unique_ids_per_sparse_core must be > 0"));
  }

  absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition,
      int64_t* max_unique_ids_per_partition) override {
    if (max_ids_per_partition == nullptr ||
        max_unique_ids_per_partition == nullptr) {
      return absl::InternalError("Setting the max ids/uniques failed.");
    }
    *max_ids_per_partition = max_ids_per_sparse_core_;
    *max_unique_ids_per_partition = max_unique_ids_per_sparse_core_;
    return absl::OkStatus();
  }

 private:
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulWithStaticBufferSize"),
                XlaSparseDenseMatmulWithStaticBufferSizeOp);

class XlaSparseDenseMatmulGradWithSgdAndStaticBufferSizeOp
    : public XlaSparseDenseMatmulGradWithSgdAndCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulGradWithSgdAndStaticBufferSizeOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithSgdAndCsrInputOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_sparse_core",
                                     &max_ids_per_sparse_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_ids_per_sparse_core",
                                     &max_unique_ids_per_sparse_core_));

    OP_REQUIRES(
        ctx, max_ids_per_sparse_core_ > 0,
        absl::InvalidArgumentError("max_ids_per_sparse_core must be > 0"));
    OP_REQUIRES(ctx, max_unique_ids_per_sparse_core_ > 0,
                absl::InvalidArgumentError(
                    "max_unique_ids_per_sparse_core must be > 0"));
  }

  absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition,
      int64_t* max_unique_ids_per_partition) override {
    if (max_ids_per_partition == nullptr ||
        max_unique_ids_per_partition == nullptr) {
      return absl::InternalError("Setting the max ids/uniques failed.");
    }
    *max_ids_per_partition = max_ids_per_sparse_core_;
    *max_unique_ids_per_partition = max_unique_ids_per_sparse_core_;
    return absl::OkStatus();
  }

 private:
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize"),
                XlaSparseDenseMatmulGradWithSgdAndStaticBufferSizeOp);

class XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOp
    : public XlaSparseDenseMatmulGradWithAdamAndCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithAdamAndCsrInputOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_sparse_core",
                                     &max_ids_per_sparse_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_ids_per_sparse_core",
                                     &max_unique_ids_per_sparse_core_));

    OP_REQUIRES(
        ctx, max_ids_per_sparse_core_ > 0,
        absl::InvalidArgumentError("max_ids_per_sparse_core must be > 0"));
    OP_REQUIRES(ctx, max_unique_ids_per_sparse_core_ > 0,
                absl::InvalidArgumentError(
                    "max_unique_ids_per_sparse_core must be > 0"));
  }

  absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition,
      int64_t* max_unique_ids_per_partition) override {
    if (max_ids_per_partition == nullptr ||
        max_unique_ids_per_partition == nullptr) {
      return absl::InternalError("Setting the max ids/uniques failed.");
    }
    *max_ids_per_partition = max_ids_per_sparse_core_;
    *max_unique_ids_per_partition = max_unique_ids_per_sparse_core_;
    return absl::OkStatus();
  }

 private:
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize"),
                XlaSparseDenseMatmulGradWithAdamAndStaticBufferSizeOp);

class XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOp
    : public XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithAdagradAndCsrInputOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_sparse_core",
                                     &max_ids_per_sparse_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_ids_per_sparse_core",
                                     &max_unique_ids_per_sparse_core_));

    OP_REQUIRES(
        ctx, max_ids_per_sparse_core_ > 0,
        absl::InvalidArgumentError("max_ids_per_sparse_core must be > 0"));
    OP_REQUIRES(ctx, max_unique_ids_per_sparse_core_ > 0,
                absl::InvalidArgumentError(
                    "max_unique_ids_per_sparse_core must be > 0"));
  }

  absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition,
      int64_t* max_unique_ids_per_partition) override {
    if (max_ids_per_partition == nullptr ||
        max_unique_ids_per_partition == nullptr) {
      return absl::InternalError("Setting the max ids/uniques failed.");
    }
    *max_ids_per_partition = max_ids_per_sparse_core_;
    *max_unique_ids_per_partition = max_unique_ids_per_sparse_core_;
    return absl::OkStatus();
  }

 private:
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize"),
                XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSizeOp);

class XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOp
    : public XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithFtrlAndCsrInputOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_sparse_core",
                                     &max_ids_per_sparse_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_ids_per_sparse_core",
                                     &max_unique_ids_per_sparse_core_));

    OP_REQUIRES(
        ctx, max_ids_per_sparse_core_ > 0,
        absl::InvalidArgumentError("max_ids_per_sparse_core must be > 0"));
    OP_REQUIRES(ctx, max_unique_ids_per_sparse_core_ > 0,
                absl::InvalidArgumentError(
                    "max_unique_ids_per_sparse_core must be > 0"));
  }

  absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition,
      int64_t* max_unique_ids_per_partition) override {
    if (max_ids_per_partition == nullptr ||
        max_unique_ids_per_partition == nullptr) {
      return absl::InternalError("Setting the max ids/uniques failed.");
    }
    *max_ids_per_partition = max_ids_per_sparse_core_;
    *max_unique_ids_per_partition = max_unique_ids_per_sparse_core_;
    return absl::OkStatus();
  }

 private:
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
};

REGISTER_XLA_OP(Name("XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize"),
                XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSizeOp);

class XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOp
    : public XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp {
 public:
  explicit XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOp(
      OpKernelConstruction* ctx)
      : XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInputOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_sparse_core",
                                     &max_ids_per_sparse_core_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_unique_ids_per_sparse_core",
                                     &max_unique_ids_per_sparse_core_));

    OP_REQUIRES(
        ctx, max_ids_per_sparse_core_ > 0,
        absl::InvalidArgumentError("max_ids_per_sparse_core must be > 0"));
    OP_REQUIRES(ctx, max_unique_ids_per_sparse_core_ > 0,
                absl::InvalidArgumentError(
                    "max_unique_ids_per_sparse_core must be > 0"));
  }

  absl::Status GetMaxIdsAndUniques(
      int64_t num_samples_per_sparse_core, int64_t feature_width,
      int64_t* max_ids_per_partition,
      int64_t* max_unique_ids_per_partition) override {
    if (max_ids_per_partition == nullptr ||
        max_unique_ids_per_partition == nullptr) {
      return absl::InternalError("Setting the max ids/uniques failed.");
    }
    *max_ids_per_partition = max_ids_per_sparse_core_;
    *max_unique_ids_per_partition = max_unique_ids_per_sparse_core_;
    return absl::OkStatus();
  }

 private:
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
};

REGISTER_XLA_OP(
    Name("XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize"),
    XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSizeOp);
}  // anonymous namespace
}  // namespace tensorflow
