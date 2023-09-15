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
#include "tensorflow/core/tpu/kernels/sparse_core_preprocess_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_stats_handler.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

namespace tensorflow {

Status ValidateInputs(const Tensor& indices_or_row_splits, const Tensor& values,
                      const Tensor& weights, int sample_count) {
  if (values.dims() != 1 || weights.dims() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Values and weights input should have dimension as 1. But got ",
        values.dims(), " for values and ", weights.dims(), " for weights."));
  }
  if (values.NumElements() != weights.NumElements()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Values and weights should have same elements. But got ",
                     values.NumElements(), " elements for values and ",
                     weights.NumElements(), " elements for weights."));
  }
  if (indices_or_row_splits.NumElements() == 0) {
    // Dense tensor.
    if (values.NumElements() != sample_count) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dense tensor input should have values elements number the same as "
          "the sample count. But got ",
          values.NumElements(), " elements for values and sample count as ",
          sample_count, "."));
    }
  } else if (indices_or_row_splits.dims() == 2 &&
             indices_or_row_splits.NumElements() > 0) {
    // TODO(pineapplejuice233): Add checking logic for sparse tensor input.
  } else if (indices_or_row_splits.dims() == 1 &&
             indices_or_row_splits.NumElements() > 0) {
    // Ragged tensor.
    if (indices_or_row_splits.NumElements() != sample_count + 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Ragged tensor input should have row_splits elements number the same "
          "as the sample count + 1. But got ",
          indices_or_row_splits.NumElements(),
          " elements for row_splits and sample count as ", sample_count, "."));
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid indices_or_row_splits input, Got dimension of ",
                     indices_or_row_splits.dims(), " and size of ",
                     indices_or_row_splits.NumElements(), "."));
  }
  return OkStatus();
}

Status ComputeRowIdsBeforePadding(const Tensor& indices_or_row_splits,
                                  const int32 total_id_count,
                                  Tensor* row_ids_before_padding) {
  // The only difference between dense tensor, sparse tensor and ragged tensor
  // is the row ids output.
  if (indices_or_row_splits.NumElements() == 0) {
    // Dense tensor to COO format.
    // Row ids are just the index ids.
    int32* row_ids_before_padding_ptr =
        row_ids_before_padding->flat<int32>().data();
    for (int32 i = 0; i < total_id_count; ++i) {
      *(row_ids_before_padding_ptr + i) = i;
    }
  } else if (indices_or_row_splits.dims() == 2 &&
             indices_or_row_splits.NumElements() > 0) {
    // Sparse tensor to COO format.
    // TODO(pineapplejuice233): should we support arbitrary rank of sparse tensor and
    // convert it to 2D?
    // For 2D sparse tensor, as we always combine on the last dimension.
    // The row ids are just the sample ids which is the first dim of the
    // indices.
    int32* row_ids_before_padding_ptr =
        row_ids_before_padding->flat<int32>().data();
    for (int32 i = 0; i < total_id_count; ++i) {
      *(row_ids_before_padding_ptr + i) =
          indices_or_row_splits.tensor<int32, 2>()(i, 0);
    }
  } else if (indices_or_row_splits.dims() == 1 &&
             indices_or_row_splits.NumElements() > 0) {
    // Ragged tensor to COO format.
    const int32* indices_or_row_splits_ptr =
        indices_or_row_splits.flat<int32>().data();
    int32* row_ids_before_padding_ptr =
        row_ids_before_padding->flat<int32>().data();
    int32 current_row_id = -1;
    for (int32 i = 0; i < total_id_count; ++i) {
      while (i == *(indices_or_row_splits_ptr + 1 + current_row_id)) {
        current_row_id += 1;
      }
      *(row_ids_before_padding_ptr + i) = current_row_id;
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid indices_or_row_splits input, Got dimension of ",
                     indices_or_row_splits.dims(), " and size of ",
                     indices_or_row_splits.NumElements(), "."));
  }
  return OkStatus();
}

// Convert the input sparse/dense/ragged tensor into COO format and normalize
// the combiner. Note the COO tensor it produces only contains three 1D tensors
// and no partitioning is performed on these tensors.
class ConvertToCooTensorOp : public OpKernel {
 public:
  explicit ConvertToCooTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_count", &sample_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ValidateInputCombiner(combiner_));
  }
  ~ConvertToCooTensorOp() override = default;
  ConvertToCooTensorOp(const ConvertToCooTensorOp&) = delete;
  ConvertToCooTensorOp& operator=(const ConvertToCooTensorOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "Compute ConvertToCooTensorOp";

    const Tensor* indices_or_row_splits;
    OP_REQUIRES_OK(ctx,
                   ctx->input("indices_or_row_splits", &indices_or_row_splits));
    const Tensor* values;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values));
    const Tensor* weights;
    OP_REQUIRES_OK(ctx, ctx->input("weights", &weights));

    OP_REQUIRES_OK(ctx, ValidateInputs(*indices_or_row_splits, *values,
                                       *weights, sample_count_));

    const int32 total_id_count = values->NumElements();

    Tensor row_ids_before_dedup;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT32, TensorShape({total_id_count}),
                                      &row_ids_before_dedup));

    OP_REQUIRES_OK(
        ctx, ComputeRowIdsBeforePadding(*indices_or_row_splits, total_id_count,
                                        &row_ids_before_dedup));

    std::vector<float> gains_rescale(sample_count_, 0.0f);

    // Compute the rescaled gains
    auto combiner_scale_contribution_fn =
        GetCombinerScaleContributionFunction(combiner_);

    auto combiner_scale_transform_fn =
        GetCombinerScaleTransformFunction(combiner_);

    const int32* row_ids_before_dedup_ptr =
        row_ids_before_dedup.flat<int32>().data();
    const int32* values_ptr = values->flat<int32>().data();
    const float* weights_ptr = weights->flat<float>().data();

    // Dedup the ids within one sample by just checking the adjacent ids. This
    // will NOT result in a full deduplication.
    std::vector<int32> row_ids;
    std::vector<int32> col_ids;
    std::vector<float> gains;
    row_ids.reserve(total_id_count);
    col_ids.reserve(total_id_count);
    gains.reserve(total_id_count);

    for (int token_id = 0; token_id < total_id_count; ++token_id) {
      // Compute the gain rescale before doing the dedup.
      const int32 row_id = *(row_ids_before_dedup_ptr + token_id);
      const int32 col_id = *(values_ptr + token_id);
      const float gain = *(weights_ptr + token_id);
      gains_rescale[row_id] += combiner_scale_contribution_fn(gain);
      if (!row_ids.empty() && row_ids.back() == row_id &&
          col_ids.back() == col_id) {
        gains.back() = gains.back() + gain;
      } else {
        row_ids.push_back(row_id);
        col_ids.push_back(col_id);
        gains.push_back(gain);
      }
    }

    absl::c_transform(gains_rescale, gains_rescale.begin(),
                      combiner_scale_transform_fn);

    const int32 output_id_count = row_ids.size();

    Tensor* gains_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gains", TensorShape({output_id_count}),
                                        &gains_tensor));
    Tensor* row_ids_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("row_ids", TensorShape({output_id_count}),
                                  &row_ids_tensor));
    Tensor* col_ids_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("col_ids", TensorShape({output_id_count}),
                                  &col_ids_tensor));

    int32* row_ids_tensor_ptr = row_ids_tensor->flat<int32>().data();
    int32* col_ids_tensor_ptr = col_ids_tensor->flat<int32>().data();
    float* gains_tensor_ptr = gains_tensor->flat<float>().data();

    // Rescale the gain so that we can always do 'sum' combine on it later.
    for (int token_id = 0; token_id < output_id_count; ++token_id) {
      *(row_ids_tensor_ptr + token_id) = row_ids[token_id];
      *(col_ids_tensor_ptr + token_id) = col_ids[token_id];
      *(gains_tensor_ptr + token_id) =
          gains[token_id] * gains_rescale[row_ids[token_id]];
    }
    VLOG(1) << "Compute ConvertToCooTensorOp done";
  }

 private:
  int sample_count_ = 1;
  std::string combiner_;
};

REGISTER_KERNEL_BUILDER(Name("ConvertToCooTensor").Device(DEVICE_CPU),
                        ConvertToCooTensorOp)

GetMinibatchesInCsrWithPhysicalReplicaOp::
    GetMinibatchesInCsrWithPhysicalReplicaOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_replica", &num_replica_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_count", &sample_count_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("max_minibatches_per_sc", &max_minibatches_per_sc_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ids_per_chip_per_sample",
                                   &max_ids_per_chip_per_sample_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("table_vocab_size", &table_vocab_size_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_width", &feature_width_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_sc_per_chip", &num_sc_per_chip_));

  device_name_ = ctx->device()->name();

  OP_REQUIRES(ctx, sample_count_ % num_sc_per_chip_ == 0,
              absl::InvalidArgumentError(absl::StrCat(
                  "sample_count ", sample_count_,
                  " is not divisible by the number of sparsecores per chip ",
                  num_sc_per_chip_)));

  // Create default instance of stats handler. May get overwritten by subclass.
  sprase_core_ops_stats_handler_ =
      std::make_unique<SparseCoreOpsStatsHandler>();
}

void GetMinibatchesInCsrWithPhysicalReplicaOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "Compute GetMinibatchesInCsrWithPhysicalReplicaOp";

  const Tensor* row_ids;
  OP_REQUIRES_OK(ctx, ctx->input("row_ids", &row_ids));
  const Tensor* col_ids;
  OP_REQUIRES_OK(ctx, ctx->input("col_ids", &col_ids));
  const Tensor* gains;
  OP_REQUIRES_OK(ctx, ctx->input("gains", &gains));
  const Tensor* splits;
  OP_REQUIRES_OK(ctx, ctx->input("splits", &splits));
  const Tensor* id_counts;
  OP_REQUIRES_OK(ctx, ctx->input("id_counts", &id_counts));

  // TODO(patn): Allow clients to provide the max_ids and max_uniques directly
  // making program_key optional. This would be useful if there's a need to
  // use this op without the bridge.
  const Tensor* program_key_t;
  OP_REQUIRES_OK(ctx, ctx->input("program_key", &program_key_t));
  tstring program_key = program_key_t->vec<tstring>()(0);
  int64_t per_sparse_core_batch_size = sample_count_ / num_sc_per_chip_;

  int64_t max_ids_per_partition = -1;
  int64_t max_unique_ids_per_partition = -1;

  GetMaxIdsAndUniques(ctx, program_key, table_name_, per_sparse_core_batch_size,
                      feature_width_, &max_ids_per_partition,
                      &max_unique_ids_per_partition);

  const int32* row_ids_tensor_ptr = row_ids->flat<int32>().data();
  const int32* col_ids_tensor_ptr = col_ids->flat<int32>().data();
  const float* gains_tensor_ptr = gains->flat<float>().data();
  const int64* splits_tensor_ptr = splits->flat<int64>().data();
  const int32* id_counts_tensor_ptr = id_counts->flat<int32>().data();

  const int num_physical_replica = num_replica_ * num_sc_per_chip_;

  size_t xla_pad_size = stream_executor::tpu::OpsApiFn()
                            ->TpuUtil_GetXlaPadSizeFromTpuTopologyFn();

  OP_REQUIRES(ctx, sample_count_ % num_sc_per_chip_ == 0,
              absl::InvalidArgumentError(
                  absl::StrCat("Sample_count has to be multiply of "
                               "num_sc_per_replica which is 4, but got ",
                               sample_count_, " samples.")));

  const int max_division_level = GetMinibatchMaxDivisionLevel();

  const int32 kMaxDivisions = 1 << max_division_level;

  int64 binary_splits = 0;
  for (int i = 0; i < splits->NumElements(); ++i) {
    binary_splits |= *(splits_tensor_ptr + i);
  }

  std::vector<int> bucket_splits =
      ConvertBinarySplitsToBucketSplits(binary_splits, max_division_level);

  const int32 num_minibatch_per_sc = bucket_splits.size() + 1;
  sprase_core_ops_stats_handler_->Record(StatsType::NUM_MINIBATCHES_PER_SC,
                                         num_minibatch_per_sc, device_name_,
                                         table_name_);

  OP_REQUIRES(
      ctx, num_minibatch_per_sc <= max_minibatches_per_sc_,
      absl::InvalidArgumentError(absl::StrCat(
          "The number of minibatches per sparse core is ", num_minibatch_per_sc,
          ". But the max minibatches per sparse core is set to be ",
          max_minibatches_per_sc_, " which is smaller.")));
  VLOG(2) << "GetMinibatchesInCsrWithPhysicalReplicaOp: "
          << "program_key ='" << program_key << "'"
          << ", table_name = " << table_name_
          << ", max_ids = " << max_ids_per_partition
          << ", max_uniques = " << max_unique_ids_per_partition
          << ", num_minibatch_per_sc = " << num_minibatch_per_sc;

  const int total_num_minibatch = num_minibatch_per_sc * num_sc_per_chip_;

  bucket_splits.insert(bucket_splits.begin(), 0);
  bucket_splits.push_back(kMaxDivisions);

  const int32 max_ids_per_chip = max_ids_per_chip_per_sample_ * sample_count_;

  const int32 padded_row_pointers_size_per_sc =
      xla::RoundUpTo<int32>(num_physical_replica, xla_pad_size);

  Tensor* row_pointers_tensor;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(
                     "row_pointers",
                     TensorShape({max_minibatches_per_sc_ * num_sc_per_chip_ *
                                  padded_row_pointers_size_per_sc}),
                     &row_pointers_tensor));

  Tensor* sorted_sample_ids_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("sorted_sample_ids",
                                           TensorShape({max_ids_per_chip}),
                                           &sorted_sample_ids_tensor));
  Tensor* sorted_token_ids_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("sorted_token_ids",
                                           TensorShape({max_ids_per_chip}),
                                           &sorted_token_ids_tensor));
  Tensor* sorted_gains_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output("sorted_gains", TensorShape({max_ids_per_chip}),
                                &sorted_gains_tensor));
  int32* row_pointers_tensor_ptr = row_pointers_tensor->flat<int32>().data();
  int32* sorted_sample_ids_tensor_ptr =
      sorted_sample_ids_tensor->flat<int32>().data();
  int32* sorted_token_ids_tensor_ptr =
      sorted_token_ids_tensor->flat<int32>().data();
  float* sorted_gains_tensor_ptr = sorted_gains_tensor->flat<float>().data();

  int32 global_index = 0;
  int32 row_pointers_index = 0;
  for (int sc_id = 0; sc_id < num_sc_per_chip_; ++sc_id) {
    for (int i = 1; i < bucket_splits.size(); ++i) {
      for (int replica_id = 0; replica_id < num_physical_replica;
           ++replica_id) {
        const int global_division_id =
            sc_id * num_physical_replica + replica_id;
        const int start_division_pos =
            global_division_id * kMaxDivisions + bucket_splits[i - 1];
        const int end_division_pos =
            global_division_id * kMaxDivisions + bucket_splits[i];
        const int token_id_count = *(id_counts_tensor_ptr + end_division_pos) -
                                   *(id_counts_tensor_ptr + start_division_pos);

        const int token_id_start_pos =
            *(id_counts_tensor_ptr + start_division_pos);

        std::copy_n(col_ids_tensor_ptr + token_id_start_pos, token_id_count,
                    sorted_token_ids_tensor_ptr + global_index);
        std::copy_n(row_ids_tensor_ptr + token_id_start_pos, token_id_count,
                    sorted_sample_ids_tensor_ptr + global_index);
        std::copy_n(gains_tensor_ptr + token_id_start_pos, token_id_count,
                    sorted_gains_tensor_ptr + global_index);

        global_index += token_id_count;

        *(row_pointers_tensor_ptr + row_pointers_index) = global_index;
        int32 num_ids_to_pad_per_replica =
            xla::RoundUpTo<int32>(global_index, xla_pad_size) - global_index;
        std::fill_n(sorted_token_ids_tensor_ptr + global_index,
                    num_ids_to_pad_per_replica, kXlaPadValue);
        std::fill_n(sorted_sample_ids_tensor_ptr + global_index,
                    num_ids_to_pad_per_replica, kXlaPadValue);
        std::fill_n(sorted_gains_tensor_ptr + global_index,
                    num_ids_to_pad_per_replica, kXlaPadValue);
        global_index += num_ids_to_pad_per_replica;
        ++row_pointers_index;
      }
      // Pad the row_pointers to be memory aligned.
      int32 num_row_pointers_to_pad =
          xla::RoundUpTo<int32>(row_pointers_index, xla_pad_size) -
          row_pointers_index;
      std::fill_n(row_pointers_tensor_ptr + row_pointers_index,
                  num_row_pointers_to_pad, global_index);
      row_pointers_index += num_row_pointers_to_pad;
    }
  }

  int32 ids_unpadded_size = global_index;

  OP_REQUIRES(ctx, ids_unpadded_size <= max_ids_per_chip,
              absl::InvalidArgumentError(absl::StrCat(
                  "Got ", ids_unpadded_size,
                  " ids after padding but the max_ids_per_chip is set to be ",
                  max_ids_per_chip, " which is smaller.")));

  int32 row_pointers_unpadded_size =
      total_num_minibatch * padded_row_pointers_size_per_sc;

  Tensor* num_minibatches_per_physical_sparse_core_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(
               "num_minibatches_per_physical_sparse_core", TensorShape({}),
               &num_minibatches_per_physical_sparse_core_tensor));

  Tensor* row_pointers_unpadded_size_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output("row_pointers_unpadded_size", TensorShape({}),
                                &row_pointers_unpadded_size_tensor));

  Tensor* ids_unpadded_size_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("ids_unpadded_size", TensorShape({}),
                                           &ids_unpadded_size_tensor));

  num_minibatches_per_physical_sparse_core_tensor->flat<int32>()(0) =
      num_minibatch_per_sc;
  row_pointers_unpadded_size_tensor->flat<int32>()(0) =
      row_pointers_unpadded_size;
  ids_unpadded_size_tensor->flat<int32>()(0) = ids_unpadded_size;

  VLOG(1) << "Compute GetMinibatchesInCsrWithPhysicalReplicaOp done";
}

#ifdef LIBTPU_ON_GCE
REGISTER_KERNEL_BUILDER(
    Name("GetMinibatchesInCsrWithPhysicalReplica").Device(DEVICE_CPU),
    GetMinibatchesInCsrWithPhysicalReplicaOp)
#endif

}  // namespace tensorflow
