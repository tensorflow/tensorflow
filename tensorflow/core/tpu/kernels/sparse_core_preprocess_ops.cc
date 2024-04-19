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
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "hwy/base.h"  // from @com_google_highway
#include "hwy/contrib/sort/order.h"  // from @com_google_highway
#include "hwy/contrib/sort/vqsort.h"  // from @com_google_highway
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_stats_handler.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

namespace tensorflow {

Status ValidateInputs(const Tensor& indices_or_row_splits, const Tensor& values,
                      const Tensor& weights, int sample_count) {
  if (values.dims() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Values input should have dimension 1. But got dimension ",
                     values.dims(), "."));
  }
  switch (weights.dims()) {
    case 0:
      break;
    case 1:
      if (values.NumElements() != weights.NumElements()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Values and weights should have same elements. But got ",
            values.NumElements(), " elements for values and ",
            weights.NumElements(), " elements for weights."));
      }
      break;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Weights input should have dimension 0 or 1. But got dimension ",
          weights.dims(), "."));
  }
  // The indices_or_row_splits input for dense tensor is strictly 0 element
  // with dimension 1.
  if (indices_or_row_splits.NumElements() == 0 &&
      indices_or_row_splits.dims() == 1) {
    // Dense tensor with 0 element is also valid.
    if (values.NumElements() != 0 && values.NumElements() != sample_count) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dense tensor input should have values elements number the same as "
          "the sample count. But got ",
          values.NumElements(), " elements for values and sample count as ",
          sample_count, "."));
    }
    // 0 element indices with dimension as 2 is also valid for empty sparse
    // tensor.
  } else if (indices_or_row_splits.dims() == 2 &&
             indices_or_row_splits.NumElements() >= 0) {
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
  return absl::OkStatus();
}

Status ComputeRowIdsBeforePadding(const Tensor& indices_or_row_splits,
                                  const int32 total_id_count,
                                  int32* row_ids_before_padding) {
  // The only difference between dense tensor, sparse tensor and ragged tensor
  // is the row ids output.
  if (indices_or_row_splits.NumElements() == 0) {
    // Dense tensor to COO format.
    // Row ids are just the index ids.
    for (int32 i = 0; i < total_id_count; ++i) {
      *(row_ids_before_padding + i) = i;
    }
  } else if (indices_or_row_splits.dims() == 2 &&
             indices_or_row_splits.NumElements() > 0) {
    // Sparse tensor to COO format.
    // TODO(pineapplejuice233): should we support arbitrary rank of sparse tensor and
    // convert it to 2D?
    // For 2D sparse tensor, as we always combine on the last dimension.
    // The row ids are just the sample ids which is the first dim of the
    // indices.
    auto indices_matrix = indices_or_row_splits.matrix<int32>();
    int32 previous_row_id = -1;
    for (int32 i = 0; i < total_id_count; ++i) {
      int32 current_row_id = indices_matrix(i, 0);
      if (current_row_id < previous_row_id) {
        return absl::InvalidArgumentError(
            "Invalid indices_or_row_splits input, indices of SparseTensor need "
            "to be sorted in ascending order.");
      }
      *(row_ids_before_padding + i) = current_row_id;
    }
  } else if (indices_or_row_splits.dims() == 1 &&
             indices_or_row_splits.NumElements() > 0) {
    // Ragged tensor to COO format.
    const int32* indices_or_row_splits_ptr =
        indices_or_row_splits.flat<int32>().data();
    int32 current_row_id = -1;
    for (int32 i = 0; i < total_id_count; ++i) {
      while (i == *(indices_or_row_splits_ptr + 1 + current_row_id)) {
        current_row_id += 1;
      }
      *(row_ids_before_padding + i) = current_row_id;
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid indices_or_row_splits input, Got dimension of ",
                     indices_or_row_splits.dims(), " and size of ",
                     indices_or_row_splits.NumElements(), "."));
  }
  return absl::OkStatus();
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

    auto row_ids_before_dedup = std::make_unique<int32[]>(total_id_count);

    OP_REQUIRES_OK(
        ctx, ComputeRowIdsBeforePadding(*indices_or_row_splits, total_id_count,
                                        row_ids_before_dedup.get()));

    // Compute the rescaled gains for non-sum combiners.
    std::optional<std::vector<float>> gains_rescale =
        combiner_ != "sum"
            ? std::make_optional<std::vector<float>>(sample_count_, 0.0f)
            : std::nullopt;

    auto combiner_scale_contribution_fn =
        GetCombinerScaleContributionFunction(combiner_);

    auto combiner_scale_transform_fn =
        GetCombinerScaleTransformFunction(combiner_);

    const int32* row_ids_before_dedup_ptr = row_ids_before_dedup.get();
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

    if (weights->NumElements() == 1) {
      // Broadcast the same weight to all tokens.
      const float gain = *weights_ptr;
      const float rescaled_gain = combiner_scale_contribution_fn(gain);
      for (int token_id = 0; token_id < total_id_count; ++token_id) {
        const int32 row_id = *(row_ids_before_dedup_ptr + token_id);
        const int32 col_id = *(values_ptr + token_id);
        if (gains_rescale.has_value()) {
          // Compute the gain rescale before doing the dedup.
          (*gains_rescale)[row_id] += rescaled_gain;
        }
        if (!row_ids.empty() && row_ids.back() == row_id &&
            col_ids.back() == col_id) {
          gains.back() = gains.back() + gain;
        } else {
          row_ids.push_back(row_id);
          col_ids.push_back(col_id);
          gains.push_back(gain);
        }
      }
    } else {
      for (int token_id = 0; token_id < total_id_count; ++token_id) {
        const int32 row_id = *(row_ids_before_dedup_ptr + token_id);
        const int32 col_id = *(values_ptr + token_id);
        const float gain = *(weights_ptr + token_id);
        if (gains_rescale.has_value()) {
          // Compute the gain rescale before doing the dedup.
          (*gains_rescale)[row_id] += combiner_scale_contribution_fn(gain);
        }
        if (!row_ids.empty() && row_ids.back() == row_id &&
            col_ids.back() == col_id) {
          gains.back() = gains.back() + gain;
        } else {
          row_ids.push_back(row_id);
          col_ids.push_back(col_id);
          gains.push_back(gain);
        }
      }
    }

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

    if (gains_rescale.has_value()) {
      // Rescale the gain so that we can always do 'sum' combine on it later.
      absl::c_transform(*gains_rescale, gains_rescale->begin(),
                        combiner_scale_transform_fn);
      for (int token_id = 0; token_id < output_id_count; ++token_id) {
        *(row_ids_tensor_ptr + token_id) = row_ids[token_id];
        *(col_ids_tensor_ptr + token_id) = col_ids[token_id];
        *(gains_tensor_ptr + token_id) =
            gains[token_id] * (*gains_rescale)[row_ids[token_id]];
      }
    } else {
      std::copy(row_ids.begin(), row_ids.end(), row_ids_tensor_ptr);
      std::copy(col_ids.begin(), col_ids.end(), col_ids_tensor_ptr);
      std::copy(gains.begin(), gains.end(), gains_tensor_ptr);
    }
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
  sparse_core_ops_stats_handler_ =
      std::make_unique<SparseCoreOpsStatsHandler>();
}

void GetMinibatchesInCsrWithPhysicalReplicaOp::Compute(OpKernelContext* ctx) {
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

  OP_REQUIRES_OK(ctx, GetMaxIdsAndUniquesExternal(
                          program_key, table_name_, per_sparse_core_batch_size,
                          feature_width_, &max_ids_per_partition,
                          &max_unique_ids_per_partition));

  const int32* row_ids_tensor_ptr = row_ids->flat<int32>().data();
  const int32* col_ids_tensor_ptr = col_ids->flat<int32>().data();
  const float* gains_tensor_ptr = gains->flat<float>().data();
  const int64* splits_tensor_ptr = splits->flat<int64>().data();
  const int32* id_counts_tensor_ptr = id_counts->flat<int32>().data();

  const int32_t total_id_count = row_ids->NumElements();

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
  sparse_core_ops_stats_handler_->Record(StatsType::NUM_MINIBATCHES_PER_SC,
                                         num_minibatch_per_sc, device_name_,
                                         table_name_);

  OP_REQUIRES(
      ctx, num_minibatch_per_sc <= max_minibatches_per_sc_,
      absl::InvalidArgumentError(absl::StrCat(
          "The number of minibatches per sparse core is ", num_minibatch_per_sc,
          ". But the max minibatches per sparse core is set to be ",
          max_minibatches_per_sc_, " which is smaller.")));
  VLOG(2) << "GetMinibatchesInCsrWithPhysicalReplicaOp: "
          << "program_key = '" << program_key << "'"
          << ", table_name = '" << table_name_ << "'"
          << ", max_ids = " << max_ids_per_partition
          << ", max_uniques = " << max_unique_ids_per_partition
          << ", num_minibatch_per_sc = " << num_minibatch_per_sc;

  const int total_num_minibatch = num_minibatch_per_sc * num_sc_per_chip_;

  bucket_splits.insert(bucket_splits.begin(), 0);
  bucket_splits.push_back(kMaxDivisions);

  const int32 max_ids_per_chip = max_ids_per_chip_per_sample_ * sample_count_;

  OP_REQUIRES(
      ctx, max_ids_per_chip % xla_pad_size == 0,
      absl::InvalidArgumentError(absl::StrCat(
          "The max_ids_per_chip is set to be ", max_ids_per_chip,
          " which is not divisible by the xla_pad_size ", xla_pad_size, " .")));

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

  // This packed id count is used to track how many ids we have packed into
  // the output tensor and based on this we would know how many ids that we
  // dropped.
  int32_t packed_id_count = 0;

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

        if (global_index + token_id_count > max_ids_per_chip) {
          if (allow_id_dropping_for_minibatching_) {
            const int32_t copy_id_count =
                std::min(max_ids_per_chip - global_index, token_id_count);
            std::copy_n(col_ids_tensor_ptr + token_id_start_pos, copy_id_count,
                        sorted_token_ids_tensor_ptr + global_index);
            std::copy_n(row_ids_tensor_ptr + token_id_start_pos, copy_id_count,
                        sorted_sample_ids_tensor_ptr + global_index);
            std::copy_n(gains_tensor_ptr + token_id_start_pos, copy_id_count,
                        sorted_gains_tensor_ptr + global_index);
            packed_id_count += copy_id_count;
            global_index = max_ids_per_chip;
          } else {
            const int32_t remain_id_count = total_id_count - packed_id_count;
            ctx->CtxFailure(absl::InvalidArgumentError(absl::StrCat(
                "The max_ids_per_chip is set to be ", max_ids_per_chip,
                " which is not going to fit all ids. The remaining id count "
                "is ",
                remain_id_count,
                " . Please consider setting the "
                "sparse_core_allow_id_dropping_for_minibatching to be "
                "true. ")));
            return;
          }
        } else {
          std::copy_n(col_ids_tensor_ptr + token_id_start_pos, token_id_count,
                      sorted_token_ids_tensor_ptr + global_index);
          std::copy_n(row_ids_tensor_ptr + token_id_start_pos, token_id_count,
                      sorted_sample_ids_tensor_ptr + global_index);
          std::copy_n(gains_tensor_ptr + token_id_start_pos, token_id_count,
                      sorted_gains_tensor_ptr + global_index);

          global_index += token_id_count;
          packed_id_count += token_id_count;
        }

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

  int32_t ids_unpadded_size = global_index;

  if (packed_id_count < total_id_count) {
    const int32_t dropped_id_count = total_id_count - packed_id_count;
    LOG(WARNING) << "Dropping " << dropped_id_count
                 << " ids so that the produced CsrWrappedCooTensor can be fit "
                    "in static bound of "
                 << max_ids_per_chip
                 << " . This could potentially impact the model quality.";
  }

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
}

#ifdef LIBTPU_ON_GCE
REGISTER_KERNEL_BUILDER(
    Name("GetMinibatchesInCsrWithPhysicalReplica").Device(DEVICE_CPU),
    GetMinibatchesInCsrWithPhysicalReplicaOp)
#endif

GetMinibatchSplitsWithPhysicalReplicaOp::
    GetMinibatchSplitsWithPhysicalReplicaOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_replica", &num_replica_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_count", &sample_count_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("table_vocab_size", &table_vocab_size_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_width", &feature_width_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_sc_per_chip", &num_sc_per_chip_));
  OP_REQUIRES(ctx, sample_count_ % num_sc_per_chip_ == 0,
              absl::InvalidArgumentError(absl::StrCat(
                  "sample_count ", sample_count_,
                  " is not divisible by the number of sparsecores per chip ",
                  num_sc_per_chip_)));
  device_name_ = ctx->device()->name();

  // Create default instance of stats handler. May get overwritten by subclass.
  sparse_core_ops_stats_handler_ =
      std::make_unique<SparseCoreOpsStatsHandler>();
}

void GetMinibatchSplitsWithPhysicalReplicaOp::Compute(OpKernelContext* ctx) {
  // TODO(patn): Allow clients to provide the max_ids and max_uniques directly
  // making program_key optional. This would be useful if there's a need to
  // use this op without the bridge.
  const Tensor* program_key_t;
  OP_REQUIRES_OK(ctx, ctx->input("program_key", &program_key_t));
  tstring program_key = program_key_t->vec<tstring>()(0);

  int32 per_sc_sample_count = sample_count_ / num_sc_per_chip_;

  int64_t max_ids_per_partition = -1;
  int64_t max_unique_ids_per_partition = -1;

  OP_REQUIRES_OK(
      ctx, GetMaxIdsAndUniquesExternal(
               program_key, table_name_, per_sc_sample_count, feature_width_,
               &max_ids_per_partition, &max_unique_ids_per_partition));

  sparse_core_ops_stats_handler_->Record(StatsType::MAX_IDS_PER_PARTITION,
                                         max_ids_per_partition, device_name_,
                                         table_name_);
  sparse_core_ops_stats_handler_->Record(
      StatsType::MAX_UNIQUE_IDS_PER_PARTITION, max_unique_ids_per_partition,
      device_name_, table_name_);

  const Tensor* row_ids;
  OP_REQUIRES_OK(ctx, ctx->input("row_ids", &row_ids));
  const Tensor* col_ids;
  OP_REQUIRES_OK(ctx, ctx->input("col_ids", &col_ids));
  const Tensor* gains;
  OP_REQUIRES_OK(ctx, ctx->input("gains", &gains));

  const int32 total_id_count = row_ids->NumElements();

  const int32* row_ids_ptr = row_ids->flat<int32>().data();
  const int32* col_ids_ptr = col_ids->flat<int32>().data();
  const float* gains_ptr = gains->flat<float>().data();

#ifndef NDEBUG
  // row_ids are typically computed by ConvertToCooTensorOp, so we
  // expect them to be sorted. (It doesn't really matter whether they're
  // ascending or descending, but here, we check for the former.)
  for (int i = 1; i < total_id_count; i++) {
    OP_REQUIRES(ctx, row_ids_ptr[i - 1] <= row_ids_ptr[i],
                absl::InvalidArgumentError(
                    "row ids need to be sorted in ascending order."));
  }
#endif

  const int num_physical_replica = num_replica_ * num_sc_per_chip_;

  OP_REQUIRES(ctx, sample_count_ % num_sc_per_chip_ == 0,
              absl::InvalidArgumentError(
                  absl::StrCat("Sample_count has to be multiply of "
                               "num_sc_per_chip which is 4, but got ",
                               sample_count_, " samples.")));

  const int max_division_level = GetMinibatchMaxDivisionLevel();

  const int32 kMaxDivisions = 1 << max_division_level;

  // The id counts tensor is the running sum of the number of ids for all
  // buckets for all the replicas on each SparseCore.
  // This is used in later minibatch forming op to craft each minibatch.
  Tensor* id_counts_tensor;
  OP_REQUIRES_OK(
      ctx,
      ctx->allocate_output(
          "id_counts",
          TensorShape(
              {kMaxDivisions * num_sc_per_chip_ * num_physical_replica + 1}),
          &id_counts_tensor));
  int32* id_counts_tensor_ptr = id_counts_tensor->flat<int32>().data();
  *id_counts_tensor_ptr = 0;

  const int32_t division_size =
      (table_vocab_size_ + kMaxDivisions - 1) / kMaxDivisions;

  // Index pointers into the original row_ids/col_ids/gains arrays.
  uint32_t index = 0;
  // Splits which should be interpreted as binary format.
  // E.g. splits = 11 with table size 1024 indicates:
  //                0001011 -> 0001 01 1
  //      which mean split at level 0 section 0, level 1 section 0 and level
  //      2 section 0. the split points are [128, 256, 512].
  int64 pre_merge_splits = 0;
  int64 after_merge_splits = 0;
  // Vector of uint64_t storing the col ids in the upper 32 bit and the index
  // to the original id array in the lower 32 bit.
  std::vector<std::vector<uint64_t>> col_ids_index_list(
      num_sc_per_chip_, std::vector<uint64_t>());

  // Vector stores the mapping between the index of the id which it can be
  // deduped.
  // For example:
  //   [0, 1, 1, 1] means that third and fourth id can be deduped with the
  //   second id.
  std::vector<uint32_t> dedup_ids_index_mapping(total_id_count);

  std::vector<bool> is_id_dropped(total_id_count, false);

  // The gains after the deduplication. If the same ids are in the same
  // sample, we will remove that id and add the gains.
  std::vector<float> gains_after_dedup(total_id_count);

  // Array which stores the id counts and unique id counts for each minibatch
  // bucket on all physical replicas.
  std::vector<int32_t> total_id_counter(num_physical_replica *
                                        (kMaxDivisions + 1));
  std::vector<int32_t> total_unique_id_counter(num_physical_replica *
                                               (kMaxDivisions + 1));
  std::vector<int32_t> record_total_id_counter(num_physical_replica *
                                               (kMaxDivisions + 1));
  std::vector<int32_t> record_total_unique_id_counter(num_physical_replica *
                                                      (kMaxDivisions + 1));

  // Array which keeps track of the index of each physical replica and each
  // bucket.
  std::vector<int32_t> per_physical_replica_bucket_index(num_physical_replica *
                                                         kMaxDivisions);

  // Id counts for each sc input.
  std::vector<int32_t> per_sc_id_count(num_sc_per_chip_, 0);

  // Keep track of the maximum number of (unique) ids we see fo this current
  // batch. If it gets too close to the configured max, we can increase
  // the value in the FDO configs.
  int32_t this_max_ids = 0;
  int32_t this_max_uniques = 0;
  // Row ids(sample ids) are already sorted.
  for (int sc_id = 0; sc_id < num_sc_per_chip_; ++sc_id) {
    col_ids_index_list[sc_id].reserve(total_id_count);
    while (index < total_id_count &&
           *(row_ids_ptr + index) < (sc_id + 1) * per_sc_sample_count) {
      col_ids_index_list[sc_id].push_back(
          (static_cast<uint64_t>(*(col_ids_ptr + index)) << 32) + index);
      ++index;
    }
    // Perform high speed sorting based on col ids.
    hwy::VQSort(col_ids_index_list[sc_id].data(),
                col_ids_index_list[sc_id].size(), hwy::SortAscending());

    memset(total_id_counter.data(), 0,
           num_physical_replica * (kMaxDivisions + 1) * sizeof(int32_t));
    memset(total_unique_id_counter.data(), 0,
           num_physical_replica * (kMaxDivisions + 1) * sizeof(int32_t));
    memset(record_total_id_counter.data(), 0,
           num_physical_replica * (kMaxDivisions + 1) * sizeof(int32_t));
    memset(record_total_unique_id_counter.data(), 0,
           num_physical_replica * (kMaxDivisions + 1) * sizeof(int32_t));

    // Loop through the col ids to count the ids and unique ids.
    int32_t previous_col_id = -1;
    int32_t previous_row_id = -1;
    uint32_t previous_id_array_index = 0;
    for (uint64_t item : col_ids_index_list[sc_id]) {
      int32 col_id = item >> 32;
      uint32_t id_array_index = item & 0xffffffff;
      int32_t row_id = *(row_ids_ptr + id_array_index);
      // If the row ids and col ids are both same as the previous one,
      // dedup the id by adding the gains.
      if (row_id != previous_row_id || col_id != previous_col_id) {
        dedup_ids_index_mapping[id_array_index] = id_array_index;
        gains_after_dedup[id_array_index] = *(gains_ptr + id_array_index);
        int32_t replica_id = col_id % num_physical_replica;
        int32_t bucket_id;
        if (allow_id_shuffling_for_minibatching_) {
          bucket_id = CalculateBucketIdWithHashing(col_id, kMaxDivisions);
        } else {
          bucket_id = std::min(col_id / division_size, kMaxDivisions - 1);
        }
        uint32_t id_counter_index =
            replica_id * (kMaxDivisions + 1) + bucket_id + 1;
        record_total_id_counter[id_counter_index]++;
        if (col_id != previous_col_id)
          record_total_unique_id_counter[id_counter_index]++;

        if (allow_id_dropping_for_minibatching_ &&
            (total_id_counter[id_counter_index] == max_ids_per_partition ||
             total_unique_id_counter[id_counter_index] ==
                 max_unique_ids_per_partition)) {
          // Marking this id as not used.
          is_id_dropped[id_array_index] = true;
        } else {
          total_id_counter[id_counter_index]++;
          if (col_id != previous_col_id)
            total_unique_id_counter[id_counter_index]++;
        }
      } else {
        // Dedup the id if both row id and col id is the same.
        uint32_t parent_idx = dedup_ids_index_mapping[previous_id_array_index];
        dedup_ids_index_mapping[id_array_index] = parent_idx;
        gains_after_dedup[parent_idx] += *(gains_ptr + id_array_index);
      }
      previous_col_id = col_id;
      previous_id_array_index = id_array_index;
      previous_row_id = row_id;
    }

    for (int replica_id = 0; replica_id < num_physical_replica; ++replica_id) {
      absl::Span<int32_t> id_counter = absl::MakeSpan(
          total_id_counter.data() + replica_id * (kMaxDivisions + 1),
          kMaxDivisions + 1);
      absl::Span<int32_t> unique_id_counter = absl::MakeSpan(
          total_unique_id_counter.data() + replica_id * (kMaxDivisions + 1),
          kMaxDivisions + 1);
      absl::Span<int32_t> record_id_counter = absl::MakeSpan(
          record_total_id_counter.data() + replica_id * (kMaxDivisions + 1),
          kMaxDivisions + 1);
      absl::Span<int32_t> record_unique_id_counter =
          absl::MakeSpan(record_total_unique_id_counter.data() +
                             replica_id * (kMaxDivisions + 1),
                         kMaxDivisions + 1);
      for (int i = 1; i < kMaxDivisions + 1; ++i) {
        // Check if the smallest division is larger than the max_ids and
        // max_unique_ids.
        OP_REQUIRES(ctx,
                    id_counter[i] <= max_ids_per_partition &&
                        unique_id_counter[i] <= max_unique_ids_per_partition,
                    absl::InvalidArgumentError(absl::StrCat(
                        "Table ", table_name_, " has too many ids for replica ",
                        replica_id, " on sparse core ", sc_id,
                        ". The max_ids_per_partition is ",
                        max_ids_per_partition, " but got ", id_counter[i],
                        " ids. The max_unique_ids_per_partition is ",
                        max_unique_ids_per_partition, " but got ",
                        unique_id_counter[i], " unique ids.",
                        " Consider making the max_division_level higher.")));
        // Save the running sum of the id counts.
        const int global_division_id =
            sc_id * num_physical_replica + replica_id;
        *(id_counts_tensor_ptr + global_division_id * kMaxDivisions + i) =
            *(id_counts_tensor_ptr + global_division_id * kMaxDivisions + i -
              1) +
            id_counter[i];
        id_counter[i] += id_counter[i - 1];
        unique_id_counter[i] += unique_id_counter[i - 1];
        record_id_counter[i] += record_id_counter[i - 1];
        record_unique_id_counter[i] += record_unique_id_counter[i - 1];
      }
      this_max_ids = std::max(this_max_ids, record_id_counter[kMaxDivisions]);
      this_max_uniques =
          std::max(this_max_uniques, record_unique_id_counter[kMaxDivisions]);
      per_sc_id_count[sc_id] += id_counter[kMaxDivisions];

      for (int level = 0; level < max_division_level; ++level) {
        // Skip this level if the previous level doesn't split.
        if (level > 0 && (pre_merge_splits >> ((1LL << (level - 1)) - 1)) == 0)
          continue;
        int32_t section_size = 1 << (max_division_level - level);
        for (int section = 0; section < (1 << level); ++section) {
          // Skip this section if the corresponding section on the previous
          // level doesn't split.
          int pre_start_bit_pos = level > 0 ? (1 << (level - 1)) - 1 : 0;
          if (level > 0 && (pre_merge_splits &
                            (1LL << (pre_start_bit_pos + (section >> 1)))) == 0)
            continue;
          int32 id_count = id_counter[(section + 1) * section_size] -
                           id_counter[section * section_size];
          int32 unique_id_count =
              unique_id_counter[(section + 1) * section_size] -
              unique_id_counter[section * section_size];
          // If the number of ids or unique ids exceeds the limit, We need to
          // split.
          if (id_count > max_ids_per_partition ||
              unique_id_count > max_unique_ids_per_partition) {
            int start_bit_pos = (1 << level) - 1;
            pre_merge_splits =
                pre_merge_splits | (1LL << (start_bit_pos + section));
          }
        }
      }
      // Convert the binary representation of the splits into index of
      // buckets.
      std::vector<int> per_replica_splits = ConvertBinarySplitsToBucketSplits(
          pre_merge_splits, max_division_level);

      per_replica_splits.insert(per_replica_splits.begin(), 0);
      per_replica_splits.push_back(kMaxDivisions);

      std::vector<int> merged_per_replica_splits;
      // Iterate through all the buckets and merge them greedly.
      int start_index = 0;
      for (int i = 1; i < per_replica_splits.size(); ++i) {
        if (unique_id_counter[per_replica_splits[i]] -
                    unique_id_counter[per_replica_splits[start_index]] <=
                max_unique_ids_per_partition &&
            id_counter[per_replica_splits[i]] -
                    id_counter[per_replica_splits[start_index]] <=
                max_ids_per_partition) {
          continue;
        } else {
          merged_per_replica_splits.push_back(per_replica_splits[i - 1]);
          start_index = i - 1;
        }
      }
      // Convert the indexes of the buckets back to the binary representation.
      after_merge_splits |= ConvertBucketSplitsToBinarySplits(
          merged_per_replica_splits, max_division_level);
    }
  }

  int64_t updated_total_id_count = absl::c_accumulate(per_sc_id_count, 0);

  int64_t dropped_id_count = absl::c_accumulate(is_id_dropped, 0);
  sparse_core_ops_stats_handler_->Record(
      StatsType::DROPPED_ID_COUNT, dropped_id_count, device_name_, table_name_);

  if (dropped_id_count > 0) {
    LOG(WARNING)
        << "Table " << table_name_ << " is dropping "
        << total_id_count - updated_total_id_count
        << " ids from the input batch so that the minibatching can happen.";
  }

  Tensor* sorted_row_ids_tensor;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output("sorted_row_ids",
                                      TensorShape({updated_total_id_count}),
                                      &sorted_row_ids_tensor));
  Tensor* sorted_col_ids_tensor;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output("sorted_col_ids",
                                      TensorShape({updated_total_id_count}),
                                      &sorted_col_ids_tensor));
  Tensor* sorted_gains_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(
                          "sorted_gains", TensorShape({updated_total_id_count}),
                          &sorted_gains_tensor));

  int32_t* sorted_row_ids_tensor_ptr =
      sorted_row_ids_tensor->flat<int32_t>().data();
  int32_t* sorted_col_ids_tensor_ptr =
      sorted_col_ids_tensor->flat<int32_t>().data();
  float* sorted_gains_tensor_ptr = sorted_gains_tensor->flat<float>().data();

  for (int sc_id = 0; sc_id < num_sc_per_chip_; ++sc_id) {
    memset(per_physical_replica_bucket_index.data(), 0,
           num_physical_replica * kMaxDivisions * sizeof(int32_t));
    for (uint64_t item : col_ids_index_list[sc_id]) {
      uint32_t id_array_index = item & 0xffffffff;
      // Skip deduped ids.
      if (is_id_dropped[id_array_index] ||
          id_array_index != dedup_ids_index_mapping[id_array_index]) {
        continue;
      }
      int32_t col_id = item >> 32;
      int32_t replica_id = col_id % num_physical_replica;
      int32_t bucket_id;
      int32_t main_index;
      if (allow_id_shuffling_for_minibatching_) {
        bucket_id = CalculateBucketIdWithHashing(col_id, kMaxDivisions);
      } else {
        bucket_id = std::min(col_id / division_size, kMaxDivisions - 1);
      }
      main_index =
          per_physical_replica_bucket_index[replica_id * kMaxDivisions +
                                            bucket_id] +
          *(id_counts_tensor_ptr +
            (sc_id * num_physical_replica + replica_id) * kMaxDivisions +
            bucket_id);
      ++per_physical_replica_bucket_index[replica_id * kMaxDivisions +
                                          bucket_id];
      *(sorted_row_ids_tensor_ptr + main_index) =
          *(row_ids_ptr + id_array_index) % per_sc_sample_count;
      *(sorted_col_ids_tensor_ptr + main_index) = col_id / num_physical_replica;
      // Use the updated gains instead.
      *(sorted_gains_tensor_ptr + main_index) =
          gains_after_dedup[id_array_index];
    }
  }

  sparse_core_ops_stats_handler_->Record(
      StatsType::IDS_PER_PARTITION, this_max_ids, device_name_, table_name_);
  sparse_core_ops_stats_handler_->Record(StatsType::UNIQUE_IDS_PER_PARTITION,
                                         this_max_uniques, device_name_,
                                         table_name_);

  CalculateHeadroom(this_max_ids, this_max_uniques, program_key,
                    max_ids_per_partition, max_unique_ids_per_partition,
                    dropped_id_count);

  Tensor* splits_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output("splits", TensorShape({}), &splits_tensor));
  splits_tensor->flat<int64>()(0) = after_merge_splits;

  Tensor* max_ids_tensor;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output("max_ids", TensorShape({}), &max_ids_tensor));
  max_ids_tensor->flat<int32>()(0) = this_max_ids;

  Tensor* max_uniques_tensor;
  OP_REQUIRES_OK(ctx, ctx->allocate_output("max_uniques", TensorShape({}),
                                           &max_uniques_tensor));
  max_uniques_tensor->flat<int32>()(0) = this_max_uniques;
}

#ifdef LIBTPU_ON_GCE
REGISTER_KERNEL_BUILDER(
    Name("GetMinibatchSplitsWithPhysicalReplica").Device(DEVICE_CPU),
    GetMinibatchSplitsWithPhysicalReplicaOp)
#endif

StoreMinibatchStatisticsInFdoOp::StoreMinibatchStatisticsInFdoOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_replica", &num_replica_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_count", &sample_count_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_width", &feature_width_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("num_sc_per_chip", &num_sc_per_chip_));
  OP_REQUIRES(ctx, sample_count_ % num_sc_per_chip_ == 0,
              absl::InvalidArgumentError(absl::StrCat(
                  "sample_count ", sample_count_,
                  " is not divisible by the number of sparsecores per chip ",
                  num_sc_per_chip_)));
  device_name_ = ctx->device()->name();
}

void StoreMinibatchStatisticsInFdoOp::Compute(OpKernelContext* ctx) {
  const Tensor* program_key_t;
  OP_REQUIRES_OK(ctx, ctx->input("program_key", &program_key_t));
  tstring program_key = program_key_t->vec<tstring>()(0);

  const Tensor* max_ids_t;
  OP_REQUIRES_OK(ctx, ctx->input("max_ids", &max_ids_t));
  int64_t max_ids = max_ids_t->scalar<int64>()();
  const Tensor* max_uniques_t;
  OP_REQUIRES_OK(ctx, ctx->input("max_uniques", &max_uniques_t));
  int64_t max_uniques = max_uniques_t->scalar<int64>()();

  int32 per_sc_sample_count = sample_count_ / num_sc_per_chip_;

  int64_t max_ids_per_partition = -1;
  int64_t max_unique_ids_per_partition = -1;

  OP_REQUIRES_OK(
      ctx, GetMaxIdsAndUniquesExternal(
               program_key, table_name_, per_sc_sample_count, feature_width_,
               &max_ids_per_partition, &max_unique_ids_per_partition));

  CalculateHeadroom(max_ids, max_uniques, program_key, max_ids_per_partition,
                    max_unique_ids_per_partition);
}

#ifdef LIBTPU_ON_GCE
REGISTER_KERNEL_BUILDER(
    Name("StoreMinibatchStatisticsInFdo").Device(DEVICE_CPU),
    StoreMinibatchStatisticsInFdoOp)
#endif

}  // namespace tensorflow
