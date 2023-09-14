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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_stats_handler.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

namespace tensorflow {

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
