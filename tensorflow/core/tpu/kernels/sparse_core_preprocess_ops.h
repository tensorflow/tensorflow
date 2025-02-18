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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_PREPROCESS_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_PREPROCESS_OPS_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/kernels/sparse_core_ops_stats_handler.h"

namespace tensorflow {

// Struct to describe an embedding lookup input data.
struct EmbeddingLookupInput {
  // Which replica it belongs.
  int32 replica_id;
  // Token id.
  int32 token_id;
  // Sample id.
  int32 sample_id;
  // Gain.
  float gain;

  EmbeddingLookupInput(int32 replica_id, int32 token_id, int32 sample_id,
                       float gain)
      : replica_id(replica_id),
        token_id(token_id),
        sample_id(sample_id),
        gain(gain) {}
};

absl::Status ValidateInputs(const Tensor& indices_or_row_splits,
                            const Tensor& values, const Tensor& weights,
                            int sample_count);

// Compute the row id list before padding.
absl::Status ComputeRowIdsBeforePadding(const Tensor& indices_or_row_splits,
                                        int32 total_id_count,
                                        int32 sample_count,
                                        int32* row_ids_before_padding);

class GetMinibatchesInCsrWithPhysicalReplicaOp : public OpKernel {
 public:
  explicit GetMinibatchesInCsrWithPhysicalReplicaOp(OpKernelConstruction* ctx);
  ~GetMinibatchesInCsrWithPhysicalReplicaOp() override = default;
  GetMinibatchesInCsrWithPhysicalReplicaOp(
      const GetMinibatchesInCsrWithPhysicalReplicaOp&) = delete;
  GetMinibatchesInCsrWithPhysicalReplicaOp& operator=(
      const GetMinibatchesInCsrWithPhysicalReplicaOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 protected:
  int sample_count_ = 1;
  int feature_width_ = 1;
  int64_t num_sc_per_chip_;
  std::string table_name_;
  std::unique_ptr<SparseCoreOpsStatsHandler> sparse_core_ops_stats_handler_;

  bool allow_id_dropping_for_minibatching_ = false;

 private:
  int num_replica_ = 1;
  int max_minibatches_per_sc_ = 1;
  int max_ids_per_chip_per_sample_ = 1;
  int table_vocab_size_ = 1;
  std::string device_name_;
};

class GetMinibatchSplitsWithPhysicalReplicaOp : public OpKernel {
 public:
  explicit GetMinibatchSplitsWithPhysicalReplicaOp(OpKernelConstruction* ctx);
  ~GetMinibatchSplitsWithPhysicalReplicaOp() override = default;
  GetMinibatchSplitsWithPhysicalReplicaOp(
      const GetMinibatchSplitsWithPhysicalReplicaOp&) = delete;
  GetMinibatchSplitsWithPhysicalReplicaOp& operator=(
      const GetMinibatchSplitsWithPhysicalReplicaOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual void CalculateHeadroom(int32 this_max_ids, int32 this_max_uniques,
                                 tstring program_key,
                                 int64_t max_ids_per_partition,
                                 int64_t max_unique_ids_per_partition,
                                 int32_t dropped_id_count) {}
  virtual inline int32_t CalculateBucketIdWithHashing(int32_t col_id,
                                                      int32_t num_buckets) {
    // TODO(pineapplejuice233): Add a proper hashing function here.
    return col_id % num_buckets;
  }

  std::string device_name_;
  std::string table_name_;
  std::unique_ptr<SparseCoreOpsStatsHandler> sparse_core_ops_stats_handler_;
  bool allow_id_dropping_for_minibatching_ = false;
  bool allow_id_shuffling_for_minibatching_ = false;

 private:
  int num_replica_ = 1;
  int sample_count_ = 1;
  int table_vocab_size_ = 1;
  int feature_width_ = 1;
  int64_t num_sc_per_chip_;
};

class StoreMinibatchStatisticsInFdoOp : public OpKernel {
 public:
  explicit StoreMinibatchStatisticsInFdoOp(OpKernelConstruction* ctx);
  ~StoreMinibatchStatisticsInFdoOp() override = default;
  StoreMinibatchStatisticsInFdoOp(const StoreMinibatchStatisticsInFdoOp&) =
      delete;
  StoreMinibatchStatisticsInFdoOp& operator=(
      const StoreMinibatchStatisticsInFdoOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual void CalculateHeadroom(int32 this_max_ids, int32 this_max_uniques,
                                 tstring program_key,
                                 int64_t max_ids_per_partition,
                                 int64_t max_unique_ids_per_partition) {}
  std::string device_name_;
  std::string table_name_;

 private:
  int num_replica_ = 1;
  int sample_count_ = 1;
  int feature_width_ = 1;
  int64_t num_sc_per_chip_;
};

// TODO(pineapplejuice233): Unify this op with ConvertToListOfCooTensorsV2Op.
class ConvertToListOfSparseCoreCooTensorsOp : public OpKernel {
 public:
  explicit ConvertToListOfSparseCoreCooTensorsOp(OpKernelConstruction* ctx);
  ~ConvertToListOfSparseCoreCooTensorsOp() override = default;
  ConvertToListOfSparseCoreCooTensorsOp(
      const ConvertToListOfSparseCoreCooTensorsOp&) = delete;
  ConvertToListOfSparseCoreCooTensorsOp& operator=(
      const ConvertToListOfSparseCoreCooTensorsOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  void WriteToOutputTensor(int32* row_ids, int32* col_ids, float* gains,
                           int32* row_ids_tensor_ptr, int32* col_ids_tensor_ptr,
                           float* gains_tensor_ptr, int32_t begin_index,
                           int32_t end_index, int32_t sc_id,
                           std::optional<std::vector<float>> gains_rescale);
  int sample_count_;
  int num_sc_per_chip_;
  int per_sc_sample_count_;
  int row_offset_;
  int col_offset_;
  int col_shift_;
  int num_sc_shards_;
  int stacked_table_sample_count_;
  int num_sc_shards_bit_mod_;
  int num_sc_shards_bit_mod_inv_;
  int per_sc_row_offset_;
  int per_sc_stacked_table_sample_count_;
  std::string combiner_;
};

class SortListOfSparseCoreCooTensorsOp : public OpKernel {
 public:
  explicit SortListOfSparseCoreCooTensorsOp(OpKernelConstruction* ctx);
  ~SortListOfSparseCoreCooTensorsOp() override = default;
  SortListOfSparseCoreCooTensorsOp(const SortListOfSparseCoreCooTensorsOp&) =
      delete;
  SortListOfSparseCoreCooTensorsOp& operator=(
      const SortListOfSparseCoreCooTensorsOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  int32_t num_sc_per_chip_;
  int32_t feature_width_;
  int32_t num_replica_;
  int32_t num_physical_replica_;
  int32_t num_physical_replica_bit_;
  int32_t max_ids_per_sparse_core_;
  int32_t max_unique_ids_per_sparse_core_;
  std::string table_name_;
  std::vector<int32_t> sample_count_list_;
  std::vector<int32_t> col_offset_list_;
  std::map<int32_t, std::vector<int32_t>> col_offset_to_feature_id_;
};

class ConvertToSparseCoreCsrWrappedCooTensorOp : public OpKernel {
 public:
  explicit ConvertToSparseCoreCsrWrappedCooTensorOp(OpKernelConstruction* ctx);
  ~ConvertToSparseCoreCsrWrappedCooTensorOp() override = default;
  ConvertToSparseCoreCsrWrappedCooTensorOp(
      const ConvertToSparseCoreCsrWrappedCooTensorOp&) = delete;
  ConvertToSparseCoreCsrWrappedCooTensorOp& operator=(
      const ConvertToSparseCoreCsrWrappedCooTensorOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  int32_t num_sc_per_chip_;
  int32_t table_vocab_size_;
  int32_t feature_width_;
  int32_t num_replica_;
  int32_t sample_count_per_sc_;
  int32_t max_minibatches_per_sc_;
  int32_t max_ids_per_chip_per_sample_;
  bool allow_id_dropping_;
  std::string table_name_;
  std::string device_name_;
};

class GetStatsFromListOfSparseCoreCooTensorsOp : public OpKernel {
 public:
  explicit GetStatsFromListOfSparseCoreCooTensorsOp(OpKernelConstruction* ctx);
  ~GetStatsFromListOfSparseCoreCooTensorsOp() override = default;
  GetStatsFromListOfSparseCoreCooTensorsOp(
      const GetStatsFromListOfSparseCoreCooTensorsOp&) = delete;
  GetStatsFromListOfSparseCoreCooTensorsOp& operator=(
      const GetStatsFromListOfSparseCoreCooTensorsOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  int32_t num_sc_per_chip_;
  int32_t feature_width_;
  int32_t num_replica_;
  int32_t num_physical_replica_;
  int32_t num_physical_replica_bit_;
  std::string table_name_;
  std::vector<int32_t> sample_count_list_;
  std::vector<int32_t> col_offset_list_;
  std::map<int32_t, std::vector<int32_t>> col_offset_to_feature_id_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_PREPROCESS_OPS_H_
