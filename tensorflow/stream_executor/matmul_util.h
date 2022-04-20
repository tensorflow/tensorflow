/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_STREAM_EXECUTOR_MATMUL_UTIL_H_
#define TENSORFLOW_STREAM_EXECUTOR_MATMUL_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/stream_executor/blas.h"

namespace stream_executor {
template <typename T>
DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  DeviceMemory<T> typed(wrapped);
  return typed;
}

// Reads the maximum number of algorithms for GEMM autotuning from the
// environment variable TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS. If no value is set,
// return the default value.
int MatmulMaxAutotuneAlgorithmCount();

// Get a workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64_t GetWorkspaceLimit(int64_t default_value_in_bytes);

// Encapsulates information which defines a unique
// batched matmul operation.
class BatchMatmulParameters {
 public:
  BatchMatmulParameters(bool trans_a, bool trans_b, bool adj_a, bool adj_b,
                        uint64_t m, uint64_t n, uint64_t k,
                        uint64_t batch_count, bool broadcast_a,
                        bool broadcast_b, tensorflow::DataType dtype_ab,
                        tensorflow::DataType dtype_cd, int device_id,
                        blas::Epilogue epilog = blas::Epilogue::kDefault)
      : trans_a_(trans_a),
        trans_b_(trans_b),
        adj_a_(adj_a),
        adj_b_(adj_b),
        m_(m),
        n_(n),
        k_(k),
        batch_count_(batch_count),
        broadcast_a_(broadcast_a),
        broadcast_b_(broadcast_b),
        dtype_ab_(dtype_ab),
        dtype_cd_(dtype_cd),
        device_id_(device_id),
        epilog_(epilog) {
    allow_tf32_ = tensorflow::tensor_float_32_execution_enabled();
    hash_code_ = trans_a;
    hash_code_ = tensorflow::Hash64Combine(hash_code_, trans_b);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, adj_a);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, adj_b);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, m);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, n);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, k);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, batch_count);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, broadcast_a);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, broadcast_b);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, dtype_ab);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, dtype_cd);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, allow_tf32_);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, device_id);
  }

  bool operator==(const BatchMatmulParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const BatchMatmulParameters& other) const {
    return !(*this == other);
  }
  uint64_t hash() const { return hash_code_; }

  std::string ToString() const {
    // clang-format off
    return absl::StrCat(
        trans_a_, ", ", trans_b_, ", ", adj_a_, ", ", adj_b_, ", ",
        m_, ", ", n_, ", ", k_, ", ", batch_count_, ", ",
        broadcast_a_, ", ", broadcast_b_, ", ",
        dtype_ab_, ", ", dtype_cd_, ", ", allow_tf32_, ", ", device_id_, ", ",
        epilog_);
    // clang-format on
  }

  template <typename H>
  friend H AbslHashValue(H h, const BatchMatmulParameters& bmp) {
    return H::combine(std::move(h), bmp.trans_a_, bmp.trans_b_, bmp.adj_a_,
                      bmp.adj_b_, bmp.m_, bmp.n_, bmp.k_, bmp.batch_count_,
                      bmp.broadcast_a_, bmp.broadcast_b_, bmp.dtype_ab_,
                      bmp.dtype_cd_, bmp.allow_tf32_, bmp.device_id_,
                      bmp.epilog_);
  }

  blas::Epilogue GetEpilogOp() const { return epilog_; }

 private:
  typedef std::tuple<bool, bool, bool, bool, int64_t, int64_t, int64_t, int64_t,
                     bool, bool, tensorflow::DataType, tensorflow::DataType,
                     bool, int, blas::Epilogue>
      ParameterDataType;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(trans_a_, trans_b_, adj_a_, adj_b_, m_, n_, k_,
                           batch_count_, broadcast_a_, broadcast_b_, dtype_ab_,
                           dtype_cd_, allow_tf32_, device_id_, epilog_);
  }

  bool trans_a_;
  bool trans_b_;
  bool adj_a_;
  bool adj_b_;
  uint64_t m_;
  uint64_t n_;
  uint64_t k_;
  uint64_t batch_count_;
  bool broadcast_a_;
  bool broadcast_b_;
  tensorflow::DataType dtype_ab_;
  tensorflow::DataType dtype_cd_;
  bool allow_tf32_;
  int device_id_;
  blas::Epilogue epilog_;
  uint64_t hash_code_;
};

// Thread-safe map from matmul parameters to their corresponding plan and
// algorithms.
class BlasLtMatmulPlanMap {
 public:
  const blas::PlanAndAlgorithms* Find(
      const BatchMatmulParameters& params) const {
    absl::MutexLock lock(&mu_);
    auto iter = params_plan_map_.find(params);
    if (iter == params_plan_map_.end()) {
      return nullptr;
    }
    return &iter->second;
  }
  const blas::PlanAndAlgorithms* Insert(const BatchMatmulParameters& params,
                                        blas::PlanAndAlgorithms value) {
    absl::MutexLock lock(&mu_);
    return &params_plan_map_.emplace(params, std::move(value)).first->second;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<BatchMatmulParameters, blas::PlanAndAlgorithms>
      params_plan_map_ ABSL_GUARDED_BY(mu_);
};

struct BatchMatmulPlanMapSingleton {
  static BlasLtMatmulPlanMap* GetInstance() {
    static BlasLtMatmulPlanMap* instance = new BlasLtMatmulPlanMap();
    return instance;
  }
};

port::StatusOr<blas::ComputationType> GetBlasComputationType(
    const tensorflow::DataType& dtype);

port::StatusOr<const blas::PlanAndAlgorithms*> GetPlanAndAlgorithms(
    Stream* stream, BatchMatmulParameters matmul_parameters, int64_t batch_size,
    tensorflow::DataType dtype, blas::MatrixDescriptor lhs_matrix,
    blas::MatrixDescriptor rhs_matrix, blas::MatrixDescriptor output_matrix);

port::StatusOr<blas::BlasLtMatmulPlanParams> CreatePlanParams(
    int64_t batch_size, tensorflow::DataType dtype, blas::Epilogue epilog,
    blas::MatrixDescriptor lhs_matrix, blas::MatrixDescriptor rhs_matrix,
    blas::MatrixDescriptor output_matrix);


}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_MATMUL_UTIL_H_
