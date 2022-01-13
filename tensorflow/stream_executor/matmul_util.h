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
#include <vector>

#include "tensorflow/core/kernels/gpu_utils.h"
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

using BlasScratchAllocator = tensorflow::GpuScratchAllocator;

static inline int64_t GetBlasWorkspaceLimit(const string& envvar_in_mb,
                                            int64_t default_value_in_bytes) {
  return tensorflow::GetWorkspaceLimit(envvar_in_mb, default_value_in_bytes);
}

// Encapsulate all of the shape, dtype etc. information that defines a unique
// batched matmul operation.
class BatchMatmulParameters {
 public:
  BatchMatmulParameters(bool trans_a, bool trans_b, bool adj_a, bool adj_b,
                        uint64 m, uint64 n, uint64 k, uint64 batch_count,
                        bool broadcast_a, bool broadcast_b,
                        tensorflow::DataType dtype_ab,
                        tensorflow::DataType dtype_cd, bool allow_tf32,
                        int device_id)
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
        allow_tf32_(allow_tf32),
        device_id_(device_id) {
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
    hash_code_ = tensorflow::Hash64Combine(hash_code_, allow_tf32);
    hash_code_ = tensorflow::Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const BatchMatmulParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const BatchMatmulParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    // clang-format off
    return tensorflow::strings::StrCat(
        trans_a_, ", ", trans_b_, ", ", adj_a_, ", ", adj_b_, ", ",
        m_, ", ", n_, ", ", k_, ", ", batch_count_, ", ",
        broadcast_a_, ", ", broadcast_b_, ", ",
        dtype_ab_, ", ", dtype_cd_, ", ", allow_tf32_, ", ", device_id_);
    // clang-format on
  }

 private:
  typedef std::tuple<bool, bool, bool, bool, int64_t, int64_t, int64_t, int64_t,
                     bool, bool, tensorflow::DataType, tensorflow::DataType,
                     bool, int>
      ParameterDataType;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(trans_a_, trans_b_, adj_a_, adj_b_, m_, n_, k_,
                           batch_count_, broadcast_a_, broadcast_b_, dtype_ab_,
                           dtype_cd_, allow_tf32_, device_id_);
  }

  bool trans_a_;
  bool trans_b_;
  bool adj_a_;
  bool adj_b_;
  uint64 m_;
  uint64 n_;
  uint64 k_;
  uint64 batch_count_;
  bool broadcast_a_;
  bool broadcast_b_;
  tensorflow::DataType dtype_ab_;
  tensorflow::DataType dtype_cd_;
  bool allow_tf32_;
  int device_id_;
  uint64 hash_code_;
};

static inline bool GetBlasComputationType(const tensorflow::DataType& dtype,
                                          bool allow_tf32,
                                          blas::ComputationType* compute_type) {
  using blas::ComputationType;
  static bool use_f32_for_f16_computation =
      tensorflow::MatmulDoFP32ComputationFP16Input();
  ComputationType f32_type =
      allow_tf32 ? ComputationType::kTF32AsF32 : ComputationType::kF32;
  switch (dtype) {
    case tensorflow::DT_HALF:
    case tensorflow::DT_BFLOAT16:
      *compute_type =
          use_f32_for_f16_computation ? f32_type : ComputationType::kF16;
      return true;
    case tensorflow::DT_FLOAT:
      *compute_type = f32_type;
      return true;
    case tensorflow::DT_DOUBLE:
      *compute_type = ComputationType::kF64;
      return true;
    case tensorflow::DT_COMPLEX64:
      *compute_type = f32_type;
      return true;
    case tensorflow::DT_COMPLEX128:
      *compute_type = ComputationType::kComplexF64;
      return true;
    default:
      // Unsupported compute_type, return false.
      return false;
  }
}

// Thread-safe map from matmul parameters to their corresponding plan and
// algorithms.
template <typename Parameters>
class BlasLtMatmulPlanMap {
 public:
  const blas::PlanAndAlgorithms* Find(const Parameters& params) {
    tensorflow::mutex_lock lock(mu_);
    auto iter = params_plan_map_.find(params);
    if (iter == params_plan_map_.end()) {
      return nullptr;
    }
    return &iter->second;
  }
  const blas::PlanAndAlgorithms* Insert(const Parameters& params,
                                        blas::PlanAndAlgorithms value) {
    tensorflow::mutex_lock lock(mu_);
    return &params_plan_map_.emplace(params, std::move(value)).first->second;
  }

 private:
  struct Hasher {
    std::size_t operator()(const Parameters& parameter) const {
      return parameter.hash();
    }
  };

  mutable tensorflow::mutex mu_;
  std::unordered_map<Parameters, blas::PlanAndAlgorithms, Hasher>
      params_plan_map_ GUARDED_BY(mu_);
};

template <typename Parameters>
struct BlasLtPlanMapSingleton {
  typedef BlasLtMatmulPlanMap<Parameters> PlanMapType;
  static PlanMapType* GetInstance() {
    static PlanMapType* instance = new PlanMapType();
    return instance;
  }
};

typedef BlasLtPlanMapSingleton<BatchMatmulParameters>
    BatchMatmulPlanMapSingleton;

// A dummy type to group matmul autotune results together.
struct BatchMatmulAutoTuneGroup {
  static string name() { return "MatmulLt"; }
};

typedef tensorflow::AutotuneSingleton<
    BatchMatmulAutoTuneGroup, BatchMatmulParameters, blas::AlgorithmConfig>
    AutoTuneBatchMatmul;

template <typename Scalar>
struct CoefficientType {
  typedef Scalar type;
};
template <>
struct CoefficientType<Eigen::half> {
  typedef float type;
};

inline tensorflow::Status FromExecutorStatus(const port::Status& s) {
  return s.ok() ? tensorflow::Status::OK()
                : tensorflow::Status(static_cast<tensorflow::error::Code>(
                                         static_cast<int>(s.code())),
                                     s.error_message());
}

template <typename T>
inline tensorflow::Status FromExecutorStatus(const port::StatusOr<T>& s) {
  return FromExecutorStatus(s.status());
}

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_MATMUL_UTIL_H_
