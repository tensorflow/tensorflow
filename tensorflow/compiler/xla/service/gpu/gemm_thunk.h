/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_THUNK_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/matmul_util.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace xla {
namespace gpu {

// This is thread-compatible.
class GemmThunk : public Thunk {
 public:
  // Constructs a thunk that computes "output = (lhs <dot> rhs) * alpha" using
  // BLAS gemm (alpha is stored in the instruction GemmBackendConfig).
  GemmThunk(ThunkInfo thunk_info, GemmConfig config,
            const BufferAllocation::Slice& lhs_buffer,
            const BufferAllocation::Slice& rhs_buffer,
            const BufferAllocation::Slice& output_buffer);

  GemmThunk(const GemmThunk&) = delete;
  GemmThunk& operator=(const GemmThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const GemmConfig config_;
  const BufferAllocation::Slice lhs_buffer_;
  const BufferAllocation::Slice rhs_buffer_;
  const BufferAllocation::Slice output_buffer_;
};

// Run the given GEMM instruction `gemm` subject to the configuration
// in `gemm_config` and the passed buffers.
//
// If `algorithm` is provided, it overrides the one specified in `config`.
Status RunGemm(
    const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    se::Stream* stream, se::ScratchAllocator* scratch_allocator,
    se::blas::IBlasLtMatmulAlgorithm* const algorithm_being_profiled,
    se::blas::ProfileResult* profile_result = nullptr,
    absl::optional<se::blas::AlgorithmType> algorithm = absl::nullopt);

// A class for storing and retrieving algorithms in cublasLT autotuning
class BlasPlansAutotuneCache {
 public:
  BlasPlansAutotuneCache() {}
  bool Find(const se::BatchMatmulParameters& params,
            se::blas::AlgorithmConfig* config) const;
  void Insert(const se::BatchMatmulParameters& params,
              const se::blas::AlgorithmConfig& config);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<se::BatchMatmulParameters, se::blas::AlgorithmConfig>
      blas_plans_algorithms_map_ ABSL_GUARDED_BY(mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(BlasPlansAutotuneCache);
};

struct BlasPlansAutotuneCacheSingleton {
  static BlasPlansAutotuneCache* GetInstance() {
    static BlasPlansAutotuneCache* instance = new BlasPlansAutotuneCache();
    return instance;
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_THUNK_H_
