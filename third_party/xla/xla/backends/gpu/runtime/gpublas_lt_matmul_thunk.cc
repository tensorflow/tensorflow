/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
<<<<<<< HEAD
#include "xla/service/gpu/autotuning/autotuner_util.h"
=======
#include "xla/service/gpu/stream_executor_util.h"
>>>>>>> upstream/master
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

<<<<<<< HEAD
struct MatmulPlanCache {

  static MatmulPlanCache& i(const se::Stream *stream) {
    static absl::Mutex m(absl::kConstInit);
    // Each GPU gets different cache instance
    static std::vector< std::unique_ptr< MatmulPlanCache > > meta(8);
    absl::MutexLock lock(&m);
    size_t dev_id = stream->parent()->device_ordinal();
    if (dev_id >= meta.size()) meta.resize(dev_id + 1);
    auto& res = meta[dev_id];
    if (!res) res.reset(new MatmulPlanCache());
    return *res;
  }

  template < class Func >
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan *> 
          GetOrCreate(const std::string& key, Func&& create) {
    // each GPU has a different mutex => hence different GPU instances can
    // create matmul plans in parallel
    absl::MutexLock lock(mutex_.get()); 
    auto res = map_.emplace(key, se::gpu::BlasLt::MatmulPlanPtr{});
    if(res.second) { // new entry inserted
      TF_ASSIGN_OR_RETURN(res.first->second, create());
    } 
    return res.first->second.get();
  }

private:
  MatmulPlanCache() : mutex_(std::make_unique< absl::Mutex >()) { }

private:
  std::unique_ptr< absl::Mutex > mutex_;
  absl::flat_hash_map<std::string, se::gpu::BlasLt::MatmulPlanPtr> map_;
};


CublasLtMatmulThunk::CublasLtMatmulThunk(
    const HloInstruction *instr, GemmConfig gemm_config,
=======
CublasLtMatmulThunk::CublasLtMatmulThunk(const CublasLtMatmulThunk& rhs)
    : Thunk(Kind::kCublasLtMatmul, {}),
      gemm_config_(rhs.gemm_config_),
      epilogue_(rhs.epilogue_),
      algorithm_idx_(rhs.algorithm_idx_),
      canonical_hlo_(rhs.canonical_hlo_),
      a_(rhs.a_),
      b_(rhs.b_),
      c_(rhs.c_),
      d_(rhs.d_),
      bias_(rhs.bias_),
      aux_(rhs.aux_),
      a_scale_(rhs.a_scale_),
      b_scale_(rhs.b_scale_),
      c_scale_(rhs.c_scale_),
      d_scale_(rhs.d_scale_),
      d_amax_(rhs.d_amax_),
      workspace_(rhs.workspace_) {}

CublasLtMatmulThunk::CublasLtMatmulThunk(
    const HloInstruction* instr, GemmConfig gemm_config,
>>>>>>> upstream/master
    se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
    BufferAllocation::Slice a, BufferAllocation::Slice b,
    BufferAllocation::Slice c, BufferAllocation::Slice d,
    BufferAllocation::Slice bias, BufferAllocation::Slice aux,
    BufferAllocation::Slice a_scale, BufferAllocation::Slice b_scale,
    BufferAllocation::Slice c_scale, BufferAllocation::Slice d_scale,
    BufferAllocation::Slice d_amax,
<<<<<<< HEAD
    std::optional<const BufferAllocation::Slice> workspace_buffer)
    : Thunk(Kind::kCublasLtMatmul, Thunk::ThunkInfo::WithProfileAnnotation(instr)),
      gemm_config_(std::move(gemm_config)),
      epilogue_(epilogue),
      algorithm_idx_(algorithm_idx),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      c_buffer_(c_buffer),
      d_buffer_(d_buffer),
      bias_buffer_(bias_buffer),
      aux_buffer_(aux_buffer),
      a_scale_buffer_(a_scale),
      b_scale_buffer_(b_scale),
      c_scale_buffer_(c_scale),
      d_scale_buffer_(d_scale),
      d_amax_buffer_(d_amax),
      workspace_buffer_(workspace_buffer) {

  canonical_hlo_ = xla::gpu::AutotuneCacheKey("nope", *instr).GetHlo();
}

absl::Status CublasLtMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {

  TF_ASSIGN_OR_RETURN(auto *plan, GetCachedMatmulPlan(params));

  VLOG(2) << params.stream->parent()->device_ordinal() << 
          ": cublas_lt_matmul for: " << canonical_hlo_;
=======
    std::optional<const BufferAllocation::Slice> workspace)
    : Thunk(Kind::kCublasLtMatmul,
            instr ? Thunk::ThunkInfo::WithProfileAnnotation(instr)
                  : Thunk::ThunkInfo{}),
      gemm_config_(std::move(gemm_config)),
      epilogue_(epilogue),
      algorithm_idx_(algorithm_idx),
      a_(a),
      b_(b),
      c_(c),
      d_(d),
      bias_(bias),
      aux_(aux),
      a_scale_(a_scale),
      b_scale_(b_scale),
      c_scale_(c_scale),
      d_scale_(d_scale),
      d_amax_(d_amax),
      workspace_(workspace) {
  // The tests creating CublasLtMatmulThunk directly might not provide the
  // pointer to the actual instruction, in this case Matmul plans are not
  // cached.
  if (instr != nullptr) {
    canonical_hlo_ = xla::gpu::AutotuneCacheKey("unused", *instr).GetHlo();
  }
}

absl::Status CublasLtMatmulThunk::ExecuteOnStreamInternal(
    se::Stream* stream, const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(auto* plan, GetCachedMatmulPlan(params));
>>>>>>> upstream/master

  const BufferAllocations& allocs = *params.buffer_allocations;

  se::DeviceMemoryBase bias, a_scale, b_scale, c_scale, d_scale, d_amax, aux,
      workspace;
  if (bias_.allocation() != nullptr) {
    bias = allocs.GetDeviceAddress(bias_);
  }
  if (a_scale_.allocation() != nullptr) {
    a_scale = allocs.GetDeviceAddress(a_scale_);
  }
  if (b_scale_.allocation() != nullptr) {
    b_scale = allocs.GetDeviceAddress(b_scale_);
  }
  if (c_scale_.allocation() != nullptr) {
    c_scale = allocs.GetDeviceAddress(c_scale_);
  }
  if (d_scale_.allocation() != nullptr) {
    d_scale = allocs.GetDeviceAddress(d_scale_);
  }
  if (d_amax_.allocation() != nullptr) {
    d_amax = allocs.GetDeviceAddress(d_amax_);
  }
  if (aux_.allocation() != nullptr) {
    aux = allocs.GetDeviceAddress(aux_);
  }
  if (workspace_.has_value()) {
    workspace = allocs.GetDeviceAddress(workspace_.value());
  }
 

  return plan->ExecuteOnStream(
<<<<<<< HEAD
      params.stream, allocs.GetDeviceAddress(a_buffer_),
      allocs.GetDeviceAddress(b_buffer_), allocs.GetDeviceAddress(c_buffer_),
      allocs.GetDeviceAddress(d_buffer_), bias, aux, a_scale, b_scale, c_scale,
      d_scale, d_amax, {}, workspace);
}


auto CublasLtMatmulThunk::GetCachedMatmulPlan(
    const ExecuteParams& params) -> absl::StatusOr<se::gpu::BlasLt::MatmulPlan *> {

  auto& cache = MatmulPlanCache::i(params.stream);

  auto create = [&]() -> absl::StatusOr<se::gpu::BlasLt::MatmulPlanPtr>  {
    VLOG(2) << this << ": Adding new MatmulPlan for stream: " << params.stream << 
                       " instr: " << canonical_hlo_;
    
    TF_ASSIGN_OR_RETURN(auto plan, se::gpu::BlasLt::GetMatmulPlan(
                params.stream, gemm_config_, epilogue_));
    
    int64_t max_workspace = workspace_buffer_.has_value()
                          ? workspace_buffer_.value().size() : 0;
    int64_t num_algorithms = algorithm_idx_ == se::blas::kDefaultAlgorithm ?
                          1 : 128;
    TF_ASSIGN_OR_RETURN(auto algorithms,
       plan->GetAlgorithms(params.stream, num_algorithms, max_workspace));

    TF_RETURN_IF_ERROR(plan->SetAlgorithm(algorithms[algorithm_idx_]));
    return std::move(plan);
  };
  return cache.GetOrCreate(canonical_hlo_, create);
=======
      stream, allocs.GetDeviceAddress(a_), allocs.GetDeviceAddress(b_),
      allocs.GetDeviceAddress(c_), allocs.GetDeviceAddress(d_), bias, aux,
      a_scale, b_scale, c_scale, d_scale, d_amax, workspace);
}

absl::StatusOr<se::gpu::BlasLt::MatmulPlan*>
CublasLtMatmulThunk::GetCachedMatmulPlan(const ExecuteParams& params) {
  auto* blas_lt = se::gpu::BlasLt::Get(params.stream);
  auto create = [&]() -> absl::StatusOr<se::gpu::BlasLt::MatmulPlanPtr> {
    VLOG(2) << this << ": Adding new MatmulPlan for stream: " << params.stream
            << " instr: " << canonical_hlo_;

    TF_ASSIGN_OR_RETURN(auto plan,
                        blas_lt->GetMatmulPlan(gemm_config_, epilogue_));
    // if workspace buffer is not provided, consider onlt the algorithms which
    // do not require a scratch space
    int64_t max_workspace =
        workspace_.has_value() ? workspace_.value().size() : 0;

    // If autotuning is disabled, there is no point on retrieving all
    // algorithms, it's enough to get the default one only.
    int64_t num_algorithms =
        algorithm_idx_ == 0 ? 1 : GemmConfig::kNumAlgorithms;
    TF_ASSIGN_OR_RETURN(
        auto algorithms,
        plan->GetAlgorithms(params.stream, num_algorithms, max_workspace));

    TF_RETURN_IF_ERROR(plan->SetAlgorithm(algorithms[algorithm_idx_]));
    return std::move(plan);
  };
  return blas_lt->GetOrCreateMatmulPlan(canonical_hlo_, create);
>>>>>>> upstream/master
}

absl::Status CublasLtMatmulThunk::Initialize(const InitializeParams& params) {
  if (!params.executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLASLT support");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
