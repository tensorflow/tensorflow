/* Copyright 2023 The OpenXLA Authors.
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

#include <algorithm>
#include <cstdint>
#include <utility>
#include <sstream>

#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/stream_executor/gpu/gpu_blas_lt_gemm_runner.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"


namespace stream_executor {
namespace gpu {

namespace {

bool AutotuneEnabled() {
  static std::atomic_bool result{[] {
    bool value = false;
    tsl::ReadBoolFromEnvVar("TF_MATMUL_AUTOTUNE_ENABLE",
         /*default_value=*/false, &value);
    return value;
  }()};
  return result;
}

} // namespace


bool operator ==(const GroupedGemmConfig& rhs, const GroupedGemmConfig& lhs) {
  return AsTuple(rhs) == AsTuple(lhs);
}

bool operator ==(const StridedGemmConfig& rhs, const StridedGemmConfig& lhs) {
  return AsTuple(rhs) == AsTuple(lhs);
}


std::ostream& operator <<(std::ostream& os, const StridedGemmConfig& cfg) {
    return os << "trans_a/b: " << (int)cfg.trans_a << "/" << (int)cfg.trans_b <<
      " m: " << cfg.m << " n: " << cfg.n << " k: " << cfg.k <<
      " batch_count: " << cfg.batch_count <<
      " lda: " << cfg.lda << " ldb: " << cfg.ldb << " ldc: " << cfg.ldc <<
      " stride_a: " << cfg.stride_a << " stride_b: " << cfg.stride_b << 
      " stride_c: " << cfg.stride_c << 
      " type_a: " << (int)cfg.type_a << " type_b: " << (int)cfg.type_b << 
      " type_c: " << (int)cfg.type_c << 
      " alpha: " << cfg.alpha << " beta: " << cfg.beta;
}

BlasLtGemmRunner::BlasLtGemmRunner(StreamExecutor *parent) :
    mutex_(std::make_unique< absl::Mutex >())
 
    { }

BlasLtGemmRunner::~BlasLtGemmRunner() { }

/*static*/ BlasLtGemmRunner& BlasLtGemmRunner::i(const Stream *stream) {
    static absl::Mutex m(absl::kConstInit);
    // Each GPU gets different cache instance
    static std::vector<std::unique_ptr< BlasLtGemmRunner >> meta(8);
    absl::MutexLock lock(&m);
    size_t dev_id = stream->parent()->device_ordinal();
    if (dev_id >= meta.size()) {
        meta.resize(dev_id + 1);
    }
    auto &res = meta[dev_id];
    if (!res) {
        res.reset(new BlasLtGemmRunner(stream->parent()));
    }
    return *res;
}

template < class TuneFunc >
absl::StatusOr< gpu::BlasLt::MatmulAlgorithm > BlasLtGemmRunner::Autotune(
    const std::vector< gpu::BlasLt::MatmulAlgorithm >& algorithms,
    TuneFunc&& benchmark_func) {
  gpu::BlasLt::MatmulAlgorithm best_algo;
  float best_ms = std::numeric_limits< float >::max(), total_ms = 0;
  uint32_t n_warmups = 1, n_iters = 5, n_total = n_warmups + n_iters, i = 0;
  
  if (AutotuneEnabled()) {
    for (uint32_t j = 0; j < algorithms.size(); j++) {
      const auto& algo = algorithms[j];
      if (!benchmark_func(algo, nullptr).ok()) continue;
      
      blas::ProfileResult profile;
      for (i = 0, total_ms = 0; i < n_total; i++) {
        auto res = benchmark_func(algo, &profile);
        if (!res.ok() || !profile.is_valid()) { 
          VLOG(1) << j << ": gemm algorithm is not valid: " << res.error_message();
          break;
        }
        if (i >= n_warmups) total_ms += profile.elapsed_time_in_ms();
      }
      if (i < n_total) continue; // invalid algorithm
      total_ms /= n_iters;
      VLOG(2) << j << ": gemm algorithm " << profile.algorithm() << " took " 
                   << total_ms << "ms, workspace: " << algo.workspace_size;
      if (total_ms < best_ms) {
        best_ms = total_ms, best_algo = algo;
      }
    } // for algorithms
  } else if (!algorithms.empty()) {
    best_algo = algorithms[0]; // otherwise use default algorithm
  }
  if (!best_algo.opaque_algo.has_value()) {
      return absl::InternalError("No valid gemm algorithms found!");
  }
  return best_algo;
}

// absl::StatusOr< std::array< int64_t, 3 >> BlasLtGemmRunner::ContiguousStrides(
//     const port::ArraySlice<DeviceMemoryBase *>& a, 
//           const port::ArraySlice<DeviceMemoryBase *>& b, 
//     const port::ArraySlice<DeviceMemoryBase *>& c, int64_t batch_count) {

//     int64_t bsa = 0, bsb = 0, bsc = 0;
//     using CT = const uint8_t;
//     for(int64_t i = 0; i < batch_count-1; i++) {
//       int64_t da = (CT *)a[i + 1]->opaque() - (CT *)a[i]->opaque(),
//          db = (CT *)b[i + 1]->opaque() - (CT *)b[i]->opaque(),
//          dc = (CT *)c[i + 1]->opaque() - (CT *)c[i]->opaque();
//           if(i == 0) {
//         bsa = da, bsb = db, bsc = dc;
//       } else if(!(bsa == da && bsb == db && bsc == dc)) { // strides mismatch
//         return absl::InternalError("Strides are not consistent!");
//       }
//     }
//     return std::array< int64_t, 3 >{ bsa, bsb, bsc };
// }

absl::Status BlasLtGemmRunner::RunBatchedImpl(Stream& stream, 
          blas::Transpose trans_a, blas::Transpose trans_b, int64_t m, int64_t n, int64_t k,
          const void *alpha, blas::DataType type_a, const void** a, int64_t lda, 
          blas::DataType type_b, const void** b, int64_t ldb, const void *beta,
                blas::DataType type_c, void** c, int64_t ldc, int64_t batch_count,
          ScratchAllocator* allocator)
{

  TF_ASSIGN_OR_RETURN(auto compute_type, 
            gpu::GetBlasComputationType(type_a, type_c, 0));

  GroupedGemmConfig cfg{
    .m = (int64_t)m,
    .n = (int64_t)n,
    .k = (int64_t)k,
    .batch_count = (int64_t)batch_count,
    .trans_a = trans_a,
    .trans_b = trans_b,
    .alpha = alpha,
    .beta = beta,
    .type_a = type_a,
    .type_b = type_b,
    .type_c = type_c,
    .type_d = type_c, 
    .lda = (int64_t)lda,
    .ldb = (int64_t)ldb,
    .ldc = (int64_t)ldc,
    .ldd = (int64_t)ldc,
    .compute_type = compute_type,
    .a = a,
    .b = b,
    .c = const_cast< const void **>(c),
    .d = c,
  };

  absl::MutexLock lock(mutex_.get());

  auto res = grouped_gemm_map_.find(cfg);
  if(res == grouped_gemm_map_.end()) {
    // NOTE: we assume that pointers a,b,c come from the device mem
    // hence we need to block stream here
    TF_ASSIGN_OR_RETURN(auto plan_res, 
            gpu::BlasLt::CreateGroupedMatmulPlan(&stream, cfg));
    res = grouped_gemm_map_.emplace(cfg, std::move(plan_res)).first;

    size_t num_solutions = AutotuneEnabled() ? MaxSolutions : 1;
    // discard solutions with non-zero workspace if allocator is not given
    TF_ASSIGN_OR_RETURN(auto algorithms, res->second->GetAlgorithms(
            num_solutions, allocator == nullptr ? 0 : 1ull << 32));

    VLOG(1) << stream.parent() << ": new GGemm config: " << 
                grouped_gemm_map_.size() << " #valid algorithms: " << algorithms.size();
    
    TF_ASSIGN_OR_RETURN(auto best_algo, Autotune(algorithms, 
      [&](const gpu::BlasLt::MatmulAlgorithm& algo, blas::ProfileResult *profile){
      if(profile == nullptr) {
        return res->second->SetAlgorithm(algo, allocator);
      }
      return res->second->ExecuteOnStream(&stream, cfg, profile);
      }));
      TF_RETURN_IF_ERROR(res->second->SetAlgorithm(best_algo, allocator));
  }
  return res->second->ExecuteOnStream(&stream, cfg);
}

absl::Status BlasLtGemmRunner::RunStridedBatchedImpl(Stream& stream, 
      blas::Transpose trans_a, blas::Transpose trans_b, int64_t m, int64_t n, int64_t k, 
      xla::complex128 alpha, 
      blas::DataType type_a, const DeviceMemoryBase& a, int64_t lda, int64_t stride_a,
      blas::DataType type_b, const DeviceMemoryBase& b, int64_t ldb, int64_t stride_b,
      double beta,
      blas::DataType type_c, DeviceMemoryBase *c, int64_t ldc, int64_t stride_c, 
      int64_t batch_count, ScratchAllocator* allocator)
{
   StridedGemmConfig scfg{
    .m = m,
    .n = n,
    .k = k,
    .batch_count = (int64_t)batch_count,
    .trans_a = trans_a,
    .trans_b = trans_b,
    .alpha = alpha,
    .beta = beta,
    .type_a = type_a,
    .type_b = type_b,
    .type_c = type_c,
    .lda = lda,
    .ldb = ldb,
    .ldc = ldc,
    .stride_a = stride_a,
    .stride_b = stride_b,
    .stride_c = stride_c,
   };

  //  VLOG(0) << "Running " << scfg;
  absl::MutexLock lock(mutex_.get());

  auto res = strided_gemm_map_.find(scfg);
  if(res == strided_gemm_map_.end()) {

     TF_ASSIGN_OR_RETURN(auto compute_type,  // TODO:PrecisionConfig::ALG_UNSET
            gpu::GetBlasComputationType(PrecisionConfig::ALG_UNSET,
                                        type_a, type_c, 0));

    int64_t row_a = m, col_a = k, row_b = k, col_b = n;
    if (trans_a == blas::Transpose::kTranspose) std::swap(row_a, col_a);
    if (trans_b == blas::Transpose::kTranspose) std::swap(row_b, col_b);

    auto order = MatrixLayout::Order::kColumnMajor;
    GemmConfig cfg = {
        .lhs_layout = MatrixLayout(type_a, row_a, col_a, order, batch_count,
                                  lda, stride_a, trans_a),

        .rhs_layout = MatrixLayout(type_b, row_b, col_b, order, batch_count,
                                  ldb, stride_b, trans_b),

        .c_layout = MatrixLayout(type_c, m, n, order, batch_count,
                                  ldc, stride_c),
        .output_layout = MatrixLayout(type_c, m, n, order, batch_count,
                                  ldc, stride_c),
        .alpha = alpha,
        .beta = beta,
        .algorithm = {},
        .grad_x = false,
        .grad_y = false,
        .compute_type = compute_type,
    };

    TF_ASSIGN_OR_RETURN(auto plan_res, 
            gpu::BlasLt::GetMatmulPlan(&stream, cfg, gpu::BlasLt::Epilogue::kDefault));
    res = strided_gemm_map_.emplace(scfg, std::move(plan_res)).first;

    // xla::gpu::GemmAlgorithmPicker autotuner(*config_);
    // TF_ASSIGN_OR_RETURN(auto rres, autotuner.RunStandalone(
    //  gemm_canonical_str, cfg, 
    //  GemmBackendConfig::Epilogue epilogue,
    //  std::vector< Shape >&& input_shapes, const Shape& output_shape,
    //  const DebugOptions& debug_options));

    size_t num_solutions = AutotuneEnabled() ? MaxSolutions : 1;
    // discard solutions with non-zero workspace if allocator is not given
    TF_ASSIGN_OR_RETURN(auto algorithms, res->second->GetAlgorithms(
            num_solutions, allocator == nullptr ? 0 : 1ull << 32));
    VLOG(1) << &stream << " dev " << stream.parent() << '(' << 
        stream.parent()->device_ordinal() << "): new StridedBatched config: " 
      << strided_gemm_map_.size() << " #algorithms: " << algorithms.size();

    if (algorithms.empty()) {
      std::ostringstream oss;
      oss << scfg;
      VLOG(1) << "No algorithms found for: " << oss.str();
    }

    TF_ASSIGN_OR_RETURN(auto best_algo, Autotune(algorithms,
      [&](const gpu::BlasLt::MatmulAlgorithm& algo, blas::ProfileResult *profile){
      if(profile == nullptr) {
        return res->second->SetAlgorithm(algo);
      }
      return res->second->ExecuteOnStream(
          &stream, a, b, *c, *c,
          DeviceMemoryBase{}, // bias
          DeviceMemoryBase{}, // aux
          DeviceMemoryBase{}, // a_scale
          DeviceMemoryBase{}, // b_scale
          DeviceMemoryBase{}, // c_scale
          DeviceMemoryBase{}, // d_scale
          DeviceMemoryBase{}, // d_amax
          absl::nullopt,      // workspace
          allocator,          // allocator
          profile);
      }));
      res->second->SetAlgorithm(best_algo);
  }
  return res->second->ExecuteOnStream(
      &stream, a, b, *c, *c,
      DeviceMemoryBase{}, // bias
      DeviceMemoryBase{}, // aux
      DeviceMemoryBase{}, // a_scale
      DeviceMemoryBase{}, // b_scale
      DeviceMemoryBase{}, // c_scale
      DeviceMemoryBase{}, // d_scale
      DeviceMemoryBase{}, // d_amax
      absl::nullopt,      // workspace
      allocator);         // allocator
}

}  // namespace gpu

}  // namespace stream_executor
