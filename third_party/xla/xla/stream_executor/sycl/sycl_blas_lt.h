/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_BLAS_LT_H_

#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace sycl {

constexpr int64_t kOneDnnGemm = 1;

class BlasLt : public gpu::BlasLt {
 public:
  explicit BlasLt(StreamExecutor* parent) : parent_(parent) {}

  absl::Status Init() override;

  absl::StatusOr<BlasLt::MatmulPlanPtr> GetMatmulPlan(
      const gpu::GemmConfig& config, Epilogue epilogue) const override;

  absl::StatusOr<MatmulPlanPtr> GetGroupedMatmulPlan(
      gpu::GroupedGemmConfig& config,
      const std::vector<Epilogue>& epilogues) const override;

  ~BlasLt() override = default;

  struct MatmulPlan : public gpu::BlasLt::MatmulPlan {
    MatmulPlan(gpu::GemmConfig config, Epilogue epilogue)
        : config_(xla::gpu::GemmConfig(config)), epilogue_(epilogue) {}

    ~MatmulPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        const Stream* stream, size_t max_algorithm_count,
        size_t max_workspace_size) const override;

    absl::Status SetAlgorithm(const MatmulAlgorithm& algorithm) override {
      // TODO(intel-tf): Do we need a lock here?
      absl::MutexLock lock(&mu_);
      algorithm_ = algorithm;
      return absl::OkStatus();
    }

   private:
    std::optional<MatmulAlgorithm> algorithm_;  // selected algorithm
    xla::gpu::GemmConfig config_;
    Epilogue epilogue_;
    mutable absl::Mutex mu_;
  };

 private:
  StreamExecutor* parent_;
  mutable absl::Mutex mu_;
};

class SyclBlasSupport : public blas::BlasSupport {
 public:
  explicit SyclBlasSupport(StreamExecutor* parent);

  bool Init();

  ~SyclBlasSupport() override;

  gpu::BlasLt* GetBlasLt() override { return &blas_lt_; }

  // Sycl BlasLt does not require implementation of the following virtual
  // methods.
  absl::StatusOr<bool> IsMainStreamSet() const override {
    return absl::UnimplementedError("IsMainStreamSet is not supported");
  }

  bool DoBlasScal(Stream* stream, uint64_t elem_count, float alpha,
                  DeviceAddress<float>* x, int incx) override {
    LOG(ERROR) << "DoBlasScal is not implemented";
    return false;
  }

  bool DoBlasScal(Stream* stream, uint64_t elem_count, double alpha,
                  DeviceAddress<double>* x, int incx) override {
    LOG(ERROR) << "DoBlasScal is not implemented";
    return false;
  }

  bool DoBlasScal(Stream* stream, uint64_t elem_count, float alpha,
                  DeviceAddress<std::complex<float>>* x, int incx) override {
    LOG(ERROR) << "DoBlasScal is not implemented";
    return false;
  }

  bool DoBlasScal(Stream* stream, uint64_t elem_count, double alpha,
                  DeviceAddress<std::complex<double>>* x, int incx) override {
    LOG(ERROR) << "DoBlasScal is not implemented";
    return false;
  }

  bool DoBlasScal(Stream* stream, uint64_t elem_count,
                  std::complex<float> alpha,
                  DeviceAddress<std::complex<float>>* x, int incx) override {
    LOG(ERROR) << "DoBlasScal is not implemented";
    return false;
  }

  bool DoBlasScal(Stream* stream, uint64_t elem_count,
                  std::complex<double> alpha,
                  DeviceAddress<std::complex<double>>* x, int incx) override {
    LOG(ERROR) << "DoBlasScal is not implemented";
    return false;
  }

  bool DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
                  float alpha, const DeviceAddress<float>& a, int lda,
                  const DeviceAddress<float>& x, int incx, float beta,
                  DeviceAddress<float>* y, int incy) override {
    LOG(ERROR) << "DoBlasGemv is not implemented";
    return false;
  }

  bool DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
                  double alpha, const DeviceAddress<double>& a, int lda,
                  const DeviceAddress<double>& x, int incx, double beta,
                  DeviceAddress<double>* y, int incy) override {
    LOG(ERROR) << "DoBlasGemv is not implemented";
    return false;
  }

  bool DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
                  std::complex<float> alpha,
                  const DeviceAddress<std::complex<float>>& a, int lda,
                  const DeviceAddress<std::complex<float>>& x, int incx,
                  std::complex<float> beta,
                  DeviceAddress<std::complex<float>>* y, int incy) override {
    LOG(ERROR) << "DoBlasGemv is not implemented";
    return false;
  }

  bool DoBlasGemv(Stream* stream, blas::Transpose trans, uint64_t m, uint64_t n,
                  std::complex<double> alpha,
                  const DeviceAddress<std::complex<double>>& a, int lda,
                  const DeviceAddress<std::complex<double>>& x, int incx,
                  std::complex<double> beta,
                  DeviceAddress<std::complex<double>>* y, int incy) override {
    LOG(ERROR) << "DoBlasGemv is not implemented";
    return false;
  }

  absl::Status DoBlasGemm(Stream* stream, blas::Transpose transa,
                          blas::Transpose transb, uint64_t m, uint64_t n,
                          uint64_t k, blas::DataType dtype, const void* alpha,
                          const DeviceAddressBase& a, int lda,
                          const DeviceAddressBase& b, int ldb, const void* beta,
                          DeviceAddressBase* c, int ldc,
                          const EngineOptions& engine_options,
                          blas::CallContext context) override {
    return absl::UnimplementedError("DoBlasGemm is not implemented");
  }

  bool GetBlasGemmAlgorithms(
      Stream* stream, const gpu::MatrixDescriptor& a,
      const gpu::MatrixDescriptor& b, gpu::OutputMatrixDescriptor* c,
      const void* alpha, const void* beta,
      std::vector<blas::AlgorithmType>* out_algorithms) override {
    out_algorithms->clear();
    return true;
  }

  absl::Status DoBlasGemmWithAlgorithm(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void* alpha,
      const DeviceAddressBase& a, blas::DataType type_a, int lda,
      const DeviceAddressBase& b, blas::DataType type_b, int ldb,
      const void* beta, DeviceAddressBase* c, blas::DataType type_c, int ldc,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      const EngineOptions& engine_options,
      blas::ProfileResult* output_profile_result,
      blas::CallContext context) override {
    return absl::UnimplementedError(
        "DoBlasGemmWithAlgorithm is not implemented");
  }

  bool DoBlasGemmBatched(Stream* stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, float alpha,
                         DeviceAddressSlice<Eigen::half> a, int lda,
                         DeviceAddressSlice<Eigen::half> b, int ldb, float beta,
                         DeviceAddressSlice<Eigen::half> c, int ldc,
                         int batch_count, const EngineOptions& engine_options,
                         ScratchAllocator* scratch_allocator,
                         blas::CallContext context) override {
    LOG(ERROR) << "DoBlasGemmBatched is not implemented";
    return false;
  }

  bool DoBlasGemmBatched(Stream* stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, float alpha,
                         DeviceAddressSlice<Eigen::bfloat16> a, int lda,
                         DeviceAddressSlice<Eigen::bfloat16> b, int ldb,
                         float beta, DeviceAddressSlice<Eigen::bfloat16> c,
                         int ldc, int batch_count,
                         const EngineOptions& engine_options,
                         ScratchAllocator* scratch_allocator,
                         blas::CallContext context) override {
    LOG(ERROR) << "DoBlasGemmBatched is not implemented";
    return false;
  }

  bool DoBlasGemmBatched(Stream* stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, float alpha, DeviceAddressSlice<float> a,
                         int lda, DeviceAddressSlice<float> b, int ldb,
                         float beta, DeviceAddressSlice<float> c, int ldc,
                         int batch_count, const EngineOptions& engine_options,
                         ScratchAllocator* scratch_allocator,
                         blas::CallContext context) override {
    LOG(ERROR) << "DoBlasGemmBatched is not implemented";
    return false;
  }

  bool DoBlasGemmBatched(Stream* stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, double alpha, DeviceAddressSlice<double> a,
                         int lda, DeviceAddressSlice<double> b, int ldb,
                         double beta, DeviceAddressSlice<double> c, int ldc,
                         int batch_count, const EngineOptions& engine_options,
                         ScratchAllocator* scratch_allocator,
                         blas::CallContext context) override {
    LOG(ERROR) << "DoBlasGemmBatched is not implemented";
    return false;
  }

  bool DoBlasGemmBatched(Stream* stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, std::complex<float> alpha,
                         DeviceAddressSlice<std::complex<float>> a, int lda,
                         DeviceAddressSlice<std::complex<float>> b, int ldb,
                         std::complex<float> beta,
                         DeviceAddressSlice<std::complex<float>> c, int ldc,
                         int batch_count, const EngineOptions& engine_options,
                         ScratchAllocator* scratch_allocator,
                         blas::CallContext context) override {
    LOG(ERROR) << "DoBlasGemmBatched is not implemented";
    return false;
  }

  bool DoBlasGemmBatched(Stream* stream, blas::Transpose transa,
                         blas::Transpose transb, uint64_t m, uint64_t n,
                         uint64_t k, std::complex<double> alpha,
                         DeviceAddressSlice<std::complex<double>> a, int lda,
                         DeviceAddressSlice<std::complex<double>> b, int ldb,
                         std::complex<double> beta,
                         DeviceAddressSlice<std::complex<double>> c, int ldc,
                         int batch_count, const EngineOptions& engine_options,
                         ScratchAllocator* scratch_allocator,
                         blas::CallContext context) override {
    LOG(ERROR) << "DoBlasGemmBatched is not implemented";
    return false;
  }

  absl::Status DoBlasGemmStridedBatched(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, blas::DataType dtype,
      const void* alpha, const DeviceAddressBase& a, int lda, int64_t stride_a,
      const DeviceAddressBase& b, int ldb, int64_t stride_b, const void* beta,
      DeviceAddressBase* c, int ldc, int64_t stride_c, int batch_count,
      const EngineOptions& engine_options, blas::CallContext context) override {
    return absl::UnimplementedError(
        "DoBlasGemmStridedBatched is not implemented");
  }

  absl::Status DoBlasGemmStridedBatchedWithAlgorithm(
      Stream* stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64_t k, const void* alpha,
      const DeviceAddressBase& a, blas::DataType type_a, int lda,
      int64_t stride_a, const DeviceAddressBase& b, blas::DataType type_b,
      int ldb, int64_t stride_b, const void* beta, DeviceAddressBase* c,
      blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      const EngineOptions& engine_options,
      blas::ProfileResult* output_profile_result,
      blas::CallContext context) override {
    return absl::UnimplementedError(
        "SyclBlas::DoBlasGemmStridedBatchedWithAlgorithm is not implemented");
  }

  bool DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo,
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                  uint64_t n, float alpha, const DeviceAddress<float>& a,
                  int lda, DeviceAddress<float>* b, int ldb) override {
    LOG(ERROR) << "DoBlasTrsm is not implemented";
    return false;
  }

  bool DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo,
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                  uint64_t n, double alpha, const DeviceAddress<double>& a,
                  int lda, DeviceAddress<double>* b, int ldb) override {
    LOG(ERROR) << "DoBlasTrsm is not implemented";
    return false;
  }

  bool DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo,
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                  uint64_t n, std::complex<float> alpha,
                  const DeviceAddress<std::complex<float>>& a, int lda,
                  DeviceAddress<std::complex<float>>* b, int ldb) override {
    LOG(ERROR) << "DoBlasTrsm is not implemented";
    return false;
  }

  bool DoBlasTrsm(Stream* stream, blas::Side side, blas::UpperLower uplo,
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                  uint64_t n, std::complex<double> alpha,
                  const DeviceAddress<std::complex<double>>& a, int lda,
                  DeviceAddress<std::complex<double>>* b, int ldb) override {
    LOG(ERROR) << "DoBlasTrsm is not implemented";
    return false;
  }

  bool DoBlasTrsmBatched(Stream* stream, blas::Side side, blas::UpperLower uplo,
                         blas::Transpose transa, blas::Diagonal diag,
                         uint64_t m, uint64_t n, float alpha,
                         const DeviceAddress<float*>& as, int lda,
                         DeviceAddress<float*>* bs, int ldb,
                         int batch_count) override {
    LOG(ERROR) << "DoBlasTrsmBatched is not implemented";
    return false;
  }

  bool DoBlasTrsmBatched(Stream* stream, blas::Side side, blas::UpperLower uplo,
                         blas::Transpose transa, blas::Diagonal diag,
                         uint64_t m, uint64_t n, double alpha,
                         const DeviceAddress<double*>& as, int lda,
                         DeviceAddress<double*>* bs, int ldb,
                         int batch_count) override {
    LOG(ERROR) << "DoBlasTrsmBatched is not implemented";
    return false;
  }

  bool DoBlasTrsmBatched(Stream* stream, blas::Side side, blas::UpperLower uplo,
                         blas::Transpose transa, blas::Diagonal diag,
                         uint64_t m, uint64_t n, std::complex<float> alpha,
                         const DeviceAddress<std::complex<float>*>& as, int lda,
                         DeviceAddress<std::complex<float>*>* bs, int ldb,
                         int batch_count) override {
    LOG(ERROR) << "DoBlasTrsmBatched is not implemented";
    return false;
  }

  bool DoBlasTrsmBatched(Stream* stream, blas::Side side, blas::UpperLower uplo,
                         blas::Transpose transa, blas::Diagonal diag,
                         uint64_t m, uint64_t n, std::complex<double> alpha,
                         const DeviceAddress<std::complex<double>*>& as,
                         int lda, DeviceAddress<std::complex<double>*>* bs,
                         int ldb, int batch_count) override {
    LOG(ERROR) << "DoBlasTrsmBatched is not implemented";
    return false;
  }

  absl::Status GetVersion(std::string* version) override {
    return absl::OkStatus();
  }

 private:
  BlasLt blas_lt_;
};

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_BLAS_LT_H_
