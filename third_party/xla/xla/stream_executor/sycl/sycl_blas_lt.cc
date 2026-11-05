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

#include "xla/stream_executor/sycl/sycl_blas_lt.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>  // NOLINT

#include "oneapi/mkl.hpp"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/onemkl_util.h"
#include "xla/stream_executor/sycl/sycl_matmul_utils.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace sycl {
absl::Status BlasLt::Init() { return absl::OkStatus(); }

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& config,
                           Epilogue epilogue) const
    -> absl::StatusOr<MatmulPlanPtr> {
  absl::MutexLock lock(&mu_);
  return std::make_unique<MatmulPlan>(config, epilogue);
}

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, const gpu::BlasLt::MemoryArgs& args,
    blas::ProfileResult* profile_result) const {
  absl::MutexLock lock(&mu_);
  ABSL_ASSIGN_OR_RETURN(auto epilogue, sycl_gemm::AsSYCLEpilogue(epilogue_));
  if (!algorithm_) {
    return absl::InvalidArgumentError("Algorithm not set");
  }
  int64_t algorithm = std::any_cast<int64_t>(algorithm_->opaque_algo);
  absl::Status status =
      RunGemm(config_,
              args.a,          // se::DeviceAddressBase lhs
              args.b,          // se::DeviceAddressBase rhs
              args.c,          // se::DeviceAddressBase c
              args.d,          // se::DeviceAddressBase output
              args.bias,       // se::DeviceAddressBase bias
              args.workspace,  // se::DeviceAddressBase workspace
              stream, epilogue, algorithm, args.scratch_allocator);
  return status;
}

auto BlasLt::MatmulPlan::GetAlgorithms(size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  absl::MutexLock lock(&mu_);
  std::vector<MatmulAlgorithm> algorithms;
  algorithms.push_back({/*opaque_algo*/ kOneDnnGemm, /*workspace_size*/ 0});
  return std::move(algorithms);
}

SyclBlasSupport::SyclBlasSupport(StreamExecutor* parent) : blas_lt_(parent) {}

SyclBlasSupport::~SyclBlasSupport() {}

namespace {

// Multiplies every element of the vector x by alpha on the stream's SYCL
// queue. The scaling factor has its own template parameter because a complex
// vector can be scaled by either a complex or a real number.
template <typename ScalarT, typename T>
bool ScalImpl(Stream* stream, uint64_t elem_count, ScalarT alpha,
              DeviceAddress<T>* x, int incx) {
  if (stream == nullptr || x == nullptr) {
    LOG(ERROR) << "DoBlasScal: stream and x must not be null";
    return false;
  }
  if (elem_count == 0) {
    return true;
  }
  // oneMKL takes the element count as a signed 64-bit integer.
  if (elem_count >
      static_cast<uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    LOG(ERROR) << "DoBlasScal: elem_count " << elem_count
               << " does not fit in int64_t";
    return false;
  }
  auto* stream_queue =
      static_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  if (stream_queue == nullptr) {
    LOG(ERROR) << "DoBlasScal: stream has no SYCL queue";
    return false;
  }
  absl::StatusOr<::sycl::event> status = ExecMklFunc([&] {
    return oneapi::mkl::blas::scal(*stream_queue,
                                   static_cast<std::int64_t>(elem_count), alpha,
                                   gpu::GpuMemoryMutable(x), incx);
  });
  if (status.ok()) {
    return true;
  }
  LOG(ERROR) << status.status();
  return false;
}

}  // namespace

bool SyclBlasSupport::DoBlasScal(Stream* stream, uint64_t elem_count,
                                 float alpha, DeviceAddress<float>* x,
                                 int incx) {
  return ScalImpl(stream, elem_count, alpha, x, incx);
}

bool SyclBlasSupport::DoBlasScal(Stream* stream, uint64_t elem_count,
                                 double alpha, DeviceAddress<double>* x,
                                 int incx) {
  return ScalImpl(stream, elem_count, alpha, x, incx);
}

bool SyclBlasSupport::DoBlasScal(Stream* stream, uint64_t elem_count,
                                 float alpha,
                                 DeviceAddress<std::complex<float>>* x,
                                 int incx) {
  return ScalImpl(stream, elem_count, alpha, x, incx);
}

bool SyclBlasSupport::DoBlasScal(Stream* stream, uint64_t elem_count,
                                 double alpha,
                                 DeviceAddress<std::complex<double>>* x,
                                 int incx) {
  return ScalImpl(stream, elem_count, alpha, x, incx);
}

bool SyclBlasSupport::DoBlasScal(Stream* stream, uint64_t elem_count,
                                 std::complex<float> alpha,
                                 DeviceAddress<std::complex<float>>* x,
                                 int incx) {
  return ScalImpl(stream, elem_count, alpha, x, incx);
}

bool SyclBlasSupport::DoBlasScal(Stream* stream, uint64_t elem_count,
                                 std::complex<double> alpha,
                                 DeviceAddress<std::complex<double>>* x,
                                 int incx) {
  return ScalImpl(stream, elem_count, alpha, x, incx);
}

bool SyclBlasSupport::Init() { return true; }

static void RegisterSyclBlasSupport() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          stream_executor::sycl::kSyclPlatformId, "syclBLAS",
          [](StreamExecutor* parent) -> blas::BlasSupport* {
            auto* blas = new SyclBlasSupport(parent);
            if (!blas->Init()) {
              delete blas;
              return nullptr;
            }
            return blas;
          });
  if (!status.ok()) {
    std::cerr << "Unable to register sycl_blas factory: " << status.message()
              << std::endl;
  }
}

}  // namespace sycl
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    syclblas, stream_executor::sycl::RegisterSyclBlasSupport());
