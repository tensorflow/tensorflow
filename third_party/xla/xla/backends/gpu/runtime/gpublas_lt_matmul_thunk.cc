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
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

CublasLtMatmulThunk::CublasLtMatmulThunk(const CublasLtMatmulThunk& rhs)
    : Thunk(Kind::kCublasLtMatmul, {}),
      gemm_config_(rhs.gemm_config_),
      epilogue_(rhs.epilogue_),
      algorithm_idx_(rhs.algorithm_idx_),
      autotune_workspace_size_(rhs.autotune_workspace_size_),
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
    Thunk::ThunkInfo thunk_info, std::string canonical_hlo,
    GemmConfig gemm_config, se::gpu::BlasLt::Epilogue epilogue,
    int64_t algorithm_idx, int64_t autotune_workspace_size, ShapedSlice a,
    ShapedSlice b, ShapedSlice c, ShapedSlice d,
    std::optional<ShapedSlice> bias, std::optional<ShapedSlice> aux,
    std::optional<ShapedSlice> a_scale, std::optional<ShapedSlice> b_scale,
    std::optional<ShapedSlice> c_scale, std::optional<ShapedSlice> d_scale,
    std::optional<ShapedSlice> d_amax,
    std::optional<const ShapedSlice> workspace)
    : Thunk(Kind::kCublasLtMatmul, std::move(thunk_info)),
      gemm_config_(std::move(gemm_config)),
      epilogue_(epilogue),
      algorithm_idx_(algorithm_idx),
      autotune_workspace_size_(autotune_workspace_size),
      canonical_hlo_(std::move(canonical_hlo)),
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
      workspace_(workspace) {}

absl::Status CublasLtMatmulThunk::ExecuteOnStreamInternal(
    se::Stream* stream, const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(auto* plan, GetCachedMatmulPlan(params));

  VLOG(3) << "Running cublas_lt matmul thunk";
  const BufferAllocations& allocs = *params.buffer_allocations;

  se::DeviceAddressBase bias, a_scale, b_scale, c_scale, d_scale, d_amax, aux,
      workspace;
  if (bias_.has_value()) {
    bias = allocs.GetDeviceAddress(bias_->slice);
  }
  if (a_scale_.has_value()) {
    a_scale = allocs.GetDeviceAddress(a_scale_->slice);
  }
  if (b_scale_.has_value()) {
    b_scale = allocs.GetDeviceAddress(b_scale_->slice);
  }
  if (c_scale_.has_value()) {
    c_scale = allocs.GetDeviceAddress(c_scale_->slice);
  }
  if (d_scale_.has_value()) {
    d_scale = allocs.GetDeviceAddress(d_scale_->slice);
  }
  if (d_amax_.has_value()) {
    d_amax = allocs.GetDeviceAddress(d_amax_->slice);
  }
  if (aux_.has_value()) {
    aux = allocs.GetDeviceAddress(aux_->slice);
  }
  if (workspace_.has_value()) {
    workspace = allocs.GetDeviceAddress(workspace_->slice);
  }

  return plan->ExecuteOnStream(
      stream, allocs.GetDeviceAddress(a_.slice),
      allocs.GetDeviceAddress(b_.slice), allocs.GetDeviceAddress(c_.slice),
      allocs.GetDeviceAddress(d_.slice), bias, aux, a_scale, b_scale, c_scale,
      d_scale, d_amax, workspace);
}

absl::StatusOr<se::gpu::BlasLt::MatmulPlan*>
CublasLtMatmulThunk::GetCachedMatmulPlan(const ExecuteParams& params) {
  auto* blas_lt = se::gpu::BlasLt::Get(params.stream);
  auto create = [&]() -> absl::StatusOr<se::gpu::BlasLt::MatmulPlanPtr> {
    VLOG(2) << this << ": Adding new MatmulPlan for stream: " << params.stream
            << " instr: " << canonical_hlo_;

    TF_ASSIGN_OR_RETURN(auto plan,
                        blas_lt->GetMatmulPlan(gemm_config_, epilogue_));

    // Set the workspace size to the size that was used for autotuning, so
    // algorithm index will be the same as returned by GetAlgorithms called
    // during autotuning.
    int64_t max_workspace = autotune_workspace_size_;

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
}

absl::Status CublasLtMatmulThunk::Initialize(const InitializeParams& params) {
  if (!params.executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLASLT support");
  }
  return absl::OkStatus();
}

Thunk::BufferUses CublasLtMatmulThunk::buffer_uses() const {
  Thunk::BufferUses res{
      BufferUse::Read(a_.slice, a_.shape),
      BufferUse::Read(b_.slice, b_.shape),
      BufferUse::Read(c_.slice, c_.shape),
      BufferUse::Write(d_.slice, d_.shape),
  };
  if (bias_.has_value()) {
    res.push_back(BufferUse::Read(bias_->slice, bias_->shape));
  }
  if (aux_.has_value()) {
    res.push_back(BufferUse::Write(aux_->slice, aux_->shape));
  }
  if (a_scale_.has_value()) {
    res.push_back(BufferUse::Read(a_scale_->slice, a_scale_->shape));
  }
  if (b_scale_.has_value()) {
    res.push_back(BufferUse::Read(b_scale_->slice, b_scale_->shape));
  }
  if (c_scale_.has_value()) {
    res.push_back(BufferUse::Read(c_scale_->slice, c_scale_->shape));
  }
  if (d_scale_.has_value()) {
    res.push_back(BufferUse::Read(d_scale_->slice, d_scale_->shape));
  }
  if (d_amax_.has_value()) {
    res.push_back(BufferUse::Write(d_amax_->slice, d_amax_->shape));
  }
  if (workspace_.has_value()) {
    res.push_back(BufferUse::Write(workspace_->slice, workspace_->shape));
  }
  return res;
}

absl::StatusOr<ThunkProto> CublasLtMatmulThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CublasLtMatmulThunkProto* cublas_lt_matmul_thunk =
      proto.mutable_cublas_lt_matmul_thunk();
  *cublas_lt_matmul_thunk->mutable_gemm_config() = gemm_config_.ToProto();
  cublas_lt_matmul_thunk->set_epilogue(
      stream_executor::gpu::BlasLt::EpilogueToProto(epilogue_));
  cublas_lt_matmul_thunk->set_algorithm_idx(algorithm_idx_);
  cublas_lt_matmul_thunk->set_autotune_workspace_size(autotune_workspace_size_);
  cublas_lt_matmul_thunk->set_canonical_hlo(canonical_hlo_);
  ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_a(), a_.ToProto());
  ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_b(), b_.ToProto());
  ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_c(), c_.ToProto());
  ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_d(), d_.ToProto());
  if (bias_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_bias(), bias_->ToProto());
  }
  if (aux_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_aux(), aux_->ToProto());
  }
  if (a_scale_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_a_scale(),
                     a_scale_->ToProto());
  }
  if (b_scale_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_b_scale(),
                     b_scale_->ToProto());
  }
  if (c_scale_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_c_scale(),
                     c_scale_->ToProto());
  }
  if (d_scale_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_d_scale(),
                     d_scale_->ToProto());
  }
  if (d_amax_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_d_amax(),
                     d_amax_->ToProto());
  }
  if (workspace_.has_value()) {
    ASSIGN_OR_RETURN(*cublas_lt_matmul_thunk->mutable_workspace(),
                     workspace_->ToProto());
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<Thunk>> CublasLtMatmulThunk::FromProto(
    Thunk::ThunkInfo thunk_info, const CublasLtMatmulThunkProto& proto,
    absl::Span<const BufferAllocation> allocations) {
  ASSIGN_OR_RETURN(
      stream_executor::gpu::GemmConfig gemm_config,
      stream_executor::gpu::GemmConfig::FromProto(proto.gemm_config()));
  ASSIGN_OR_RETURN(
      stream_executor::gpu::BlasLt::Epilogue epilogue,
      stream_executor::gpu::BlasLt::EpilogueFromProto(proto.epilogue()));
  ASSIGN_OR_RETURN(ShapedSlice a,
                   ShapedSlice::FromProto(proto.a(), allocations));
  ASSIGN_OR_RETURN(ShapedSlice b,
                   ShapedSlice::FromProto(proto.b(), allocations));
  ASSIGN_OR_RETURN(ShapedSlice c,
                   ShapedSlice::FromProto(proto.c(), allocations));
  ASSIGN_OR_RETURN(ShapedSlice d,
                   ShapedSlice::FromProto(proto.d(), allocations));

  std::optional<ShapedSlice> bias;
  if (proto.has_bias()) {
    ASSIGN_OR_RETURN(bias, ShapedSlice::FromProto(proto.bias(), allocations));
  }
  std::optional<ShapedSlice> aux;
  if (proto.has_aux()) {
    ASSIGN_OR_RETURN(aux, ShapedSlice::FromProto(proto.aux(), allocations));
  }
  std::optional<ShapedSlice> a_scale;
  if (proto.has_a_scale()) {
    ASSIGN_OR_RETURN(a_scale,
                     ShapedSlice::FromProto(proto.a_scale(), allocations));
  }
  std::optional<ShapedSlice> b_scale;
  if (proto.has_b_scale()) {
    ASSIGN_OR_RETURN(b_scale,
                     ShapedSlice::FromProto(proto.b_scale(), allocations));
  }
  std::optional<ShapedSlice> c_scale;
  if (proto.has_c_scale()) {
    ASSIGN_OR_RETURN(c_scale,
                     ShapedSlice::FromProto(proto.c_scale(), allocations));
  }
  std::optional<ShapedSlice> d_scale;
  if (proto.has_d_scale()) {
    ASSIGN_OR_RETURN(d_scale,
                     ShapedSlice::FromProto(proto.d_scale(), allocations));
  }
  std::optional<ShapedSlice> d_amax;
  if (proto.has_d_amax()) {
    ASSIGN_OR_RETURN(d_amax,
                     ShapedSlice::FromProto(proto.d_amax(), allocations));
  }
  std::optional<ShapedSlice> workspace;
  if (proto.has_workspace()) {
    ASSIGN_OR_RETURN(workspace,
                     ShapedSlice::FromProto(proto.workspace(), allocations));
  }
  return std::make_unique<CublasLtMatmulThunk>(
      std::move(thunk_info), std::move(proto.canonical_hlo()),
      xla::gpu::GemmConfig(std::move(gemm_config)), std::move(epilogue),
      proto.algorithm_idx(), proto.autotune_workspace_size(), std::move(a),
      std::move(b), std::move(c), std::move(d), std::move(bias), std::move(aux),
      std::move(a_scale), std::move(b_scale), std::move(c_scale),
      std::move(d_scale), std::move(d_amax), std::move(workspace));
}

}  // namespace gpu
}  // namespace xla
