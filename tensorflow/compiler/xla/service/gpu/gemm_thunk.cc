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

#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"

#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

BlasScratchAllocator::BlasScratchAllocator(
    int device_ordinal, se::DeviceMemoryAllocator *memory_allocator)
    : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

int64 BlasScratchAllocator::GetMemoryLimitInBytes() {
  static const int64 max_scratch_size = tensorflow::GetBlasWorkspaceLimit(
      "TF_CUBLAS_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default
  return max_scratch_size;
}

StatusOr<se::DeviceMemory<uint8>> BlasScratchAllocator::AllocateBytes(
    int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8>(buffer_addr);
}

GpuGemmConfig GetGpuGemmConfig(const HloInstruction *gemm) {
  GpuGemmConfig config;
  config.output_shape = gemm->shape();
  config.lhs_shape = gemm->operand(0)->shape();
  config.rhs_shape = gemm->operand(1)->shape();
  auto backend_config_or = gemm->backend_config<GemmBackendConfig>();
  config.backend_config = std::move(backend_config_or.ValueOrDie());
  return config;
}

GemmThunk::GemmThunk(ThunkInfo thunk_info, GpuGemmConfig config,
                     const BufferAllocation::Slice &lhs_buffer,
                     const BufferAllocation::Slice &rhs_buffer,
                     const BufferAllocation::Slice &output_buffer,
                     bool implements_whole_instruction)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      implements_whole_instruction_(implements_whole_instruction) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams &params) {
  auto get_device_address = [&](const BufferAllocation::Slice &slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  se::DeviceMemoryBase lhs_data = get_device_address(lhs_buffer_);
  se::DeviceMemoryBase rhs_data = get_device_address(rhs_buffer_);
  se::DeviceMemoryBase output_data = get_device_address(output_buffer_);

  auto &buffer_allocations = *params.buffer_allocations;
  BlasScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                         buffer_allocations.memory_allocator());

  VLOG(3) << "Running GEMM thunk";
  return RunGemm(config_, lhs_data, rhs_data, output_data, params.stream,
                 implements_whole_instruction_, profile_index(),
                 &scratch_allocator, params.profiler);
}

template <typename Element, typename AlphaType>
static bool DoGemmWithAlgorithm(
    int64 batch_size, MatrixDescriptor lhs_matrix, MatrixDescriptor rhs_matrix,
    MatrixDescriptor output_matrix, AlphaType alpha, double beta,
    se::Stream *stream, absl::optional<se::blas::AlgorithmType> algorithm,
    se::blas::ProfileResult *output_profile_result) {
  DCHECK(!output_matrix.transpose);

  PrimitiveType type = primitive_util::NativeToPrimitiveType<Element>();

  // Converts from an XLA PrimitiveType to a blas::ComputationType, which is
  // used to specify the precision with which matmul computations should be
  // performed, separately from the precision of the inputs and result.
  se::blas::ComputationType computation_type;
  switch (type) {
    case F16:
      // Use F32 as computation type for F16 as we currently only implement
      // the cuDNN pseudo half configuration for half precision.
      computation_type = se::blas::ComputationType::kF32;
      break;
    case F32:
      computation_type = se::blas::ComputationType::kF32;
      break;
    case F64:
      computation_type = se::blas::ComputationType::kF64;
      break;
    case C64:
      computation_type = se::blas::ComputationType::kComplexF32;
      break;
    case C128:
      computation_type = se::blas::ComputationType::kComplexF64;
      break;
    default:
      return false;
  }

  se::DeviceMemory<Element> lhs_data(lhs_matrix.data);
  se::DeviceMemory<Element> rhs_data(rhs_matrix.data);
  se::DeviceMemory<Element> output_data(output_matrix.data);

  auto lhs_transpose = lhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto rhs_transpose = rhs_matrix.transpose ? se::blas::Transpose::kTranspose
                                            : se::blas::Transpose::kNoTranspose;
  auto k = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  if (algorithm) {
    // Autotuning is disabled for batch_size != 1.
    CHECK_EQ(1, batch_size);
    return stream
        ->ThenBlasGemmWithAlgorithm(
            lhs_transpose, rhs_transpose, output_matrix.num_rows,
            output_matrix.num_cols,
            /*size of reduce dim=*/k,
            /*alpha=*/static_cast<Element>(alpha), lhs_data,
            /*leading dim of LHS=*/lhs_matrix.num_rows, rhs_data,
            /*leading dim of RHS=*/rhs_matrix.num_rows,
            /*beta=*/static_cast<Element>(beta), &output_data,
            /*leading dim of output=*/output_matrix.num_rows, computation_type,
            *algorithm, output_profile_result)
        .ok();
  }

  if (batch_size != 1) {
    int64 lhs_stride = lhs_matrix.num_rows * lhs_matrix.num_cols;
    int64 rhs_stride = rhs_matrix.num_rows * rhs_matrix.num_cols;
    int64 output_stride = output_matrix.num_rows * output_matrix.num_cols;
    return stream
        ->ThenBlasGemmStridedBatched(
            lhs_transpose, rhs_transpose, output_matrix.num_rows,
            output_matrix.num_cols, /*size of reduce dim=*/k,
            /*alpha=*/alpha, lhs_data,
            /*leading dim of LHS=*/lhs_matrix.num_rows, lhs_stride, rhs_data,
            /*leading dim of RHS=*/rhs_matrix.num_rows, rhs_stride,
            /*beta=*/beta, &output_data,
            /*leading dim of output=*/output_matrix.num_rows, output_stride,
            batch_size)
        .ok();
  }

  return stream
      ->ThenBlasGemm(
          lhs_transpose, rhs_transpose, output_matrix.num_rows,
          output_matrix.num_cols, /*size of reduce dim=*/k, /*alpha=*/alpha,
          lhs_data, /*leading dim of LHS=*/lhs_matrix.num_rows, rhs_data,
          /*leading dim of RHS=*/rhs_matrix.num_rows, /*beta=*/beta,
          &output_data, /*leading dim of output=*/output_matrix.num_rows)
      .ok();
}

template <typename ElemType>
static bool DoGemmLt(
    int64 batch_size, MatrixDescriptor lhs_matrix, MatrixDescriptor rhs_matrix,
    MatrixDescriptor output_matrix, se::Stream *stream,
    se::ScratchAllocator *scratch_allocator,
    std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm> &profiled_algorithm,
    se::blas::ProfileResult *output_profile_result) {
  LOG(INFO) << "CublasLT called";
  DCHECK(!output_matrix.transpose);
  tensorflow::DataType dtype = tensorflow::DataTypeToEnum<ElemType>::value;
  bool allow_tf32 = tensorflow::tensor_float_32_execution_enabled();
  int device_id = stream->parent()->device_ordinal();
  bool trans_x = lhs_matrix.transpose;
  bool trans_y = rhs_matrix.transpose;

  int64 m = output_matrix.num_rows;
  int64 n = output_matrix.num_cols;
  auto k = lhs_matrix.transpose ? lhs_matrix.num_rows : lhs_matrix.num_cols;

  tensorflow::BatchMatmulParameters matmul_parameters(
      trans_x, trans_y, /*adj_x*/ false, /*adj_y*/ false, m, n, k, batch_size,
      /*broadcast_a*/ false, /*broadcast_b*/ false, dtype, dtype, allow_tf32,
      device_id);

  const auto *plan_and_algorithms =
      tensorflow::BatchMatmulPlanMapSingleton::GetInstance()->Find(
          matmul_parameters);

  const auto &plan = plan_and_algorithms->plan;
  const auto &algorithms = plan_and_algorithms->algorithms;

  std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm> algorithm(nullptr);
  if (!profiled_algorithm.get()) {
    se::blas::AlgorithmConfig algorithm_config(se::blas::kNoAlgorithm);
    tensorflow::AutoTuneBatchMatmul::GetInstance()->Find(matmul_parameters,
                                                         &algorithm_config);

    se::blas::AlgorithmType algorithm_idx = algorithm_config.algorithm();
    algorithm.reset(algorithms[algorithm_idx].get());
  } else {
    algorithm.reset(profiled_algorithm.get());
  }
  // The BlasLtMatmul routines (unlike BlasGemm, BlasGemmBatched etc.) take
  // alpha and beta with the same type as the matrices.
  ElemType alpha(1.0);
  ElemType beta(0.0);
  se::DeviceMemory<ElemType> lhs_data(lhs_matrix.data);
  se::DeviceMemory<ElemType> rhs_data(rhs_matrix.data);
  se::DeviceMemory<ElemType> output_data(output_matrix.data);

  return stream
      ->ThenBlasLtMatmul(plan.get(), alpha, rhs_data, lhs_data, beta,
                         &output_data, scratch_allocator, algorithm.get(), {},
                         output_profile_result)
      .ok();
}

Status PopulateInputOutputMatrices(const GpuGemmConfig &gemm_config,
                                   se::DeviceMemoryBase lhs_buffer,
                                   se::DeviceMemoryBase rhs_buffer,
                                   se::DeviceMemoryBase output_buffer,
                                   MatrixDescriptor &lhs_matrix,
                                   MatrixDescriptor &rhs_matrix,
                                   MatrixDescriptor &output_matrix) {
  VLOG(2) << "Populate I/O matrices";
  const Shape &output_shape = gemm_config.output_shape;
  const Shape &lhs_shape = gemm_config.lhs_shape;
  const Shape &rhs_shape = gemm_config.rhs_shape;
  const GemmBackendConfig &backend_config = gemm_config.backend_config;

  const DotDimensionNumbers &dim_nums = backend_config.dot_dimension_numbers();
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2, output_shape.rank());

  int64 row_dim = dim_nums.lhs_batch_dimensions_size();
  int64 col_dim = dim_nums.lhs_batch_dimensions_size() + 1;

  int64 batch_size = backend_config.batch_size();

  // Check that the batch dims don't cover the last two dims.
  for (int64 batch_dim : dim_nums.lhs_batch_dimensions()) {
    CHECK_NE(row_dim, batch_dim);
    CHECK_NE(col_dim, batch_dim);
  }

  // Verify that the non-batch dimensions are minor-most. This is required for
  // efficient access.
  for (const auto *shape : {&lhs_shape, &rhs_shape, &output_shape}) {
    CHECK_LT(shape->layout().minor_to_major(row_dim), 2);
    CHECK_LT(shape->layout().minor_to_major(col_dim), 2);
  }

  int64 output_num_rows = output_shape.dimensions(row_dim);
  int64 output_num_cols = output_shape.dimensions(col_dim);

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert dot between row-major matrices to that
  // between column-major matrices. The key insight for the conversion is that,
  // in linear storage, matrix M in column-major order is identical to the
  // transpose of M in row-major order. In other words,
  //
  //   column-major(M) = row-major(M^T).
  //
  // Leveraging this insight, we can perform dot between row-major matrices as
  // follows.
  //
  // row-major(C)
  //   = row-major(A x B) = column-major((A x B)^T) = column-major(B^T x A^T)
  //   = gemm(column-major(B^T), column-major(A^T))
  //   = gemm(row-major(B), row-major(A))
  //
  // Although we do not modify the content of A and B in linear memory, we
  // should use the dimensions of B^T and A^T when calling gemm. For example,
  // the leading dimension of the LHS matrix of gemm is the number of rows in
  // B^T and thus the number of columns in B.
  auto make_descriptor = [&](se::DeviceMemoryBase data, const Shape &shape,
                             bool transpose) -> MatrixDescriptor {
    bool is_row_major = LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch = LayoutUtil::Minor(shape.layout(), row_dim) !=
                           LayoutUtil::Minor(output_shape.layout(), row_dim);
    return MatrixDescriptor{
        data, static_cast<bool>(transpose ^ layout_mismatch),
        shape.dimensions(row_dim + static_cast<int64>(is_row_major)),
        shape.dimensions(row_dim + static_cast<int64>(!is_row_major))};
  };

  lhs_matrix = make_descriptor(
      lhs_buffer, lhs_shape, dim_nums.lhs_contracting_dimensions(0) == row_dim);
  rhs_matrix = make_descriptor(
      rhs_buffer, rhs_shape, dim_nums.rhs_contracting_dimensions(0) == col_dim);

  if (LayoutUtil::Minor(output_shape.layout(), row_dim) != 0) {
    std::swap(lhs_matrix, rhs_matrix);
    std::swap(output_num_cols, output_num_rows);
  }

  MatrixDescriptor out_matrix{output_buffer, /*needs_transpose=*/false,
                              output_num_rows, output_num_cols};
  output_matrix = out_matrix;

  return Status::OK();
}

Status RunGemm(const GpuGemmConfig &gemm_config,
               se::DeviceMemoryBase lhs_buffer, se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream *stream,
               bool implements_whole_instruction,
               absl::optional<int64> profile_index,
               BlasScratchAllocator *scratch_allocator,
               HloExecutionProfiler *profiler,
               se::blas::ProfileResult *profile_result,
               absl::optional<se::blas::AlgorithmType> algorithm,
               const std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm>
                   &profiled_algorithm) {
  VLOG(2) << "Executing a GemmThunk";
  MatrixDescriptor lhs_matrix;
  MatrixDescriptor rhs_matrix;
  MatrixDescriptor output_matrix;
  const Shape &output_shape = gemm_config.output_shape;
  int64 batch_size = gemm_config.backend_config.batch_size();
  CHECK(PopulateInputOutputMatrices(gemm_config, lhs_buffer, lhs_buffer,
                                    output_buffer, lhs_matrix, rhs_matrix,
                                    output_matrix)
            .ok());
  bool launch_ok = false;
  // The BlasLtMatmul routines are only supported from CUDA 11.0 onward.
  if (batch_size != 1) {
    //#if GOOGLE_CUDA && CUDA_VERSION >= 11000
    launch_ok = [&]() {
      std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm> best_algo(nullptr);
      if (!profiled_algorithm.get()) {
        best_algo.reset(profiled_algorithm.get());
      }
      switch (output_shape.element_type()) {
        case F16:

          return DoGemmLt<Eigen::half>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, stream,
              scratch_allocator, best_algo,
              /*output_profile_result=*/profile_result);
        case F32:

          return DoGemmLt<float>(batch_size, lhs_matrix, rhs_matrix,
                                 output_matrix, stream, scratch_allocator,
                                 best_algo,
                                 /*output_profile_result=*/profile_result);
        case F64:

          return DoGemmLt<double>(batch_size, lhs_matrix, rhs_matrix,
                                  output_matrix, stream, scratch_allocator,
                                  best_algo,
                                  /*output_profile_result=*/profile_result);
        case C64:
          return DoGemmLt<complex64>(batch_size, lhs_matrix, rhs_matrix,
                                     output_matrix, stream, scratch_allocator,
                                     best_algo,
                                     /*output_profile_result=*/profile_result);
        case C128:
          return DoGemmLt<complex128>(batch_size, lhs_matrix, rhs_matrix,
                                      output_matrix, stream, scratch_allocator,
                                      best_algo,
                                      /*output_profile_result=*/profile_result);
        default:
          return false;
      }
    }();
  } else {
    //#else   // if not GOOGLE_CUDA or CUDA_VERSION < 11000
    const GemmBackendConfig &backend_config = gemm_config.backend_config;
    auto best_algorithm = [&]() -> absl::optional<se::blas::AlgorithmType> {
      if (algorithm) {
        return *algorithm;
      }
      if (backend_config.algorithm_case() ==
          GemmBackendConfig::ALGORITHM_NOT_SET) {
        return absl::nullopt;
      }
      return backend_config.selected_algorithm();
    }();

    complex128 alpha = {backend_config.alpha_real(),
                        backend_config.alpha_imag()};
    double beta = backend_config.beta();

    launch_ok = [&]() {
      switch (output_shape.element_type()) {
        case F16:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemmWithAlgorithm<Eigen::half, double>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, alpha.real(),
              beta, stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        case F32:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemmWithAlgorithm<float, double>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, alpha.real(),
              beta, stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        case F64:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemmWithAlgorithm<double, double>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, alpha.real(),
              beta, stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        case C64:
          return DoGemmWithAlgorithm<complex64, complex64>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix,
              static_cast<complex64>(alpha), beta, stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        case C128:
          return DoGemmWithAlgorithm<complex128, complex128>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, alpha, beta,
              stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        default:
          return false;
      }
    }();
  }  //#endif  // not GOOGLE_CUDA or CUDA_VERSION < 11000
  if (!launch_ok) {
    return InternalError("Unable to launch cuBLAS gemm on stream %p", stream);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
