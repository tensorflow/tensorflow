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
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

BlasScratchAllocator::BlasScratchAllocator(
    int device_ordinal, se::DeviceMemoryAllocator *memory_allocator)
    : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

int64_t BlasScratchAllocator::GetMemoryLimitInBytes() {
  static const int64_t max_scratch_size = se::GetBlasWorkspaceLimit(
      "TF_CUBLAS_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default
  return max_scratch_size;
}

StatusOr<se::DeviceMemory<uint8_t>> BlasScratchAllocator::AllocateBytes(
    int64_t byte_size) {
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
  return se::DeviceMemory<uint8_t>(buffer_addr);
}

#if GOOGLE_CUDA && CUDA_VERSION >= 11000
StatusOr<se::blas::BlasLtMatmulPlanParams> CreatePlanParams(
    int64_t batch_size, se::blas::DataType blas_dtype,
    tensorflow::DataType dtype, se::blas::MatrixDescriptor lhs_matrix,
    se::blas::MatrixDescriptor rhs_matrix,
    se::blas::MatrixDescriptor output_matrix) {
  se::blas::BlasLtMatmulPlanParams plan_params;
  int64_t m = output_matrix.num_rows;
  int64_t n = output_matrix.num_cols;
  int64_t k = lhs_matrix.reduced_dim();

  plan_params.ab_type = blas_dtype;
  plan_params.c_type = blas_dtype;
  bool allow_tf32 = tensorflow::tensor_float_32_execution_enabled();
  se::blas::ComputationType computation_type;
  if (!se::GetBlasComputationType(dtype, allow_tf32, &computation_type)) {
    return InternalError("Unsupported dtype for batched matmul 2");
  }
  plan_params.computation_type = computation_type;

  plan_params.pointer_mode = se::blas::PointerMode::kHost;
  plan_params.epilogue = se::blas::Epilogue::kDefault;

  plan_params.transa = lhs_matrix.transpose;
  plan_params.transb = rhs_matrix.transpose;
  plan_params.m = m;
  plan_params.n = n;
  plan_params.k = k;
  plan_params.lda = lhs_matrix.num_rows;
  plan_params.ldb = rhs_matrix.num_rows;
  plan_params.ldc = output_matrix.num_rows;
  plan_params.batch_count = batch_size;

  bool broadcast = batch_size == 1;
  int64_t lhs_stride = broadcast ? 0 : lhs_matrix.stride;
  int64_t rhs_stride = broadcast ? 0 : rhs_matrix.stride;
  plan_params.stride_a = lhs_stride;
  plan_params.stride_b = rhs_stride;
  plan_params.stride_c = output_matrix.stride;

  if (VLOG_IS_ON(4)) {
    bool trans_x = lhs_matrix.transpose == se::blas::Transpose::kTranspose;
    bool trans_y = rhs_matrix.transpose == se::blas::Transpose::kTranspose;
    std::string transString[] = {"kNoTranspose", "kTranspose"};
    VLOG(4) << "plan_params.transa: " << transString[trans_x ? 1 : 0]
            << " plan_params.transb: " << transString[trans_y ? 1 : 0]
            << " plan_params.m: " << plan_params.m
            << " plan_params.n: " << plan_params.n
            << " plan_params.k: " << plan_params.k
            << " plan_params.lda: " << plan_params.lda
            << " plan_params.ldb: " << plan_params.ldb
            << " plan_params.ldc: " << plan_params.ldc
            << " plan_params.batch_count: " << plan_params.batch_count
            << " plan_params.stride_a: " << plan_params.stride_a
            << " plan_params.stride_b: " << plan_params.stride_b
            << " plan_params.stride_c: " << plan_params.stride_c;
  }
  return plan_params;
}
#endif

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
                 &scratch_allocator, nullptr);
}

// Converts from an XLA PrimitiveType to a blas::ComputationType, which is
// used to specify the precision with which matmul computations should be
// performed, separately from the precision of the inputs and result.
static absl::optional<se::blas::ComputationType> ComputationTypeFromPrimitive(
    PrimitiveType type) {
  switch (type) {
    case F16:  // Use F32 computation for higher precision.
    case BF16:
    case F32:
      return se::blas::ComputationType::kF32;
    case F64:
      return se::blas::ComputationType::kF64;
    case C64:
      return se::blas::ComputationType::kComplexF32;
    case C128:
      return se::blas::ComputationType::kComplexF64;
    case S32:
      return se::blas::ComputationType::kI32;
    default:
      return absl::nullopt;
  }
}

template <typename Input, typename Output>
static bool DoGemmWithAlgorithm(
    int64_t batch_size, se::blas::MatrixDescriptor lhs,
    se::blas::MatrixDescriptor rhs, se::blas::MatrixDescriptor output_matrix,
    Output alpha, Output beta, se::Stream *stream,
    se::blas::AlgorithmType algorithm,
    se::blas::ProfileResult *output_profile_result) {
  CHECK(output_matrix.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  se::blas::ComputationType computation_type =
      *ComputationTypeFromPrimitive(output_type);
  se::DeviceMemory<Output> output_data(output_matrix.data);

  if (batch_size != 1) {
    return stream
        ->ThenBlasGemmStridedBatchedWithAlgorithm(
            lhs.transpose, rhs.transpose, output_matrix.num_rows,
            output_matrix.num_cols,
            /*size of reduce dim=*/lhs.reduced_dim(),
            /*alpha=*/alpha, lhs.cast<Input>(), lhs.stride,
            /*leading dim of LHS=*/lhs.num_rows, rhs.cast<Input>(),
            /*leading dim of RHS=*/rhs.num_rows, rhs.stride,
            /*beta=*/beta, &output_data,
            /*leading dim of output=*/output_matrix.num_rows,
            output_matrix.stride, batch_size, computation_type, algorithm,
            output_profile_result)
        .ok();
  } else {
    return stream
        ->ThenBlasGemmWithAlgorithm(
            lhs.transpose, rhs.transpose, output_matrix.num_rows,
            output_matrix.num_cols,
            /*size of reduce dim=*/lhs.reduced_dim(),
            /*alpha=*/alpha, lhs.cast<Input>(),
            /*lda=*/lhs.num_rows, rhs.cast<Input>(),
            /*ldb=*/rhs.num_rows,
            /*beta=*/beta, &output_data,
            /*ldc=*/output_matrix.num_rows, computation_type, algorithm,
            output_profile_result)
        .ok();
  }
}

template <typename Input>
static bool DoGemm(int64_t batch_size, const se::blas::MatrixDescriptor &lhs,
                   const se::blas::MatrixDescriptor &rhs,
                   const se::blas::MatrixDescriptor &output_matrix, Input alpha,
                   Input beta, se::Stream *stream,
                   absl::optional<se::blas::AlgorithmType> algorithm,
                   se::blas::ProfileResult *output_profile_result) {
  CHECK(output_matrix.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Input> output_data(output_matrix.data);

  if (algorithm) {
    return DoGemmWithAlgorithm<Input, Input>(batch_size, lhs, rhs,
                                             output_matrix, alpha, beta, stream,
                                             *algorithm, output_profile_result);
  }

  if (batch_size != 1) {
    return stream
        ->ThenBlasGemmStridedBatched(
            lhs.transpose, rhs.transpose, output_matrix.num_rows,
            output_matrix.num_cols, /*size of reduce dim=*/lhs.reduced_dim(),
            /*alpha=*/alpha, lhs.cast<Input>(),
            /*leading dim of LHS=*/lhs.num_rows, lhs.stride, rhs.cast<Input>(),
            /*leading dim of RHS=*/rhs.num_rows, rhs.stride,
            /*beta=*/beta, &output_data,
            /*leading dim of output=*/output_matrix.num_rows,
            output_matrix.stride, batch_size)
        .ok();
  }
  return stream
      ->ThenBlasGemm(lhs.transpose, rhs.transpose, output_matrix.num_rows,
                     output_matrix.num_cols,
                     /*size of reduce dim=*/lhs.reduced_dim(),
                     /*alpha=*/alpha, lhs.cast<Input>(),
                     /*leading dim of LHS=*/lhs.num_rows, rhs.cast<Input>(),
                     /*leading dim of RHS=*/rhs.num_rows,
                     /*beta=*/beta, &output_data,
                     /*leading dim of output=*/output_matrix.num_rows)
      .ok();
}

#if GOOGLE_CUDA && CUDA_VERSION >= 11000
template <typename Input>
static bool DoGemmLt(
    int64_t batch_size, se::blas::MatrixDescriptor lhs_matrix,
    se::blas::MatrixDescriptor rhs_matrix,
    se::blas::MatrixDescriptor output_matrix, se::Stream *stream, Input alpha,
    Input beta, se::ScratchAllocator *scratch_allocator,
    se::blas::IBlasLtMatmulAlgorithm *const algorithm_being_profiled,
    se::blas::ProfileResult *output_profile_result) {
  CHECK(output_matrix.transpose == se::blas::Transpose::kNoTranspose);
  tensorflow::DataType dtype = tensorflow::DataTypeToEnum<Input>::value;

  bool allow_tf32 = tensorflow::tensor_float_32_execution_enabled();
  int device_id = stream->parent()->device_ordinal();

  bool trans_x = lhs_matrix.transpose == se::blas::Transpose::kTranspose;
  bool trans_y = rhs_matrix.transpose == se::blas::Transpose::kTranspose;

  int64_t m = output_matrix.num_rows;
  int64_t n = output_matrix.num_cols;
  auto k = lhs_matrix.reduced_dim();
  bool broadcast = batch_size == 1;
  VLOG(2) << "matmul params: trans_x " << trans_x << " trans_y " << trans_y
          << " adj_x " << false << " adj_y " << false << " m " << m << " n "
          << n << " k " << k << " batch_size " << batch_size << " broadcast "
          << broadcast << " broadcast " << broadcast << " dtype " << dtype
          << " allow_tf32 " << allow_tf32 << " device_id " << device_id;
  se::BatchMatmulParameters matmul_parameters(
      trans_x, trans_y, false, false, m, n, k, batch_size, broadcast, broadcast,
      dtype, dtype, allow_tf32, device_id);
  const se::blas::PlanAndAlgorithms *plan_and_algorithms =
      se::BatchMatmulPlanMapSingleton::GetInstance()->Find(matmul_parameters);

  // If autotuner is disabled, BatchMatmulPlanMapSingleton is not populated.
  if (!plan_and_algorithms) {
    se::blas::DataType blas_dtype = se::blas::ToDataType<Input>::value;
    se::blas::BlasLtMatmulPlanParams plan_params =
        CreatePlanParams(batch_size, blas_dtype, dtype, lhs_matrix, rhs_matrix,
                         output_matrix)
            .ConsumeValueOrDie();

    std::unique_ptr<se::blas::IBlasLtMatmulPlan> plan =
        stream->parent()
            ->CreateBlasLtMatmulPlan(plan_params)
            .ConsumeValueOrDie();

    // When autotuner is disabled, the max_autotune_algorithm_count is 1 and
    // only algo 0 is the default algorithm config.
    static const int64_t max_scratch_size = se::GetBlasWorkspaceLimit(
        "TF_CUBLAS_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default
    std::vector<std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm>> algorithms =
        stream->parent()
            ->GetBlasLtMatmulAlgorithms(plan.get(), max_scratch_size,
                                        /* max_autotune_algorithm_count */ 1)
            .ConsumeValueOrDie();

    plan_and_algorithms =
        se::BatchMatmulPlanMapSingleton::GetInstance()->Insert(
            matmul_parameters, {std::move(plan), std::move(algorithms)});
  }

  const std::unique_ptr<se::blas::IBlasLtMatmulPlan> &plan =
      plan_and_algorithms->plan;
  const std::vector<std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm>>
      &algorithms = plan_and_algorithms->algorithms;

  // 'algorithm_being_profiled' is the algorithm that is being profiled (as the
  // name suggests). RunGemm can have two callers - gemm_algorithm_picker (at
  // compile time) & gpu_executable (at runtime). 'algorithm_being_profiled' is
  // NULL when RunGemm is called during runtime (i.e, via ExecuteOnStream from
  // gpu_executable). In that case, 'ThenBlasLtMatmul' is called with the
  // algorithm selected by the autotuner (from the AutotuneCache) or default
  // algorithm (when autotuner is disabled). When called from gemm_algorithm
  // picker, 'ThenBlasLtMatmul' is called with the 'algorithm_being_profiled'.
  se::blas::IBlasLtMatmulAlgorithm *algorithm_ptr = algorithm_being_profiled;
  if (!algorithm_being_profiled) {
    se::blas::AlgorithmConfig algorithm_config(se::blas::kNoAlgorithm);

    // When autotuner is disabled, AutoTuneBatchMatmul singleton is empty.
    if (!se::AutoTuneBatchMatmul::GetInstance()->Find(matmul_parameters,
                                                      &algorithm_config)) {
      algorithm_config.set_algorithm(0);
      VLOG(4) << "Autotuner disabled: Inserting algorithm id "
              << algorithm_config.algorithm() << " for " << trans_x << " "
              << trans_y << " " << m << " " << n << " " << k << " "
              << batch_size << " " << broadcast << " " << broadcast << " "
              << dtype << " " << allow_tf32 << " " << device_id;
      se::AutoTuneBatchMatmul::GetInstance()->Insert(matmul_parameters,
                                                     algorithm_config);
    }
    se::blas::AlgorithmType algorithm_idx = algorithm_config.algorithm();
    algorithm_ptr = algorithms[algorithm_idx].get();
  }

  se::DeviceMemory<Input> output_data(output_matrix.data);
  return stream
      ->ThenBlasLtMatmul(plan.get(), alpha, lhs_matrix.cast<Input>(),
                         rhs_matrix.cast<Input>(), beta, &output_data,
                         scratch_allocator, algorithm_ptr, {},
                         output_profile_result)
      .ok();
}
#endif

MatrixDescs PopulateInputOutputMatrices(const GpuGemmConfig &gemm_config,
                                        se::DeviceMemoryBase lhs_buffer,
                                        se::DeviceMemoryBase rhs_buffer,
                                        se::DeviceMemoryBase output_buffer) {
  const Shape &output_shape = gemm_config.output_shape;
  const Shape &lhs_shape = gemm_config.lhs_shape;
  const Shape &rhs_shape = gemm_config.rhs_shape;
  const GemmBackendConfig &backend_config = gemm_config.backend_config;
  const DotDimensionNumbers &dim_nums = backend_config.dot_dimension_numbers();
  absl::Span<const int64_t> output_batch_dims =
      (dim_nums.lhs_batch_dimensions_size() >
       dim_nums.rhs_batch_dimensions_size())
          ? dim_nums.lhs_batch_dimensions()
          : dim_nums.rhs_batch_dimensions();

  int64_t batch_size = backend_config.batch_size();
  int64_t output_row_dim = output_batch_dims.size();
  int64_t output_col_dim = output_row_dim + 1;

  if (backend_config.rhs_stride() && backend_config.lhs_stride()) {
    CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
             dim_nums.rhs_batch_dimensions_size());
  }

  int64_t output_num_rows = output_shape.dimensions(output_row_dim);
  int64_t output_num_cols = output_shape.dimensions(output_col_dim);

  auto validate_matrix = [&](const Shape &shape, auto batch_dimensions) {
    int64_t row_dim = batch_dimensions.size();
    int64_t col_dim = row_dim + 1;
    CHECK_EQ(row_dim + 2, shape.rank());

    // Check that the batch dims don't cover the last two dims.
    for (int64_t batch_dim : batch_dimensions) {
      CHECK_NE(row_dim, batch_dim);
      CHECK_NE(col_dim, batch_dim);
    }

    // Verify that the non-batch dimensions are minor-most. This is required for
    // efficient access.
    CHECK_LT(shape.layout().minor_to_major(row_dim), 2);
    CHECK_LT(shape.layout().minor_to_major(col_dim), 2);
  };

  validate_matrix(lhs_shape, dim_nums.lhs_batch_dimensions());
  validate_matrix(rhs_shape, dim_nums.rhs_batch_dimensions());
  validate_matrix(output_shape, output_batch_dims);

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
                             int64_t row_dim, bool transpose,
                             int64_t stride) -> se::blas::MatrixDescriptor {
    bool is_row_major = LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch =
        LayoutUtil::Minor(shape.layout(), row_dim) - row_dim !=
        LayoutUtil::Minor(output_shape.layout(), row_dim) -
            output_batch_dims.size();
    int64_t rows =
        shape.dimensions(row_dim + static_cast<int64_t>(is_row_major));
    int64_t cols =
        shape.dimensions(row_dim + static_cast<int64_t>(!is_row_major));
    if (stride != 0) {
      CHECK_EQ(stride, rows * cols);
    }
    return se::blas::MatrixDescriptor{data,
                                      transpose ^ layout_mismatch
                                          ? se::blas::Transpose::kTranspose
                                          : se::blas::Transpose::kNoTranspose,
                                      rows, cols, stride};
  };

  bool lhs_transpose = dim_nums.lhs_contracting_dimensions(0) ==
                       dim_nums.lhs_batch_dimensions_size();
  bool rhs_transpose = dim_nums.rhs_contracting_dimensions(0) ==
                       dim_nums.rhs_batch_dimensions_size() + 1;

  se::blas::MatrixDescriptor lhs_matrix = make_descriptor(
      lhs_buffer, lhs_shape, dim_nums.lhs_batch_dimensions_size(),
      lhs_transpose, backend_config.lhs_stride());
  se::blas::MatrixDescriptor rhs_matrix = make_descriptor(
      rhs_buffer, rhs_shape, dim_nums.rhs_batch_dimensions_size(),
      rhs_transpose, backend_config.rhs_stride());

  if (LayoutUtil::Minor(output_shape.layout(), output_row_dim) != 0) {
    std::swap(lhs_matrix, rhs_matrix);
    std::swap(output_num_cols, output_num_rows);
  }

  se::blas::MatrixDescriptor out_matrix{
      output_buffer, /*needs_transpose=*/se::blas::Transpose::kNoTranspose,
      output_num_rows, output_num_cols, output_num_rows * output_num_cols};
  return std::make_tuple(lhs_matrix, rhs_matrix, out_matrix);
}

Status RunGemm(const GpuGemmConfig &gemm_config,
               se::DeviceMemoryBase lhs_buffer, se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream *stream,
               bool implements_whole_instruction,
               absl::optional<int64_t> profile_index,
               BlasScratchAllocator *scratch_allocator,
               se::blas::IBlasLtMatmulAlgorithm *const algorithm_being_profiled,
               se::blas::ProfileResult *profile_result,
               absl::optional<se::blas::AlgorithmType> algorithm) {
  VLOG(2) << "Executing a GemmThunk";
  se::blas::MatrixDescriptor lhs_matrix, rhs_matrix, output_matrix;
  std::tie(lhs_matrix, rhs_matrix, output_matrix) = PopulateInputOutputMatrices(
      gemm_config, lhs_buffer, rhs_buffer, output_buffer);
  const Shape &output_shape = gemm_config.output_shape;
  int64_t batch_size = gemm_config.backend_config.batch_size();
  const GemmBackendConfig &backend_config = gemm_config.backend_config;
  // The BlasLtMatmul routines (unlike BlasGemm, BlasGemmBatched etc.) take
  // alpha and beta with the same type as the matrices.
  complex128 alpha = {backend_config.alpha_real(), backend_config.alpha_imag()};
  double beta = backend_config.beta();

  bool launch_ok = false;
  // The BlasLtMatmul routines are only supported from CUDA 11.0 onward.
#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11000
  std::unordered_set<PrimitiveType> enabled_types = {F16, F32, F64, C64, C128};
  if (tensorflow::EnableCublasLtGemm() &&
      enabled_types.find(output_shape.element_type()) != enabled_types.end()) {
    bool has_activation_relu =
        static_cast<se::dnn::ActivationMode>(
            backend_config.activation_mode()) == se::dnn::ActivationMode::kRelu;
    launch_ok = [&]() {
      se::blas::IBlasLtMatmulAlgorithm *best_algorithm = nullptr;
      if (algorithm_being_profiled) {
        best_algorithm = algorithm_being_profiled;
      }
      complex128 beta_cmplx = {beta, 0};
      switch (output_shape.element_type()) {
        case F16:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemmLt<Eigen::half>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, stream,
              static_cast<Eigen::half>(alpha.real()),
              static_cast<Eigen::half>(beta), scratch_allocator, best_algorithm,
              /*output_profile_result=*/profile_result);
        case F32:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemmLt<float>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, stream,
              static_cast<float>(alpha.real()), static_cast<float>(beta),
              scratch_allocator, best_algorithm,
              /*output_profile_result=*/profile_result);
        case F64:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemmLt<double>(batch_size, lhs_matrix, rhs_matrix,
                                  output_matrix, stream,
                                  static_cast<double>(alpha.real()), beta,
                                  scratch_allocator, best_algorithm,
                                  /*output_profile_result=*/profile_result);
        case C64:
          return DoGemmLt<complex64>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, stream,
              static_cast<complex64>(alpha), static_cast<complex64>(beta_cmplx),
              scratch_allocator, best_algorithm,
              /*output_profile_result=*/profile_result);
        case C128:
          return DoGemmLt<complex128>(batch_size, lhs_matrix, rhs_matrix,
                                      output_matrix, stream, alpha, beta_cmplx,
                                      scratch_allocator, best_algorithm,
                                      /*output_profile_result=*/profile_result);
        default:
          LOG(ERROR) << "Unexpected GEMM datatype: " << output_shape.ToString();
          return false;
      }
    }();
  } else {
#endif
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
    const Shape &lhs_shape = gemm_config.lhs_shape;
    const Shape &rhs_shape = gemm_config.rhs_shape;

    launch_ok = [&]() {
      switch (output_shape.element_type()) {
        case S32: {
          if (!best_algorithm) {
            LOG(ERROR) << "Only extended GEMM is supported for int32_t";
            return false;
          }
          CHECK_EQ(alpha.imag(), 0);
          if (lhs_shape.element_type() == PrimitiveType::S8 &&
              rhs_shape.element_type() == lhs_shape.element_type()) {
            return DoGemmWithAlgorithm<int8_t, int32_t>(
                batch_size, lhs_matrix, rhs_matrix, output_matrix,
                static_cast<int32_t>(alpha.real()), static_cast<int32_t>(beta),
                stream, *best_algorithm,
                /*output_profile_result=*/profile_result);
          }
          LOG(ERROR) << "For int32_t GEMM output only int8_t input is "
                        "supported, got input: "
                     << primitive_util::LowercasePrimitiveTypeName(
                            lhs_shape.element_type());
          return false;
        }
        case F16:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemm<Eigen::half>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix,
              static_cast<Eigen::half>(alpha.real()),
              static_cast<Eigen::half>(beta), stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        case BF16:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemm<Eigen::bfloat16>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix,
              static_cast<Eigen::bfloat16>(alpha.real()),
              static_cast<Eigen::bfloat16>(beta), stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        case F32:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemm<float>(batch_size, lhs_matrix, rhs_matrix,
                               output_matrix, alpha.real(), beta, stream,
                               best_algorithm,
                               /*output_profile_result=*/profile_result);
        case F64:
          CHECK_EQ(alpha.imag(), 0);
          return DoGemm<double>(batch_size, lhs_matrix, rhs_matrix,
                                output_matrix, alpha.real(), beta, stream,
                                best_algorithm,
                                /*output_profile_result=*/profile_result);
        case C64:
          return DoGemm<complex64>(batch_size, lhs_matrix, rhs_matrix,
                                   output_matrix, static_cast<complex64>(alpha),
                                   static_cast<complex64>(beta), stream,
                                   best_algorithm,
                                   /*output_profile_result=*/profile_result);
        case C128:
          return DoGemm<complex128>(
              batch_size, lhs_matrix, rhs_matrix, output_matrix, alpha,
              static_cast<complex128>(beta), stream, best_algorithm,
              /*output_profile_result=*/profile_result);
        default:
          LOG(ERROR) << "Unexpected GEMM datatype: " << output_shape.ToString();
          return false;
      }
    }();
#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11000
  }
#endif  // not GOOGLE_CUDA or CUDA_VERSION < 11000
  if (!launch_ok) {
    return InternalError("Unable to launch cuBLAS gemm on stream %p", stream);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
