// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Manjunath Kudlur <keveman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H

#if defined(EIGEN_USE_GPU)

namespace Eigen {
namespace internal {

template <typename OutExpr, typename InExpr, typename Op, typename Indices,
          bool Tileable>
class TensorExecutor<
    const TensorAssignOp<
        OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
    GpuDevice, false, Tileable> {
 public:
  typedef const TensorAssignOp<
      OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>
      Expression;
  static void run(const Expression& expr, const GpuDevice& device);
};

template <typename OutExpr, typename InExpr, typename Op, typename Indices,
          bool Tileable>
class TensorExecutor<
    const TensorAssignOp<
        OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
    GpuDevice, true, Tileable> {
 public:
  typedef const TensorAssignOp<
      OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>
      Expression;
  static void run(const Expression& expr, const GpuDevice& device);
};

template <typename InExpr, typename Op, typename Indices, bool Tileable>
class TensorExecutor<const TensorEvalToOp<const TensorReductionOp<
                         Op, const Indices, const InExpr> >,
                     GpuDevice, false, Tileable> {
 public:
  typedef const TensorEvalToOp<
      const TensorReductionOp<Op, const Indices, const InExpr> > Expression;
  static void run(const Expression& expr, const GpuDevice& device);
};

template <typename InExpr, typename Op, typename Indices, bool Tileable>
class TensorExecutor<const TensorEvalToOp<const TensorReductionOp<
                         Op, const Indices, const InExpr> >,
                     GpuDevice, true, Tileable> {
 public:
  typedef const TensorEvalToOp<
      const TensorReductionOp<Op, const Indices, const InExpr> > Expression;
  static void run(const Expression& expr, const GpuDevice& device);
};

}  // end namespace internal
}  // end namespace Eigen

#if defined(__CUDACC__)

namespace Eigen {

namespace internal {

namespace {

#define DIVUP(x, y) (((x) + (y)-1) / (y))

// Initialize output[0..size-1] with val
template <typename Output>
__global__ void InitVector(const float val, int size, Output output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
    output.coeffRef(i) = val;
  }
}

// -----------------------------------------------------------------------------
// Column Reduction kernels
// -----------------------------------------------------------------------------
template <int GRID_DIM, int BLOCK_DIM, int NUM_PER_THREAD, typename Input,
          typename Output, typename Reducer>
__global__ void ColumnReduceKernel(Reducer reducer, const Input input, int rows,
                                   int cols, Output output) {
  assert(blockDim.x == BLOCK_DIM);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  assert(gridDim.x == GRID_DIM);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  typedef typename Input::Index Index;

  const Index num_input_points = DIVUP(rows, NUM_PER_THREAD) * cols;
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;

  for (Index i = bx * BLOCK_DIM + tx; i < num_input_points;
       i += BLOCK_DIM * GRID_DIM) {
    const Index input_col = i % cols;
    const Index input_row_begin =
        ((i / cols) % DIVUP(rows, NUM_PER_THREAD)) * NUM_PER_THREAD;
    float reduced_val = reducer.bottom_value();
    for (int j = 0; j < NUM_PER_THREAD; ++j) {
      float val = ((input_col < cols) && (input_row_begin + j < rows))
                      ? input.coeff((input_row_begin + j) * cols + input_col)
                      : reducer.bottom_value();
      reduced_val = reducer(reduced_val, val);
    }
#if __CUDA_ARCH__ >= 300
    reducer.atomic_reduce(&output.coeffRef(input_col), reduced_val);
#endif
  }
}

// -----------------------------------------------------------------------------
// Row Reduction kernels
// -----------------------------------------------------------------------------
template <int GRID_DIM, int BLOCK_DIM, int NUM_PER_THREAD, typename Input,
          typename Output, typename Reducer>
__global__ void RowReduceKernel(Reducer reducer, const Input input, int rows,
                                int cols, Output output) {
  assert(BLOCK_DIM % 32 == 0);
  assert(blockDim.x == BLOCK_DIM);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  assert(gridDim.x == GRID_DIM);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  const int unroll_times = 16;
  assert(NUM_PER_THREAD % unroll_times == 0);

  typedef typename Input::Index Index;

  __shared__ float temp[BLOCK_DIM];

  const Index input_col_blocks = DIVUP(cols, BLOCK_DIM * NUM_PER_THREAD);
  const Index num_input_blocks = input_col_blocks * rows;

  const int bx = blockIdx.x;
  const int tx = threadIdx.x;

  for (Index i = bx; i < num_input_blocks; i += GRID_DIM) {
    const Index col_block = i % input_col_blocks;
    const Index row_block = i / input_col_blocks;
    const Index col_begin = col_block * BLOCK_DIM * NUM_PER_THREAD + tx;
    const Index row = row_block;
    float reduced_val = reducer.bottom_value();
    if (row < rows) {
      for (Index j = 0; j < NUM_PER_THREAD; j += unroll_times) {
        const Index last_col = col_begin + BLOCK_DIM * (j + unroll_times - 1);
        if (last_col >= cols) {
          // We can skip the last iteration of the loop since we know
          // that col >= cols there.
#pragma unroll
          for (int k = 0; k < unroll_times - 1; ++k) {
            const Index col = col_begin + BLOCK_DIM * (j + k);
            const float val = (col < cols ? input.coeff(row * cols + col)
                               : reducer.bottom_value());
            reduced_val = reducer(reduced_val, val);
          }
          break;  // col < cols for all later iterations.
        } else {
          // Faster version of the loop with no branches after unrolling.
#pragma unroll
          for (int k = 0; k < unroll_times; ++k) {
            const Index col = col_begin + BLOCK_DIM * (j + k);
            reduced_val = reducer(reduced_val, input.coeff(row * cols + col));
          }
        }
      }
    }
    temp[tx] = reduced_val;

    __syncthreads();
    const int warp_id = tx & 31;
    if (warp_id < 16) temp[tx] = reducer(temp[tx], temp[tx + 16]);
    if (warp_id < 8) temp[tx] = reducer(temp[tx], temp[tx + 8]);
    if (warp_id < 4) temp[tx] = reducer(temp[tx], temp[tx + 4]);
    if (warp_id < 2) temp[tx] = reducer(temp[tx], temp[tx + 2]);
    if (warp_id < 1) temp[tx] = reducer(temp[tx], temp[tx + 1]);

    if (warp_id == 0) {
      if (row < rows) {
#if __CUDA_ARCH__ >= 300
        reducer.atomic_reduce(&output.coeffRef(row), temp[tx]);
#endif
      }
    }

    __syncthreads();
  }
}

template <typename Input, typename Output, typename Reducer>
void ColumnReduceCuda(Reducer reducer, const GpuDevice& device,
                      const Input input, int rows, int cols, Output output) {
  const int block_size = 256;
  const int grid_size = 128;
  const int num_per_thread = 16;
  LAUNCH_CUDA_KERNEL(InitVector, 32, 1024, 0, device, reducer.bottom_value(),
                     cols, output);
  LAUNCH_CUDA_KERNEL(
      (ColumnReduceKernel<grid_size, block_size, num_per_thread>), grid_size,
      block_size, 0, device, reducer, input, rows, cols, output);
}

template <typename Input, typename Output, typename Reducer>
void RowReduceCuda(Reducer reducer, const GpuDevice& device, const Input input,
                   int rows, int cols, Output output) {
  const int block_size = 256;
  const int grid_size = 32;
  const int num_per_thread = 128;
  LAUNCH_CUDA_KERNEL(InitVector, 32, 1024, 0, device, reducer.bottom_value(),
                     rows, output);
  LAUNCH_CUDA_KERNEL((RowReduceKernel<grid_size, block_size, num_per_thread>),
                     grid_size, block_size, 0, device, reducer, input, rows,
                     cols, output);
}

// Provides arbitrary sum reductions, applying a function across the
// right argument being reduced prior to summing
template <typename F>
struct FnSumReducer {
  __host__ __device__ FnSumReducer(F f) : f_(f) {}
  __host__ __device__ float bottom_value() { return 0.0f; }
  __device__ float operator()(float x, float y) const { return x + f_(y); }
  __device__ void atomic_reduce(float* x, float y) const { atomicAdd(x, y); }

  F f_;
};

// Identity is used for the basic SumReduction
struct Identity {
  __device__ float operator()(float x) const { return x; }
};

struct CudaSumReducer : FnSumReducer<Identity> {
  __host__ __device__ CudaSumReducer() : FnSumReducer(Identity()) {}
};

struct CudaMaxReducer {
  // nvcc doesn't recognize numeric_limits<float>::lowest for some reason.
  CudaMaxReducer() {
    bottom_value_ = -3.40282347E+38F;  // std::numeric_limits<float>::lowest();
  }
  __host__ __device__ float bottom_value() { return bottom_value_; }
  __device__ float operator()(float x, float y) const { return fmax(x, y); }

  // This is equivalent to atomicMax(x, y), but CUDA does not have atomicMax for
  // float data type. Instead, this atomically compares-and-swaps the old value
  // at x with y. If the old value returned by the CAS operation was already
  // larger than y, or what was read before, it declares success and finishes,
  // otherwise repeats the procedure.
  __device__ void atomic_reduce(float* x, float y) {
    unsigned int old_val = *reinterpret_cast<unsigned int*>(x);
    while (*reinterpret_cast<float*>(&old_val) < y) {
      unsigned int current_val =
          atomicCAS(reinterpret_cast<unsigned int*>(x), old_val,
                    *reinterpret_cast<unsigned int*>(&y));
      if (old_val == current_val) {
        break;
      }
      old_val = current_val;
    }
  }
  float bottom_value_;
};

}  // end namespace

template <typename Op>
struct IsFloatSumReduction {
  static const bool value = false;
};

template <>
struct IsFloatSumReduction<SumReducer<float> > {
  static const bool value = true;
};

template <typename Op>
struct IsFloatMaxReduction {
  static const bool value = false;
};

template <>
struct IsFloatMaxReduction<MaxReducer<float> > {
  static const bool value = true;
};

template <typename Op>
struct SumOrMaxOfFloat {
  static const bool value =
      IsFloatSumReduction<Op>::value || IsFloatMaxReduction<Op>::value;
};

enum ReductionType { ROW_REDUCE, COL_REDUCE, UNOPTIMIZED };

template <typename Op, typename Expr, typename ReductionExpr>
ReductionType GetReductionType(const Expr& expr,
                               const ReductionExpr& reduction_expr,
                               const GpuDevice& device, std::size_t* rows,
                               std::size_t* cols) {
  typedef TensorEvaluator<const Expr, GpuDevice> EvalExpr;
  typedef TensorEvaluator<const ReductionExpr, GpuDevice> ReductionEvalExpr;

  if (device.majorDeviceVersion() < 3) {
    return UNOPTIMIZED;
  }
  const EvalExpr eval_expr(expr, device);

  // We only have fast reductions for sum/max of float.
  if (!SumOrMaxOfFloat<Op>::value) {
    return UNOPTIMIZED;
  }

  // For sum/max of float, if we are doing a full reduction, we can
  // use the ROW_REDUCE optimization.
  if (ReductionEvalExpr::NumReducedDims == ReductionEvalExpr::NumInputDims) {
    *rows = 1;
    *cols = array_prod(eval_expr.dimensions());
    return ROW_REDUCE;
  }

  if (ReductionEvalExpr::NumReducedDims > 1) {
    return UNOPTIMIZED;
  }

  const int dim = reduction_expr.dims()[0];
  if (static_cast<int>(ReductionEvalExpr::Layout) ==
      static_cast<int>(RowMajor)) {
    if (dim == ReductionEvalExpr::NumInputDims - 1) {
      *rows = array_prod(eval_expr.dimensions()) /
              eval_expr.dimensions()[ReductionEvalExpr::NumInputDims - 1];
      *cols = eval_expr.dimensions()[ReductionEvalExpr::NumInputDims - 1];
      if (*cols < 32) return UNOPTIMIZED;
      return ROW_REDUCE;
    } else if (dim == 0) {
      *rows = eval_expr.dimensions()[0];
      *cols = array_prod(eval_expr.dimensions()) / eval_expr.dimensions()[0];
      if (*rows < 32) return UNOPTIMIZED;
      return COL_REDUCE;
    }
  } else if (static_cast<int>(ReductionEvalExpr::Layout) ==
             static_cast<int>(ColMajor)) {
    if (dim == ReductionEvalExpr::NumInputDims - 1) {
      *rows = eval_expr.dimensions()[ReductionEvalExpr::NumInputDims - 1];
      *cols = array_prod(eval_expr.dimensions()) /
              eval_expr.dimensions()[ReductionEvalExpr::NumInputDims - 1];
      if (*rows < 32) return UNOPTIMIZED;
      return COL_REDUCE;
    } else if (dim == 0) {
      *rows = array_prod(eval_expr.dimensions()) / eval_expr.dimensions()[0];
      *cols = eval_expr.dimensions()[0];
      if (*cols < 32) return UNOPTIMIZED;
      return ROW_REDUCE;
    }
  }
  return UNOPTIMIZED;
}

template <typename Expression, typename Index, bool Vectorizable>
struct LaunchKernel;

template <typename Expression, typename Index>
struct LaunchKernel<Expression, Index, true> {
  static void launch(int num_blocks, int block_size, const GpuDevice& device,
                     const TensorEvaluator<Expression, GpuDevice>& evaluator,
                     Index size) {
    LAUNCH_CUDA_KERNEL(
        (EigenMetaKernel_Vectorizable<TensorEvaluator<Expression, GpuDevice>,
                                      Index>),
        num_blocks, block_size, 0, device, evaluator, size);
  }
};

template <typename Expression, typename Index>
struct LaunchKernel<Expression, Index, false> {
  static void launch(int num_blocks, int block_size, const GpuDevice& device,
                     const TensorEvaluator<Expression, GpuDevice>& evaluator,
                     Index size) {
    LAUNCH_CUDA_KERNEL(
        (EigenMetaKernel_NonVectorizable<TensorEvaluator<Expression, GpuDevice>,
                                         Index>),
        num_blocks, block_size, 0, device, evaluator, size);
  }
};

template <typename F, typename LHS, typename RHS, bool Compatible>
struct LaunchRowReduce;

template <typename F, typename LHS, typename RHS>
struct LaunchRowReduce<F, LHS, RHS, true> {
  static void launch(const GpuDevice& device, RHS input, std::size_t rows,
                     std::size_t cols, LHS output) {
    RowReduceCuda(F(), device, input, rows, cols, output);
  }
};

template <typename F, typename LHS, typename RHS>
struct LaunchRowReduce<F, LHS, RHS, false> {
  static void launch(const GpuDevice& device, RHS input, std::size_t rows,
                     std::size_t cols, LHS output) {}
};

template <typename F, typename LHS, typename RHS, bool Compatible>
struct LaunchColReduce;

template <typename F, typename LHS, typename RHS>
struct LaunchColReduce<F, LHS, RHS, true> {
  static void launch(const GpuDevice& device, RHS input, std::size_t rows,
                     std::size_t cols, LHS output) {
    ColumnReduceCuda(F(), device, input, rows, cols, output);
  }
};

template <typename F, typename LHS, typename RHS>
struct LaunchColReduce<F, LHS, RHS, false> {
  static void launch(const GpuDevice& device, RHS input, std::size_t rows,
                     std::size_t cols, LHS output) {}
};

template <typename Expression, typename Device, bool Vectorizable>
class TensorAssignExecutorHelper;

template <typename OutExpr, typename InExpr, typename Op, typename Indices,
          bool Vectorizable>
class TensorAssignExecutorHelper<
    const TensorAssignOp<
      OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
    GpuDevice, Vectorizable> {
 public:
  typedef const TensorAssignOp<
    OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>
    Expression;

  typedef typename Expression::Index Index;
  typedef TensorEvaluator<OutExpr, GpuDevice> LHSEval;
  typedef TensorEvaluator<const InExpr, GpuDevice> RHSEval;
  static inline void run(const Expression& expr, const GpuDevice& device) {
    std::size_t rows, cols;
    const ReductionType reduction_type =
        GetReductionType<Op>(expr.rhsExpression().expression(),
                             expr.rhsExpression(), device, &rows, &cols);
    if (reduction_type == UNOPTIMIZED) {
      TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
      const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
      if (needs_assign) {
        const int num_blocks = device.getNumCudaMultiProcessors() *
                               device.maxCudaThreadsPerMultiProcessor() /
                               device.maxCudaThreadsPerBlock();
        const int block_size = device.maxCudaThreadsPerBlock();
        const Index size = array_prod(evaluator.dimensions());
        LaunchKernel<Expression, Index, Vectorizable>::launch(
            num_blocks, block_size, device, evaluator, size);
      }
      evaluator.cleanup();
    } else {
      LHSEval output(expr.lhsExpression(), device);
      RHSEval input(expr.rhsExpression().expression(), device);
      bool lhs_needs_assign = output.evalSubExprsIfNeeded(NULL);
      bool rhs_needs_assign = input.evalSubExprsIfNeeded(NULL);
      if (lhs_needs_assign && rhs_needs_assign) {
        const bool Compatible =
            IsFloatSumReduction<Op>::value || IsFloatMaxReduction<Op>::value;
        if (reduction_type == ROW_REDUCE) {
          if (IsFloatSumReduction<Op>::value) {
            LaunchRowReduce<CudaSumReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          } else if (IsFloatMaxReduction<Op>::value) {
            LaunchRowReduce<CudaMaxReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          } else {
            // Unsupported reduction type
            assert(false && "Unsupported reduction function for ROW_REDUCE");
          }
        } else {
          if (IsFloatSumReduction<Op>::value) {
            LaunchColReduce<CudaSumReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          } else if (IsFloatMaxReduction<Op>::value) {
            LaunchColReduce<CudaMaxReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          } else {
            // Unsupported reduction type
            assert(false && "Unsupported reduction function for COL_REDUCE");
          }
        }
      }
      input.cleanup();
      output.cleanup();
    }
  }
};

template <typename OutExpr, typename InExpr, typename Op, typename Indices,
          bool Tileable>
inline void TensorExecutor<
    const TensorAssignOp<
        OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
    GpuDevice, false, Tileable>::run(const Expression& expr,
                                     const GpuDevice& device) {
  TensorAssignExecutorHelper<
      const TensorAssignOp<
          OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
      GpuDevice, false>::run(expr, device);
}

template <typename OutExpr, typename InExpr, typename Op, typename Indices,
          bool Tileable>
inline void TensorExecutor<
    const TensorAssignOp<
        OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
    GpuDevice, true, Tileable>::run(const Expression& expr,
                                    const GpuDevice& device) {
  TensorAssignExecutorHelper<
      const TensorAssignOp<
          OutExpr, TensorReductionOp<Op, Indices const, InExpr const> const>,
      GpuDevice, true>::run(expr, device);
}

template <typename T, typename Index>
struct PtrWrapper {
  EIGEN_DEVICE_FUNC PtrWrapper(T* ptr) : m_ptr(ptr) {}
  EIGEN_DEVICE_FUNC T& coeffRef(Index i) { return *(m_ptr + i); }
  T* m_ptr;
};

template <typename Expression, typename Device, bool Vectorizable>
class TensorEvalToExecutorHelper;

template <typename InExpr, typename Op, typename Indices, bool Vectorizable>
class TensorEvalToExecutorHelper<const TensorEvalToOp<const TensorReductionOp<
                                     Op, const Indices, const InExpr> >,
                                 GpuDevice, Vectorizable> {
 public:
  typedef const TensorEvalToOp<const TensorReductionOp<
      Op, const Indices, const InExpr> > Expression;
  typedef typename Expression::Index Index;
  typedef TensorEvaluator<const InExpr, GpuDevice> RHSEval;

  static inline void run(const Expression& expr, const GpuDevice& device) {
    std::size_t rows, cols;
    const ReductionType reduction_type =
        GetReductionType<Op>(expr.expression().expression(), expr.expression(),
                             device, &rows, &cols);
    if (reduction_type == UNOPTIMIZED) {
      TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
      const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
      if (needs_assign) {
        const int num_blocks = device.getNumCudaMultiProcessors() *
                               device.maxCudaThreadsPerMultiProcessor() /
                               device.maxCudaThreadsPerBlock();
        const int block_size = device.maxCudaThreadsPerBlock();
        const Index size = array_prod(evaluator.dimensions());
        LaunchKernel<Expression, Index, Vectorizable>::launch(
            num_blocks, block_size, device, evaluator, size);
      }
      evaluator.cleanup();
    } else {
      typedef typename internal::remove_const<typename Expression::Scalar>::type Scalar;
      PtrWrapper<Scalar, Index> output(expr.buffer());
      TensorEvaluator<const InExpr, GpuDevice> input(
          expr.expression().expression(), device);
      typedef PtrWrapper<Scalar, Index> LHSEval;
      typedef TensorEvaluator<const InExpr, GpuDevice> RHSEval;
      bool rhs_needs_assign = input.evalSubExprsIfNeeded(NULL);
      if (rhs_needs_assign) {
        const bool Compatible =
            IsFloatSumReduction<Op>::value || IsFloatMaxReduction<Op>::value;
        if (reduction_type == ROW_REDUCE) {
          if (IsFloatSumReduction<Op>::value) {
            LaunchRowReduce<CudaSumReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          } else if (IsFloatMaxReduction<Op>::value) {
            LaunchRowReduce<CudaMaxReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          }
        } else {
          if (IsFloatSumReduction<Op>::value) {
            LaunchColReduce<CudaSumReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          } else if (IsFloatMaxReduction<Op>::value) {
            LaunchColReduce<CudaMaxReducer, LHSEval, RHSEval,
                            Compatible>::launch(device, input, rows, cols,
                                                output);
          }
        }
      }
      input.cleanup();
    }
  }
};

template <typename InExpr, typename Op, typename Indices, bool Tileable>
inline void
TensorExecutor<const TensorEvalToOp<
                   const TensorReductionOp<Op, const Indices, const InExpr> >,
               GpuDevice, false, Tileable>::run(const Expression& expr,
                                                const GpuDevice& device) {
  TensorEvalToExecutorHelper<const TensorEvalToOp<const TensorReductionOp<
                                 Op, const Indices, const InExpr> >,
                             GpuDevice, false>::run(expr, device);
}

template <typename InExpr, typename Op, typename Indices, bool Tileable>
inline void
TensorExecutor<const TensorEvalToOp<
                   const TensorReductionOp<Op, const Indices, const InExpr> >,
               GpuDevice, true, Tileable>::run(const Expression& expr,
                                               const GpuDevice& device) {
  TensorEvalToExecutorHelper<const TensorEvalToOp<const TensorReductionOp<
                                 Op, const Indices, const InExpr> >,
                             GpuDevice, true>::run(expr, device);
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // __CUDACC__
#endif  // EIGEN_USE_GPU
#endif  // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_CUDA_H
