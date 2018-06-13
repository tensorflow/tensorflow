/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/util/matmul_autotune.h"
#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#endif
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMatMul;

namespace {
// Converts a TensorFlow Tensor to an Eigen Matrix.
template <typename T>
Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
ToEigenMatrix(const Tensor& tensor) {
  auto matrix = tensor.matrix<T>();
  return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(
      matrix.data(), matrix.dimension(0), matrix.dimension(1));
}

// Converts a TensorFlow Tensor to an Eigen Vector.
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> ToEigenVector(Tensor* tensor) {
  auto v = tensor->flat<T>();
  return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(v.data(), v.dimension(0));
}
template <typename T>
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> ToEigenVector(
    const Tensor& tensor) {
  auto v = tensor.flat<T>();
  return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(v.data(), v.dimension(0));
}
}  // namespace

// If either side can be represented as a vector, do an explicit vector
// matrix multiply and return true; else return false.
//
// Note: this uses plain Eigen and not Eigen Tensor because it is more
// efficient.
template <typename T>
bool ExplicitVectorMatrixOptimization(
    const Tensor& a, const Tensor& b,
    const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
    Tensor* out) {
  if (out->dim_size(0) == 1) {
    if (dim_pair[0].second == 0) {
      // Note: this case is optimized in Eigen Tensors.
      return false;
    } else {
      auto out_v = ToEigenVector<T>(out);
      auto a_v = ToEigenVector<T>(a);
      auto b_m = ToEigenMatrix<T>(b);
      out_v.noalias() = b_m * a_v;
    }
    return true;
  } else if (out->dim_size(1) == 1) {
    auto out_v = ToEigenVector<T>(out);
    auto a_m = ToEigenMatrix<T>(a);
    auto b_v = ToEigenVector<T>(b);
    if (dim_pair[0].first == 0) {
      out_v.noalias() = a_m.transpose() * b_v;
    } else {
      out_v.noalias() = a_m * b_v;
    }
    return true;
  }
  return false;
}
// Half is not supported.
template <>
bool ExplicitVectorMatrixOptimization<Eigen::half>(
    const Tensor& a, const Tensor& b,
    const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
    Tensor* out) {
  return false;
}

template <typename Device, typename T>
struct LaunchMatMulBase {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  typedef se::blas::AlgorithmType AlgorithmType;
#else
  typedef int64 AlgorithmType;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  static void launch(
      OpKernelContext* ctx, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      std::vector<AlgorithmType>* algorithms, bool use_aututone, Tensor* out) {
#ifndef TENSORFLOW_USE_SYCL
    // An explicit vector-matrix multiply is much better optimized than an
    // implicit one and this is a bottleneck during non-batched inference.
    bool was_vector = ExplicitVectorMatrixOptimization<T>(a, b, dim_pair, out);
    if (!was_vector) {
#endif  // TENSORFLOW_USE_SYCL
      functor::MatMulFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                          out->matrix<T>(), a.matrix<T>(),
                                          b.matrix<T>(), dim_pair);
#ifndef TENSORFLOW_USE_SYCL
    }
#endif  // TENSORFLOW_USE_SYCL
  }

  static void GetBlasGemmAlgorithm(OpKernelConstruction* ctx,
                                   std::vector<int64>* algorithms,
                                   bool* algorithm_set_flag) {}
};
// On CPUs, we ignore USE_CUBLAS
template <typename T>
struct LaunchMatMulCPU : LaunchMatMulBase<CPUDevice, T> {};

template <typename T, bool USE_CUBLAS>
struct LaunchMatMul<CPUDevice, T, USE_CUBLAS> : public LaunchMatMulCPU<T> {};

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct LaunchMatMulSYCL : LaunchMatMulBase<SYCLDevice, T> {};

template <typename T, bool USE_CUBLAS>
struct LaunchMatMul<SYCLDevice, T, USE_CUBLAS> : public LaunchMatMulSYCL<T> {};
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace {

template <typename T>
struct LaunchBlasGemv {
  static void Compute(OpKernelContext* ctx, se::Stream* stream, bool trans,
                      uint64 m, uint64 n, const se::DeviceMemory<T>& a,
                      const se::DeviceMemory<T>& b, se::DeviceMemory<T>* c,
                      se::blas::ProfileResult* output_profile) {
    const auto blas_trans = trans ? se::blas::Transpose::kTranspose
                                  : se::blas::Transpose::kNoTranspose;
    if (output_profile == nullptr) {
      bool blas_launch_status =
          stream
              ->ThenBlasGemv(blas_trans, m, n, static_cast<T>(1.0), a, m, b, 1,
                             static_cast<T>(0.0), c, 1)
              .ok();
      if (!blas_launch_status) {
        ctx->SetStatus(
            errors::Internal("Blas GEMV launch failed:  m=", m, ", n=", n));
      }
    } else {
      bool blas_launch_status =
          stream
              ->ThenBlasGemvWithProfiling(blas_trans, m, n, static_cast<T>(1.0),
                                          a, m, b, 1, static_cast<T>(0.0), c, 1,
                                          output_profile)
              .ok();
      if (!blas_launch_status) {
        ctx->SetStatus(errors::Internal(
            "Blas GEMV with profiling launch failed:  m=", m, ", n=", n));
      }
    }
  }

  static bool IsSupported() { return true; }
};

template <>
void LaunchBlasGemv<Eigen::half>::Compute(
    OpKernelContext* ctx, se::Stream* stream, bool trans, uint64 m, uint64 n,
    const se::DeviceMemory<Eigen::half>& a,
    const se::DeviceMemory<Eigen::half>& b, se::DeviceMemory<Eigen::half>* c,
    se::blas::ProfileResult* output_profile) {
  ctx->SetStatus(errors::Internal(
      "Blas GEMV launch failed: GEMV is not implemented for float16."));
}

template <>
bool LaunchBlasGemv<Eigen::half>::IsSupported() {
  return false;
}

template <typename T>
bool ShouldUseGemv(uint64 n) {
  return (LaunchBlasGemv<T>::IsSupported() && n == 1);
}

}  // namespace

bool GetCublasAutotuneComputationType(const DataType& dtype,
                                      se::blas::ComputationType* compute_type) {
  using se::blas::ComputationType;
  bool use_f32_for_f16_computation = MatmulDoFP32ComputationFP16Input();
  switch (dtype) {
    case DT_HALF:
    case DT_BFLOAT16:
      if (use_f32_for_f16_computation) {
        *compute_type = ComputationType::kF32;
      } else {
        *compute_type = ComputationType::kF16;
      }
      return false;
    case DT_FLOAT:
      *compute_type = ComputationType::kF32;
      return true;
    case DT_DOUBLE:
      *compute_type = ComputationType::kF64;
      return true;
    default:
      // Unsupported compute_type, return false.
      return false;
  }
}

// A dummy type to group matmul autotune results together.
struct MatmulAutoTuneGroup {
  static string name() { return "Matmul"; }
};
typedef AutoTuneSingleton<MatmulAutoTuneGroup, MatmulParameters,
                          se::blas::AlgorithmConfig>
    AutoTuneMatmul;

template <typename T>
struct LaunchMatMul<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      std::vector<int64>* algorithms, bool use_autotune, Tensor* out) {
    using se::blas::AlgorithmConfig;
    using se::blas::ComputationType;
    using se::blas::kDefaultAlgorithm;
    using se::blas::kDefaultBlasGemm;
    using se::blas::kDefaultBlasGemv;
    using se::blas::kNoAlgorithm;
    using se::blas::ProfileResult;
    using se::blas::Transpose;
    Transpose trans[] = {Transpose::kNoTranspose, Transpose::kTranspose};
    const uint64 m = a.dim_size(1 - dim_pair[0].first);
    const uint64 k = a.dim_size(dim_pair[0].first);
    const uint64 n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;
    auto blas_transpose_a = trans[transpose_a];
    auto blas_transpose_b = trans[transpose_b];

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto a_ptr = AsDeviceMemory(a.template flat<T>().data(),
                                a.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(b.template flat<T>().data(),
                                b.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(out->template flat<T>().data(),
                                out->template flat<T>().size());
    auto alpha = static_cast<T>(1.0);
    auto beta = static_cast<T>(0.0);

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = a.dtype();
    MatmulParameters matmul_parameters = {
        transpose_a, transpose_b, m, n, k, dtype, device_id,
    };
    AlgorithmConfig algorithm_config(kNoAlgorithm);

    ComputationType computation_type;
    bool compute_type_supported =
        GetCublasAutotuneComputationType(dtype, &computation_type);
    if (use_autotune && compute_type_supported && !algorithms->empty()) {
      ProfileResult best_result;
      // TODO(yangzihao): Unify this code with conv autotuning.
      if (!AutoTuneMatmul::GetInstance()->Find(matmul_parameters,
                                               &algorithm_config)) {
        ProfileResult profile_result;
        for (auto profile_algorithm : (*algorithms)) {
          // Cublas does
          // C = A x B
          // where A, B and C are assumed to be in column major.
          // We want the output to be in row-major, so we can compute
          // C' = B' x A' (' stands for transpose)
          bool cublas_launch_status =
              stream
                  ->ThenBlasGemmWithAlgorithm(
                      blas_transpose_b, blas_transpose_a, n, m, k, alpha, b_ptr,
                      transpose_b ? k : n, a_ptr, transpose_a ? m : k, beta,
                      &c_ptr, n, computation_type, profile_algorithm,
                      &profile_result)
                  .ok();
          if (cublas_launch_status) {
            if (profile_result.is_valid()) {
              if (profile_result.elapsed_time_in_ms() <
                  best_result.elapsed_time_in_ms()) {
                best_result = profile_result;
              }
            }
          }
        }
        // Try BlasGemmWithProfiling
        bool cublas_launch_status =
            stream
                ->ThenBlasGemmWithProfiling(
                    blas_transpose_b, blas_transpose_a, n, m, k, 1.0, b_ptr,
                    transpose_b ? k : n, a_ptr, transpose_a ? m : k, 0.0,
                    &c_ptr, n, &profile_result)
                .ok();
        if (cublas_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
          }
        }
        // Try BlasGemvWithProfiling
        if (ShouldUseGemv<T>(n)) {
          LaunchBlasGemv<T>::Compute(ctx, stream, !transpose_a,
                                     transpose_a ? m : k, transpose_a ? k : m,
                                     a_ptr, b_ptr, &c_ptr, &profile_result);
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
          }
        }
      }
      // We make sure that each matmul parameter set only gets one pass of
      // autotune. If the best result is found, assign it to algorithm_type
      // and insert it to autotune map. If all internal kernels of
      // cublasGemmEx() returns invalid results, we add kNoAlgorithm to the
      // autotune map.
      if (best_result.is_valid()) {
        algorithm_config.set_algorithm(best_result.algorithm());
      }
      AutoTuneMatmul::GetInstance()->Insert(matmul_parameters,
                                            algorithm_config);
      if (algorithm_config.algorithm() != kNoAlgorithm &&
          algorithm_config.algorithm() != kDefaultBlasGemm &&
          algorithm_config.algorithm() != kDefaultBlasGemv) {
        bool cublas_launch_status =
            stream
                ->ThenBlasGemmWithAlgorithm(
                    blas_transpose_b, blas_transpose_a, n, m, k, alpha, b_ptr,
                    transpose_b ? k : n, a_ptr, transpose_a ? m : k, beta,
                    &c_ptr, n, computation_type, algorithm_config.algorithm(),
                    nullptr)
                .ok();
        if (!cublas_launch_status) {
          ctx->SetStatus(errors::Internal(
              "Blas GEMM with algorithm launch failed : a.shape=(",
              a.dim_size(0), ", ", a.dim_size(1), "), b.shape=(", b.dim_size(0),
              ", ", b.dim_size(1), "), m=", m, ", n=", n, ", k=", k));
        }
      }
    }
    // For the following case, we use normal BlasGemm():
    //  1) We didn't set the use_autotune flag;
    //  2) compute type does not support autotune;
    //  3) no algorithm is found;
    //  4) all internal kernels in autotune return invalid results.
    //  For the following case, we use normal BlasGemv():
    //  1) We didn't set the use_autotune flag but LaunchBlasGemv is supported
    //     and n == 1.
    //  2) We set the use_autotune flag and it picked up BlasGemv() and set the
    //     algorithm_config.algorithm() to be kDefaultBlasGemv.
    if (!use_autotune || !compute_type_supported || algorithms->empty() ||
        algorithm_config.algorithm() == kNoAlgorithm ||
        algorithm_config.algorithm() == kDefaultBlasGemm ||
        algorithm_config.algorithm() == kDefaultBlasGemv) {
      if (algorithm_config.algorithm() == kDefaultBlasGemv ||
          ShouldUseGemv<T>(n)) {
        // This is a matrix*vector multiply so use GEMV to compute A * b.
        // Here we are multiplying in the natural order, so we have to flip
        // the transposition flag to compensate for the tensor being stored
        // row-major.
        // TODO(yangzihao): Add Gemv as an autotuning option too.
        LaunchBlasGemv<T>::Compute(ctx, stream, !transpose_a,
                                   transpose_a ? m : k, transpose_a ? k : m,
                                   a_ptr, b_ptr, &c_ptr, nullptr);
      } else {
        // Use C' = B' x A' (' stands for transpose)
        bool blas_launch_status =
            stream
                ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                               1.0f, b_ptr, transpose_b ? k : n, a_ptr,
                               transpose_a ? m : k, 0.0f, &c_ptr, n)
                .ok();
        if (!blas_launch_status) {
          ctx->SetStatus(errors::Internal(
              "Blas GEMM launch failed : a.shape=(", a.dim_size(0), ", ",
              a.dim_size(1), "), b.shape=(", b.dim_size(0), ", ", b.dim_size(1),
              "), m=", m, ", n=", n, ", k=", k));
        }
      }
    }
  }

  static void GetBlasGemmAlgorithm(OpKernelConstruction* ctx,
                                   std::vector<int64>* algorithms,
                                   bool* algorithm_set_flag) {
    if (*algorithm_set_flag == false) {
      auto* stream = ctx->device()->tensorflow_gpu_device_info()->stream;
      stream->parent()->GetBlasGemmAlgorithms(algorithms);
      *algorithm_set_flag = true;
    }
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T, bool USE_CUBLAS>
class MatMulOp : public OpKernel {
 public:
  explicit MatMulOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), algorithms_set_already_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    LaunchMatMul<Device, T, USE_CUBLAS>::GetBlasGemmAlgorithm(
        ctx, &algorithms_, &algorithms_set_already_);
    use_autotune_ = MatmulAutotuneEnable();
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    LaunchMatMul<Device, T, USE_CUBLAS>::launch(
        ctx, a, b, dim_pair, &algorithms_, use_autotune_, out);
  }

 private:
  std::vector<int64> algorithms_;
  bool algorithms_set_already_;
  bool use_autotune_;
  bool transpose_a_;
  bool transpose_b_;
};

namespace functor {

// Partial specialization MatMulFunctor<Device=CPUDevice, T>.
template <typename T>
struct MatMulFunctor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    MatMul<CPUDevice>(d, out, in0, in1, dim_pair);
  }
};

#ifdef TENSORFLOW_USE_SYCL
// Partial specialization MatMulFunctor<Device=SYCLDevice, T>.
template <typename T>
struct MatMulFunctor<SYCLDevice, T> {
  void operator()(
      const SYCLDevice& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    MatMul<SYCLDevice>(d, out, in0, in1, dim_pair);
  }
};
#endif  // TENSORFLOW_USE_SYCL

}  // end namespace functor

#define REGISTER_CPU_EIGEN(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatMul").Device(DEVICE_CPU).TypeConstraint<T>("T").Label("eigen"), \
      MatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);

#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("MatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      MatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>); \
  REGISTER_CPU_EIGEN(T);

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MatMul").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      MatMulOp<GPUDevice, T, true /* cublas, true by default */>); \
  REGISTER_KERNEL_BUILDER(Name("MatMul")                           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .Label("cublas"),                    \
                          MatMulOp<GPUDevice, T, true /* cublas */>)

#if defined(INTEL_MKL) && !defined(DO_NOT_USE_ML)

// MKL does not support half and int32 types for matrix-multiplication, so
// register the kernel to use default Eigen based implementations for these
// types. Registration for NO-LABEL version is in mkl_matmul_op.cc
TF_CALL_float(REGISTER_CPU_EIGEN);
TF_CALL_double(REGISTER_CPU_EIGEN);
TF_CALL_half(REGISTER_CPU);

TF_CALL_int32(REGISTER_CPU);
TF_CALL_complex64(REGISTER_CPU_EIGEN);
TF_CALL_complex128(REGISTER_CPU_EIGEN);
#else
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);

TF_CALL_int32(REGISTER_CPU);
TF_CALL_complex64(REGISTER_CPU);
TF_CALL_complex128(REGISTER_CPU);
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_half(REGISTER_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MatMul").Device(DEVICE_SYCL).TypeConstraint<T>("T"), \
      MatMulOp<SYCLDevice, T, false /* xxblas */>);              \
  REGISTER_KERNEL_BUILDER(Name("MatMul")                         \
                              .Device(DEVICE_SYCL)               \
                              .TypeConstraint<T>("T")            \
                              .Label("eigen"),                   \
                          MatMulOp<SYCLDevice, T, false /* xxblas */>)
TF_CALL_float(REGISTER_SYCL);
TF_CALL_double(REGISTER_SYCL);

#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
