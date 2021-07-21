/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_ROCM_SOLVERS_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_ROCM_SOLVERS_H_

// This header declares the class ROCmSolver, which contains wrappers of linear
// algebra solvers in the rocBlas and rocSolverDN libraries for use in TensorFlow
// kernels.

#if TENSORFLOW_USE_ROCM

#include <functional>
#include <vector>

#include "rocm/include/hip/hip_complex.h"
#include "rocm/include/rocblas.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/rocm/rocsolver_wrapper.h"
namespace tensorflow {

// Type traits to get ROCm complex types from std::complex<T>.
template <typename T>
struct ROCmComplexT {
  typedef T type;
};
template <>
struct ROCmComplexT<std::complex<float>> {
  typedef rocblas_float_complex type;
};
template <>
struct ROCmComplexT<std::complex<double>> {
  typedef rocblas_double_complex type;
};
// Converts pointers of std::complex<> to pointers of
// ROCmComplex/ROCmDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename ROCmComplexT<T>::type* ROCmComplex(const T* p) {
  return reinterpret_cast<const typename ROCmComplexT<T>::type*>(p);
}
template <typename T>
inline typename ROCmComplexT<T>::type* ROCmComplex(T* p) {
  return reinterpret_cast<typename ROCmComplexT<T>::type*>(p);
}

// Container of LAPACK info data (an array of int) generated on-device by
// a rocSolver call. One or more such objects can be passed to
// rocSolver::CopyLapackInfoToHostAsync() along with a callback to
// check the LAPACK info data after the corresponding kernels
// finish and LAPACK info has been copied from the device to the host.
class DeviceLapackInfo;

// Host-side copy of LAPACK info.
class HostLapackInfo;

template <typename Scalar>
class ScratchSpace;

class ROCmSolver {
 public:
  // This object stores a pointer to context, which must outlive it.
  explicit ROCmSolver(OpKernelContext* context);
  virtual ~ROCmSolver();

  // Allocates a temporary tensor that will live for the duration of the
  // ROCmSolver object.
  Status allocate_scoped_tensor(DataType type, const TensorShape& shape,
                                Tensor* scoped_tensor);
  Status forward_input_or_allocate_scoped_tensor(
      gtl::ArraySlice<int> candidate_input_indices, DataType type,
      const TensorShape& shape, Tensor* input_alias_or_new_scoped_tensor);

  static void CheckLapackInfoAndDeleteSolverAsync(
    std::unique_ptr<ROCmSolver> solver, 
    const std::vector<DeviceLapackInfo>& dev_lapack_info, 
    std::function<void(const Status&, const std::vector<HostLapackInfo>&)>
    info_checker_callback);  

  static void CheckLapackInfoAndDeleteSolverAsync(
    std::unique_ptr<ROCmSolver> solver,
    const std::vector<DeviceLapackInfo>& dev_lapack_info,
    AsyncOpKernel::DoneCallback done); 

  OpKernelContext* context() { return context_; }

  template <typename Scalar>
  ScratchSpace<Scalar> GetScratchSpace(const TensorShape& shape,
                                       const std::string& debug_info,
                                       bool on_host);
  template <typename Scalar>
  ScratchSpace<Scalar> GetScratchSpace(int64 size,
                                       const std::string& debug_info,
                                       bool on_host);
  // Returns a DeviceLapackInfo that will live for the duration of the
  // ROCmSolver object.
  inline DeviceLapackInfo GetDeviceLapackInfo(int64 size,
                                              const std::string& debug_info);


  // ====================================================================
  // Wrappers for ROCSolver start here
  //
  // The method names below
  // map to those in ROCSolver, which follow the naming
  // convention in LAPACK see

  // LU factorization.
  // Computes LU factorization with partial pivoting P * A = L * U.
  template <typename Scalar>
  Status getrf(int m, int n, Scalar* dev_A, int lda, int* dev_pivots, int* info);

  // Uses LU factorization to solve A * X = B.
  template <typename Scalar>
  Status getrs(const rocblas_operation trans, int n, int nrhs, Scalar* A,
               int lda, const int* dev_pivots, Scalar* B, int ldb);

  template <typename Scalar>
  Status
  getrf_batched(int m, int n, Scalar** dev_A, int lda, int* dev_pivots,
                rocblas_stride stride, DeviceLapackInfo* info, const int batch_count);

  template <typename Scalar>
  Status getrs_batched(const rocblas_operation trans, int n, int nrhs,
                       Scalar** A, int lda, int* dev_pivots,
                       rocblas_stride stride, Scalar** B, const int ldb,
                       const int batch_count);


  template <typename Scalar>
  Status Trsm(rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
              rocblas_diagonal diag, int m, int n, const Scalar* alpha,
              const Scalar* A, int lda, Scalar* B, int ldb);

 private:
  OpKernelContext* context_;  // not owned.
  hipStream_t hip_stream_;
  rocblas_handle rocm_blas_handle_;
  std::vector<TensorReference> scratch_tensor_refs_;

  TF_DISALLOW_COPY_AND_ASSIGN(ROCmSolver);
};

// Helper class to allocate scratch memory and keep track of debug info.
// Mostly a thin wrapper around Tensor & allocate_temp.
template <typename Scalar>
class ScratchSpace {
 public:
  ScratchSpace(OpKernelContext* context, int64 size, bool on_host)
      : ScratchSpace(context, TensorShape({size}), "", on_host) {}

  ScratchSpace(OpKernelContext* context, int64 size, const string& debug_info,
               bool on_host)
      : ScratchSpace(context, TensorShape({size}), debug_info, on_host) {}

  ScratchSpace(OpKernelContext* context, const TensorShape& shape,
               const string& debug_info, bool on_host)
      : context_(context), debug_info_(debug_info), on_host_(on_host) {
    AllocatorAttributes alloc_attr;
    if (on_host) {
      // Allocate pinned memory on the host to avoid unnecessary
      // synchronization.
      alloc_attr.set_on_host(true);
      alloc_attr.set_gpu_compatible(true);
    }
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<Scalar>::value, shape,
                                       &scratch_tensor_, alloc_attr));
  }

  virtual ~ScratchSpace() {}

  Scalar* mutable_data() {
    return scratch_tensor_.template flat<Scalar>().data();
  }
  const Scalar* data() const {
    return scratch_tensor_.template flat<Scalar>().data();
  }
  Scalar& operator()(int64 i) {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  const Scalar& operator()(int64 i) const {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  int64 bytes() const { return scratch_tensor_.TotalBytes(); }
  int64 size() const { return scratch_tensor_.NumElements(); }
  const string& debug_info() const { return debug_info_; }

  Tensor& tensor() { return scratch_tensor_; }
  const Tensor& tensor() const { return scratch_tensor_; }

  // Returns true if this ScratchSpace is in host memory.
  bool on_host() const { return on_host_; }

 protected:
  OpKernelContext* context() const { return context_; }

 private:
  OpKernelContext* context_;  // not owned
  const string debug_info_;
  const bool on_host_;
  Tensor scratch_tensor_;
};

class HostLapackInfo : public ScratchSpace<int> {
 public:
  HostLapackInfo(OpKernelContext* context, int64 size,
                 const std::string& debug_info)
      : ScratchSpace<int>(context, size, debug_info, /* on_host */ true) {}
};

class DeviceLapackInfo : public ScratchSpace<int> {
 public:
  DeviceLapackInfo(OpKernelContext* context, int64 size,
                   const std::string& debug_info)
      : ScratchSpace<int>(context, size, debug_info, /* on_host */ false) {}

  // Allocates a new scratch space on the host and launches a copy of the
  // contents of *this to the new scratch space. Sets success to true if
  // the copy kernel was launched successfully.
  HostLapackInfo CopyToHost(bool* success) const {
    CHECK(success != nullptr);
    HostLapackInfo copy(context(), size(), debug_info());
    auto stream = context()->op_device_context()->stream();
    se::DeviceMemoryBase wrapped_src(
        static_cast<void*>(const_cast<int*>(this->data())));
    *success =
        stream->ThenMemcpy(copy.mutable_data(), wrapped_src, this->bytes())
            .ok();
    return copy;
  }
};


template <typename Scalar>
ScratchSpace<Scalar> ROCmSolver::GetScratchSpace(const TensorShape& shape,
                                                 const std::string& debug_info,
                                                 bool on_host) {
  ScratchSpace<Scalar> new_scratch_space(context_, shape, debug_info, on_host);
  scratch_tensor_refs_.emplace_back(new_scratch_space.tensor());
  return std::move(new_scratch_space);
}

template <typename Scalar>
ScratchSpace<Scalar> ROCmSolver::GetScratchSpace(int64 size,
                                                 const std::string& debug_info,
                                                 bool on_host) {
  return GetScratchSpace<Scalar>(TensorShape({size}), debug_info, on_host);
}

inline DeviceLapackInfo ROCmSolver::GetDeviceLapackInfo(
    int64 size, const std::string& debug_info) {
  DeviceLapackInfo new_dev_info(context_, size, debug_info);
  scratch_tensor_refs_.emplace_back(new_dev_info.tensor());
  return new_dev_info;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_ROCM_SOLVERS_H_
