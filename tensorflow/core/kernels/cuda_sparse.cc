/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   =============================================================================
*/

#ifdef GOOGLE_CUDA

#include <complex>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda/include/cusparse.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Type traits to get CUDA complex types from std::complex<>.
// TODO: reuse with cuda_solvers
template <typename T>
struct CudaComplexT {
  typedef T type;
};
template <>
struct CudaComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct CudaComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};
// Converts pointers of std::complex<> to pointers of
// cuComplex/cuDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename CudaComplexT<T>::type* AsCudaComplex(const T* p) {
  return reinterpret_cast<const typename CudaComplexT<T>::type*>(p);
}
template <typename T>
inline typename CudaComplexT<T>::type* AsCudaComplex(T* p) {
  return reinterpret_cast<typename CudaComplexT<T>::type*>(p);
}

// A set of initialized handles to the underlying Cuda libraries used by
// CudaSparse. We maintain one such set of handles per unique stream.
class CudaSparseHandles {
 public:
  explicit CudaSparseHandles(cudaStream_t stream)
      : initialized_(false), stream_(stream) {}

  CudaSparseHandles(CudaSparseHandles&& rhs)
      : initialized_(rhs.initialized_),
        stream_(std::move(rhs.stream_)),
        cusparse_handle_(rhs.cusparse_handle_) {
    rhs.initialized_ = false;
  }

  CudaSparseHandles& operator=(CudaSparseHandles&& rhs) {
    if (this == &rhs) return *this;
    Release();
    stream_ = std::move(rhs.stream_);
    cusparse_handle_ = std::move(rhs.cusparse_handle_);
    initialized_ = rhs.initialized_;
    rhs.initialized_ = false;
    return *this;
  }

  ~CudaSparseHandles() { Release(); }

  Status Initialize() {
    if (initialized_) return Status::OK();
    TF_RETURN_IF_CUSPARSE_ERROR(cusparseCreate(&cusparse_handle_));
    TF_RETURN_IF_CUSPARSE_ERROR(cusparseSetStream(cusparse_handle_, stream_));
    initialized_ = true;
    return Status::OK();
  }

  cusparseHandle_t& handle() {
    DCHECK(initialized_);
    return cusparse_handle_;
  }

  const cusparseHandle_t& handle() const {
    DCHECK(initialized_);
    return cusparse_handle_;
  }

 private:
  void Release() {
    if (initialized_) {
      // This should never return anything other than success
      auto err = cusparseDestroy(cusparse_handle_);
      DCHECK(err == CUSPARSE_STATUS_SUCCESS)
          << "Failed to destroy cuSparse instance.";
      initialized_ = false;
    }
  }
  bool initialized_;
  cudaStream_t stream_;
  cusparseHandle_t cusparse_handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(CudaSparseHandles);
};

static mutex handle_map_mutex(LINKER_INITIALIZED);

using HandleMap = std::unordered_map<cudaStream_t, CudaSparseHandles>;

// Returns a singleton map used for storing initialized handles for each unique
// cuda stream.
HandleMap* GetHandleMapSingleton() {
  static HandleMap* cm = new HandleMap;
  return cm;
}

}  // namespace

CudaSparse::CudaSparse(OpKernelContext* context)
    : initialized_(false), context_(context) {
  cuda_stream_ =
      *reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                 ->stream()
                                                 ->implementation()
                                                 ->GpuStreamMemberHack());
}

Status CudaSparse::Initialize() {
  HandleMap* handle_map = GetHandleMapSingleton();
  DCHECK(handle_map);
  mutex_lock lock(handle_map_mutex);
  auto it = handle_map->find(cuda_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating CudaSparse handles for stream " << cuda_stream_;
    // Previously unseen Cuda stream. Initialize a set of Cuda sparse library
    // handles for it.
    CudaSparseHandles new_handles(cuda_stream_);
    TF_RETURN_IF_ERROR(new_handles.Initialize());
    it =
        handle_map->insert(std::make_pair(cuda_stream_, std::move(new_handles)))
            .first;
  }
  cusparse_handle_ = &it->second.handle();
  initialized_ = true;
  return Status::OK();
}

// Macro that specializes a sparse method for all 4 standard
// numeric types.
// TODO: reuse with cuda_solvers
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

// Macros to construct cusparse method names.
#define SPARSE_FN(method, sparse_prefix) cusparse##sparse_prefix##method
#define SPARSE_NAME(method, sparse_prefix) "cusparse" #sparse_prefix #method
#define BUFSIZE_FN(method, sparse_prefix) \
  cusparse##sparse_prefix##method##_bufferSizeExt

//=============================================================================
// Wrappers of cuSparse computational methods begin here.
//
// WARNING to implementers: The function signatures listed in the online docs
// are sometimes inaccurate, e.g., are missing 'const' on pointers
// to immutable arguments, while the actual headers have them as expected.
// Check the actual declarations in the cusparse.h header file.
//=============================================================================

template <typename Scalar, typename SparseFn>
static inline Status GtsvImpl(SparseFn op, cusparseHandle_t cusparse_handle,
                              int m, int n, const Scalar* dl, const Scalar* d,
                              const Scalar* du, Scalar* B, int ldb) {
  TF_RETURN_IF_CUSPARSE_ERROR(op(cusparse_handle, m, n, AsCudaComplex(dl),
                                 AsCudaComplex(d), AsCudaComplex(du),
                                 AsCudaComplex(B), ldb));
  return Status::OK();
}

#define GTSV_INSTANCE(Scalar, sparse_prefix)                                 \
  template <>                                                                \
  Status CudaSparse::Gtsv<Scalar>(int m, int n, const Scalar* dl,            \
                                  const Scalar* d, const Scalar* du,         \
                                  Scalar* B, int ldb) const {                \
    DCHECK(initialized_);                                                    \
    return GtsvImpl(SPARSE_FN(gtsv, sparse_prefix), *cusparse_handle_, m, n, \
                    dl, d, du, B, ldb);                                      \
  }

TF_CALL_LAPACK_TYPES(GTSV_INSTANCE);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
