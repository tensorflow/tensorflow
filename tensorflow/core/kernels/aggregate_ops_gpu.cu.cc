#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/aggregate_ops.h"

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization for a GPUDevice, that uses the Eigen implementation.
namespace functor {
template <typename T>
struct Add2Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2) {
    Add2EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2);
  }
};

template <typename T>
struct Add3Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3) {
    Add3EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3);
  }
};

template <typename T>
struct Add4Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4) {
    Add4EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4);
  }
};

template <typename T>
struct Add5Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5) {
    Add5EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5);
  }
};

template <typename T>
struct Add6Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6) {
    Add6EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6);
  }
};

template <typename T>
struct Add7Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6,
                  typename TTypes<T>::ConstFlat in7) {
    Add7EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7);
  }
};

template <typename T>
struct Add8Functor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8);
  }
};

template <typename T>
struct Add8pFunctor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8pEigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                          in7, in8);
  }
};

template <typename T>
struct Add9Functor<GPUDevice, T> {
  void operator()(
      const GPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9) {
    Add9EigenImpl<GPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8, in9);
  }
};

}  // end namespace functor

// Instantiate the GPU implementation for float.
template struct functor::Add2Functor<GPUDevice, float>;
template struct functor::Add3Functor<GPUDevice, float>;
template struct functor::Add4Functor<GPUDevice, float>;
template struct functor::Add5Functor<GPUDevice, float>;
template struct functor::Add6Functor<GPUDevice, float>;
template struct functor::Add7Functor<GPUDevice, float>;
template struct functor::Add8Functor<GPUDevice, float>;
template struct functor::Add8pFunctor<GPUDevice, float>;
template struct functor::Add9Functor<GPUDevice, float>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
