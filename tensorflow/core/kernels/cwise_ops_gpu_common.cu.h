#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
#define TENSORFLOW_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_

#define EIGEN_USE_GPU

#include <complex>

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef std::complex<float> complex64;

// Partial specialization of UnaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    out.device(d) = in.unaryExpr(typename Functor::func());
  }
};

// Partial specialization of BinaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor, int NDIMS>
struct BinaryFunctor<GPUDevice, Functor, NDIMS> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    out.device(d) = in0.binaryExpr(in1, typename Functor::func());
  }

  void Left(const GPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    out.device(d) = in.unaryExpr(Unary(scalar.data()));
  }

  void Right(const GPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    out.device(d) = in.unaryExpr(Unary(scalar.data()));
  }

  void BCast(const GPUDevice& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        out.device(d) = in0.binaryExpr(in1.broadcast(bcast1), func);
        return;
      }
      if (!bcast0_all_one && bcast1_all_one) {
        out.device(d) = in0.broadcast(bcast0).binaryExpr(in1, func);
        return;
      }
    }
    out.device(d) =
        in0.broadcast(bcast0).binaryExpr(in1.broadcast(bcast1), func);
  }
};

template <typename T>
struct SelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    out.device(d) = cond_flat.select(then_flat, else_flat);
  }
};

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for UnaryFunctor (e.g., functor:sqrt).
#define DEFINE_UNARY1(F, T) template struct UnaryFunctor<GPUDevice, F<T> >
#define DEFINE_UNARY2(F, T0, T1) \
  DEFINE_UNARY1(F, T0);          \
  DEFINE_UNARY1(F, T1)
#define DEFINE_UNARY3(F, T0, T1, T2) \
  DEFINE_UNARY2(F, T0, T1);          \
  DEFINE_UNARY1(F, T2)
#define DEFINE_UNARY4(F, T0, T1, T2, T3) \
  DEFINE_UNARY2(F, T0, T1);              \
  DEFINE_UNARY2(F, T2, T3)
#define DEFINE_UNARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_UNARY2(F, T0, T1);                  \
  DEFINE_UNARY3(F, T2, T3, T4)

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for BinaryFunctor.
#define DEFINE_BINARY1(F, T)                         \
  template struct BinaryFunctor<GPUDevice, F<T>, 1>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 2>; \
  template struct BinaryFunctor<GPUDevice, F<T>, 3>
#define DEFINE_BINARY2(F, T0, T1) \
  DEFINE_BINARY1(F, T0);          \
  DEFINE_BINARY1(F, T1)
#define DEFINE_BINARY3(F, T0, T1, T2) \
  DEFINE_BINARY2(F, T0, T1);          \
  DEFINE_BINARY1(F, T2)
#define DEFINE_BINARY4(F, T0, T1, T2, T3) \
  DEFINE_BINARY2(F, T0, T1);              \
  DEFINE_BINARY2(F, T2, T3)
#define DEFINE_BINARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_BINARY2(F, T0, T1);                  \
  DEFINE_BINARY3(F, T2, T3, T4)

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CWISE_OPS_GPU_COMMON_CU_H_
