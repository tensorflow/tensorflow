#ifndef TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_
#define TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor>,
                           Eigen::Aligned> Tensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor>,
                           Eigen::Aligned> ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor> >
      UnalignedTensor;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor> >
      UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>,
                           Eigen::Aligned> Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor>,
      Eigen::Aligned> Scalar;
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<const T, Eigen::Sizes<>, Eigen::RowMajor>,
      Eigen::Aligned> ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<
      T, Eigen::Sizes<>, Eigen::RowMajor> > UnalignedScalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<
      const T, Eigen::Sizes<>, Eigen::RowMajor> > UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Aligned> ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>, Eigen::Aligned>
      Vec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Aligned> ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor> > UnalignedFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor> >
      UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor> > UnalignedVec;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor> >
      UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      Matrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned> ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor> >
      UnalignedMatrix;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor> >
      UnalignedConstMatrix;
};

typedef typename TTypes<float, 1>::Tensor32Bit::Index Index32;

template <typename DSizes>
Eigen::DSizes<Index32, DSizes::count> To32BitDims(const DSizes& in) {
  Eigen::DSizes<Index32, DSizes::count> out;
  for (int i = 0; i < DSizes::count; ++i) {
    out[i] = in[i];
  }
  return out;
}

template <typename TensorType>
typename TTypes<typename TensorType::Scalar,
                TensorType::NumIndices>::Tensor32Bit
To32Bit(TensorType in) {
  typedef typename TTypes<typename TensorType::Scalar,
                          TensorType::NumIndices>::Tensor32Bit RetType;
  return RetType(in.data(), To32BitDims(in.dimensions()));
}

}  // namespace tensorflow
#endif  // TENSORFLOW_FRAMEWORK_TENSOR_TYPES_H_
