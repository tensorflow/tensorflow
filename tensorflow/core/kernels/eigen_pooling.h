/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_POOLING_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_POOLING_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_patch_3d.h"

namespace Eigen {

/** SpatialMaxPooling
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies a max-pooling over a multichannel input image.
  *
  * The input parameter is expected to be a with a rank of 4 (channels, height, width, others in col-major, and the reverse of that in row-major).
  *
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be channels, height, width, and others (in col-major, and the reverse of that if the input was row-major).
  *
  * The order of the width and height dimensions can be swapped if needed.
  *
*/
#if !defined(EIGEN_HAS_INDEX_LIST)
template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorReshapingOp<const Eigen::DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>, const TensorReductionOp<internal::MaxReducer<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>, const Eigen::array<int, 2>, const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
#else
template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorReshapingOp<const Eigen::DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>, const TensorReductionOp<internal::MaxReducer<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>, typename internal::conditional<internal::traits<Input>::Layout == ColMajor, const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >, const Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3> > >::type, const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
#endif
SpatialMaxPooling(const Input& input, DenseIndex patchRows, DenseIndex patchCols,
                  DenseIndex strideRows, DenseIndex strideCols, const PaddingType padding_type,
                  DenseIndex in_strideRows = 1, DenseIndex in_strideCols = 1)
{
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4, YOU_MADE_A_PROGRAMMING_MISTAKE);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  const DenseIndex patchRowsEff = patchRows + (patchRows - 1) * (in_strideRows - 1);
  const DenseIndex patchColsEff = patchCols + (patchCols - 1) * (in_strideCols - 1);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int idxRows = isColMajor ? 1 : 2;
  static const int idxCols = isColMajor ? 2 : 1;

  // Molds the output of the reduction into the shape expected by the user.
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, internal::traits<Input>::NumDimensions> post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxRows] = numext::ceil((in.dimension(idxRows) - patchRowsEff + 1.f) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil((in.dimension(idxCols) - patchColsEff + 1.f) / static_cast<float>(strideCols));
  } else {
    post_reduce_dims[idxRows] = numext::ceil(in.dimension(idxRows) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil(in.dimension(idxCols) / static_cast<float>(strideCols));
  }
  post_reduce_dims[3] = in.dimension(3);

#if !defined(EIGEN_HAS_INDEX_LIST)
  // nvcc doesn't support cxx11
  Eigen::array<int, 2> reduction_dims;
  if (isColMajor) {
    reduction_dims[0] = 1;
    reduction_dims[1] = 2;
  } else {
    reduction_dims[0] = 2;
    reduction_dims[1] = 3;
  }
#else
  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  typename internal::conditional<internal::traits<Input>::Layout == ColMajor, const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >, const Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3> > >::type reduction_dims;
#endif

  return input.extract_image_patches(patchRows, patchCols, strideRows, strideCols, in_strideRows, in_strideCols, padding_type, -Eigen::NumTraits<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>::highest()).maximum(reduction_dims).reshape(post_reduce_dims);
}

/** CuboidMaxPooling
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies a max-pooling over a multichannel input volume.
  *
  * The input parameter is expected to be a tensor with a rank of 5 (channels, depth, height, width, others in col-major, and the reverse of that in row-major).
  *
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be channels, depth, height, width, and others (in col-major, and the reverse of that if the input was row-major).
  *
  * The order of the depth, width and height dimensions can be swapped if needed.
  *
*/
#if !defined(EIGEN_HAS_INDEX_LIST)
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::MaxReducer<float>, const Eigen::array<int, 1>,
        const TensorReshapingOp<
            const Eigen::DSizes<DenseIndex, 3>,
            const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Input> > > >
#else
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::MaxReducer<float>,
        const Eigen::IndexList<Eigen::type2index<1> >,
        const TensorReshapingOp<
            const Eigen::DSizes<DenseIndex, 3>,
            const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Input> > > >
#endif
CuboidMaxPooling(const Input& input, DenseIndex patchPlanes,
                 DenseIndex patchRows, DenseIndex patchCols,
                 DenseIndex stridePlanes, DenseIndex strideRows,
                 DenseIndex strideCols, const PaddingType padding_type) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 5, YOU_MADE_A_PROGRAMMING_MISTAKE);
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  static const int idxPlanes = isColMajor ? 1 : 3;
  static const int idxRows = 2;
  static const int idxCols = isColMajor ? 3 : 1;

  // Molds the output of the reduction into the shape expected by the used
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output depth
  // - 3rd dim: output height
  // - 4th dim: output width
  // - 5th dim and beyond: everything else including batch size
  Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions> post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxPlanes] = numext::ceil((in.dimension(idxPlanes) - patchPlanes + 1.f) / static_cast<float>(stridePlanes));
    post_reduce_dims[idxRows] = numext::ceil((in.dimension(idxRows) - patchRows + 1.f) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil((in.dimension(idxCols) - patchCols + 1.f) / static_cast<float>(strideCols));
  } else {
    post_reduce_dims[idxPlanes] = numext::ceil(in.dimension(idxPlanes) / static_cast<float>(stridePlanes));
    post_reduce_dims[idxRows] = numext::ceil(in.dimension(idxRows) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil(in.dimension(idxCols) / static_cast<float>(strideCols));
  }
  post_reduce_dims[4] = in.dimension(4);

  Eigen::DSizes<DenseIndex, 3> pre_reduce_dims;
  pre_reduce_dims[1] = patchRows * patchCols * patchPlanes;
  if (isColMajor) {
    pre_reduce_dims[0] = post_reduce_dims[0];
    pre_reduce_dims[2] = post_reduce_dims[1] * post_reduce_dims[2] * post_reduce_dims[3] * post_reduce_dims[4];
  } else {
    pre_reduce_dims[0] = post_reduce_dims[0] * post_reduce_dims[1] * post_reduce_dims[2] * post_reduce_dims[3];
    pre_reduce_dims[2] = post_reduce_dims[4];
  }

#if !defined(EIGEN_HAS_INDEX_LIST)
  // nvcc doesn't support cxx11
  Eigen::array<int, 1> reduction_dims;
  reduction_dims[0] = 1;
#else
  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  Eigen::IndexList<Eigen::type2index<1> > reduction_dims;
#endif
  return input.extract_volume_patches(patchPlanes, patchRows, patchCols,
                                      stridePlanes, strideRows, strideCols,
                                      padding_type, -Eigen::NumTraits<float>::highest())
      .reshape(pre_reduce_dims)
      .maximum(reduction_dims)
      .reshape(post_reduce_dims);
}


/** SpatialAvgPooling
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies an average pooling over a multichannel input image.
  *
  * The input parameter is expected to be a tensor with a rank of 4 (channels, height, width, others in col-major, and the reverse of that in row-major).
  *
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be channels, height, width, and others (in col-major, and the reverse of that if the input was row-major).
  *
  * The order of the width and height dimensions can be swapped if needed.
  *
*/
namespace internal {

template <typename T> struct AvgPoolMeanReducer
{
#if (EIGEN_ARCH_i386 || EIGEN_ARCH_x86_64) && !defined(__CUDACC__)
  // We only support packet access for floats.
  static const bool PacketAccess = internal::is_same<T, float>::value;
#else
  static const bool PacketAccess = false;
#endif
  static const bool IsStateful = true;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE AvgPoolMeanReducer() : scalarCount_(0) {
    typedef typename packet_traits<T>::type Packet;
    packetCount_ = pset1<Packet>(0.0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) {
    if (t != -Eigen::NumTraits<T>::highest()) {
      (*accum) = (*accum) + t;
      scalarCount_++;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return static_cast<T>(0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    eigen_assert(scalarCount_ > 0);
    return accum / scalarCount_;
  }

#if (EIGEN_ARCH_i386 || EIGEN_ARCH_x86_64) && !defined(__CUDACC__)
#ifdef EIGEN_VECTORIZE_AVX
#define pequal(a,b) _mm256_cmp_ps(a,b,_CMP_EQ_UQ)
#define psel(a,b,false_mask) _mm256_blendv_ps(a,b,false_mask)
#else
#define pequal(a,b) _mm_cmpeq_ps(a,b)
#define psel(a,b,false_mask) _mm_or_ps(_mm_andnot_ps(false_mask, a), _mm_and_ps(false_mask, b))
#endif

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) {
    reducePacketWithType(static_cast<T>(0), p, accum);
  }

  template <typename Packet>
  void reducePacketWithType(T, const Packet& p, Packet* accum) {
    Packet skip_mask = pequal(p, pset1<Packet>(-Eigen::NumTraits<T>::highest()));
    (*accum) = padd<Packet>(*accum, psel(p, pset1<Packet>(0), skip_mask));
    packetCount_ = padd<Packet>(packetCount_, psel(pset1<Packet>(1), pset1<Packet>(0), skip_mask));
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(0);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return pdiv(vaccum, packetCount_);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return (saccum + predux(vaccum)) / (scalarCount_ + predux(packetCount_));
  }
#endif

 protected:
    typedef typename packet_traits<T>::type Packet;
    int scalarCount_;
    Packet packetCount_;
};

}  // namespace internal

#if !defined(EIGEN_HAS_INDEX_LIST)
template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorReshapingOp<const Eigen::DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>, const TensorReductionOp<internal::AvgPoolMeanReducer<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>, const Eigen::array<int, 2>, const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
#else
template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorReshapingOp<const Eigen::DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>, const TensorReductionOp<internal::AvgPoolMeanReducer<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>, typename internal::conditional<internal::traits<Input>::Layout == ColMajor, const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >, const Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3> > >::type, const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
#endif
SpatialAvgPooling(const Input& input, DenseIndex patchRows, DenseIndex patchCols,
                  DenseIndex strideRows, DenseIndex strideCols, const PaddingType padding_type,
                  DenseIndex in_strideRows = 1, DenseIndex in_strideCols = 1)
{
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4, YOU_MADE_A_PROGRAMMING_MISTAKE);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  const DenseIndex patchRowsEff = patchRows + (patchRows - 1) * (in_strideRows - 1);
  const DenseIndex patchColsEff = patchCols + (patchCols - 1) * (in_strideCols - 1);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int idxRows = isColMajor ? 1 : 2;
  static const int idxCols = isColMajor ? 2 : 1;

  // Molds the output of the reduction into the shape expected by the user.
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, internal::traits<Input>::NumDimensions> post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxRows] = numext::ceil((in.dimension(idxRows) - patchRowsEff + 1.f) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil((in.dimension(idxCols) - patchColsEff + 1.f) / static_cast<float>(strideCols));
  } else {
    post_reduce_dims[idxRows] = numext::ceil(in.dimension(idxRows) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil(in.dimension(idxCols) / static_cast<float>(strideCols));
  }
  post_reduce_dims[3] = in.dimension(3);

  typedef typename internal::remove_const<typename internal::traits<Input>::Scalar>::type CoeffReturnType;
  internal::AvgPoolMeanReducer<CoeffReturnType> mean_with_nan;

#if !defined(EIGEN_HAS_INDEX_LIST)
  // nvcc doesn't support cxx11
  Eigen::array<int, 2> reduction_dims;
  if (isColMajor) {
    reduction_dims[0] = 1;
    reduction_dims[1] = 2;
  } else {
    reduction_dims[0] = 2;
    reduction_dims[1] = 3;
  }
#else
  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  typename internal::conditional<internal::traits<Input>::Layout == ColMajor, const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >, const Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3> > >::type reduction_dims;
#endif
  return input.extract_image_patches(patchRows, patchCols, strideRows, strideCols, in_strideRows, in_strideCols, padding_type, -Eigen::NumTraits<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>::highest()).reduce(reduction_dims, mean_with_nan).reshape(post_reduce_dims);
}


/** CuboidAvgPooling
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies an average pooling over a multichannel input volume.
  *
  * The input parameter is expected to be a tensor with a rank of 5 (channels, depth, height, width, others, and the reverse of that in row-major).
  *
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be channels, depth, width, and others (in col-major, and the reverse of that if the input was row-major).
  *
  * The order of the depth, width and height dimensions can be swapped if needed.
  *
*/
#if !defined(EIGEN_HAS_INDEX_LIST)
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
    const Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>,
    const TensorReductionOp<
        internal::AvgPoolMeanReducer<float>, const Eigen::array<int, 1>,
        const TensorReshapingOp<
            const Eigen::DSizes<DenseIndex, 3>,
            const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Input> > > >
#else
template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorReshapingOp<
      const Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions>,
      const TensorReductionOp<
          internal::AvgPoolMeanReducer<float>,
          const Eigen::IndexList<Eigen::type2index<1> >,
          const TensorReshapingOp<
              const Eigen::DSizes<DenseIndex, 3>,
              const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic, const Input> > > >
#endif
CuboidAvgPooling(const Input& input, DenseIndex patchPlanes,
                 DenseIndex patchRows, DenseIndex patchCols,
                 DenseIndex stridePlanes, DenseIndex strideRows,
                 DenseIndex strideCols, const PaddingType padding_type) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 5, YOU_MADE_A_PROGRAMMING_MISTAKE);
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  static const int idxPlanes = isColMajor ? 1 : 3;
  static const int idxRows = 2;
  static const int idxCols = isColMajor ? 3 : 1;
  // Molds the output of the reduction into the shape expected by the used
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: outupt depth
  // - 3rd dim: output height
  // - 4th dim: output width
  // - 5th dim and beyond: everything else including batch size
  Eigen::DSizes<DenseIndex, internal::traits<Input>::NumDimensions> post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxPlanes] = numext::ceil((in.dimension(idxPlanes) - patchPlanes + 1.f) / static_cast<float>(stridePlanes));
    post_reduce_dims[idxRows] = numext::ceil((in.dimension(idxRows) - patchRows + 1.f) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil((in.dimension(idxCols) - patchCols + 1.f) / static_cast<float>(strideCols));
  } else {
    post_reduce_dims[idxPlanes] = numext::ceil(in.dimension(idxPlanes) / static_cast<float>(stridePlanes));
    post_reduce_dims[idxRows] = numext::ceil(in.dimension(idxRows) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil(in.dimension(idxCols) / static_cast<float>(strideCols));
  }
  post_reduce_dims[4] = in.dimension(4);

  Eigen::DSizes<DenseIndex, 3> pre_reduce_dims;
  pre_reduce_dims[1] = patchRows * patchCols * patchPlanes;
  if (isColMajor) {
    pre_reduce_dims[0] = post_reduce_dims[0];
    pre_reduce_dims[2] = post_reduce_dims[1] * post_reduce_dims[2] * post_reduce_dims[3] * post_reduce_dims[4];
  } else {
    pre_reduce_dims[0] = post_reduce_dims[0] * post_reduce_dims[1] * post_reduce_dims[2] * post_reduce_dims[3];
    pre_reduce_dims[2] = post_reduce_dims[4];
  }

  typedef typename internal::remove_const<typename internal::traits<Input>::Scalar>::type CoeffReturnType;
  internal::AvgPoolMeanReducer<CoeffReturnType> mean_with_nan;

#if !defined(EIGEN_HAS_INDEX_LIST)
  // nvcc doesn't support cxx11
  Eigen::array<int, 1> reduction_dims;
  reduction_dims[0] = 1;
#else
  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  Eigen::IndexList<Eigen::type2index<1> > reduction_dims;
#endif
  return input.extract_volume_patches(patchPlanes, patchRows, patchCols,
                                      stridePlanes, strideRows, strideCols,
                                      padding_type, -Eigen::NumTraits<float>::highest())
      .reshape(pre_reduce_dims)
      .reduce(reduction_dims, mean_with_nan)
      .reshape(post_reduce_dims);
}

} // end namespace Eigen

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_POOLING_H_
