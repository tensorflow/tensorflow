#ifndef EIGEN_CXX11_SRC_NEURAL_NETWORKS_PATCH3D_H
#define EIGEN_CXX11_SRC_NEURAL_NETWORKS_PATCH3D_H

#if not defined(__CUDACC__)
#include <type_traits>
#endif

namespace Eigen {
namespace internal {

/** Extract3DPatches
 * \ingroup CXX11_NeuralNetworksModule
 *
 * \brief Extracts 3D patches from a multichannel input volume.
 *
 * The input parameter is expected to be a tensor with a rank of 4 or more
 * (channels, depth, height, width, optional others in col-major, and the
 * reverse order in row-major).

 * The return value will be a tensor of 3 more dimension than the input tensor.
 * In col-major, the first 4 dimensions of the result are: channels, patch_depth,
 * patch_height, patch_width. The next dimensions will identify the patch
 * position on the 3D grid of extracted patches: z, y, x. The remaining
 * dimensions, if any, will be the same as the 'other' dimensions of the input
 * tensor.
 */

template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorStridingOp<
    const array<typename internal::traits<Input>::Index,
                internal::traits<Input>::NumDimensions + 3>,
    const TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions + 3>,
        const TensorPatchOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorPaddingOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            internal::traits<Input>::NumDimensions>,
                const Input> > > >
Extract3DPatches(
    const Input& input, const DenseIndex patchPlanes,
    const DenseIndex patchRows, const DenseIndex patchCols,
    const DenseIndex stridePlanes, const DenseIndex strideRows,
    const DenseIndex strideCols,
    const DenseIndex paddingZTop, const DenseIndex paddingZBottom,
    const DenseIndex paddingTop, const DenseIndex paddingBottom,
    const DenseIndex paddingLeft, const DenseIndex paddingRight,
    const typename internal::traits<Input>::Scalar padding_value = 0) {

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions >= 4, YOU_MADE_A_PROGRAMMING_MISTAKE);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int NumDims = internal::traits<Input>::NumDimensions;
  static const int ExtDims = NumDims + 3;

  // Tensor size after patch extraction. We add three dimensions to unpack the
  // linear patch index into a 3D grid over which stride() can work.
  DSizes<TensorIndex, ExtDims> pre_stride_dims;

  if (isColMajor) {
    pre_stride_dims[0] = in.dimension(0);
    pre_stride_dims[1] = patchPlanes;
    pre_stride_dims[2] = patchRows;
    pre_stride_dims[3] = patchCols;
  } else {
    pre_stride_dims[ExtDims - 1] = in.dimension(NumDims - 1);
    pre_stride_dims[ExtDims - 4] = patchCols;
    pre_stride_dims[ExtDims - 3] = patchRows;
    pre_stride_dims[ExtDims - 2] = patchPlanes;
  }

  const TensorIndex inputPlanes = isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputRows = isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const TensorIndex inputCols = isColMajor ? in.dimension(3) : in.dimension(NumDims - 4);

  array<IndexPair<TensorIndex>, NumDims> paddings;
  for (int i = 0; i < NumDims; ++i) {
    paddings[i] = IndexPair<TensorIndex>(0, 0);
  }

  paddings[isColMajor ? 1 : (NumDims - 2)] = IndexPair<TensorIndex>(paddingZTop, paddingZBottom);
  paddings[isColMajor ? 2 : (NumDims - 3)] = IndexPair<TensorIndex>(paddingTop, paddingBottom);
  paddings[isColMajor ? 3 : (NumDims - 4)] = IndexPair<TensorIndex>(paddingLeft, paddingRight);

  pre_stride_dims[isColMajor ? 4 : (ExtDims - 5)] = inputPlanes + paddingZBottom + paddingZTop - patchPlanes + 1;
  pre_stride_dims[isColMajor ? 5 : (ExtDims - 6)] = inputRows + paddingTop + paddingBottom - patchRows + 1;
  pre_stride_dims[isColMajor ? 6 : (ExtDims - 7)] = inputCols + paddingLeft + paddingRight - patchCols + 1;

  if (isColMajor) {
    for (int i = 7; i < NumDims + 3; ++i) {
      pre_stride_dims[i] = in.dimension(i - 3);
    }
  } else {
    for (int i = 0; i < NumDims - 4; ++i) {
      pre_stride_dims[i] = in.dimension(i);
    }
  }

  DSizes<TensorIndex, NumDims> patch_dims;
  if (isColMajor) {
    patch_dims[0] = in.dimension(0);
    patch_dims[1] = patchPlanes;
    patch_dims[2] = patchRows;
    patch_dims[3] = patchCols;
    for (int i = 4; i < NumDims; ++i) {
      patch_dims[i] = 1;
    }
  } else {
    patch_dims[NumDims - 1] = in.dimension(NumDims - 1);
    patch_dims[NumDims - 4] = patchCols;
    patch_dims[NumDims - 3] = patchRows;
    patch_dims[NumDims - 2] = patchPlanes;
    for (int i = 0; i < NumDims - 4; i++) {
      patch_dims[i] = 1;
    }
  }

  array<TensorIndex, NumDims + 3> strides;
  if (isColMajor) {
    // No striding within the patches.
    for (int i = 0; i < 4; ++i) {
      strides[i] = 1;
    }
    // Apply striding in the spatial patch grid dimensions only.
    strides[4] = stridePlanes;
    strides[5] = strideRows;
    strides[6] = strideCols;
    // No striding in the remaining dimensions (batches, ...).
    for (int i = 7; i < NumDims + 3; i++) {
      strides[i] = 1;
    }
  } else {
    // No striding within the patches.
    for (int i = 1; i <= 4; ++i) {
      strides[ExtDims - i] = 1;
    }
    // Apply striding in the spatial patch grid dimensions only.
    strides[ExtDims - 7] = strideCols;
    strides[ExtDims - 6] = strideRows;
    strides[ExtDims - 5] = stridePlanes;
    // No striding in the remaining dimensions (batches, ...).
    for (int i = 0; i < NumDims - 4; i++) {
      strides[i] = 1;
    }
  }

  // TODO(mjanusz): Consider getting rid of pad(), and stride() and extend
  // extract_patches to take additional parameters for padding/striding,
  // similarly to extract_image_patches.
  return input.pad(paddings, padding_value).extract_patches(patch_dims).reshape(pre_stride_dims).stride(strides);
}


template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorStridingOp<
    const array<typename internal::traits<Input>::Index,
                internal::traits<Input>::NumDimensions + 3>,
    const TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions + 3>,
        const TensorPatchOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorPaddingOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            internal::traits<Input>::NumDimensions>,
                const Input> > > >
Extract3DPatches(
    const Input& input, const DenseIndex patchPlanes,
    const DenseIndex patchRows, const DenseIndex patchCols,
    const DenseIndex stridePlanes, const DenseIndex strideRows,
    const DenseIndex strideCols, const PaddingType padding_type,
    const typename internal::traits<Input>::Scalar padding_value = 0) {
  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions >= 4, YOU_MADE_A_PROGRAMMING_MISTAKE);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int NumDims = internal::traits<Input>::NumDimensions;

  const TensorIndex inputPlanes = isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputRows = isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const TensorIndex inputCols = isColMajor ? in.dimension(3) : in.dimension(NumDims - 4);

  switch (padding_type) {
    case PADDING_VALID:
      // No padding in any dimension.
      return Extract3DPatches(input, patchPlanes, patchRows, patchCols,
                              stridePlanes, strideRows, strideCols,
                              0, 0, 0, 0, 0, 0, padding_value);
    case PADDING_SAME: {
      // The side of the tensor before striding should be just the expected
      // output times the stride.
      const TensorIndex size_z = ceil(inputPlanes / static_cast<float>(stridePlanes)) * stridePlanes;
      const TensorIndex size_y = ceil(inputRows / static_cast<float>(strideRows)) * strideRows;
      const TensorIndex size_x = ceil(inputCols / static_cast<float>(strideCols)) * strideCols;

      // The size of the patch space is going to be: padded_input_size - patch_size + 1.
      // This has to match the expected size before striding (pre_stride_dims).
      // The deltas below extend the input to the expected size.
      const TensorIndex dz = size_z + patchPlanes - 1 - inputPlanes;
      const TensorIndex dy = size_y + patchRows - 1 - inputRows;
      const TensorIndex dx = size_x + patchCols - 1 - inputCols;

      return Extract3DPatches(input, patchPlanes, patchRows, patchCols,
                              stridePlanes, strideRows, strideCols,
                              dz - dz / 2, dz / 2,
                              dy - dy / 2, dy / 2,
                              dx - dx / 2, dx / 2,
                              padding_value);
    }
    default:
      eigen_assert(false && "unexpected padding");
      // unreachable code to avoid missing return warning.
      return Extract3DPatches(input, patchPlanes, patchRows, patchCols,
                              stridePlanes, strideRows, strideCols,
                              0, 0, 0, 0, 0, 0, padding_value);
  }
}

// TODO(mjanusz): Switch this to a 'using' alias once CUDA supports C++11.
template <typename Input>
struct Extract3DPatchesType {
  typedef const TensorStridingOp< const array<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions + 3>,
      const TensorReshapingOp< const DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions + 3>,
      const TensorPatchOp< const DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>,
      const TensorPaddingOp< const array< IndexPair<typename internal::traits<Input>::Index>, internal::traits<Input>::NumDimensions>,
      const Input> > > > type;
};

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_CXX11_SRC_NEURAL_NETWORKS_PATCH3D_H
