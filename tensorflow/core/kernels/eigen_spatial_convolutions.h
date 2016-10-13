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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

namespace internal {

// TODO: Consolidate this part of the code with the image patch extraction code
// since they are both very similar.
template <typename NewDimension, DenseIndex Rows, DenseIndex Cols,
          typename ArgType, typename Device, typename Scalar_, typename Index,
          typename nocontract_t, typename contract_t, int Side, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper<
    Scalar_, Index, Side,
    TensorEvaluator<
        const TensorReshapingOp<NewDimension,
                                const TensorImagePatchOp<Rows, Cols, ArgType> >,
        Device>,
    nocontract_t, contract_t, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  typedef Scalar_ Scalar;
  typedef TensorContractionInputMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      Self;
  typedef TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;
  typedef SubMapper VectorMapper;
  typedef SubMapper LinearMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  EIGEN_DEVICE_FUNC
  TensorContractionInputMapper(
      const TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>& tensor,
      const nocontract_t&, const nocontract_t&, const contract_t&,
      const contract_t&)
      : m_impl(tensor.impl().impl()) {
    Index patch_rows;
    Index patch_depth;
    if (internal::traits<ArgType>::Layout == ColMajor) {
      patch_depth = tensor.impl().dimensions()[0];
      patch_rows = tensor.impl().dimensions()[1];
      m_patch_cols = tensor.impl().dimensions()[2];
      m_num_patches = tensor.impl().dimensions()[3];
    } else {
      const int NumDims = tensor.impl().dimensions().size();
      patch_depth = tensor.impl().dimensions()[NumDims - 1];
      patch_rows = tensor.impl().dimensions()[NumDims - 2];
      m_patch_cols = tensor.impl().dimensions()[NumDims - 3];
      m_num_patches = tensor.impl().dimensions()[NumDims - 4];
    }
    m_patch_row_inflate_strides = tensor.impl().rowInflateStride();
    m_patch_col_inflate_strides = tensor.impl().colInflateStride();

    m_colStride = patch_rows;

    m_outputRows = tensor.impl().outputRows();
    m_row_strides = tensor.impl().userRowStride();
    m_col_strides = tensor.impl().userColStride();

    m_in_row_strides = tensor.impl().userInRowStride();
    m_in_col_strides = tensor.impl().userInColStride();

    if (internal::traits<ArgType>::Layout == ColMajor) {
      m_inputRows = tensor.impl().impl().dimensions()[1];
      m_inputCols = tensor.impl().impl().dimensions()[2];
    } else {
      const int NumDims = tensor.impl().impl().dimensions().size();
      m_inputRows = tensor.impl().impl().dimensions()[NumDims - 2];
      m_inputCols = tensor.impl().impl().dimensions()[NumDims - 3];
    }

    m_rowInputStride = patch_depth;
    m_colInputStride = patch_depth * m_inputRows;
    m_patchInputStride = patch_depth * m_inputRows * m_inputCols;

    m_rowPaddingTop = tensor.impl().rowPaddingTop();
    m_colPaddingLeft = tensor.impl().colPaddingLeft();

    m_fastInputRowStride =
        internal::TensorIntDivisor<Index>(m_patch_row_inflate_strides);
    m_fastInputColStride =
        internal::TensorIntDivisor<Index>(m_patch_col_inflate_strides);
    m_fastNumPatches = internal::TensorIntDivisor<Index>(m_num_patches);
    m_fastColStride = internal::TensorIntDivisor<Index>(m_colStride);
    m_fastOutputRows = internal::TensorIntDivisor<Index>(m_outputRows);
    m_fastDimZero = internal::TensorIntDivisor<Index>(patch_depth);
  }

  EIGEN_DEVICE_FUNC
  TensorContractionInputMapper(const TensorContractionInputMapper& base_mapper)
      : m_impl(base_mapper.m_impl) {
    m_patch_cols = base_mapper.m_patch_cols;
    m_num_patches = base_mapper.m_num_patches;
    m_patch_row_inflate_strides = base_mapper.m_patch_row_inflate_strides;
    m_patch_col_inflate_strides = base_mapper.m_patch_col_inflate_strides;

    m_colStride = base_mapper.m_colStride;

    m_rowInputStride = base_mapper.m_rowInputStride;
    m_colInputStride = base_mapper.m_colInputStride;
    m_patchInputStride = base_mapper.m_patchInputStride;

    m_inputRows = base_mapper.m_inputRows;
    m_inputCols = base_mapper.m_inputCols;

    m_outputRows = base_mapper.m_outputRows;
    m_row_strides = base_mapper.m_row_strides;
    m_col_strides = base_mapper.m_col_strides;

    m_in_row_strides = base_mapper.m_in_row_strides;
    m_in_col_strides = base_mapper.m_in_col_strides;

    m_rowPaddingTop = base_mapper.m_rowPaddingTop;
    m_colPaddingLeft = base_mapper.m_colPaddingLeft;

    m_fastInputRowStride = base_mapper.m_fastInputRowStride;
    m_fastInputColStride = base_mapper.m_fastInputColStride;
    m_fastNumPatches = base_mapper.m_fastNumPatches;
    m_fastColStride = base_mapper.m_fastColStride;
    m_fastOutputRows = base_mapper.m_fastOutputRows;
    m_fastDimZero = base_mapper.m_fastDimZero;
  }

  // If true, turns off some optimizations for loading packets since the image
  // patches are "non-standard" such as there are non-trivial strides or
  // inflations in the input.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
    return m_in_row_strides != 1 || m_in_col_strides != 1 ||
           m_patch_row_inflate_strides != 1 || m_patch_col_inflate_strides != 1;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE SubMapper getSubMapper(Index i, Index j) const {
    return SubMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar operator()(Index row) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(0, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, rowIndex, colIndex, otherIndex);
  }

  // Load the coefficient at the patchIndex location instead of the usual
  // m_rowIndex,
  // m_colIndex, m_otherIndex. This is currently only used by the gpu code.
  // EIGEN_DEVICE_FUNC
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index patchIndex) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(0, rowIndex, colIndex, otherIndex);
    return loadPacket(row, rowIndex, colIndex, otherIndex);
  }

  // Load the packet at the patchIndex location instead of the usual m_rowIndex,
  // m_colIndex, m_otherIndex. This is currently only used by the gpu code.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row, Index patchIndex) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, rowIndex, colIndex, otherIndex);
    return loadPacket(row, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE const TensorEvaluator<ArgType, Device>& impl() const {
    return m_impl;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchDepth() const { return m_rowInputStride; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRows() const { return m_colStride; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchCols() const { return m_patch_cols; }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet packetNoPadding(const Index depth,
                                             const Index baseIndex) const {
    const Index inputIndex = depth + baseIndex;
    return m_impl.template packet<Unaligned>(inputIndex);
  }

 private:
  friend class TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar loadCoeff(Index patchId, Index rowIndex,
                                       Index colIndex, Index otherIndex) const {
    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;

    const Index colOffset = patchOffset / m_fastColStride;
    const Index inputCol = colIndex + colOffset * m_in_col_strides;
    const Index origInputCol =
        (m_patch_col_inflate_strides == 1)
            ? inputCol
            : ((inputCol >= 0) ? (inputCol / m_fastInputColStride) : 0);
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputRow = rowIndex + rowOffset * m_in_row_strides;
    const Index origInputRow =
        (m_patch_row_inflate_strides == 1)
            ? inputRow
            : ((inputRow >= 0) ? (inputRow / m_fastInputRowStride) : 0);
    if (origInputCol < 0 || origInputRow < 0 || origInputCol >= m_inputCols ||
        origInputRow >= m_inputRows ||
        (inputCol != origInputCol * m_patch_col_inflate_strides) ||
        (inputRow != origInputRow * m_patch_row_inflate_strides)) {
      return Scalar(0);
    }
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + origInputRow * m_rowInputStride +
                             origInputCol * m_colInputStride + otherIndex;
    return m_impl.coeff(inputIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar loadCoeffStandard(Index patchId, Index rowIndex,
                                               Index colIndex,
                                               Index otherIndex) const {
    eigen_assert(!nonStandardPatches());

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;

    const Index colOffset = patchOffset / m_fastColStride;
    const Index inputCol = colIndex + colOffset;
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputRow = rowIndex + rowOffset;
    if (inputCol < 0 || inputCol >= m_inputCols || inputRow < 0 ||
        inputRow >= m_inputRows) {
      return Scalar(0);
    }
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.coeff(inputIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index patchId, Index rowIndex,
                                        Index colIndex,
                                        Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    if (nonStandardPatches()) {
      return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
    }
    return loadPacketStandard(patchId, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketStandard(Index patchId, Index rowIndex,
                                                Index colIndex,
                                                Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    eigen_assert(!nonStandardPatches());

    if ((patchDepth() % packetSize) == 0) {
      return loadPacketFast(patchId, rowIndex, colIndex, otherIndex);
    } else {
      const Index patchOffsets[2] = {
          patchId / m_fastDimZero, (patchId + packetSize - 1) / m_fastDimZero};

      const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                   patchOffsets[1] / m_fastColStride};

      const Index inputCols[2] = {colIndex + colOffsets[0],
                                  colIndex + colOffsets[1]};
      if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
        // all zeros
        return internal::pset1<Packet>(Scalar(0));
      }

      if (inputCols[0] == inputCols[1]) {
        const Index rowOffsets[2] = {
            patchOffsets[0] - colOffsets[0] * m_colStride,
            patchOffsets[1] - colOffsets[1] * m_colStride};
        eigen_assert(rowOffsets[0] <= rowOffsets[1]);
        const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                    rowIndex + rowOffsets[1]};

        if (inputRows[0] >= m_inputRows || inputRows[1] < 0) {
          // all zeros
          return internal::pset1<Packet>(Scalar(0));
        }

        if (inputRows[0] >= 0 && inputRows[1] < m_inputRows) {
          // no padding
          const Index depth = patchId - patchOffsets[0] * patchDepth();
          const Index inputIndex = depth + inputRows[0] * m_rowInputStride +
                                   inputCols[0] * m_colInputStride + otherIndex;
          return m_impl.template packet<Unaligned>(inputIndex);
        }
      }
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index patchId, Index rowIndex,
                                            Index colIndex,
                                            Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    eigen_assert(!nonStandardPatches());
    eigen_assert((patchDepth() % packetSize) == 0);
    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;
    eigen_assert((patchId + packetSize - 1) / m_fastDimZero == patchOffset);

    const Index colOffset = patchOffset / m_fastColStride;
    const Index inputCol = colIndex + colOffset;
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputRow = rowIndex + rowOffset;
    if (inputCol < 0 || inputRow < 0 || inputCol >= m_inputCols ||
        inputRow >= m_inputRows) {
      // all zeros
      return internal::pset1<Packet>(Scalar(0));
    }
    // no padding
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.template packet<Unaligned>(inputIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet packetWithPossibleZero(
      Index patchId, Index rowIndex, Index colIndex, Index otherIndex) const {
    const int packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_ALIGN_MAX
    typename internal::remove_const<Scalar>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = loadCoeff(patchId + i, rowIndex, colIndex, otherIndex);
    }
    Packet rslt = internal::pload<Packet>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void computeBaseIndices(
      Index patchIndex, Index& rowIndex, Index& colIndex,
      Index& otherIndex) const {
    const int NumInputDims = array_size<
        typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
    otherIndex = (NumInputDims == 3) ? 0 : patchIndex / m_fastNumPatches;
    const Index patch2DIndex = (NumInputDims == 3)
                                   ? patchIndex
                                   : (patchIndex - otherIndex * m_num_patches);
    otherIndex *= m_patchInputStride;
    colIndex = patch2DIndex / m_fastOutputRows;
    rowIndex = patch2DIndex - colIndex * m_outputRows;
    colIndex = colIndex * m_col_strides - m_colPaddingLeft;
    rowIndex = rowIndex * m_row_strides - m_rowPaddingTop;
  }

  Index m_patch_cols;                 // number of colums in the patch
  Index m_num_patches;                // number of patches to extract.
  Index m_patch_row_inflate_strides;  // the strides for row inflation in the
                                      // image patch
  Index m_patch_col_inflate_strides;  // the strides for col inflation in the
                                      // image patch
  // Fast representation of inflation strides.
  internal::TensorIntDivisor<Index> m_fastInputRowStride;
  internal::TensorIntDivisor<Index> m_fastInputColStride;

  Index m_otherStride;
  Index m_colStride;
  internal::TensorIntDivisor<Index> m_fastNumPatches;
  internal::TensorIntDivisor<Index> m_fastColStride;

  Index m_rowInputStride;    // row stride in the input tensor
  Index m_colInputStride;    // col stride in the input tensor
  Index m_patchInputStride;  // patch stride in the input tensor

  Index m_inputRows;  // Number of rows in the input tensor
  Index m_inputCols;  // Number of cols in the input tensor

  Index m_outputRows;  // Number of patch rows

  Index m_row_strides;  // User specified row stride
  Index m_col_strides;  // User specified col stride

  Index m_in_row_strides;  // User specified input row stride
  Index m_in_col_strides;  // User specified input col stride

  Index m_rowPaddingTop;   // Row padding
  Index m_colPaddingLeft;  // Column padding

  internal::TensorIntDivisor<Index> m_fastOutputRows;
  internal::TensorIntDivisor<Index> m_fastDimZero;

  const TensorEvaluator<ArgType, Device> m_impl;
};

template <typename NewDimension, DenseIndex Rows, DenseIndex Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, int Side, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper<
    Scalar, Index, Side,
    TensorEvaluator<
        const TensorReshapingOp<NewDimension,
                                const TensorImagePatchOp<Rows, Cols, ArgType> >,
        Device>,
    nocontract_t, contract_t, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  typedef TensorContractionInputMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      ParentMapper;
  typedef TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      Self;
  typedef Self LinearMapper;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const ParentMapper& base_mapper, Index vert_offset, Index horiz_offset)
      : m_base_mapper(base_mapper),
        m_depth_offset(vert_offset),
        m_col_offset(horiz_offset) {
    m_base_mapper.computeBaseIndices(m_col_offset, m_rowIndex, m_colIndex,
                                     m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const Self& base_mapper, Index vert_offset, Index horiz_offset)
      : m_base_mapper(base_mapper.m_base_mapper),
        m_depth_offset(vert_offset + base_mapper.m_depth_offset),
        m_col_offset(horiz_offset + base_mapper.m_col_offset) {
    m_base_mapper.computeBaseIndices(m_col_offset, m_rowIndex, m_colIndex,
                                     m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_base_mapper.loadCoeff(i + m_depth_offset, m_rowIndex, m_colIndex,
                                   m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i,
                                                          Index j) const {
    return m_base_mapper(i + m_depth_offset, j + m_col_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    return m_base_mapper.loadPacket(i + m_depth_offset, m_rowIndex, m_colIndex,
                                    m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i,
                                                          Index j) const {
    return m_base_mapper.template loadPacket<Alignment>(i + m_depth_offset,
                                                        j + m_col_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar
  loadCoeffStandard(Index i) const {
    return m_base_mapper.loadCoeffStandard(i + m_depth_offset, m_rowIndex,
                                           m_colIndex, m_otherIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index i) const {
    return m_base_mapper.loadPacketFast(i + m_depth_offset, m_rowIndex,
                                        m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet
  loadPacketStandard(Index i) const {
    return m_base_mapper.loadPacketStandard(i + m_depth_offset, m_rowIndex,
                                            m_colIndex, m_otherIndex);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC bool aligned(Index) const {
    return false;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
    return m_base_mapper.nonStandardPatches();
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchDepth() const {
    return m_base_mapper.m_rowInputStride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRows() const {
    return m_base_mapper.m_colStride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchCols() const {
    return m_base_mapper.m_patch_cols;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet packetNoPadding(const Index depth,
                                             const Index baseIndex) const {
    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.template packet<Unaligned>(inputIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padRow(const Index row) const {
    const Index r = m_rowIndex + row;
    return r < 0 || r >= m_base_mapper.m_inputRows;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padCol(const Index col) const {
    const Index c = m_colIndex + col;
    return c < 0 || c >= m_base_mapper.m_inputCols;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index baseIndex(const Index row, const Index col) const {
    const Index r = m_rowIndex + row;
    const Index c = m_colIndex + col;
    return r * m_base_mapper.m_rowInputStride +
           c * m_base_mapper.m_colInputStride + m_otherIndex;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index rowOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return patchOffset - colOffset * m_base_mapper.m_colStride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index colOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return colOffset;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index depthOffset() const {
    const Index patchOffset = m_depth_offset % m_base_mapper.patchDepth();
    return patchOffset;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LinearMapper
  getLinearMapper(Index i, Index j) const {
    return LinearMapper(m_base_mapper, i + m_depth_offset, j + m_col_offset);
  }

 private:
  const ParentMapper& m_base_mapper;  // that was a reference before
  Index m_depth_offset;               // First row in the input matrix
  Index m_col_offset;                 // First col in the input matrix

  Index m_rowIndex;  // precomputed row index corresponding to the col offset
  Index m_colIndex;  // precomputed col index corresponding to the col offset
  Index
      m_otherIndex;  // precomputed other index corresponding to the col offset
};

template <typename NewDimension, DenseIndex Rows, DenseIndex Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment,
          int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<
            const TensorReshapingOp<
                NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
            Device>,
        nocontract_t, contract_t, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;
  typedef SubMapper DataMapper;

  EIGEN_DEVICE_FUNC
  static inline Index ceil_div(Index a, Index b) { return (a + b - 1) / b; }

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

    EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE);
    typedef typename packet_traits<Scalar>::type Packet;

    const Index packet_cols4 = (cols / 4) * 4;
    const Index peeled_k = (depth / packet_size) * packet_size;
    const bool non_standard_patches = rhs.nonStandardPatches();

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if ((packet_size % 4) == 0 && !non_standard_patches) {
        const Index patch_depth = rhs.patchDepth();
        if ((patch_depth % packet_size) == 0) {
          const Index patch_cols = rhs.patchCols();
          const Index patch_rows = rhs.patchRows();

          const Index startCol = rhs.colOffset();
          const Index max_cols = std::min<Index>(
              ceil_div(peeled_k, patch_rows * patch_depth) + startCol,
              patch_cols);

          for (Index c = startCol; c < max_cols; ++c) {
            eigen_assert(k < peeled_k);
            const Index startRow = (c == startCol) ? rhs.rowOffset() : 0;
            const Index max_rows = std::min<Index>(
                ceil_div(peeled_k - c * patch_rows * patch_depth, patch_depth) +
                    startRow,
                patch_rows);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);
            for (Index r = startRow; r < max_rows; ++r) {
              eigen_assert(k < peeled_k);
              const bool pad0 = pad_col0 || dm0.padRow(r);
              const bool pad1 = pad_col1 || dm1.padRow(r);
              const bool pad2 = pad_col2 || dm2.padRow(r);
              const bool pad3 = pad_col3 || dm3.padRow(r);

              const Index idx0 = dm0.baseIndex(r, c);
              const Index idx1 = dm1.baseIndex(r, c);
              const Index idx2 = dm2.baseIndex(r, c);
              const Index idx3 = dm3.baseIndex(r, c);

              const Index startDepth =
                  ((c == startCol) && (r == startRow)) ? rhs.depthOffset() : 0;
              const Index max_depth =
                  std::min<Index>(peeled_k - c * patch_rows * patch_depth -
                                      r * patch_depth + startDepth,
                                  patch_depth);
              eigen_assert((max_depth - startDepth) % packet_size == 0);
              for (Index d = startDepth; d < max_depth; d += packet_size) {
                eigen_assert(k < peeled_k);
                PacketBlock<Packet, 4> kernel;
                kernel.packet[0] = pad0 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx0);
                kernel.packet[1] = pad1 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx1);
                kernel.packet[2] = pad2 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx2);
                kernel.packet[3] = pad3 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx3);
                ptranspose(kernel);
                pstoreu(block + 0 * packet_size, kernel.packet[0]);
                pstoreu(block + 1 * packet_size, kernel.packet[1]);
                pstoreu(block + 2 * packet_size, kernel.packet[2]);
                pstoreu(block + 3 * packet_size, kernel.packet[3]);
                block += 4 * packet_size;
                k += packet_size;
              }
            }
          }

          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 4> kernel;
            kernel.packet[0] = dm0.loadPacketFast(k);
            kernel.packet[1] = dm1.loadPacketFast(k);
            kernel.packet[2] = dm2.loadPacketFast(k);
            kernel.packet[3] = dm3.loadPacketFast(k);
            ptranspose(kernel);
            pstoreu(block + 0 * packet_size, kernel.packet[0]);
            pstoreu(block + 1 * packet_size, kernel.packet[1]);
            pstoreu(block + 2 * packet_size, kernel.packet[2]);
            pstoreu(block + 3 * packet_size, kernel.packet[3]);
            block += 4 * packet_size;
          }
        } else {
          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 4> kernel;
            kernel.packet[0] = dm0.loadPacketStandard(k);
            kernel.packet[1] = dm1.loadPacketStandard(k);
            kernel.packet[2] = dm2.loadPacketStandard(k);
            kernel.packet[3] = dm3.loadPacketStandard(k);
            ptranspose(kernel);
            pstoreu(block + 0 * packet_size, kernel.packet[0]);
            pstoreu(block + 1 * packet_size, kernel.packet[1]);
            pstoreu(block + 2 * packet_size, kernel.packet[2]);
            pstoreu(block + 3 * packet_size, kernel.packet[3]);
            block += 4 * packet_size;
          }
        }
      }
      if (!rhs.nonStandardPatches()) {
        for (; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // copy the remaining columns one at a time (nr==1)
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Special case for non-vectorized types such as float16.
template <typename NewDimension, DenseIndex Rows, DenseIndex Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment, int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<
            const TensorReshapingOp<
                NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
            Device>,
        nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered,
        Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered,
      Alignment>
      SubMapper;
  typedef SubMapper DataMapper;

  EIGEN_DEVICE_FUNC
  static inline Index ceil_div(Index a, Index b) { return (a + b - 1) / b; }

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

    EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE);

    const Index packet_cols4 = (cols / 4) * 4;

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      if (!rhs.nonStandardPatches()) {
        for (Index k = 0; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (Index k = 0; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // copy the remaining columns one at a time (nr==1)
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

}  // end namespace internal

/** SpatialConvolution
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies a 2D convolution over a multichannel input image.
  *
  * The input parameter is expected to be a tensor with a rank of 3 or more
 * (channels, height, width, and optionally others)
  * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_height, kernel_width)
  * The input and the kernel must both be in col-major layout. The result will
 * also be in col-major layout.
  *
  * If col_in_stride, row_in_stride > 1, then applies convolution with holes
 * (aka atrous convolution), sampling every col_in_stride, row_in_stride input
 * pixels.
  *
  * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be filters, height, width (and
 * others if applicable).
  *
  * It is possible to swap the order of the width and height dimensions provided
 * that the same order is used in the input, the kernel, and the output.
  *
  */
template <typename Input, typename Kernel>
EIGEN_DEVICE_FUNC
    EIGEN_ALWAYS_INLINE static const typename internal::conditional<
        internal::traits<Input>::Layout == ColMajor,
        TensorReshapingOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorContractionOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            1>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const Kernel>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic,
                                             const Input> > > >,
        TensorReshapingOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorContractionOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            1>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const Kernel> > > >::type
    SpatialConvolution(const Input& input, const Kernel& kernel,
                       const DenseIndex row_stride = 1,
                       const DenseIndex col_stride = 1,
                       const PaddingType padding_type = PADDING_SAME,
                       const DenseIndex row_in_stride = 1,
                       const DenseIndex col_in_stride = 1) {
  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
                   internal::traits<Kernel>::NumDimensions,
                   internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);

  EIGEN_STATIC_ASSERT(
      internal::traits<Input>::Layout == internal::traits<Kernel>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE);
  const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  const int NumDims = internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  const DenseIndex kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const DenseIndex kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex InputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case PADDING_VALID:
      out_height = numext::ceil((InputRows - kernelRowsEff + 1.f) /
                                static_cast<float>(row_stride));
      out_width = numext::ceil((InputCols - kernelColsEff + 1.f) /
                               static_cast<float>(col_stride));
      break;
    case PADDING_SAME:
      out_height = numext::ceil(InputRows / static_cast<float>(row_stride));
      out_width = numext::ceil(InputCols / static_cast<float>(col_stride));
      break;
    default:
      eigen_assert(false && "unexpected padding");
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = out_height * out_width;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[0] = out_height * out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_height;
    post_contract_dims[2] = out_width;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_height;
    post_contract_dims[NumDims - 3] = out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }
  // TODO(yangke): choose() is defined in TensorContraction.h -- consider
  // moving it to somewhere more "common".
  return choose(
      Cond<internal::traits<Input>::Layout == ColMajor>(),
      kernel.reshape(kernel_dims)
          .contract(input
                        .extract_image_patches(
                            kernelRows, kernelCols, row_stride, col_stride,
                            row_in_stride, col_in_stride, padding_type)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(post_contract_dims),
      input
          .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
                                 row_in_stride, col_in_stride, padding_type)
          .reshape(pre_contract_dims)
          .contract(kernel.reshape(kernel_dims), contract_dims)
          .reshape(post_contract_dims));
}

}  // end namespace Eigen

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_
