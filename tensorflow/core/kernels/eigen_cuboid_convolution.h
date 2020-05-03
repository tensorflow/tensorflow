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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_CUBOID_CONVOLUTION_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_CUBOID_CONVOLUTION_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_volume_patch.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#include "tensorflow/core/kernels/eigen_convolution_helpers.h"

namespace Eigen {

namespace internal {

// WARNING: Most of the code here implicitly assumes that the matrix is in
// ColMajor layout. This is guaranteed by the tensor contraction (see
// TensorContraction.h).
//
// Inside Eigen a tensor contraction is represented by a matrix multiplication.
// We don't want to actually extract volume patches and reshape the result into
// a matrix (this involves allocating huge extra memory), so the patch
// extraction and reshape operations are implicit.
//
// TensorContractionInputMapper takes a matrix index and returns the coefficient
// (or the packet) of the "virtual tensor", that would be at that index if we
// were to actually reshape the result of patch extraction.
//
// TensorContractionSubMapper provides a similar view into the "virtual matrix"
// at the given vertical and horizontal offsets.
//
// "Virtual matrix" dimensions:
//   *0: kernelChannels * kernelPlanes * kernelRows * kernelCols
//    1: out_planes * out_height * out_width * OTHERS (e.g batches, etc...)
//
// *) extracted patches are continuous in memory (innermost dimension assuming
//    col major layout)
//
// With this dimensions:
//   row - offset within a single patch (in code: patchId)
//   col - index of the extracted patch (in code: patchIndex)
//         patchIndex ∈ [0..num_patches * OTHERS] (batch and other dimensions)
//
template <typename NewDimension, Index Planes, Index Rows, Index Cols,
          typename ArgType, typename Device, typename Scalar_, typename Index,
          typename nocontract_t, typename contract_t, int Side, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper<
    Scalar_, Index, Side,
    TensorEvaluator<const TensorReshapingOp<NewDimension,
                                            const TensorVolumePatchOp<
                                                Planes, Rows, Cols, ArgType> >,
                    Device>,
    nocontract_t, contract_t, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  typedef Scalar_ Scalar;
  typedef TensorContractionInputMapper<
      Scalar, Index, Side,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      Self;
  typedef TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
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
              NewDimension,
              const TensorVolumePatchOp<Planes, Rows, Cols, ArgType> >,
          Device>& tensor,
      const nocontract_t&, const nocontract_t&, const contract_t&,
      const contract_t&)
      : m_impl(tensor.impl().impl()) {
    if (internal::traits<ArgType>::Layout == ColMajor) {
      m_patch_depth = tensor.impl().dimensions()[0];
      m_patch_planes = tensor.impl().dimensions()[1];
      m_patch_rows = tensor.impl().dimensions()[2];
      m_patch_cols = tensor.impl().dimensions()[3];
      m_num_patches = tensor.impl().dimensions()[4];
    } else {
      const int NumDims = tensor.impl().dimensions().size();
      m_patch_depth = tensor.impl().dimensions()[NumDims - 1];
      m_patch_planes = tensor.impl().dimensions()[NumDims - 2];
      m_patch_rows = tensor.impl().dimensions()[NumDims - 3];
      m_patch_cols = tensor.impl().dimensions()[NumDims - 4];
      m_num_patches = tensor.impl().dimensions()[NumDims - 5];
    }

    // Strides for navigating through the single patch.
    m_patch_plane_stride = m_patch_depth;
    m_patch_row_stride = m_patch_planes * m_patch_plane_stride;
    m_patch_col_stride = m_patch_rows * m_patch_row_stride;

    // Strides for the output tensor.
    // IMPORTANT: These strides are used to locate an element in a patch at a
    // depth zero (channel), which is not quite the same as "traditional"
    // stride.
    m_rowStride = m_patch_planes;
    m_colStride = m_patch_rows * m_rowStride;
    m_patchStride = m_colStride * m_patch_cols * m_patch_depth;
    m_otherStride = m_patchStride * m_num_patches;

    m_outputPlanes = tensor.impl().outputPlanes();
    m_outputRows = tensor.impl().outputRows();
    m_outputCols = tensor.impl().outputCols();

    m_outputPlanesRows = m_outputPlanes * m_outputRows;

    m_plane_strides = tensor.impl().userPlaneStride();
    m_row_strides = tensor.impl().userRowStride();
    m_col_strides = tensor.impl().userColStride();

    m_in_plane_strides = tensor.impl().userInPlaneStride();
    m_in_row_strides = tensor.impl().userInRowStride();
    m_in_col_strides = tensor.impl().userInColStride();

    m_patch_plane_inflate_strides = tensor.impl().planeInflateStride();
    m_patch_row_inflate_strides = tensor.impl().rowInflateStride();
    m_patch_col_inflate_strides = tensor.impl().colInflateStride();

    if (internal::traits<ArgType>::Layout == ColMajor) {
      m_inputDepth = tensor.impl().impl().dimensions()[0];
      m_inputPlanes = tensor.impl().impl().dimensions()[1];
      m_inputRows = tensor.impl().impl().dimensions()[2];
      m_inputCols = tensor.impl().impl().dimensions()[3];
    } else {
      const int NumDims = tensor.impl().impl().dimensions().size();
      m_inputDepth = tensor.impl().impl().dimensions()[NumDims - 1];
      m_inputPlanes = tensor.impl().impl().dimensions()[NumDims - 2];
      m_inputRows = tensor.impl().impl().dimensions()[NumDims - 3];
      m_inputCols = tensor.impl().impl().dimensions()[NumDims - 4];
    }

    // Strides for navigating through the input tensor.
    m_planeInputStride = m_inputDepth;
    m_rowInputStride = m_inputDepth * m_inputPlanes;
    m_colInputStride = m_inputDepth * m_inputRows * m_inputPlanes;
    m_patchInputStride =
        m_inputDepth * m_inputRows * m_inputCols * m_inputPlanes;

    m_planePaddingTop = tensor.impl().planePaddingTop();
    m_rowPaddingTop = tensor.impl().rowPaddingTop();
    m_colPaddingLeft = tensor.impl().colPaddingLeft();

    m_fastNumPatches = internal::TensorIntDivisor<Index>(m_num_patches);

    m_fastPatchPlaneStride =
        internal::TensorIntDivisor<Index>(m_patch_plane_stride);
    m_fastPatchRowStride =
        internal::TensorIntDivisor<Index>(m_patch_row_stride);
    m_fastPatchColStride =
        internal::TensorIntDivisor<Index>(m_patch_col_stride);

    m_fastInputPlaneStride =
        internal::TensorIntDivisor<Index>(m_patch_plane_inflate_strides);
    m_fastInputRowStride =
        internal::TensorIntDivisor<Index>(m_patch_row_inflate_strides);
    m_fastInputColStride =
        internal::TensorIntDivisor<Index>(m_patch_col_inflate_strides);

    m_fastRowStride = internal::TensorIntDivisor<Index>(m_rowStride);
    m_fastColStride = internal::TensorIntDivisor<Index>(m_colStride);

    m_fastDimZero = internal::TensorIntDivisor<Index>(m_patch_depth);
    m_fastOutputRows = internal::TensorIntDivisor<Index>(m_outputRows);
    m_fastOutputPlanes = internal::TensorIntDivisor<Index>(m_outputPlanes);
    m_fastOutputRows = internal::TensorIntDivisor<Index>(m_outputRows);
    m_fastOutputCols = internal::TensorIntDivisor<Index>(m_outputCols);

    m_fastOutputPlanesRows =
        internal::TensorIntDivisor<Index>(m_outputPlanesRows);
  }

  EIGEN_DEVICE_FUNC
  TensorContractionInputMapper(const TensorContractionInputMapper& base_mapper)
      : m_impl(base_mapper.m_impl) {
    m_patch_depth = base_mapper.m_patch_depth;
    m_patch_planes = base_mapper.m_patch_planes;
    m_patch_rows = base_mapper.m_patch_rows;
    m_patch_cols = base_mapper.m_patch_cols;
    m_num_patches = base_mapper.m_num_patches;

    m_patch_plane_stride = base_mapper.m_patch_plane_stride;
    m_patch_row_stride = base_mapper.m_patch_row_stride;
    m_patch_col_stride = base_mapper.m_patch_col_stride;

    m_rowStride = base_mapper.m_rowStride;
    m_colStride = base_mapper.m_colStride;
    m_patchStride = base_mapper.m_patchStride;
    m_otherStride = base_mapper.m_otherStride;

    m_planeInputStride = base_mapper.m_planeInputStride;
    m_rowInputStride = base_mapper.m_rowInputStride;
    m_colInputStride = base_mapper.m_colInputStride;
    m_patchInputStride = base_mapper.m_patchInputStride;
    m_otherInputStride = base_mapper.m_otherInputStride;

    m_inputDepth = base_mapper.m_inputDepth;
    m_inputPlanes = base_mapper.m_inputPlanes;
    m_inputRows = base_mapper.m_inputRows;
    m_inputCols = base_mapper.m_inputCols;

    m_outputPlanes = base_mapper.m_outputPlanes;
    m_outputRows = base_mapper.m_outputRows;
    m_outputCols = base_mapper.m_outputCols;

    m_plane_strides = base_mapper.m_plane_strides;
    m_row_strides = base_mapper.m_row_strides;
    m_col_strides = base_mapper.m_col_strides;

    m_in_plane_strides = base_mapper.m_in_plane_strides;
    m_in_row_strides = base_mapper.m_in_row_strides;
    m_in_col_strides = base_mapper.m_in_col_strides;

    m_patch_plane_inflate_strides = base_mapper.m_patch_plane_inflate_strides;
    m_patch_row_inflate_strides = base_mapper.m_patch_row_inflate_strides;
    m_patch_col_inflate_strides = base_mapper.m_patch_col_inflate_strides;

    m_planePaddingTop = base_mapper.m_planePaddingTop;
    m_rowPaddingTop = base_mapper.m_rowPaddingTop;
    m_colPaddingLeft = base_mapper.m_colPaddingLeft;

    m_outputPlanesRows = base_mapper.m_outputPlanesRows;

    m_fastNumPatches = base_mapper.m_fastNumPatches;
    m_fastPatchPlaneStride = base_mapper.m_fastPatchPlaneStride;
    m_fastPatchRowStride = base_mapper.m_fastPatchRowStride;
    m_fastPatchColStride = base_mapper.m_fastPatchColStride;
    m_fastInputPlaneStride = base_mapper.m_fastInputPlaneStride;
    m_fastInputRowStride = base_mapper.m_fastInputRowStride;
    m_fastInputColStride = base_mapper.m_fastInputColStride;
    m_fastRowStride = base_mapper.m_fastRowStride;
    m_fastColStride = base_mapper.m_fastColStride;
    m_fastOutputPlanes = base_mapper.m_fastOutputPlanes;
    m_fastOutputRows = base_mapper.m_fastOutputRows;
    m_fastOutputCols = base_mapper.m_fastOutputCols;
    m_fastDimZero = base_mapper.m_fastDimZero;
    m_fastOutputPlanesRows = base_mapper.m_fastOutputPlanesRows;
  }

  // If true, turns off some optimizations for loading packets since the image
  // patches are "non-standard" such as there are non-trivial strides or
  // inflations in the input.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
    return m_in_plane_strides != 1 || m_in_row_strides != 1 ||
           m_in_col_strides != 1 || m_patch_plane_inflate_strides != 1 ||
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
    Index planeIndex, rowIndex, colIndex, otherIndex;
    computeBaseIndices(0, planeIndex, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, planeIndex, rowIndex, colIndex, otherIndex);
  }

  // Load the coefficient at the patchIndex location instead of the usual
  // m_rowIndex, m_colIndex, m_otherIndex. This is currently only used by the
  // gpu code.
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index patchIndex) const {
    Index planeIndex, rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, planeIndex, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, planeIndex, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row) const {
    Index planeIndex, rowIndex, colIndex, otherIndex;
    computeBaseIndices(0, planeIndex, rowIndex, colIndex, otherIndex);
    return loadPacket(row, planeIndex, rowIndex, colIndex, otherIndex);
  }

  // Load the packet at the patchIndex location instead of the usual m_rowIndex,
  // m_colIndex, m_otherIndex. This is currently only used by the gpu code.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row, Index patchIndex) const {
    Index planeIndex, rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, planeIndex, rowIndex, colIndex, otherIndex);
    return loadPacket(row, planeIndex, rowIndex, colIndex, otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE const TensorEvaluator<ArgType, Device>& impl() const {
    return m_impl;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchDepth() const { return m_planeInputStride; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchPlanes() const { return m_rowStride; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRows() const { return m_patch_rows; }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchCols() const { return m_patch_cols; }

 private:
  friend class TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;

  // Load coefficient from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar loadCoeff(Index patchId, Index planeIndex,
                                       Index rowIndex, Index colIndex,
                                       Index otherIndex) const {
    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;

    const Index colOffset = patchOffset / m_fastColStride;
    const Index inputCol = colIndex + colOffset * m_in_col_strides;
    const Index origInputCol =
        (m_patch_col_inflate_strides == 1)
            ? inputCol
            : ((inputCol >= 0) ? (inputCol / m_fastInputColStride) : 0);

    const Index rowOffset =
        (patchOffset - colOffset * m_colStride) / m_fastRowStride;
    const Index inputRow = rowIndex + rowOffset * m_in_row_strides;
    const Index origInputRow =
        (m_patch_row_inflate_strides == 1)
            ? inputRow
            : ((inputRow >= 0) ? (inputRow / m_fastInputRowStride) : 0);

    const Index planeOffset =
        patchOffset - colOffset * m_colStride - rowOffset * m_rowStride;
    const Index inputPlane = planeIndex + planeOffset * m_in_plane_strides;
    const Index origInputPlane =
        (m_patch_plane_inflate_strides == 1)
            ? inputPlane
            : ((inputPlane >= 0) ? (inputPlane / m_fastInputPlaneStride) : 0);

    if (origInputCol < 0 || origInputRow < 0 || origInputPlane < 0 ||
        origInputCol >= m_inputCols || origInputRow >= m_inputRows ||
        origInputPlane >= m_inputPlanes ||
        (inputCol != origInputCol * m_patch_col_inflate_strides) ||
        (inputRow != origInputRow * m_patch_row_inflate_strides) ||
        (inputPlane != origInputPlane * m_patch_plane_inflate_strides)) {
      return Scalar(0);
    }

    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + origInputPlane * m_planeInputStride +
                             origInputRow * m_rowInputStride +
                             origInputCol * m_colInputStride + otherIndex;

    return m_impl.coeff(inputIndex);
  }

  // This is the same as loadCoeff(...), but optimized for all `inflate_strides`
  // and `in_strides` equal to 1 (template specialization without templates).
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar loadCoeffStandard(Index patchId, Index planeIndex,
                                               Index rowIndex, Index colIndex,
                                               Index otherIndex) const {
    eigen_assert(!nonStandardPatches());

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;

    const Index colOffset = patchOffset / m_fastColStride;
    const Index rowOffset =
        (patchOffset - colOffset * m_colStride) / m_fastRowStride;
    const Index planeOffset =
        patchOffset - colOffset * m_colStride - rowOffset * m_rowStride;

    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    const Index inputPlane = planeIndex + planeOffset;

    if (inputCol < 0 || inputCol >= m_inputCols || inputRow < 0 ||
        inputRow >= m_inputRows || inputPlane < 0 ||
        inputPlane >= m_inputPlanes) {
      return Scalar(0);
    }

    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputPlane * m_planeInputStride +
                             inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;

    return m_impl.coeff(inputIndex);
  }

  // Load packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index patchId, Index planeIndex,
                                        Index rowIndex, Index colIndex,
                                        Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;

    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId <
                 patchDepth() * patchPlanes() * patchRows() * patchCols());

    if (nonStandardPatches()) {
      return packetWithPossibleZero(patchId, planeIndex, rowIndex, colIndex,
                                    otherIndex);
    }
    typedef decltype(m_impl) TensorEvaluatorT;
    return loadPacketStandard<Packet, TensorEvaluatorT>(
        patchId, planeIndex, rowIndex, colIndex, otherIndex);
  }

  // Helper function to load a 'partial' packet - this is the single row part of
  // a packet that is split across two rows (but single column). In the
  // 'partial' packet, the elements corresponding to the row (specified through
  // rowOffset) are loaded and the rest of the elements are zero-filled into the
  // 'partial' packet. This function is called from
  // loadPacketStandardFromSingleColumnTwoRows(). This code path is exercised
  // only when the packet type supports masked load and when the partial packet
  // load is available in the TensorEvaluator.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPartialPacketStandard(
      Index planeIndex, Index rowIndex, Index colIndex, Index otherIndex,
      Index patchId, const Index span[], const Index patchOffsets[],
      Index colOffset, Index rowOffset) const {
    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    const Index planeOffsets[2] = {
        patchOffsets[0] - colOffset * m_colStride - rowOffset * m_rowStride,
        patchOffsets[1] - colOffset * m_colStride - rowOffset * m_rowStride};
    const Index inputPlanes[2] = {planeIndex + planeOffsets[0],
                                  planeIndex + planeOffsets[1]};

    if (inputRow >= m_inputRows || inputRow < 0 || inputCol >= m_inputCols ||
        inputCol < 0 || inputPlanes[0] >= m_inputPlanes || inputPlanes[1] < 0) {
      // Partial packet is all zeros
      return internal::pset1<Packet>(Scalar(0));
    } else if (inputPlanes[0] >= 0 && inputPlanes[1] < m_inputPlanes) {
      // From inputIndex-span[0], we need to load elements starting from index
      // span[0] all the way upto (and including) span[1].
      const Index depth = patchId - patchOffsets[0] * patchDepth();
      const Index inputIndex = depth + inputPlanes[0] * m_planeInputStride +
                               inputRow * m_rowInputStride +
                               inputCol * m_colInputStride + otherIndex;
      return m_impl.template partialPacket<Packet>(
          inputIndex - span[0], mask<Packet>(span[0], span[1] + 1));
    } else {
      // Using slow path for this partial packet.
      // We need to load elements starting from index span[0] all the way upto
      // (and including) span[1]. We split this load into 3 parts:
      // 0 : span[0]-1 - Zeros will be loaded for these indices
      // span[0] : span[1] - Elements will be loaded here for these indices
      // span[1]+1 : packetSize-1 - Zeross will be loaded for these indices
      const Index packetSize = internal::unpacket_traits<Packet>::size;
      EIGEN_ALIGN_MAX
      typename internal::remove_const<Scalar>::type values[packetSize];
      for (int i = 0; i < span[0]; ++i) values[i] = Scalar(0);
      for (int i = span[0]; i < span[1] + 1; ++i)
        values[i] = loadCoeff(patchId - span[0] + i, planeIndex, rowIndex,
                              colIndex, otherIndex);
      for (int i = span[1] + 1; i < packetSize; ++i) values[i] = Scalar(0);
      return internal::pload<Packet>(values);
    }
  }

  // Helper function to load a packet that is split across two rows (but single
  // column). If required, this function is called from loadPacketStandard()
  // when the packet type supports masked load and when the partial packet load
  // is available in the TensorEvaluator.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketStandardFromSingleColumnTwoRows(
      Index patchId, Index planeIndex, Index rowIndex, Index colIndex,
      Index otherIndex, const Index patchOffsets[], const Index colOffsets[],
      const Index rowOffsets[]) const {
    eigen_assert(colOffsets[1] == colOffsets[0] &&
                 rowOffsets[1] == rowOffsets[0] + 1);
    const Index packetSize = internal::unpacket_traits<Packet>::size;

    // Packet to load will be split into 2 parts where each part spans a single
    // row and both the parts span the same column.
    // First determine where to split.
    const Index patchIdSplit =
        (((rowOffsets[1] * m_rowStride) + (colOffsets[0] * m_colStride)) *
         m_patch_depth) -
        1;
    const Index patchOffsetSplit = patchIdSplit / m_fastDimZero;

    // patchIds[i]:          patchId corresponding to partial packet i
    // spans[i]:             Start and end indices corresponding to the elements
    //                       to be loaded for partial packet i
    // patchOffsets2Cols[i]: patchOffsets corresponding to partial packet i
    const Index patchIds[2] = {patchId, patchIdSplit + 1};
    const Index spans[2][2] = {{0, patchIdSplit - patchId},
                               {patchIdSplit - patchId + 1, packetSize - 1}};
    const Index patchOffsets2Cols[2][2] = {
        {patchOffsets[0], patchOffsetSplit},
        {patchOffsetSplit + 1, patchOffsets[1]}};

    // Load partial packets and do bit-wise OR to generate required packet
    return internal::por<Packet>(
        loadPartialPacketStandard(planeIndex, rowIndex, colIndex, otherIndex,
                                  patchIds[0], spans[0], patchOffsets2Cols[0],
                                  colOffsets[0], rowOffsets[0]),
        loadPartialPacketStandard(planeIndex, rowIndex, colIndex, otherIndex,
                                  patchIds[1], spans[1], patchOffsets2Cols[1],
                                  colOffsets[1], rowOffsets[1]));
  }

  // Helper function to load a packet that is present in a single column and
  // row. If required, this function is called from loadPacketStandard().
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketStandardFromSingleColumnSingleRow(
      Index patchId, Index planeIndex, Index rowIndex, Index colIndex,
      Index otherIndex, const Index patchOffsets[], const Index colOffsets[],
      const Index rowOffsets[], const Index inputCols[],
      const Index inputRows[]) const {
    eigen_assert(colOffsets[1] == colOffsets[0] &&
                 rowOffsets[1] == rowOffsets[0]);
    const Index planeOffsets[2] = {
        patchOffsets[0] - colOffsets[0] * m_colStride -
            rowOffsets[0] * m_rowStride,
        patchOffsets[1] - colOffsets[1] * m_colStride -
            rowOffsets[1] * m_rowStride};
    eigen_assert(planeOffsets[0] <= planeOffsets[1]);
    const Index inputPlanes[2] = {planeIndex + planeOffsets[0],
                                  planeIndex + planeOffsets[1]};

    if (inputPlanes[0] >= m_inputPlanes || inputPlanes[1] < 0) {
      return internal::pset1<Packet>(Scalar(0));
    }
    if (inputPlanes[0] >= 0 && inputPlanes[1] < m_inputPlanes) {
      const Index depth = patchId - patchOffsets[0] * patchDepth();
      const Index inputIndex = depth + inputPlanes[0] * m_planeInputStride +
                               inputRows[0] * m_rowInputStride +
                               inputCols[0] * m_colInputStride + otherIndex;
      return m_impl.template packet<Unaligned>(inputIndex);
    }
    return packetWithPossibleZero(patchId, planeIndex, rowIndex, colIndex,
                                  otherIndex);
  }

  // Load standard packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  // This function will be called if partial packet loading is not available
  // for the TensorEvaluator or if the packet type does not support masked
  // load.
  template <typename PacketT, typename TensorEvaluatorT>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<
      !TensorEvaluatorHasPartialPacket<TensorEvaluatorT, PacketT, Index>::value,
      PacketT>::type
  loadPacketStandard(Index patchId, Index planeIndex, Index rowIndex,
                     Index colIndex, Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId <
                 patchDepth() * patchPlanes() * patchRows() * patchCols());
    eigen_assert(!nonStandardPatches());

    if ((patchDepth() % packetSize) == 0) {
      return loadPacketFast(patchId, planeIndex, rowIndex, colIndex,
                            otherIndex);
    } else {
      // Offsets and input calculation here are identical to
      // loadCoeffStandard(...), but repeated twice.

      const Index patchOffsets[2] = {
          patchId / m_fastDimZero, (patchId + packetSize - 1) / m_fastDimZero};

      const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                   patchOffsets[1] / m_fastColStride};
      eigen_assert(colOffsets[0] <= colOffsets[1]);

      const Index inputCols[2] = {colIndex + colOffsets[0],
                                  colIndex + colOffsets[1]};
      if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
        return internal::pset1<Packet>(Scalar(0));
      }

      if (inputCols[0] == inputCols[1]) {
        const Index rowOffsets[2] = {
            (patchOffsets[0] - colOffsets[0] * m_colStride) / m_fastRowStride,
            (patchOffsets[1] - colOffsets[1] * m_colStride) / m_fastRowStride};
        eigen_assert(rowOffsets[0] <= rowOffsets[1]);
        const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                    rowIndex + rowOffsets[1]};

        if (inputRows[0] >= m_inputRows || inputRows[1] < 0) {
          return internal::pset1<Packet>(Scalar(0));
        }

        if (inputRows[0] == inputRows[1]) {
          return loadPacketStandardFromSingleColumnSingleRow(
              patchId, planeIndex, rowIndex, colIndex, otherIndex, patchOffsets,
              colOffsets, rowOffsets, inputCols, inputRows);
        }
      }
    }

    return packetWithPossibleZero(patchId, planeIndex, rowIndex, colIndex,
                                  otherIndex);
  }

  // Load standard packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  // This function will be called if partial packet loading is available for
  // the TensorEvaluator and if the packet type supports masked load.
  // The only difference between this and the other case is that if the packet
  // to load is split across two rows (but in same column), then in this case
  // instead of going to the slow (element-by-element) load, we load two packets
  // - each containing elements from one of the rows (rest of the elements of
  // the packets are zeroes), and then combine these two packets to generate the
  // required packet. The idea is to enable fast load (if possible) of these
  // 'partial' packets.
  template <typename PacketT, typename TensorEvaluatorT>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<
      TensorEvaluatorHasPartialPacket<TensorEvaluatorT, PacketT, Index>::value,
      PacketT>::type
  loadPacketStandard(Index patchId, Index planeIndex, Index rowIndex,
                     Index colIndex, Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId <
                 patchDepth() * patchPlanes() * patchRows() * patchCols());
    eigen_assert(!nonStandardPatches());

    if ((patchDepth() % packetSize) == 0) {
      return loadPacketFast(patchId, planeIndex, rowIndex, colIndex,
                            otherIndex);
    } else {
      // Offsets and input calculation here are identical to
      // loadCoeffStandard(...), but repeated twice.

      const Index patchOffsets[2] = {
          patchId / m_fastDimZero, (patchId + packetSize - 1) / m_fastDimZero};

      const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                   patchOffsets[1] / m_fastColStride};
      eigen_assert(colOffsets[0] <= colOffsets[1]);

      const Index inputCols[2] = {colIndex + colOffsets[0],
                                  colIndex + colOffsets[1]};
      if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
        return internal::pset1<Packet>(Scalar(0));
      }

      if (inputCols[0] == inputCols[1]) {
        const Index rowOffsets[2] = {
            (patchOffsets[0] - colOffsets[0] * m_colStride) / m_fastRowStride,
            (patchOffsets[1] - colOffsets[1] * m_colStride) / m_fastRowStride};
        eigen_assert(rowOffsets[0] <= rowOffsets[1]);
        const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                    rowIndex + rowOffsets[1]};

        if (inputRows[0] >= m_inputRows || inputRows[1] < 0) {
          return internal::pset1<Packet>(Scalar(0));
        }

        if (inputRows[0] == inputRows[1]) {
          return loadPacketStandardFromSingleColumnSingleRow(
              patchId, planeIndex, rowIndex, colIndex, otherIndex, patchOffsets,
              colOffsets, rowOffsets, inputCols, inputRows);
        }
        if (inputRows[0] + 1 == inputRows[1]) {
          return loadPacketStandardFromSingleColumnTwoRows(
              patchId, planeIndex, rowIndex, colIndex, otherIndex, patchOffsets,
              colOffsets, rowOffsets);
        }
      }
    }

    return packetWithPossibleZero(patchId, planeIndex, rowIndex, colIndex,
                                  otherIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index patchId, Index planeIndex,
                                            Index rowIndex, Index colIndex,
                                            Index otherIndex) const {
    const Index packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(patchId <
                 patchDepth() * patchPlanes() * patchRows() * patchCols());

    eigen_assert(!nonStandardPatches());
    eigen_assert((patchDepth() % packetSize) == 0);

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;
    eigen_assert((patchId + packetSize - 1) / m_fastDimZero == patchOffset);

    const Index colOffset = patchOffset / m_fastColStride;
    const Index rowOffset =
        (patchOffset - colOffset * m_colStride) / m_fastRowStride;
    const Index planeOffset =
        patchOffset - colOffset * m_colStride - rowOffset * m_rowStride;

    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    const Index inputPlane = planeIndex + planeOffset;

    if (inputCol < 0 || inputRow < 0 || inputPlane < 0 ||
        inputCol >= m_inputCols || inputRow >= m_inputRows ||
        inputPlane >= m_inputPlanes) {
      return internal::pset1<Packet>(Scalar(0));
    }

    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputPlane * m_planeInputStride +
                             inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.template packet<Unaligned>(inputIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet
  packetWithPossibleZero(Index patchId, Index planeIndex, Index rowIndex,
                         Index colIndex, Index otherIndex) const {
    const int packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_ALIGN_MAX
    typename internal::remove_const<Scalar>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] =
          loadCoeff(patchId + i, planeIndex, rowIndex, colIndex, otherIndex);
    }
    Packet rslt = internal::pload<Packet>(values);
    return rslt;
  }

  // Precompute the indices (plane, row, col, other) of the first element of
  // the given patch index, within the output tensor of the TensorVolumePatchOp.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void computeBaseIndices(
      Index patchIndex, Index& planeIndex, Index& rowIndex, Index& colIndex,
      Index& otherIndex) const {
    const size_t NumInputDims = array_size<
        typename TensorEvaluator<ArgType, Device>::Dimensions>::value;

    // Check if patchIndex might contain batch and other dimensions.
    otherIndex = (NumInputDims == 4) ? 0 : patchIndex / m_fastNumPatches;

    // Compute index of the patch within the batch (and other dimensions).
    const Index patch3DIndex = (NumInputDims == 4)
                                   ? patchIndex
                                   : (patchIndex - otherIndex * m_num_patches);

    otherIndex *= m_patchInputStride;

    colIndex = patch3DIndex / m_fastOutputPlanesRows;
    rowIndex =
        (patch3DIndex - colIndex * m_outputPlanesRows) / m_fastOutputPlanes;
    planeIndex =
        patch3DIndex - (colIndex * m_outputRows + rowIndex) * m_outputPlanes;

    colIndex = colIndex * m_col_strides - m_colPaddingLeft;
    rowIndex = rowIndex * m_row_strides - m_rowPaddingTop;
    planeIndex = planeIndex * m_plane_strides - m_planePaddingTop;
  }

  Index m_patch_depth;   // number of channels in the patch
  Index m_patch_planes;  // number of planes in the patch
  Index m_patch_rows;    // number of rows in the patch
  Index m_patch_cols;    // number of columns in the patch
  Index m_num_patches;   // number of patches to extract

  // Strides for navigating through the single patch.
  Index m_patch_plane_stride;
  Index m_patch_row_stride;
  Index m_patch_col_stride;

  // Strides for the output tensor (depth is not the part of the stride).
  Index m_rowStride;
  Index m_colStride;
  Index m_patchStride;
  Index m_otherStride;

  Index m_planeInputStride;  // Plane stride in the input tensor
  Index m_rowInputStride;    // Row stride in the input tensor
  Index m_colInputStride;    // Col stride in the input tensor
  Index m_patchInputStride;  // Patch stride in the input tensor
  Index m_otherInputStride;

  Index m_inputDepth;   // Depth of the input tensor
  Index m_inputPlanes;  // Number of planes in the input tensor
  Index m_inputRows;    // Number of rows in the input tensor
  Index m_inputCols;    // Number of cols in the input tensor

  Index m_outputPlanes;      // Number of output planes
  Index m_outputRows;        // Number of output rows
  Index m_outputCols;        // Number of output cols
  Index m_outputPlanesRows;  // Cached outputPlanes * outputRows.

  Index m_plane_strides;  // User specified plane stride
  Index m_row_strides;    // User specified row stride
  Index m_col_strides;    // User specified col stride

  // User specified plane/row/col atrous convolution strides.
  Index m_in_plane_strides;
  Index m_in_row_strides;
  Index m_in_col_strides;

  // User specified plane/row/col inflation strides in the image patch.
  Index m_patch_plane_inflate_strides;
  Index m_patch_row_inflate_strides;
  Index m_patch_col_inflate_strides;

  Index m_planePaddingTop;  // Plane padding
  Index m_rowPaddingTop;    // Row padding
  Index m_colPaddingLeft;   // Column padding

  // Fast representation of various divisors.
  internal::TensorIntDivisor<Index> m_fastNumPatches;

  internal::TensorIntDivisor<Index> m_fastPatchPlaneStride;
  internal::TensorIntDivisor<Index> m_fastPatchRowStride;
  internal::TensorIntDivisor<Index> m_fastPatchColStride;

  internal::TensorIntDivisor<Index> m_fastInputPlaneStride;
  internal::TensorIntDivisor<Index> m_fastInputRowStride;
  internal::TensorIntDivisor<Index> m_fastInputColStride;

  internal::TensorIntDivisor<Index> m_fastRowStride;
  internal::TensorIntDivisor<Index> m_fastColStride;

  internal::TensorIntDivisor<Index> m_fastDimZero;  // aka output depth
  internal::TensorIntDivisor<Index> m_fastOutputPlanes;
  internal::TensorIntDivisor<Index> m_fastOutputRows;
  internal::TensorIntDivisor<Index> m_fastOutputCols;
  internal::TensorIntDivisor<Index> m_fastOutputPlanesRows;

  const TensorEvaluator<ArgType, Device> m_impl;
};

template <typename NewDimension, Index Planes, Index Rows, Index Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, int Side, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper<
    Scalar, Index, Side,
    TensorEvaluator<const TensorReshapingOp<NewDimension,
                                            const TensorVolumePatchOp<
                                                Planes, Rows, Cols, ArgType> >,
                    Device>,
    nocontract_t, contract_t, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  typedef TensorContractionInputMapper<
      Scalar, Index, Side,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      ParentMapper;
  typedef TensorContractionSubMapper<
      Scalar, Index, Side,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
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
    m_base_mapper.computeBaseIndices(m_col_offset, m_planeIndex, m_rowIndex,
                                     m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const Self& base_mapper, Index vert_offset, Index horiz_offset)
      : m_base_mapper(base_mapper.m_base_mapper),
        m_depth_offset(vert_offset + base_mapper.m_depth_offset),
        m_col_offset(horiz_offset + base_mapper.m_col_offset) {
    m_base_mapper.computeBaseIndices(m_col_offset, m_planeIndex, m_rowIndex,
                                     m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_base_mapper.loadCoeff(i + m_depth_offset, m_planeIndex, m_rowIndex,
                                   m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i,
                                                          Index j) const {
    return m_base_mapper(i + m_depth_offset, j + m_col_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    return m_base_mapper.loadPacket(i + m_depth_offset, m_planeIndex,
                                    m_rowIndex, m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i,
                                                          Index j) const {
    return m_base_mapper.template loadPacket<Alignment>(i + m_depth_offset,
                                                        j + m_col_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar
  loadCoeffStandard(Index i) const {
    return m_base_mapper.loadCoeffStandard(
        i + m_depth_offset, m_planeIndex, m_rowIndex, m_colIndex, m_otherIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index i) const {
    return m_base_mapper.loadPacketFast(i + m_depth_offset, m_planeIndex,
                                        m_rowIndex, m_colIndex, m_otherIndex);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet
  loadPacketStandard(Index i) const {
    typedef decltype(m_base_mapper.m_impl) TensorEvaluatorT;
    return m_base_mapper.template loadPacketStandard<Packet, TensorEvaluatorT>(
        i + m_depth_offset, m_planeIndex, m_rowIndex, m_colIndex, m_otherIndex);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC bool aligned(Index) const {
    return false;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
    return m_base_mapper.nonStandardPatches();
  }

  // Max(Col|Row|Plane|Depth): compute the upper limit for the column, row,
  // plane and depth index respectively that fits into the peeled_k elements
  // starting at m_depth_offset.

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxCol(const Index peeled_k) const {
    const Index max_col =
        fastPatchColStride().divide(m_depth_offset + peeled_k);
    return std::min<Index>(1 + max_col, patchCols());
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxRow(const Index peeled_k,
                                   const Index col) const {
    const Index max_row = fastPatchRowStride().divide(
        m_depth_offset + peeled_k - col * patchColStride());
    return std::min<Index>(1 + max_row, patchRows());
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxPlane(const Index peeled_k, const Index col,
                                     const Index row) const {
    const Index max_plane = fastPatchPlaneStride().divide(
        m_depth_offset + peeled_k - col * patchColStride() -
        row * patchRowStride());
    return std::min<Index>(1 + max_plane, patchPlanes());
  }

  // MaxDepth uses only the remaining number of elements in the peeled_k.
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index maxDepth(const Index num_elements,
                                     const Index start_depth) const {
    return std::min<Index>(start_depth + num_elements, patchDepth());
  }

  // Every register matters in this code, so sometimes to prevent register
  // spilling, instead of the variable that you would expect to see, we use
  // another one, that is guaranteed to have the same value. E.g. patch depth is
  // always the same as input depth, and it's also the same as input plane
  // stride. Bunch of other parameters have similar relations.

  typedef internal::TensorIntDivisor<Index> IndexDivisor;

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchDepth() const {
    eigen_assert(m_base_mapper.m_patch_depth ==
                     m_base_mapper.m_planeInputStride &&
                 "Patch depth must be equal to plane input stride.");
    return m_base_mapper.m_planeInputStride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchPlanes() const {
    eigen_assert(m_base_mapper.m_patch_planes == m_base_mapper.m_rowStride &&
                 "Patch planes must be equal to row stride.");
    return m_base_mapper.m_rowStride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRows() const {
    return m_base_mapper.m_patch_rows;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchCols() const {
    return m_base_mapper.m_patch_cols;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchPlaneStride() const {
    eigen_assert(patchDepth() == m_base_mapper.m_patch_plane_stride &&
                 "Patch depth must be equal to patch plane stride.");
    return patchDepth();
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchRowStride() const {
    return m_base_mapper.m_patch_row_stride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index patchColStride() const {
    return m_base_mapper.m_patch_col_stride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE IndexDivisor fastPatchPlaneStride() const {
    eigen_assert(patchDepth() == m_base_mapper.m_patch_plane_stride &&
                 "Patch depth must be equal to patch plane stride.");
    return m_base_mapper.m_fastDimZero;  // patch_depth
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE IndexDivisor fastPatchRowStride() const {
    return m_base_mapper.m_fastPatchRowStride;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE IndexDivisor fastPatchColStride() const {
    return m_base_mapper.m_fastPatchColStride;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Packet packetNoPadding(const Index depth,
                                             const Index baseIndex) const {
    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.template packet<Unaligned>(inputIndex);
  }
  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar coeffNoPadding(const Index depth,
                                            const Index baseIndex) const {
    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.coeff(inputIndex);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE bool padPlane(const Index plane) const {
    const Index p = m_planeIndex + plane;
    return p < 0 || p >= m_base_mapper.m_inputPlanes;
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
  EIGEN_ALWAYS_INLINE Index baseIndex(const Index plane, const Index row,
                                      const Index col) const {
    const Index p = m_planeIndex + plane;
    const Index r = m_rowIndex + row;
    const Index c = m_colIndex + col;
    return p * m_base_mapper.m_planeInputStride +
           r * m_base_mapper.m_rowInputStride +
           c * m_base_mapper.m_colInputStride + m_otherIndex;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index planeOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    const Index rowOffset =
        (patchOffset - colOffset * m_base_mapper.m_colStride) /
        m_base_mapper.m_fastRowStride;
    const Index planeOffset = patchOffset -
                              colOffset * m_base_mapper.m_colStride -
                              rowOffset * m_base_mapper.m_rowStride;
    return planeOffset;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index rowOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    const Index rowOffset =
        (patchOffset - colOffset * m_base_mapper.m_colStride) /
        m_base_mapper.m_fastRowStride;
    return rowOffset;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index colOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return colOffset;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Index depthOffset() const {
    return m_depth_offset % patchDepth();
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LinearMapper
  getLinearMapper(Index i, Index j) const {
    return LinearMapper(m_base_mapper, i + m_depth_offset, j + m_col_offset);
  }

 private:
  const ParentMapper m_base_mapper;  // Keeping a copy instead of a reference
                                     // performs better in benchmarks.

  Index m_depth_offset;  // First row in the input matrix
  Index m_col_offset;    // First col in the input matrix

  // Knowing that: col_offset == patchIndex * OTHERS, we keep precomputed base
  // indices for the first element in a patch specified by col_offset
  // (see computeBaseIndices(...) for details).
  Index m_planeIndex;
  Index m_rowIndex;
  Index m_colIndex;
  Index m_otherIndex;
};

// Arrange a block of the right input matrix (in our case it's always a "virtual
// matrix" constructed from extracted volume patches) in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0  E0 F0 G0 H0 ... Z0
// A1 B1 C1 D1  E1 F1 G1 H1 ... Z1
// A2 B2 C2 D2  E2 F2 G2 H2 ... Z2
// A3 B3 C3 D3  E3 F3 G3 H3 ... Z3
// A4 B4 C4 D4  E4 F4 G4 H4 ... Z4
// A5 B5 C5 D5  E5 F5 G5 H5 ... Z5
// A6 B6 C6 D6  E6 F6 G6 H6 ... Z6
// A7 B7 C7 D7  E7 F7 G7 H7 ... Z7
// A8 ...
// ...
//
// *) A, B, C, ... - patches extracted from the original input.
// *) A0, A1, A2 ... - values from the same patch at different offsets.
//
// The traversal (packed rhs memory) order (B0 besides A0 in memory):
// A0 B0 C0 D0 A1 B1 C1 D1 ...
// E0 F0 G0 H0 E1 F1 G1 H1 ...
// ...
// Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 ... <- doesn't belong to any block (nr = 4)
//
// This traversal order must be the same as in default gemm_pack_rhs defined in
// GeneralBlockPanelKernel.h.
//
// *) nr - number of registers along the 'n' dimension.
//    See GeneralBlockPanelKernel.h and "Anatomy of High-Performance Matrix
//    Multiplication" paper.
//
// TODO(ezhulenev): Add support for squeezing reads along two innermost
// dimensions (see eigen_spatial_convolutions).
template <typename NewDimension, Index Planes, Index Rows, Index Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment,
          int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            NewDimension, const TensorVolumePatchOp<
                                              Planes, Rows, Cols, ArgType> >,
                        Device>,
        nocontract_t, contract_t, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;

  typedef SubMapper DataMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE);

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

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
        // FAST PATH:
        // Iterate over patch columns, rows and planes if we know that a single
        // packet do not span across multiple planes, rows or columns.
        if ((rhs.patchDepth() % packet_size) == 0) {
          const Index start_col = rhs.colOffset();
          const Index max_col = rhs.maxCol(peeled_k);

          for (Index c = start_col; c < max_col; ++c) {
            eigen_assert(k <= peeled_k);

            const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
            const Index max_row = rhs.maxRow(peeled_k, c);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);

            for (Index r = start_row; r < max_row; ++r) {
              eigen_assert(k <= peeled_k);

              const Index start_plane = ((c == start_col) && (r == start_row))
                                            ? rhs.planeOffset()
                                            : 0;
              const Index max_plane = rhs.maxPlane(peeled_k, c, r);

              const bool pad_row0 = pad_col0 || dm0.padRow(r);
              const bool pad_row1 = pad_col1 || dm1.padRow(r);
              const bool pad_row2 = pad_col2 || dm2.padRow(r);
              const bool pad_row3 = pad_col3 || dm3.padRow(r);

              for (Index p = start_plane; p < max_plane; ++p) {
                eigen_assert(k <= peeled_k);

                const bool pad0 = pad_row0 || dm0.padPlane(p);
                const bool pad1 = pad_row1 || dm1.padPlane(p);
                const bool pad2 = pad_row2 || dm2.padPlane(p);
                const bool pad3 = pad_row3 || dm3.padPlane(p);

                const Index idx0 = dm0.baseIndex(p, r, c);
                const Index idx1 = dm1.baseIndex(p, r, c);
                const Index idx2 = dm2.baseIndex(p, r, c);
                const Index idx3 = dm3.baseIndex(p, r, c);

                const Index start_depth =
                    ((c == start_col) && (r == start_row) && (p == start_plane))
                        ? rhs.depthOffset()
                        : 0;
                const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);
                eigen_assert((max_depth - start_depth) % packet_size == 0);

                for (Index d = start_depth; d < max_depth; d += packet_size) {
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
          }

          // The loop above should fill peeled_k elements.
          eigen_assert(peeled_k == k);

        } else {
          // Packet can span multiple planes, rows or columns, so we have to go
          // though the slower "standard" path.
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

      // Copy the remaining coefficients of the column block after the peeled_k.
      if (!non_standard_patches) {
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

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Template specialization for packet_size = 2. We must special-case packet
// blocks with nr > packet_size, e.g. PacketBlock<Packet2d, 4>.
//
// TODO(ezhulenev): Add support for squeezing reads along two innermost
// dimensions (see eigen_spatial_convolutions).
template <typename NewDimension, Index Planes, Index Rows, Index Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment, int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            NewDimension, const TensorVolumePatchOp<
                                              Planes, Rows, Cols, ArgType> >,
                        Device>,
        nocontract_t, contract_t, /*packet_size*/ 2, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, /*packet_size*/ 2, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;
  typedef SubMapper DataMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE);

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

    const int packet_size = 2;

    const Index packet_cols4 = (cols / 4) * 4;
    const Index peeled_k = (depth / packet_size) * packet_size;
    const bool non_standard_patches = rhs.nonStandardPatches();

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if (!non_standard_patches) {
        // FAST PATH:
        // Iterate over patch columns, rows and planes if we know that a single
        // packet do not span across multiple planes, rows or columns.
        if ((rhs.patchDepth() % packet_size) == 0) {
          const Index start_col = rhs.colOffset();
          const Index max_col = rhs.maxCol(peeled_k);

          for (Index c = start_col; c < max_col; ++c) {
            eigen_assert(k <= peeled_k);

            const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
            const Index max_row = rhs.maxRow(peeled_k, c);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);

            for (Index r = start_row; r < max_row; ++r) {
              eigen_assert(k <= peeled_k);

              const Index start_plane = ((c == start_col) && (r == start_row))
                                            ? rhs.planeOffset()
                                            : 0;
              const Index max_plane = rhs.maxPlane(peeled_k, c, r);

              const bool pad_row0 = dm0.padRow(r);
              const bool pad_row1 = dm1.padRow(r);
              const bool pad_row2 = dm2.padRow(r);
              const bool pad_row3 = dm3.padRow(r);

              for (Index p = start_plane; p < max_plane; ++p) {
                eigen_assert(k <= peeled_k);

                const bool pad0 = pad_col0 || pad_row0 || dm0.padPlane(p);
                const bool pad1 = pad_col1 || pad_row1 || dm1.padPlane(p);
                const bool pad2 = pad_col2 || pad_row2 || dm2.padPlane(p);
                const bool pad3 = pad_col3 || pad_row3 || dm3.padPlane(p);

                const Index idx0 = dm0.baseIndex(p, r, c);
                const Index idx1 = dm1.baseIndex(p, r, c);
                const Index idx2 = dm2.baseIndex(p, r, c);
                const Index idx3 = dm3.baseIndex(p, r, c);

                const Index start_depth =
                    ((c == start_col) && (r == start_row) && (p == start_plane))
                        ? rhs.depthOffset()
                        : 0;
                const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);
                eigen_assert((max_depth - start_depth) % packet_size == 0);

                for (Index d = start_depth; d < max_depth; d += packet_size) {
                  eigen_assert(k < peeled_k);
                  PacketBlock<Packet, 2> kernel0;
                  PacketBlock<Packet, 2> kernel1;
                  kernel0.packet[0] = pad0 ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, idx0);
                  kernel0.packet[1] = pad1 ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, idx1);
                  kernel1.packet[0] = pad2 ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, idx2);
                  kernel1.packet[1] = pad3 ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, idx3);
                  ptranspose(kernel0);
                  ptranspose(kernel1);
                  pstoreu(block + 0 * packet_size, kernel0.packet[0]);
                  pstoreu(block + 1 * packet_size, kernel1.packet[0]);
                  pstoreu(block + 2 * packet_size, kernel0.packet[1]);
                  pstoreu(block + 3 * packet_size, kernel1.packet[1]);
                  block += 4 * packet_size;
                  k += packet_size;
                }
              }
            }
          }

          // The loop above should fill peeled_k elements.
          eigen_assert(peeled_k == k);

        } else {
          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 2> kernel0;
            PacketBlock<Packet, 2> kernel1;
            kernel0.packet[0] = dm0.loadPacketStandard(k);
            kernel0.packet[1] = dm1.loadPacketStandard(k);
            kernel1.packet[0] = dm2.loadPacketStandard(k);
            kernel1.packet[1] = dm3.loadPacketStandard(k);
            ptranspose(kernel0);
            ptranspose(kernel1);
            pstoreu(block + 0 * packet_size, kernel0.packet[0]);
            pstoreu(block + 1 * packet_size, kernel1.packet[0]);
            pstoreu(block + 2 * packet_size, kernel0.packet[1]);
            pstoreu(block + 3 * packet_size, kernel1.packet[1]);
            block += 4 * packet_size;
          }
        }
      }

      // Copy the remaining coefficients of the column block after the peeled_k.
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

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Special case for non-vectorized types such as float16 (packet_size = 1).
template <typename NewDimension, Index Planes, Index Rows, Index Cols,
          typename ArgType, typename Device, typename Scalar, typename Index,
          typename nocontract_t, typename contract_t, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment, int nr>
struct gemm_pack_rhs<
    Scalar, Index,
    TensorContractionSubMapper<
        Scalar, Index, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            NewDimension, const TensorVolumePatchOp<
                                              Planes, Rows, Cols, ArgType> >,
                        Device>,
        nocontract_t, contract_t, /*packet_size*/ 1, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  typedef TensorContractionSubMapper<
      Scalar, Index, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered,
      Alignment>
      SubMapper;
  typedef SubMapper DataMapper;

  EIGEN_STATIC_ASSERT((nr == 4), YOU_MADE_A_PROGRAMMING_MISTAKE);

  EIGEN_DEVICE_FUNC
  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0) const {
    eigen_assert(stride == 0);
    eigen_assert(offset == 0);

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

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
// Pack a block of the right input matrix (in our case it's always a "virtual
// matrix" constructed from extracted image patches) in contiguous block in
// column-major storage order. Knowing the properties of the original patch op
// we can do it more efficient than the default gemm_pack_colmajor_block.
//
// TODO(ezhulenev): gemm_pack_colmajor_block for spatial convolutions supports
// squeezing reads along the 2 innermost dimensions, add it here if needed.
template <typename NewDimension, Index Planes, Index Rows, Index Cols,
          typename ArgType, typename Device, typename Scalar,
          typename StorageIndex, typename nocontract_t, typename contract_t,
          int packet_size, bool inner_dim_contiguous, bool inner_dim_reordered,
          int Alignment>
struct gemm_pack_colmajor_block<
    Scalar, StorageIndex,
    TensorContractionSubMapper<
        Scalar, StorageIndex, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            NewDimension, const TensorVolumePatchOp<
                                              Planes, Rows, Cols, ArgType> >,
                        Device>,
        nocontract_t, contract_t, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    ColMajor> {
  typedef TensorContractionSubMapper<
      Scalar, StorageIndex, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          NewDimension, const TensorVolumePatchOp<
                                            Planes, Rows, Cols, ArgType> >,
                      Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;

  typedef SubMapper DataMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper& rhs, StorageIndex rows,
                  StorageIndex cols) {
    const bool standard_patches = !rhs.nonStandardPatches();

    if (standard_patches && rhs.patchDepth() % packet_size == 0) {
      packStandardPatches<true>(block, rhs, rows, cols);

    } else if (standard_patches) {
      packStandardPatches<false>(block, rhs, rows, cols);

    } else {
      // With non-standard patches we don't do any vectorized loads.
      // TODO(ezhulenev): It doesn't look like that we should completely give up
      // on packets. Make this code path faster!
      for (StorageIndex col = 0; col < cols; ++col) {
        SubMapper lm = rhs.getLinearMapper(0, col);
        for (StorageIndex i = 0; i < rows; ++i) {
          *block = lm(i);
          ++block;
        }
      }
    }
  }

 private:
  // Pack standard volume patches:
  //
  // - patch_depth_is_multiple_of_packet_size=true: We are guaranteed to have
  //   depth dimension size to be a multiple of packet size, so we can skip all
  //   non vectorized loads and checks.
  //
  template <bool patch_depth_is_multiple_of_packet_size>
  EIGEN_ALWAYS_INLINE void packStandardPatches(Scalar* block,
                                               const DataMapper& rhs,
                                               StorageIndex rows,
                                               StorageIndex cols) {
    eigen_assert(!rhs.nonStandardPatches());

    // Give vectorized_rows the name used in all other gemm_pack_rhs above.
    const Index peeled_k = (rows / packet_size) * packet_size;

    const Index start_col = rhs.colOffset();
    const Index max_col = rhs.maxCol(peeled_k);

    for (StorageIndex col = 0; col < cols; ++col) {
      SubMapper lm = rhs.getLinearMapper(0, col);

      Index k = 0;
      for (Index c = start_col; c < max_col; ++c) {
        eigen_assert(k <= peeled_k);

        const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
        const Index max_row = rhs.maxRow(peeled_k, c);
        const bool pad_col = lm.padCol(c);

        for (Index r = start_row; r < max_row; ++r) {
          eigen_assert(k <= peeled_k);

          const Index start_plane =
              ((c == start_col) && (r == start_row)) ? rhs.planeOffset() : 0;
          const Index max_plane = rhs.maxPlane(peeled_k, c, r);
          const bool pad_row = pad_col || lm.padRow(r);

          for (Index p = start_plane; p < max_plane; ++p) {
            eigen_assert(k <= peeled_k);

            const Index start_depth =
                ((c == start_col) && (r == start_row) && (p == start_plane))
                    ? rhs.depthOffset()
                    : 0;
            const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);

            const bool pad = pad_col || pad_row || lm.padPlane(p);
            const Index base_idx = lm.baseIndex(p, r, c);

            if (patch_depth_is_multiple_of_packet_size)
              eigen_assert((max_depth - start_depth) % packet_size == 0);

            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            const Index max_vectorized_depth =
                patch_depth_is_multiple_of_packet_size
                    ? max_depth
                    : max_depth - packet_size;

            Index d = start_depth;

            // 1. Process depth dimension with vectorized instructions.
            for (; d < max_vectorized_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet packet = pad ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, packet);
              block += packet_size;
              k += packet_size;
            }

            // 2. Finish with coefficients.
            if (!patch_depth_is_multiple_of_packet_size) {
              for (; d < max_depth; d++) {
                eigen_assert(k < peeled_k);
                *block = pad ? Scalar(0) : rhs.coeffNoPadding(d, base_idx);
                ++block;
                ++k;
              }
            }
          }
        }
      }

      // The loop above should fill peeled_k elements.
      eigen_assert(peeled_k == k);

      // Fill remaining elements using loadCoeffStandard.
      for (; k < rows; ++k) {
        *block = lm.loadCoeffStandard(k);
        ++block;
      }
    }
  }
};
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

}  // namespace internal

/** CuboidConvolution
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies a 3D convolution over a multichannel input voxel block.
 *
 * The input parameter is expected to be a tensor with a rank of 4 or more
 * (channels, depth, height, width, and optionally others).
 * The kernel parameter is expected to be a 5D tensor (filters, channels,
 * kernel_depth, kernel_height, kernel_width).
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be filters, depth, height, width
 * (and others if applicable).
 *
 * The input and kernel have to be in the same layout, and both row-major and
 * col-major are supported. The shapes given above are for col-major layout.
 * For row-major, all dimensions should be reversed.
 *
 * It is possible to swap the order of the depth, width, and height dimensions
 * provided that the same order is used in the input, the kernel, and the
 * output.
 */
template <typename Input, typename Kernel>
EIGEN_ALWAYS_INLINE static const typename internal::conditional<
    internal::traits<Input>::Layout == ColMajor,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const Kernel>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const Input> > > >,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const Input> >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const Kernel> > > >::type
CuboidConvolution(const Input& input, const Kernel& kernel,
                  const Index stridePlanes = 1, const Index strideRows = 1,
                  const Index strideCols = 1,
                  const PaddingType padding_type = PADDING_SAME) {
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
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int NumDims = internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result.
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[4];
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[3];

  // Spatial size of the kernel.
  const TensorIndex kernelPlanes =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[4] : kern.dimensions()[0];

  if (isColMajor) {
    eigen_assert(kernelChannels == in.dimension(0));
  } else {
    eigen_assert(kernelChannels == in.dimension(NumDims - 1));
  }

  const TensorIndex inputPlanes =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputRows =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const TensorIndex inputCols =
      isColMajor ? in.dimension(3) : in.dimension(NumDims - 4);

  TensorIndex out_planes;
  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case PADDING_VALID:
      out_planes = Eigen::divup(inputPlanes - kernelPlanes + 1,
                                static_cast<TensorIndex>(stridePlanes));
      out_height = Eigen::divup(inputRows - kernelRows + 1,
                                static_cast<TensorIndex>(strideRows));
      out_width = Eigen::divup(inputCols - kernelCols + 1,
                               static_cast<TensorIndex>(strideCols));
      break;
    case PADDING_SAME:
      out_planes =
          Eigen::divup(inputPlanes, static_cast<TensorIndex>(stridePlanes));
      out_height =
          Eigen::divup(inputRows, static_cast<TensorIndex>(strideRows));
      out_width = Eigen::divup(inputCols, static_cast<TensorIndex>(strideCols));
      break;
    default:
      out_planes = 0;
      out_height = 0;
      out_width = 0;
      eigen_assert(false && "unexpected padding");
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelPlanes * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelPlanes * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }

  // Molds the output of the patch extraction result into a 2D tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] =
        kernelChannels * kernelPlanes * kernelRows * kernelCols;
    pre_contract_dims[1] = out_planes * out_height * out_width;
    for (int i = 4; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] =
        kernelChannels * kernelPlanes * kernelRows * kernelCols;
    pre_contract_dims[0] = out_planes * out_height * out_width;
    for (int i = 0; i < NumDims - 4; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  // Molds the output of the contraction into the shape expected by the user
  // (assuming ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output depth
  // - 3nd dim: output height
  // - 4rd dim: output width
  // - 5th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_planes;
    post_contract_dims[2] = out_height;
    post_contract_dims[3] = out_width;
    for (int i = 4; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_planes;
    post_contract_dims[NumDims - 3] = out_height;
    post_contract_dims[NumDims - 4] = out_width;
    for (int i = 0; i < NumDims - 4; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  return choose(
      Cond<internal::traits<Input>::Layout == ColMajor>(),
      kernel.reshape(kernel_dims)
          .contract(input
                        .extract_volume_patches(
                            kernelPlanes, kernelRows, kernelCols, stridePlanes,
                            strideRows, strideCols, padding_type)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(post_contract_dims),
      input
          .extract_volume_patches(kernelPlanes, kernelRows, kernelCols,
                                  stridePlanes, strideRows, strideCols,
                                  padding_type)
          .reshape(pre_contract_dims)
          .contract(kernel.reshape(kernel_dims), contract_dims)
          .reshape(post_contract_dims));
}

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_CUBOID_CONVOLUTION_H_
