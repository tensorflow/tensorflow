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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Note the following header is used in both TF and TFLite. Particularly, it's
// used for float TFLite Conv2D.
#include "tensorflow/core/kernels/eigen_spatial_convolutions-inl.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"

namespace Eigen {
namespace internal {

// After we vectorized all loads from the underlying tensor using Packet ops, we
// have to finalize coefficients that do not fit into a packet.
template <typename Scalar, typename DataMapper, int packet_size,
          bool masked_load_store>
struct FinalizeDataMapperCoeffs {
  EIGEN_ALWAYS_INLINE static Index finalize(Scalar* block,
                                            const DataMapper& rhs,
                                            Index base_idx, Index depth,
                                            Index max_depth, bool pad = false) {
    const Index num_coeffs = max_depth - depth;
    eigen_assert(num_coeffs <= packet_size);

    for (; depth < max_depth; ++depth) {
      *block = pad ? Scalar(0) : rhs.coeffNoPadding(depth, base_idx);
      ++block;
    }

    return num_coeffs;
  }
};

template <typename Scalar, typename DataMapper, int packet_size>
struct FinalizeDataMapperCoeffs<Scalar, DataMapper, packet_size,
                                /*masked_load_store=*/true> {
  EIGEN_ALWAYS_INLINE static Index finalize(Scalar* block,
                                            const DataMapper& rhs,
                                            Index base_idx, Index depth,
                                            Index max_depth, bool pad = false) {
    Index num_coeffs = max_depth - depth;
    eigen_assert(num_coeffs <= packet_size);
    if (num_coeffs == 0) return 0;

    using Packet = typename packet_traits<Scalar>::type;
    Packet p = pad ? pset1<Packet>(Scalar(0))
                   : rhs.partialPacketNoPadding(depth, base_idx, num_coeffs);
    internal::pstoreu(block, p, mask<Packet>(0, num_coeffs));

    return num_coeffs;
  }
};

// Pack a block of the right input matrix (in our case it's always a
// "virtual matrix" constructed from extracted image patches) in contiguous
// block in column-major storage order. Knowing the properties of the
// original patch op we can do it more efficient than the default
// gemm_pack_colmajor_block.
template <typename NewDimension, Index Rows, Index Cols, typename ArgType,
          typename Device, typename Scalar, typename StorageIndex,
          typename nocontract_t, typename contract_t, int packet_size,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
struct gemm_pack_colmajor_block<
    Scalar, StorageIndex,
    TensorContractionSubMapper<
        Scalar, StorageIndex, Rhs,
        TensorEvaluator<
            const TensorReshapingOp<
                NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
            Device>,
        nocontract_t, contract_t, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    ColMajor> {
  typedef TensorContractionSubMapper<
      Scalar, StorageIndex, Rhs,
      TensorEvaluator<
          const TensorReshapingOp<
              NewDimension, const TensorImagePatchOp<Rows, Cols, ArgType> >,
          Device>,
      nocontract_t, contract_t, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>
      SubMapper;

  typedef SubMapper DataMapper;
  typedef typename packet_traits<Scalar>::type Packet;

  using CoeffFinalizer = FinalizeDataMapperCoeffs<
      Scalar, DataMapper, packet_size,
      TensorEvaluatorHasPartialPacket<typename DataMapper::TensorEvaluatorT,
                                      Packet, Index>::value &&
          unpacket_traits<Packet>::masked_store_available>;

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper rhs, StorageIndex rows,
                  StorageIndex cols) {
    const bool standard_patches = !rhs.nonStandardPatches();

    if (standard_patches && (rhs.patchDepth() % packet_size == 0)) {
      // Single packet always belong to single patch (row, col).
      if (rhs.hasPadding()) {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/true,
                            /*has_padding=*/true>(block, rhs, rows, cols);
      } else {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/true,
                            /*has_padding=*/false>(block, rhs, rows, cols);
      }

    } else if (standard_patches) {
      // Single packet can span across multiple patch rows or columns.
      if (rhs.hasPadding()) {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/false,
                            /*has_padding=*/true>(block, rhs, rows, cols);
      } else {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/false,
                            /*has_padding=*/false>(block, rhs, rows, cols);
      }

    } else if (rhs.patchDepth() % packet_size == 0) {
      // Single packet always belong to single patch (row, col).
      packNonStandardPatches</*patch_depth_is_multiple_of_packet_size*/
                             true>(block, rhs, rows, cols);

    } else {
      // Single packet can span across multiple patch rows or columns.
      packNonStandardPatches</*patch_depth_is_multiple_of_packet_size*/
                             false>(block, rhs, rows, cols);
    }
  }

 private:
  // (A) Standard image patches:
  //
  //  (1) patch_row_inflate_strides == 1    AND
  //  (2) patch_col_inflate_strides == 1
  //
  // Standard patches guarantee that two inner most dimensions (depth and rows)
  // are contiguous in memory and we can try to squeeze reads from them.
  //
  // (B) Non standard image patches: in_row/in_col and patch_row/patch_col
  // strides can be not equal to 1, and for each [row, col] inside a patch we
  // have to do additional computations to find corresponding row and col in the
  // input tensor. Also we can no longer squeeze reads from inner dimensions.
  //
  // Additional parameters:
  // - patch_depth_is_multiple_of_packet_size=true: We are guaranteed to have
  //   depth dimension size to be a multiple of packet size, so we can skip all
  //   non vectorized loads and checks, because it's guaranteed that block size
  //   will be a multiple of a packet size (see TensorContractionBlocking).
  //
  // - has_padding: Input tensor has non-zero padding. In this case for each
  //   patch col and row we need to check that it doesn't correspond to the
  //   padded region of original input.
  template <bool patch_depth_is_multiple_of_packet_size, bool has_padding>
  EIGEN_ALWAYS_INLINE void packStandardPatches(Scalar* block,
                                               const DataMapper rhs,
                                               StorageIndex rows,
                                               StorageIndex cols) {
    eigen_assert(!rhs.nonStandardPatches());

    // Give vectorized_rows the name used in all other gemm_pack_rhs above.
    const StorageIndex peeled_k = (rows / packet_size) * packet_size;

    const StorageIndex start_col = rhs.colOffset();
    const StorageIndex max_col = rhs.maxCol(peeled_k);
    const StorageIndex rhs_depth_offset = rhs.depthOffset();

    for (StorageIndex col = 0; col < cols; ++col) {
      SubMapper lm = rhs.getLinearMapper(0, col);

      StorageIndex k = 0;
      for (Index c = start_col; c < max_col; ++c) {
        eigen_assert(k <= peeled_k);

        const StorageIndex start_row = (c == start_col) ? rhs.rowOffset() : 0;
        const StorageIndex max_row = rhs.maxRow(peeled_k, c);
        const bool pad_col = has_padding && lm.padCol(c);

        eigen_assert(has_padding || !lm.padCol(c));
        eigen_assert(has_padding || !lm.padAnyRow(start_row, max_row - 1));

        // We can squeeze reads for all rows in [start_row, max_row) range.
        if (!has_padding ||
            (!pad_col && !lm.padAnyRow(start_row, max_row - 1))) {
          const StorageIndex start_depth =
              (c == start_col) ? rhs_depth_offset : 0;

          const StorageIndex max_depth =
              std::min<StorageIndex>(start_depth + (peeled_k - k),
                                     (max_row - start_row) * rhs.patchDepth());

          const StorageIndex base_idx = lm.baseIndex(start_row, c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            eigen_assert((max_depth - start_depth) % packet_size == 0);
            StorageIndex d = start_depth;

            const StorageIndex unrolled_depth = max_depth - 4 * packet_size;
            for (; d <= unrolled_depth; d += 4 * packet_size) {
              eigen_assert(k < peeled_k);

              Packet p0 = rhs.packetNoPadding(d + 0 * packet_size, base_idx);
              Packet p1 = rhs.packetNoPadding(d + 1 * packet_size, base_idx);
              Packet p2 = rhs.packetNoPadding(d + 2 * packet_size, base_idx);
              Packet p3 = rhs.packetNoPadding(d + 3 * packet_size, base_idx);

              internal::pstoreu(block + 0 * packet_size, p0);
              internal::pstoreu(block + 1 * packet_size, p1);
              internal::pstoreu(block + 2 * packet_size, p2);
              internal::pstoreu(block + 3 * packet_size, p3);

              block += 4 * packet_size;
              k += 4 * packet_size;
            }

            for (; d < max_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              internal::pstoreu(block, rhs.packetNoPadding(d, base_idx));
              block += packet_size;
              k += packet_size;
            }

          } else {
            StorageIndex d = start_depth;

            const StorageIndex unrolled_depth = max_depth - 4 * packet_size;
            for (; d <= unrolled_depth; d += 4 * packet_size) {
              eigen_assert(k < peeled_k);

              Packet p0 = rhs.packetNoPadding(d + 0 * packet_size, base_idx);
              Packet p1 = rhs.packetNoPadding(d + 1 * packet_size, base_idx);
              Packet p2 = rhs.packetNoPadding(d + 2 * packet_size, base_idx);
              Packet p3 = rhs.packetNoPadding(d + 3 * packet_size, base_idx);

              internal::pstoreu(block + 0 * packet_size, p0);
              internal::pstoreu(block + 1 * packet_size, p1);
              internal::pstoreu(block + 2 * packet_size, p2);
              internal::pstoreu(block + 3 * packet_size, p3);

              block += 4 * packet_size;
              k += 4 * packet_size;
            }

            const StorageIndex vectorized_depth = max_depth - packet_size;
            for (; d <= vectorized_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              internal::pstoreu(block, rhs.packetNoPadding(d, base_idx));
              block += packet_size;
              k += packet_size;
            }

            eigen_assert(k <= peeled_k);
            const Index num_coeffs =
                CoeffFinalizer::finalize(block, rhs, base_idx, d, max_depth);

            k += num_coeffs;
            block += num_coeffs;
            eigen_assert(k <= peeled_k);
          }

          // Go to the next column.
          continue;
        }

        // If we are not allowed to squeeze reads along the `row` and `depth`
        // dimensions, we must process rows one by one.
        for (StorageIndex r = start_row; r < max_row; ++r) {
          eigen_assert(k <= peeled_k);

          const StorageIndex start_depth =
              ((c == start_col) && (r == start_row)) ? rhs_depth_offset : 0;
          const StorageIndex max_depth =
              rhs.maxDepth(peeled_k - k, start_depth);

          const bool pad = has_padding && (pad_col || lm.padRow(r));
          eigen_assert(has_padding || !lm.padRow(r));

          const StorageIndex base_idx = lm.baseIndex(r, c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            eigen_assert((max_depth - start_depth) % packet_size == 0);
            StorageIndex d = start_depth;

            for (; d < max_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet p = (has_padding && pad)
                                   ? pset1<Packet>(Scalar(0))
                                   : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

          } else {
            StorageIndex d = start_depth;

            const StorageIndex vectorized_depth = max_depth - packet_size;
            for (; d <= vectorized_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet p = (has_padding && pad)
                                   ? pset1<Packet>(Scalar(0))
                                   : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

            eigen_assert(k <= peeled_k);
            const Index num_coeffs = CoeffFinalizer::finalize(
                block, rhs, base_idx, d, max_depth, has_padding && pad);

            k += num_coeffs;
            block += num_coeffs;
            eigen_assert(k <= peeled_k);
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

  template <bool patch_depth_is_multiple_of_packet_size>
  EIGEN_ALWAYS_INLINE void packNonStandardPatches(Scalar* block,
                                                  const DataMapper rhs,
                                                  StorageIndex rows,
                                                  StorageIndex cols) {
    eigen_assert(rhs.nonStandardPatches());

    // Give vectorized_rows the name used in all other gemm_pack_rhs above.
    const StorageIndex peeled_k = (rows / packet_size) * packet_size;

    const StorageIndex start_col = rhs.colOffset();
    const StorageIndex max_col = rhs.maxCol(peeled_k);
    const StorageIndex rhs_depth_offset = rhs.depthOffset();

    // Original input column and row after applying all non-standard strides and
    // dilations. Computed by padOrSkip{Row,Col}.
    Index orig_c = 0;
    Index orig_r = 0;

    for (StorageIndex col = 0; col < cols; ++col) {
      SubMapper lm = rhs.getLinearMapper(0, col);

      StorageIndex k = 0;
      for (Index c = start_col; c < max_col; ++c) {
        eigen_assert(k <= peeled_k);

        const StorageIndex start_row = (c == start_col) ? rhs.rowOffset() : 0;
        const StorageIndex max_row = rhs.maxRow(peeled_k, c);
        const bool pad_or_skip_col = lm.padOrSkipCol(c, &orig_c);

        for (StorageIndex r = start_row; r < max_row; ++r) {
          eigen_assert(k <= peeled_k);

          const StorageIndex start_depth =
              ((c == start_col) && (r == start_row)) ? rhs_depth_offset : 0;
          const StorageIndex max_depth =
              rhs.maxDepth(peeled_k - k, start_depth);

          const bool pad_or_skip =
              pad_or_skip_col || lm.padOrSkipRow(r, &orig_r);
          const StorageIndex base_idx = lm.origBaseIndex(orig_r, orig_c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            eigen_assert((max_depth - start_depth) % packet_size == 0);
            StorageIndex d = start_depth;

            for (; d < max_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet p = pad_or_skip ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

          } else {
            const StorageIndex vectorized_depth = max_depth - packet_size;
            StorageIndex d = start_depth;
            for (; d <= vectorized_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet p = pad_or_skip ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

            eigen_assert(k <= peeled_k);
            const Index num_coeffs = CoeffFinalizer::finalize(
                block, rhs, base_idx, d, max_depth, pad_or_skip);

            k += num_coeffs;
            block += num_coeffs;
            eigen_assert(k <= peeled_k);
          }
        }
      }

      // The loop above should fill peeled_k elements.
      eigen_assert(peeled_k == k);

      // Fill remaining elements using loadCoeff.
      for (; k < rows; ++k) {
        *block = lm(k);
        ++block;
      }
    }
  }
};
}  // namespace internal
}  // namespace Eigen
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_
