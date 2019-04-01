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

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"

namespace Eigen {
namespace internal {
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

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper rhs, StorageIndex rows,
                  StorageIndex cols) {
    const bool standard_patches = !rhs.nonStandardPatches();

    if (standard_patches && (rhs.patchDepth() % packet_size == 0)) {
      // Single packet always belong to single patch (row, col).
      packStandardPatches</*patch_depth_is_multiple_of_packet_size*/ true>(
          block, rhs, rows, cols);

    } else if (standard_patches) {
      // Single packet can span across multiple patch rows or columns.
      packStandardPatches</*patch_depth_is_multiple_of_packet_size*/ false>(
          block, rhs, rows, cols);

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
  // Pack standard image patches:
  //
  // - patch_depth_is_multiple_of_packet_size=true: We are guaranteed to have
  //   depth dimension size to be a multiple of packet size, so we can skip all
  //   non vectorized loads and checks.
  template <bool patch_depth_is_multiple_of_packet_size>
  EIGEN_ALWAYS_INLINE void packStandardPatches(Scalar* block,
                                               const DataMapper rhs,
                                               StorageIndex rows,
                                               StorageIndex cols) {
    eigen_assert(!rhs.nonStandardPatches());

    // Give vectorized_rows the name used in all other gemm_pack_rhs above.
    const StorageIndex peeled_k = (rows / packet_size) * packet_size;

    const StorageIndex start_col = rhs.colOffset();
    const StorageIndex max_col = rhs.maxCol(peeled_k);

    for (StorageIndex col = 0; col < cols; ++col) {
      SubMapper lm = rhs.getLinearMapper(0, col);

      StorageIndex k = 0;
      for (Index c = start_col; c < max_col; ++c) {
        eigen_assert(k <= peeled_k);

        const StorageIndex start_row = (c == start_col) ? rhs.rowOffset() : 0;
        const StorageIndex max_row = rhs.maxRow(peeled_k, c);
        const bool pad_col = lm.padCol(c);

        // We can squeeze reads for all rows in [start_row, max_row) range.
        if (!pad_col && !lm.padAnyRow(start_row, max_row - 1)) {
          const StorageIndex start_depth =
              (c == start_col) ? rhs.depthOffset() : 0;

          const StorageIndex max_depth =
              std::min<StorageIndex>(start_depth + (peeled_k - k),
                                     (max_row - start_row) * rhs.patchDepth());

          const StorageIndex base_idx = lm.baseIndex(start_row, c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            eigen_assert((max_depth - start_depth) % packet_size == 0);
            StorageIndex d = start_depth;

            for (; d < max_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              internal::pstoreu(block, rhs.packetNoPadding(d, base_idx));
              block += packet_size;
              k += packet_size;
            }

          } else {
            StorageIndex d = start_depth;
            const StorageIndex vectorized_depth = max_depth - packet_size;

            for (; d <= vectorized_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              internal::pstoreu(block, rhs.packetNoPadding(d, base_idx));
              block += packet_size;
              k += packet_size;
            }
            for (; d < max_depth; d++) {
              eigen_assert(k < peeled_k);
              *block = rhs.coeffNoPadding(d, base_idx);
              ++block;
              ++k;
            }
          }

          // Go to the next column.
          continue;
        }

        // If we are not allowed to squeeze reads along the `row` and `depth`
        // dimensions, we must process rows one by one.
        for (StorageIndex r = start_row; r < max_row; ++r) {
          eigen_assert(k <= peeled_k);

          const StorageIndex start_depth =
              ((c == start_col) && (r == start_row)) ? rhs.depthOffset() : 0;
          const StorageIndex max_depth =
              rhs.maxDepth(peeled_k - k, start_depth);

          const bool pad = pad_col || lm.padRow(r);
          const StorageIndex base_idx = lm.baseIndex(r, c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            eigen_assert((max_depth - start_depth) % packet_size == 0);
            StorageIndex d = start_depth;

            for (; d < max_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet p = pad ? pset1<Packet>(Scalar(0))
                                   : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

          } else {
            const StorageIndex max_vectorized_depth = max_depth - packet_size;
            StorageIndex d = start_depth;
            for (; d < max_vectorized_depth; d += packet_size) {
              eigen_assert(k < peeled_k);
              const Packet p = pad ? pset1<Packet>(Scalar(0))
                                   : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }
            for (; d < max_depth; d++) {
              eigen_assert(k < peeled_k);
              *block = pad ? Scalar(0) : rhs.coeffNoPadding(d, base_idx);
              ++block;
              ++k;
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
}  // end namespace internal
}  // end namespace Eigen
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

// Note the following header is used in both TF and TFLite. Particularly, it's
// used for float TFLite Conv2D.
#include "tensorflow/core/kernels/eigen_spatial_convolutions-inl.h"

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_SPATIAL_CONVOLUTIONS_H_
