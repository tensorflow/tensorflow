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

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_SLICE_UTIL_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_SLICE_UTIL_H_

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

// Some hackery to invoke eigen tensor to copy over tensor slices with variable
// dimension tensors.
// TODO(yangke): get rid of that once the variable dimension tensor support is
// in.
static const int kTensorSliceMaxRank = 8;

// Create a tensor map with the given shape: we support up to 8 dimensions. If
// the shape has less than 8 dimensions, we pad the remaining dimension with 1.
template <typename T>
Eigen::TensorMap<Eigen::Tensor<T, kTensorSliceMaxRank, Eigen::RowMajor>>
GetEigenTensorMapFromTensorShape(const TensorShape& shape, T* data) {
  Eigen::DSizes<Eigen::DenseIndex, kTensorSliceMaxRank> dsizes =
      shape.AsEigenDSizesWithPadding<kTensorSliceMaxRank>();
  Eigen::TensorMap<Eigen::Tensor<T, kTensorSliceMaxRank, Eigen::RowMajor>> eig(
      data, dsizes);
  return eig;
}

// For everything except string, a standard Eigen cast and assignment works
template <typename DstT>
struct CopyThatWorksWithStringPointer {
  template <typename SrcTensor, typename DstTensor, typename Shape>
  static void Copy(const SrcTensor& s, Shape s_start, Shape len, DstTensor& d,
                   Shape d_start) {
    d.slice(d_start, len) = s.slice(s_start, len).template cast<DstT>();
  }
};

// Eigen makes it extremely difficult to dereference a tensor of string* into
// string, so we roll our own loop instead.
template <>
struct CopyThatWorksWithStringPointer<tstring> {
  template <typename SrcTensor, typename DstTensor, typename Shape>
  static void Copy(const SrcTensor& s, Shape s_start, Shape len, DstTensor& d,
                   Shape d_start) {
    typedef typename SrcTensor::Index Index;
    static_assert(kTensorSliceMaxRank == 8,
                  "If kTensorSliceMaxRank changes, modify the loop below.");
    for (Index i0 = 0; i0 < len[0]; i0++) {
      for (Index i1 = 0; i1 < len[1]; i1++) {
        for (Index i2 = 0; i2 < len[2]; i2++) {
          for (Index i3 = 0; i3 < len[3]; i3++) {
            for (Index i4 = 0; i4 < len[4]; i4++) {
              for (Index i5 = 0; i5 < len[5]; i5++) {
                for (Index i6 = 0; i6 < len[6]; i6++) {
                  for (Index i7 = 0; i7 < len[7]; i7++) {
                    d(d_start[0] + i0, d_start[1] + i1, d_start[2] + i2,
                      d_start[3] + i3, d_start[4] + i4, d_start[5] + i5,
                      d_start[6] + i6, d_start[7] + i7) =
                        *s(s_start[0] + i0, s_start[1] + i1, s_start[2] + i2,
                           s_start[3] + i3, s_start[4] + i4, s_start[5] + i5,
                           s_start[6] + i6, s_start[7] + i7);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

// Checkpointing of half is done by storing the raw 16 bits as a signed 32bit
// integer. To restore the checkpoint we need to do the reverse operation by
// reinterpreting the integer as a 16 bit float. This prevents us from using
// the default cast operation.
template <>
struct CopyThatWorksWithStringPointer<Eigen::half> {
  template <typename SrcTensor, typename DstTensor, typename Shape>
  static void Copy(const SrcTensor& s, Shape s_start, Shape len, DstTensor& d,
                   Shape d_start) {
    typedef typename SrcTensor::Index Index;
    static_assert(kTensorSliceMaxRank == 8,
                  "If kTensorSliceMaxRank changes, modify the loop below.");
    for (Index i0 = 0; i0 < len[0]; i0++) {
      for (Index i1 = 0; i1 < len[1]; i1++) {
        for (Index i2 = 0; i2 < len[2]; i2++) {
          for (Index i3 = 0; i3 < len[3]; i3++) {
            for (Index i4 = 0; i4 < len[4]; i4++) {
              for (Index i5 = 0; i5 < len[5]; i5++) {
                for (Index i6 = 0; i6 < len[6]; i6++) {
                  for (Index i7 = 0; i7 < len[7]; i7++) {
                    d(d_start[0] + i0, d_start[1] + i1, d_start[2] + i2,
                      d_start[3] + i3, d_start[4] + i4, d_start[5] + i5,
                      d_start[6] + i6, d_start[7] + i7) =
                        Eigen::numext::bit_cast<Eigen::half, uint16_t>(
                            s(s_start[0] + i0, s_start[1] + i1, s_start[2] + i2,
                              s_start[3] + i3, s_start[4] + i4, s_start[5] + i5,
                              s_start[6] + i6, s_start[7] + i7));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

// Given a tensor described by "shape", two slices "slice_s" and "slice_d",
// and two pointers "ptr_s" and "ptr_d", where "ptr_s" points to a chunk of
// memory that stores the data for "slice_s" and "ptr_d" points to a chunk of
// memory that stores the data for "slice_d". This function copies the data
// that belongs to the intersection of the two slices from slice_s to
// slice_d.  Uses Tensor cast<DstT>() to convert from SrcT to DstT. Returns true
// iff the two slices share any intersection (and thus some data is copied).
// TODO(yangke): figure out if we can make it private.
template <typename SrcT, typename DstT>
static bool CopyDataFromTensorSliceToTensorSlice(const TensorShape& shape,
                                                 const TensorSlice& slice_s,
                                                 const TensorSlice& slice_d,
                                                 const SrcT* ptr_s,
                                                 DstT* ptr_d) {
  CHECK_LE(shape.dims(), kTensorSliceMaxRank)
      << "Only tensors of size up to " << kTensorSliceMaxRank
      << " are supported";
  // We need to compute the intersection of the two slices.
  TensorSlice inter;
  if (!slice_s.Intersect(slice_d, &inter)) {
    // There is no intersection: returns false.
    return false;
  } else {
    // We need to compute the applied shapes after applying slice_s and
    // slice_d.
    TensorShape shp_s, shp_d;
    Status s;
    s = slice_s.SliceTensorShape(shape, &shp_s);
    if (!s.ok()) {
      LOG(WARNING) << s;
      return false;
    }
    s = slice_d.SliceTensorShape(shape, &shp_d);
    if (!s.ok()) {
      LOG(WARNING) << s;
      return false;
    }

    // We need to compute the relative slice of "inter" w.r.t. both slice_s and
    // slice_d.
    TensorSlice rel_s, rel_d;
    slice_s.ComputeRelative(inter, &rel_s);
    slice_d.ComputeRelative(inter, &rel_d);

    // Get the eigen tensor maps to the data.
    auto t_s = GetEigenTensorMapFromTensorShape(shp_s, ptr_s);
    auto t_d = GetEigenTensorMapFromTensorShape(shp_d, ptr_d);

    Eigen::DSizes<Eigen::DenseIndex, kTensorSliceMaxRank> s_start, s_len,
        d_start, d_len;

    rel_s.FillIndicesAndSizes<kTensorSliceMaxRank>(shp_s, &s_start, &s_len);
    rel_d.FillIndicesAndSizes<kTensorSliceMaxRank>(shp_d, &d_start, &d_len);
    CopyThatWorksWithStringPointer<DstT>::Copy(t_s, s_start, s_len, t_d,
                                               d_start);
    return true;
  }
}

}  // namespace

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_SLICE_UTIL_H_
