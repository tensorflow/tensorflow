#ifndef TENSORFLOW_UTIL_TENSOR_SLICE_UTIL_H_
#define TENSORFLOW_UTIL_TENSOR_SLICE_UTIL_H_

#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

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
  CHECK_LE(shape.dims(), kTensorSliceMaxRank) << "Only tensors of size up to "
                                              << kTensorSliceMaxRank
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
    t_d.slice(d_start, d_len) = t_s.slice(s_start, s_len).template cast<DstT>();
    return true;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_SLICE_UTIL_H_
