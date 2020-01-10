#include "tensorflow/core/util/tensor_slice_set.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_slice_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

namespace checkpoint {

TensorSliceSet::TensorSliceSet(const TensorShape& shape, DataType type)
    : shape_(shape), type_(type) {}

TensorSliceSet::~TensorSliceSet() {}

Status TensorSliceSet::Register(const TensorSlice& slice,
                                     const string& tag, const float* data) {
  TensorShape result_shape;
  TF_RETURN_IF_ERROR(slice.SliceTensorShape(shape_, &result_shape));
  string str = slice.DebugString();
  // We check if there is any intersection between this slice and any of the
  // registered slices.
  for (const auto x : slices_) {
    if (slice.Overlaps(x.second.slice)) {
      return errors::Internal("Overlapping slices: existing slice = ", x.first,
                              ", new slice = ", str);
    }
  }
  // No overlap: we can now insert the slice
  TensorSliceSet::SliceInfo info = {slice, tag, data,
                                    result_shape.num_elements()};
  slices_.insert(std::make_pair(str, info));
  return Status::OK();
}

// TODO(yangke): merge Query() with QueryMeta()
bool TensorSliceSet::Query(const TensorSlice& slice, float* data) const {
  Status s;
  string str = slice.DebugString();
  // First we check if there is an exactly match (this is the dominant case).
  const TensorSliceSet::SliceInfo* info = gtl::FindOrNull(slices_, str);
  if (info) {
    if (data) {
      std::copy_n(info->data, info->num_floats, data);
    }
    return true;
  } else {
    // We didn't find any exact match but there is still a posibility that
    // mutliple existing slices can be patched together to output the slice.
    // We figure this out by computing the intersection of each of the existing
    // slices with the query slice, and check if the union of all these
    // intersections cover the entire slice. We rely on the fact that the
    // existing slices don't have any intersection among themselves.
    TensorShape target_shape;
    Status s;
    s = slice.SliceTensorShape(shape_, &target_shape);
    if (!s.ok()) {
      LOG(WARNING) << s;
      return false;
    }
    int64 total_size = target_shape.num_elements();

    int64 overlap_size = 0;
    TensorSlice intersection;
    TensorShape inter_shape;
    for (const auto x : slices_) {
      if (slice.Intersect(x.second.slice, &intersection)) {
        s = intersection.SliceTensorShape(shape_, &inter_shape);
        if (!s.ok()) {
          LOG(WARNING) << s;
          return false;
        }
        overlap_size += inter_shape.num_elements();
      }
    }
    if (total_size == overlap_size) {
      // We have it!
      // Now we need to copy the data to "data"
      if (data) {
        for (const auto x : slices_) {
          CopyDataFromTensorSliceToTensorSlice(shape_, x.second.slice, slice,
                                               x.second.data, data);
        }
      }
      return true;
    } else {
      // We don't have all the data for the asked tensor slice
      return false;
    }
  }
}

bool TensorSliceSet::QueryMeta(
    const TensorSlice& slice,
    std::vector<std::pair<TensorSlice, string>>* results) const {
  results->clear();
  Status s;
  string str = slice.DebugString();
  // First we check if there is an exactly match (this is the dominant case).
  const TensorSliceSet::SliceInfo* info = gtl::FindOrNull(slices_, str);
  if (info) {
    results->emplace_back(std::make_pair(info->slice, info->tag));
    return true;
  } else {
    // We didn't find any exact match but there is still a posibility that
    // multiple existing slices can be patched together to output the slice.
    // We figure this out by computing the intersection of each of the existing
    // slices with the query slice, and check if the union of all these
    // intersections cover the entire slice. We rely on the fact that the
    // existing slices don't have any intersection among themselves.
    TensorShape target_shape;
    Status s;
    s = slice.SliceTensorShape(shape_, &target_shape);
    if (!s.ok()) {
      LOG(WARNING) << s;
      return false;
    }
    int64 total_size = target_shape.num_elements();

    int64 overlap_size = 0;
    TensorSlice intersection;
    TensorShape inter_shape;
    for (const auto x : slices_) {
      if (slice.Intersect(x.second.slice, &intersection)) {
        s = intersection.SliceTensorShape(shape_, &inter_shape);
        if (!s.ok()) {
          LOG(WARNING) << s;
          return false;
        }
        overlap_size += inter_shape.num_elements();
        results->emplace_back(std::make_pair(x.second.slice, x.second.tag));
      }
    }
    if (total_size == overlap_size) {
      // We have it!
      return true;
    } else {
      // We don't have all the data for the asked tensor slice
      results->clear();
      return false;
    }
  }
}

}  // namespace checkpoint

}  // namespace tensorflow
