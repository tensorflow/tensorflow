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

#include "tensorflow/core/util/tensor_slice_set.h"

#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

TensorSliceSet::TensorSliceSet(const TensorShape& shape, DataType type)
    : shape_(shape), type_(type) {}

TensorSliceSet::~TensorSliceSet() {}

Status TensorSliceSet::Register(const TensorSlice& slice, const string& tag) {
  TensorShape result_shape;
  TF_RETURN_IF_ERROR(slice.SliceTensorShape(shape_, &result_shape));
  string str = slice.DebugString();

  if (slices_.empty()) {
    slices_hull_ = slice;
  } else {
    // We check if there is any intersection between this slice and any of the
    // registered slices.
    if (slices_hull_.Overlaps(slice)) {
      for (const auto& x : slices_) {
        if (slice.Overlaps(x.second.slice)) {
          return errors::Internal("Overlapping slices: existing slice = ",
                                  x.first, ", new slice = ", str);
        }
      }
    }
    // No overlap: we can now insert the slice
    slices_hull_.UpdateToCover(slice);
  }

  TensorSliceSet::SliceInfo info = {slice, tag, result_shape.num_elements()};
  slices_.insert(std::make_pair(str, info));
  return OkStatus();
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
    // We didn't find any exact match but there is still a possibility that
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
    int64_t total_size = target_shape.num_elements();

    int64_t overlap_size = 0;
    TensorSlice intersection;
    TensorShape inter_shape;
    for (const auto& x : slices_) {
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

Status RegisterTensorSlice(
    const string& name, const TensorShape& shape, DataType type,
    const string& tag, const TensorSlice& slice,
    std::unordered_map<string, TensorSliceSet*>* tensor_slices) {
  DCHECK_NE(tensor_slices, nullptr);
  TensorSliceSet* tss = gtl::FindPtrOrNull(*tensor_slices, name);
  // Create a tensor slice set if needed
  if (!tss) {
    tss = new TensorSliceSet(shape, type);
    tensor_slices->insert(std::make_pair(name, tss));
  } else {
    // Check if the shapes match
    const TensorShape& tss_shape(tss->shape());
    if (!shape.IsSameSize(tss_shape)) {
      return errors::Internal("Incompatible tensor shapes detected for tensor ",
                              name, ": existing = ", tss_shape.DebugString(),
                              ", new = ", shape.DebugString());
    }
    if (type != tss->type()) {
      return errors::Internal("Incompatible tensor types detected for tensor ",
                              name,
                              ": existing = ", DataTypeString(tss->type()),
                              ", new = ", DataTypeString(type));
    }
  }
  // Register the tensor slices without the actual data.
  return tss->Register(slice, tag);
}

}  // namespace checkpoint

}  // namespace tensorflow
