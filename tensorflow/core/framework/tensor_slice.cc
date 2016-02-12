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

#include "tensorflow/core/framework/tensor_slice.h"
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

TensorSlice::TensorSlice(int dim) { SetFullSlice(dim); }

TensorSlice::TensorSlice(const TensorSliceProto& proto) {
  starts_.reserve(proto.extent_size());
  lengths_.reserve(proto.extent_size());
  for (const auto& e : proto.extent()) {
    starts_.push_back(e.start());
    lengths_.push_back(GetExtentLength(e));
  }
}

TensorSlice::TensorSlice(std::initializer_list<std::pair<int, int>> extents) {
  starts_.reserve(extents.size());
  lengths_.reserve(extents.size());
  for (const auto& e : extents) {
    starts_.push_back(e.first);
    lengths_.push_back(e.second);
  }
}

Status TensorSlice::Parse(const string& str, TensorSlice* slice) {
  std::vector<string> items = str_util::Split(str, ':', str_util::SkipEmpty());
  slice->starts_.reserve(items.size());
  slice->lengths_.reserve(items.size());
  for (const string& x : items) {
    int s, l;
    if (x == "-") {
      // "everything"
      s = 0;
      l = kFullExtent;
    } else {
      char junk;
      if (sscanf(x.c_str(), "%d,%d%c", &s, &l, &junk) != 2) {
        return errors::InvalidArgument(
            "Expected a pair of numbers or '-' "
            "but got '",
            x, "': string = ", str);
      }
      if (s < 0 || l <= 0) {
        return errors::InvalidArgument(
            "Expected non-negative start and "
            "positive length but got start = ",
            s, ", length = ", l, ": string = ", str);
      }
    }
    slice->starts_.push_back(s);
    slice->lengths_.push_back(l);
  }

  return Status::OK();
}

void TensorSlice::Clear() {
  starts_.clear();
  lengths_.clear();
}

void TensorSlice::SetFullSlice(int dim) {
  Clear();
  starts_.reserve(dim);
  lengths_.reserve(dim);
  for (int d = 0; d < dim; ++d) {
    starts_.push_back(0);
    lengths_.push_back(kFullExtent);
  }
}

void TensorSlice::Extend(int dim) {
  int old_dim = dims();
  DCHECK_LE(old_dim, dim);
  starts_.resize(dim);
  lengths_.resize(dim);
  for (int d = old_dim; d < dim; ++d) {
    starts_[d] = 0;
    lengths_[d] = kFullExtent;
  }
}

void TensorSlice::AsProto(TensorSliceProto* proto) const {
  for (int d = 0; d < dims(); ++d) {
    TensorSliceProto::Extent* e = proto->add_extent();
    // We only need to record the explicit slice for non-full slices
    if (!IsFullAt(d)) {
      e->set_start(starts_[d]);
      e->set_length(lengths_[d]);
    }
  }
}

string TensorSlice::DebugString() const {
  string buffer;
  bool first = true;
  for (int d = 0; d < dims(); ++d) {
    if (!first) {
      buffer.append(":");
    }
    string s;
    if (IsFullAt(d)) {
      buffer.append("-");
    } else {
      strings::StrAppend(&buffer, starts_[d], ",", lengths_[d]);
    }
    first = false;
  }
  return buffer;
}

bool TensorSlice::Intersect(const TensorSlice& other,
                            TensorSlice* result) const {
  // First, if two slices have different ranks, they obviously don't overlap
  // -- in fact they are not compatible.
  if (dims() != other.dims()) {
    return false;
  }

  // Setting the result to the right dimension
  if (result) {
    result->SetFullSlice(dims());
  }
  // The two slices overlap if they overlap in all dimensions.
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      if (result) {
        result->set_start(d, other.start(d));
        result->set_length(d, other.length(d));
      }
    } else if (other.IsFullAt(d)) {
      if (result) {
        result->set_start(d, start(d));
        result->set_length(d, length(d));
      }
    } else {
      // If we have an intersection here, it should have a start that is the
      // max of the two starts and an end that is the min of the two ends.
      int s = std::max(start(d), other.start(d));
      int l = std::min(end(d), other.end(d)) - s;
      if (l > 0) {
        // We have a real intersection
        if (result) {
          result->set_start(d, s);
          result->set_length(d, l);
        }
      } else {
        // We don't have an intersection for this dimension -- thus we don't
        // have any intersection at all.
        if (result) {
          result->Clear();
        }
        return false;
      }
    }
  }
  // If we are here, we know there is overlap in every dimension.
  return true;
}

void TensorSlice::ComputeRelative(const TensorSlice& sub,
                                  TensorSlice* relative) const {
  DCHECK_EQ(dims(), sub.dims());
  relative->SetFullSlice(dims());
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      relative->set_start(d, sub.start(d));
      relative->set_length(d, sub.length(d));
    } else {
      // Otherwise the relative start is the difference between the start of
      // sub and the start of base
      relative->set_start(d, sub.start(d) - start(d));
      relative->set_length(d, sub.length(d));
    }
  }
}

void TensorSlice::UpdateToCover(const TensorSlice& other) {
  DCHECK_EQ(dims(), other.dims());
  for (int d = 0; d < dims(); ++d) {
    if (!IsFullAt(d)) {
      if (other.IsFullAt(d)) {
        starts_[d] = 0;
        lengths_[d] = kFullExtent;
      } else {
        const auto new_end = std::max(end(d), other.end(d));
        set_start(d, std::min(start(d), other.start(d)));
        set_length(d, new_end - start(d));
      }
    }
  }
}

// static
bool TensorSlice::HasExtentLength(const TensorSliceProto::Extent& extent) {
  return extent.has_length_case() == TensorSliceProto::Extent::kLength;
}

// static
int64 TensorSlice::GetExtentLength(const TensorSliceProto::Extent& extent) {
  if (!HasExtentLength(extent)) return -1;
  return extent.length();
}

Status TensorSlice::SliceTensorShape(const TensorShape& shape,
                                     TensorShape* result_shape) const {
  result_shape->Clear();
  // Mismatching ranks: we can't apply the slice at all.
  if (shape.dims() != dims()) {
    return errors::Internal("Mismatching ranks: shape = ", shape.DebugString(),
                            ", slice = ", DebugString());
  }
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      result_shape->AddDim(shape.dim_size(d));
    } else {
      // Check if the extent applies to the dimension
      if (end(d) <= shape.dim_size(d)) {
        // Yes: the end is within the range of the dim -- we adjust the result
        // shape so that its size along this dimension is the length of the
        // slice.
        result_shape->AddDim(length(d));
      } else {
        // The extent doesn't apply to the dimension
        result_shape->Clear();
        return errors::Internal("Extent in dimension ", d,
                                " out of bounds: shape = ", shape.DebugString(),
                                ", slice = ", DebugString());
      }
    }
  }
  // If we are here, we have successfully applied the shape.
  return Status::OK();
}

const int TensorSlice::kFullExtent = -1;

}  // namespace tensorflow
