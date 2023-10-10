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

#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {

// TensorShape and PartialTensorShape should have no fields beyond
// TensorShapeRep.  In particular, their sizes should be the same.
static_assert(sizeof(TensorShapeRep) == sizeof(TensorShape),
              "TensorShape must have no fields beyond TensorShapeRep");
static_assert(sizeof(TensorShapeRep) == sizeof(PartialTensorShape),
              "PartialTensorShape must have no fields beyond TensorShapeRep");

template <class Shape>
static void AppendTo(const TensorShapeBase<Shape>& s,
                     gtl::InlinedVector<int64, 8>* vals) {
  for (auto dim : s) {
    vals->push_back(dim.size);
  }
}

void TensorShape::CheckDimsEqual(int NDIMS) const {
  CHECK_EQ(NDIMS, dims()) << "Asking for tensor of " << NDIMS << " dimensions"
                          << " from a tensor of " << dims() << " dimensions";
}

void TensorShape::CheckDimsAtMost(int NDIMS) const {
  CHECK_GE(NDIMS, dims()) << "Asking for tensor of at most " << NDIMS
                          << " dimensions from a tensor of " << dims()
                          << " dimensions";
}

// TODO(slebedev): Consider merging IsValid implementations.
template <class Shape>
bool TensorShapeBase<Shape>::IsValid() {
  // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
  // unknown_shape() set, and it seems hard to remove this without backwards
  // compatibility issues.
  if (kIsPartial && unknown_rank()) return dims() == 0;
  int64_t num_elements = 1;
  if (dims() > MaxDimensions()) return false;
  for (auto d : dim_sizes()) {
    if (d < (kIsPartial ? -1 : 0)) return false;
    if (d == -1) {
      num_elements = -1;
    } else if (!kIsPartial || num_elements >= 0) {
      num_elements = MultiplyWithoutOverflow(num_elements, d);
      if (num_elements < 0) return false;
    }
  }
  return true;
}

template <class Shape>
bool TensorShapeBase<Shape>::IsValid(const TensorShapeProto& proto) {
  // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
  // unknown_shape() set, and it seems hard to remove this without backwards
  // compatibility issues.
  if (kIsPartial && proto.unknown_rank()) return proto.dim_size() == 0;
  int64_t num_elements = 1;
  if (proto.dim().size() > MaxDimensions()) return false;
  for (const auto& d : proto.dim()) {
    if (d.size() < (kIsPartial ? -1 : 0)) return false;
    if (d.size() == -1) {
      num_elements = -1;
    } else if (!kIsPartial || num_elements >= 0) {
      num_elements = MultiplyWithoutOverflow(num_elements, d.size());
      if (num_elements < 0) return false;
    }
  }
  return true;
}

template <class Shape>
Status TensorShapeBase<Shape>::IsValidShape(const TensorShapeProto& proto) {
  // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
  // unknown_shape() set, and it seems hard to remove this without backwards
  // compatibility issues.
  if (kIsPartial && proto.unknown_rank()) {
    if (proto.dim_size() > 0) {
      return errors::InvalidArgument(
          "An unknown shape must not have any dimensions set.");
    }
    return OkStatus();
  }
  int64_t num_elements = 1;
  if (proto.dim().size() > MaxDimensions()) {
    return errors::InvalidArgument("Shape ", DebugString(proto),
                                   " has too many dimensions");
  }
  for (const auto& d : proto.dim()) {
    if (d.size() < (kIsPartial ? -1 : 0)) {
      if (kIsPartial) {
        return errors::InvalidArgument(
            "Shape ", DebugString(proto),
            " has dimensions with values below -1 (where -1 means unknown)");
      } else {
        return errors::InvalidArgument("Shape ", DebugString(proto),
                                       " is not fully defined");
      }
    }
    if (d.size() == -1) {
      num_elements = -1;
    } else if (!kIsPartial || num_elements >= 0) {
      num_elements = MultiplyWithoutOverflow(num_elements, d.size());
      if (num_elements < 0) {
        return errors::InvalidArgument(
            "Shape ", DebugString(proto),
            " is too large (more than 2**63 - 1 entries)");
      }
    }
  }
  return OkStatus();
}

template <class Shape>
TensorShapeBase<Shape>::TensorShapeBase(const TensorShapeProto& proto) {
  set_tag(REP16);
  set_data_type(DT_INVALID);
  // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
  // unknown_shape() set, and it seems hard to remove this without backwards
  // compatibility issues.
  if (kIsPartial && proto.unknown_rank()) {
    set_ndims_byte(kUnknownRank);
    set_num_elements(-1);
  } else {
    set_ndims_byte(0);
    set_num_elements(1);
    for (const auto& d : proto.dim()) {
      AddDim(d.size());
    }
  }
}

template <class Shape>
Status TensorShapeBase<Shape>::BuildTensorShapeBase(
    const TensorShapeProto& proto, TensorShapeBase* out) {
  out->set_tag(REP16);
  out->set_data_type(DT_INVALID);
  // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
  // unknown_shape() set, and it seems hard to remove this without backwards
  // compatibility issues.
  if (kIsPartial && proto.unknown_rank()) {
    out->set_ndims_byte(kUnknownRank);
    out->set_num_elements(-1);
  } else {
    out->set_ndims_byte(0);
    out->set_num_elements(1);
    int64_t num_elements_excluding_zero_dims = 1;
    Status s = OkStatus();
    for (const auto& d : proto.dim()) {
      s = out->AddDimWithStatus(d.size());
      if (!s.ok()) {
        return s;
      }
      // If one of the dimensions has size 0, multiplying the dimensions in
      // ascending order isn't sufficient to prevent all multiplication orders
      // from overflowing. To do that, we need to check that there would be no
      // overflow if all zero-length dimensions were multiplied last, which is
      // equivalent to ensuring that there's no overflow if zero-length
      // dimensions are skipped. Unknown dimensions are also ignored.
      if (d.size() > 0) {
        num_elements_excluding_zero_dims =
            MultiplyWithoutOverflow(num_elements_excluding_zero_dims, d.size());
        if (TF_PREDICT_FALSE(num_elements_excluding_zero_dims < 0)) {
          return errors::InvalidArgument(
              "Encountered overflow when multiplying shape dimensions");
        }
      }
    }
  }
  return OkStatus();
}

template <class Shape>
TensorShapeBase<Shape>::TensorShapeBase(gtl::ArraySlice<int64_t> dim_sizes) {
  set_tag(REP16);
  set_data_type(DT_INVALID);
  TF_CHECK_OK(InitDims(dim_sizes));
}

template <class Shape>
Status TensorShapeBase<Shape>::BuildTensorShapeBase(
    gtl::ArraySlice<int64_t> dim_sizes, TensorShapeBase* out) {
  out->set_tag(REP16);
  out->set_data_type(DT_INVALID);
  return out->InitDims(dim_sizes);
}

// Returns true iff partial is true and val is < 0.
// REQUIRES: val < kMaxRep16
// REQUIRES: partial || val >= 0
static inline bool Set16(bool partial, uint16* dst, int dim, int64_t val) {
  if (partial) {
    if (val < 0) {
      dst[dim] = std::numeric_limits<uint16>::max();
      return true;
    }
  }
  dst[dim] = val;
  return false;
}

template <class Shape>
Status TensorShapeBase<Shape>::InitDims(gtl::ArraySlice<int64_t> dim_sizes) {
  DCHECK_EQ(tag(), REP16);

  // Allow sizes that are under kint64max^0.25 so that 4-way multiplication
  // below cannot overflow.
  static const int64_t kMaxSmall = 0xd744;
  static_assert(kMaxSmall * kMaxSmall * kMaxSmall * kMaxSmall <= kint64max,
                "bad overflow check");
  bool large_size = false;
  for (auto s : dim_sizes) {
    if (s > kMaxSmall) {
      large_size = true;
      break;
    }
  }

  if (!kIsPartial && !large_size) {
    for (auto s : dim_sizes) {
      if (TF_PREDICT_FALSE(s < 0)) {
        return errors::InvalidArgument(
            "Expected shape dimensions to be non-negative, got ", s);
      }
    }
  }

  if (!large_size) {
    // Every size fits in 16 bits; use fast-paths for dims in {1,2,3,4}.
    uint16* dst = as16()->dims_;
    switch (dim_sizes.size()) {
      case 1: {
        set_ndims_byte(1);
        const int64_t size = dim_sizes[0];
        const bool neg = Set16(kIsPartial, dst, 0, size);
        set_num_elements(neg ? -1 : size);
        return OkStatus();
      }
      case 2: {
        set_ndims_byte(2);
        const int64_t size0 = dim_sizes[0];
        const int64_t size1 = dim_sizes[1];
        bool neg = Set16(kIsPartial, dst, 0, size0);
        neg |= Set16(kIsPartial, dst, 1, size1);
        set_num_elements(neg ? -1 : (size0 * size1));
        return OkStatus();
      }
      case 3: {
        set_ndims_byte(3);
        const int64_t size0 = dim_sizes[0];
        const int64_t size1 = dim_sizes[1];
        const int64_t size2 = dim_sizes[2];
        bool neg = Set16(kIsPartial, dst, 0, size0);
        neg |= Set16(kIsPartial, dst, 1, size1);
        neg |= Set16(kIsPartial, dst, 2, size2);
        set_num_elements(neg ? -1 : (size0 * size1 * size2));
        return OkStatus();
      }
      case 4: {
        set_ndims_byte(4);
        const int64_t size0 = dim_sizes[0];
        const int64_t size1 = dim_sizes[1];
        const int64_t size2 = dim_sizes[2];
        const int64_t size3 = dim_sizes[3];
        bool neg = Set16(kIsPartial, dst, 0, size0);
        neg |= Set16(kIsPartial, dst, 1, size1);
        neg |= Set16(kIsPartial, dst, 2, size2);
        neg |= Set16(kIsPartial, dst, 3, size3);
        set_num_elements(neg ? -1 : (size0 * size1 * size2 * size3));
        return OkStatus();
      }
    }
  }

  set_ndims_byte(0);
  set_num_elements(1);
  Status status = OkStatus();
  for (int64_t s : dim_sizes) {
    status.Update(AddDimWithStatus(internal::SubtleMustCopy(s)));
    if (!status.ok()) {
      return status;
    }
  }

  return status;
}

template <class Shape>
TensorShapeBase<Shape>::TensorShapeBase() {
  set_tag(REP16);
  set_data_type(DT_INVALID);
  if (kIsPartial) {
    set_ndims_byte(kUnknownRank);
    set_num_elements(-1);
  } else {
    set_ndims_byte(0);
    set_num_elements(1);
  }
}

void TensorShapeRep::DestructorOutOfLine() {
  DCHECK(tag() == REP_OUT_OF_LINE);
  delete as64()->dims_;
}

void TensorShapeRep::SlowCopyFrom(const TensorShapeRep& b) {
  if (b.tag() != REP_OUT_OF_LINE) {
    if (tag() == REP_OUT_OF_LINE) {
      delete as64()->dims_;
    }
    memcpy(buf(), b.buf(), sizeof(u_.buf));
    // memcpy above implicitly also does:
    //   set_tag(b.tag());
    //   set_ndims_byte(b.ndims_byte());
    //   set_data_type(b.data_type());
  } else {
    set_ndims_byte(b.ndims_byte());
    set_data_type(b.data_type());
    if (tag() == REP_OUT_OF_LINE) {
      // vector already allocated
      *(as64()->dims_) = *(b.as64()->dims_);
    } else {
      set_tag(REP_OUT_OF_LINE);
      as64()->dims_ = new gtl::InlinedVector<int64_t, 4>(*(b.as64()->dims_));
    }
  }
}

template <class Shape>
int64_t TensorShapeBase<Shape>::dim_size(int d) const {
  if (unknown_rank()) return -1;
  CHECK_GE(d, 0);                  // Crash OK
  if (d > 0) CHECK_LT(d, dims());  // Crash OK
  if (tag() == REP16) {
    uint16 dim = as16()->dims_[d];
    if (kIsPartial && dim == kUnknownRep16) return -1;
    return dim;
  } else if (tag() == REP32) {
    uint32 dim = as32()->dims_[d];
    if (kIsPartial && dim == kUnknownRep32) return -1;
    return dim;
  } else {
    return (*as64()->dims_)[d];
  }
}

void TensorShapeRep::Clear() {
  ClearAllButDataType();
  set_data_type(DT_INVALID);
}

void TensorShapeRep::ClearAllButDataType() {
  if (tag() == REP_OUT_OF_LINE) {
    delete as64()->dims_;
  }
  set_tag(REP16);
  set_ndims_byte(0);
  // Leaves data_type alone
  set_num_elements(1);
}

template <class Shape>
Status TensorShapeBase<Shape>::RecomputeNumElements() {
  if (unknown_rank()) {
    set_num_elements(-1);
    return OkStatus();
  }
  int64_t n = 1;
  for (auto dim : *this) {
    if (kIsPartial && dim.size < 0) {
      n = -1;
      break;
    }
    n = MultiplyWithoutOverflow(n, dim.size);
    if (TF_PREDICT_FALSE(n < 0)) {
      return errors::InvalidArgument(
          "Shape ", this->DebugString(),
          " results in overflow when computing number of elements");
    }
  }
  set_num_elements(n);
  return OkStatus();
}

template <class Shape>
void TensorShapeBase<Shape>::AddDim(int64_t size) {
  if (!kIsPartial) CHECK_GE(size, 0);
  if (unknown_rank()) return;
  CHECK_LT(ndims_byte(), MaxDimensions()) << "Too many dimensions in tensor";
  int64_t new_num_elements;
  if (kIsPartial && (num_elements() < 0 || size < 0)) {
    new_num_elements = -1;
  } else {
    new_num_elements = MultiplyWithoutOverflow(num_elements(), size);
    CHECK_LE(0, new_num_elements);
  }
  UnsafeAddDim(size, new_num_elements);
}

template <class Shape>
Status TensorShapeBase<Shape>::AddDimWithStatus(int64_t size) {
  if (!kIsPartial) {
    if (TF_PREDICT_FALSE(size < 0)) {
      return errors::InvalidArgument("Expected a non-negative size, got ",
                                     size);
    }
  }

  if (unknown_rank()) {
    return OkStatus();
  }

  if (TF_PREDICT_FALSE(ndims_byte() >= MaxDimensions())) {
    return errors::InvalidArgument("Too many dimensions in tensor");
  }

  int64_t new_num_elements;
  if (kIsPartial && (num_elements() < 0 || size < 0)) {
    new_num_elements = -1;
  } else {
    new_num_elements = MultiplyWithoutOverflow(num_elements(), size);
    if (TF_PREDICT_FALSE(new_num_elements < 0)) {
      return errors::InvalidArgument("Encountered overflow when multiplying ",
                                     num_elements(), " with ", size,
                                     ", result: ", new_num_elements);
    }
  }

  UnsafeAddDim(size, new_num_elements);
  return OkStatus();
}

template <class Shape>
void TensorShapeBase<Shape>::UnsafeAddDim(int64_t size,
                                          int64_t new_num_elements) {
  const int nd = ndims_byte();
  if (tag() == REP16 && nd < 6 && size < kMaxRep16) {
    as16()->dims_[nd] =
        kIsPartial && size < 0 ? kUnknownRep16 : static_cast<uint16>(size);
  } else if (tag() == REP32 && nd < 3 && size < kMaxRep32) {
    as32()->dims_[nd] =
        kIsPartial && size < 0 ? kUnknownRep32 : static_cast<uint32>(size);
  } else if (tag() == REP_OUT_OF_LINE) {
    as64()->dims_->push_back(size);
  } else {
    // Need to change representation
    gtl::InlinedVector<int64_t, 8> vals;
    AppendTo(*this, &vals);
    vals.push_back(size);
    // We know we can't be REP16.  See if we have a small enough
    // number of dimensions and each dimension's size is small enough
    // to allow REP32.
    bool can_be_rep32 = (vals.size() <= 3);
    if (can_be_rep32) {
      for (size_t i = 0; i < vals.size(); i++) {
        if (vals[i] >= kMaxRep32) {
          can_be_rep32 = false;
          break;
        }
      }
    }
    if (can_be_rep32) {
      set_tag(REP32);
      for (size_t d = 0; d < vals.size(); d++) {
        as32()->dims_[d] = kIsPartial && vals[d] < 0
                               ? kUnknownRep32
                               : static_cast<uint32>(vals[d]);
      }
    } else {
      set_tag(REP_OUT_OF_LINE);
      as64()->dims_ =
          new gtl::InlinedVector<int64_t, 4>(vals.begin(), vals.end());
    }
  }
  set_ndims_byte(nd + 1);
  set_num_elements(new_num_elements);
}

template <class Shape>
void TensorShapeBase<Shape>::AppendShape(const TensorShapeBase& shape) {
  for (auto d : shape) AddDim(d.size);
}

template <class Shape>
Status TensorShapeBase<Shape>::AppendShapeWithStatus(
    const TensorShapeBase& shape) {
  Status s = OkStatus();
  for (auto d : shape) {
    s.Update(AddDimWithStatus(d.size));
    if (!s.ok()) {
      return s;
    }
  }
  return s;
}

template <class Shape>
void TensorShapeBase<Shape>::InsertDim(int d, int64_t size) {
  CHECK_GE(d, 0);
  CHECK_LE(d, dims());
  if (!kIsPartial) CHECK_GE(size, 0);
  CHECK_LT(dims(), MaxDimensions());
  gtl::InlinedVector<int64_t, 8> vals;
  AppendTo(*this, &vals);
  vals.insert(vals.begin() + d, size);
  ClearAllButDataType();
  for (auto dval : vals) {
    AddDim(dval);
  }
}

template <class Shape>
Status TensorShapeBase<Shape>::InsertDimWithStatus(int d, int64_t size) {
  if (!kIsPartial) {
    if (TF_PREDICT_FALSE(size < 0)) {
      return errors::InvalidArgument("Expected a non-negative size, got ",
                                     size);
    }
  }

  if (TF_PREDICT_FALSE(d < 0)) {
    return errors::Internal("The insertion index must be non-negative, got ",
                            d);
  }
  if (TF_PREDICT_FALSE(d > dims())) {
    return errors::Internal("The insertion index must be at most ", dims(),
                            " got ", d);
  }
  if (TF_PREDICT_FALSE(dims() >= MaxDimensions())) {
    return errors::Internal("Shape has ", dims(),
                            " dimensions which is the maximum allowed");
  }

  gtl::InlinedVector<int64_t, 8> vals;
  AppendTo(*this, &vals);
  vals.insert(vals.begin() + d, size);
  ClearAllButDataType();

  Status s = OkStatus();
  for (auto dval : vals) {
    s.Update(AddDimWithStatus(dval));
    if (!s.ok()) {
      return s;
    }
  }
  return s;
}

template <class Shape>
gtl::InlinedVector<int64_t, 4> TensorShapeBase<Shape>::dim_sizes() const {
  gtl::InlinedVector<int64_t, 4> result;
  for (auto dim : *this) {
    result.push_back(dim.size);
  }
  return result;
}

template <class Shape>
void TensorShapeBase<Shape>::set_dim(int d, int64_t size) {
  CHECK_GE(d, 0);
  CHECK_LT(d, dims());
  if (!kIsPartial) {
    CHECK_GE(size, 0);
  }
  if (tag() == REP16 && size < kMaxRep16) {
    as16()->dims_[d] =
        kIsPartial && size < 0 ? kUnknownRep16 : static_cast<uint16>(size);
  } else if (tag() == REP32 && size < kMaxRep32) {
    as32()->dims_[d] =
        kIsPartial && size < 0 ? kUnknownRep32 : static_cast<uint32>(size);
  } else if (tag() == REP_OUT_OF_LINE) {
    (*as64()->dims_)[d] = size;
  } else {
    // Must upgrade
    gtl::InlinedVector<int64_t, 8> vals;
    AppendTo(*this, &vals);
    vals[d] = size;
    ClearAllButDataType();
    for (auto dval : vals) {
      AddDim(dval);
    }
  }
  TF_CHECK_OK(RecomputeNumElements());
}

template <class Shape>
Status TensorShapeBase<Shape>::SetDimWithStatus(int d, int64_t size) {
  if (TF_PREDICT_FALSE(d < 0)) {
    return errors::InvalidArgument("Index must be non-negative, got ", d);
  }
  if (TF_PREDICT_FALSE(d >= dims())) {
    return errors::InvalidArgument("Index must be less than ", dims(), ", got ",
                                   d);
  }
  if (TF_PREDICT_FALSE(!kIsPartial && size < 0)) {
    return errors::InvalidArgument("Expected a non-negative size, got ", size);
  }

  if (tag() == REP16 && size < kMaxRep16) {
    as16()->dims_[d] =
        kIsPartial && size < 0 ? kUnknownRep16 : static_cast<uint16>(size);
  } else if (tag() == REP32 && size < kMaxRep32) {
    as32()->dims_[d] =
        kIsPartial && size < 0 ? kUnknownRep32 : static_cast<uint32>(size);
  } else if (tag() == REP_OUT_OF_LINE) {
    (*as64()->dims_)[d] = size;
  } else {
    // Must upgrade
    gtl::InlinedVector<int64_t, 8> vals;
    AppendTo(*this, &vals);
    vals[d] = size;
    ClearAllButDataType();

    Status s = OkStatus();
    for (auto dval : vals) {
      s.Update(AddDimWithStatus(dval));
      if (!s.ok()) {
        return s;
      }
    }
  }

  return RecomputeNumElements();
}

template <class Shape>
void TensorShapeBase<Shape>::RemoveDimRange(int begin, int end) {
  if (unknown_rank()) return;
  begin = begin < 0 ? dims() + begin + 1 : begin;
  end = end < 0 ? dims() + end + 1 : end;
  CHECK_GE(begin, 0);
  CHECK_LE(begin, dims());
  CHECK_GE(end, 0);
  CHECK_LE(end, dims());
  if (begin >= end) return;
  gtl::InlinedVector<int64_t, 8> vals;
  AppendTo(*this, &vals);
  vals.erase(vals.begin() + begin, vals.begin() + end);
  ClearAllButDataType();
  for (auto dval : vals) {
    AddDim(dval);
  }
  TF_CHECK_OK(RecomputeNumElements());
}

template <class Shape>
Status TensorShapeBase<Shape>::RemoveDimRangeWithStatus(int begin, int end) {
  if (unknown_rank()) {
    return OkStatus();
  }

  begin = begin < 0 ? dims() + begin + 1 : begin;
  end = end < 0 ? dims() + end + 1 : end;

  if (TF_PREDICT_FALSE(begin < 0)) {
    return errors::Internal("Start index must be non-negative, got ", begin);
  }
  if (TF_PREDICT_FALSE(begin > dims())) {
    return errors::Internal("Start index must be less than ", dims(), ", got ",
                            begin);
  }
  if (TF_PREDICT_FALSE(end < 0)) {
    return errors::Internal("End index must be non-negative, got ", end);
  }
  if (TF_PREDICT_FALSE(end > dims())) {
    return errors::Internal("End index must be less than ", dims(), ", got ",
                            end);
  }

  if (begin >= end) {
    return OkStatus();
  }

  gtl::InlinedVector<int64_t, 8> vals;
  AppendTo(*this, &vals);
  vals.erase(vals.begin() + begin, vals.begin() + end);
  ClearAllButDataType();

  Status s = OkStatus();
  for (auto dval : vals) {
    s.Update(AddDimWithStatus(dval));
    if (!s.ok()) {
      return s;
    }
  }

  return RecomputeNumElements();
}

bool TensorShape::IsSameSize(const TensorShape& b) const {
  if (b.dims() != dims()) return false;
  for (int d = 0; d < dims(); d++) {
    if (dim_size(d) != b.dim_size(d)) return false;
  }
  return true;
}

template <class Shape>
void TensorShapeBase<Shape>::AsProto(TensorShapeProto* proto) const {
  proto->Clear();
  if (unknown_rank()) {
    proto->set_unknown_rank(true);
  } else {
    for (int i = 0; i < dims(); i++) {
      proto->add_dim()->set_size(dim_size(i));
    }
  }
}

template <class Shape>
TensorShapeProto TensorShapeBase<Shape>::AsProto() const {
  TensorShapeProto out;
  AsProto(&out);
  return out;
}

template <class Shape>
TensorShapeIter<Shape> TensorShapeBase<Shape>::begin() const {
  return TensorShapeIter<Shape>(static_cast<const Shape*>(this), 0);
}

template <class Shape>
TensorShapeIter<Shape> TensorShapeBase<Shape>::end() const {
  const int max_dim = unknown_rank() ? -1 : dims();
  return TensorShapeIter<Shape>(static_cast<const Shape*>(this), max_dim);
}

string TensorShapeRep::DebugString() const {
  const auto& shape = *static_cast<const PartialTensorShape*>(this);
  if (shape.unknown_rank()) return "<unknown>";
  string s = "[";
  for (int i = 0; i < shape.dims(); i++) {
    if (i > 0) strings::StrAppend(&s, ",");
    int64_t dim = shape.dim_size(i);
    if (dim < 0) {
      strings::StrAppend(&s, "?");
    } else {
      strings::StrAppend(&s, dim);
    }
  }
  strings::StrAppend(&s, "]");
  return s;
}

string TensorShapeRep::DebugString(const TensorShapeProto& proto) {
  string s;
  if (proto.unknown_rank()) {
    strings::StrAppend(&s, "<unknown>");
    if (proto.dim_size() == 0) return s;
  }
  strings::StrAppend(&s, "[");
  bool first = true;
  for (const auto& d : proto.dim()) {
    if (!first) strings::StrAppend(&s, ",");
    if (d.size() == -1) {
      strings::StrAppend(&s, "?");
    } else {
      strings::StrAppend(&s, d.size());
    }
    first = false;
  }
  strings::StrAppend(&s, "]");
  return s;
}

bool TensorShapeUtils::StartsWith(const TensorShape& shape,
                                  const TensorShape& prefix) {
  if (shape.dims() < prefix.dims()) return false;
  for (int i = 0; i < prefix.dims(); ++i) {
    if (shape.dim_size(i) != prefix.dim_size(i)) return false;
  }
  return true;
}

bool TensorShapeUtils::EndsWith(const TensorShape& shape,
                                const TensorShape& suffix) {
  const int suffix_size = suffix.dims();
  if (shape.dims() < suffix_size) return false;
  for (int i = 0; i < suffix_size; ++i) {
    if (shape.dim_size(shape.dims() - suffix_size + i) != suffix.dim_size(i)) {
      return false;
    }
  }
  return true;
}

template <typename T, class Shape>
Status MakeShapeHelper(const T* dims, int64_t n, Shape* out) {
  out->Clear();
  if (n > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Too many dimensions");
  }
  if (n < 0) {
    return errors::InvalidArgument("Negative number of dimensions ", n);
  }
  for (int64_t i = 0; i < n; ++i) {
    T dim = internal::SubtleMustCopy(dims[i]);
    int64_t new_num_elements;
    if (dim < 0) {
      if (!out->kIsPartial) {
        return errors::InvalidArgument("Dimension ", dim, " must be >= 0");
      }
      if (dim < -1) {
        return errors::InvalidArgument("Dimension ", dim, " must be >= -1");
      }
      dim = -1;
      new_num_elements = -1;
    } else if (out->num_elements() < 0) {
      new_num_elements = -1;
    } else {
      new_num_elements = MultiplyWithoutOverflow(out->num_elements(), dim);
      if (TF_PREDICT_FALSE(new_num_elements < 0)) {
        TensorShapeProto proto;
        for (int64_t j = 0; j < n; ++j) {
          proto.add_dim()->set_size(internal::SubtleMustCopy(dims[j]));
        }
        return errors::InvalidArgument(
            "Shape ", TensorShape::DebugString(proto),
            " would have more than 2**63 - 1 elements");
      }
    }
    out->UnsafeAddDim(dim, new_num_elements);
  }
  return OkStatus();
}

#define MAKE_SHAPE(T, Shape)                                                 \
  Status TensorShapeUtils::MakeShape(const T* dims, int64_t n, Shape* out) { \
    return MakeShapeHelper(dims, n, out);                                    \
  }                                                                          \
  Status TensorShapeUtils::MakeShape(gtl::ArraySlice<T> shape, Shape* out) { \
    return MakeShapeHelper(shape.data(), shape.size(), out);                 \
  }
MAKE_SHAPE(int32, TensorShape)
MAKE_SHAPE(int64_t, TensorShape)
MAKE_SHAPE(int32, PartialTensorShape)
MAKE_SHAPE(int64_t, PartialTensorShape)
#undef MAKE_SHAPE

string TensorShapeUtils::ShapeListString(
    const gtl::ArraySlice<TensorShape>& shapes) {
  string result = "[";
  bool first = true;
  for (const TensorShape& shape : shapes) {
    strings::StrAppend(&result, (first ? "" : ", "), shape.DebugString());
    first = false;
  }
  strings::StrAppend(&result, "]");
  return result;
}

PartialTensorShape PartialTensorShape::Concatenate(int64_t size) const {
  PartialTensorShape out = *this;
  out.AddDim(size);
  return out;
}

Status PartialTensorShape::ConcatenateWithStatus(
    int64_t size, PartialTensorShape* out) const {
  *out = *this;
  return out->AddDimWithStatus(size);
}

PartialTensorShape PartialTensorShape::Concatenate(
    const PartialTensorShape& shape) const {
  if (unknown_rank() || shape.unknown_rank()) {
    return PartialTensorShape();
  }
  PartialTensorShape out = *this;
  for (auto dim : shape) out.AddDim(dim.size);
  return out;
}

Status PartialTensorShape::ConcatenateWithStatus(
    const PartialTensorShape& shape, PartialTensorShape* out) const {
  if (unknown_rank() || shape.unknown_rank()) {
    *out = PartialTensorShape();
    return OkStatus();
  }
  *out = *this;
  for (auto dim : shape) {
    Status s = out->AddDimWithStatus(dim.size);
    if (!s.ok()) return s;
  }

  return OkStatus();
}

Status PartialTensorShape::MergeWith(const PartialTensorShape& shape,
                                     PartialTensorShape* result) const {
  if (unknown_rank()) {
    *result = shape;
    return OkStatus();
  }
  if (shape.unknown_rank()) {
    *result = *this;
    return OkStatus();
  }
  const int dims_ = dims();
  if (dims_ != shape.dims()) {
    return errors::InvalidArgument(
        "PartialTensorShape: Incompatible ranks during merge: ", dims_, " vs. ",
        shape.dims());
  }

  if (result == this) {
    return errors::Internal(
        "PartialTensorShape::MergeWith: Cannot output result to itself");
  }

  result->Clear();
  Status s = OkStatus();
  for (int i = 0; i < dims_; ++i) {
    const int64_t dim0 = dim_size(i);
    const int64_t dim1 = shape.dim_size(i);
    if (dim0 >= 0 && dim1 >= 0 && dim0 != dim1) {
      return errors::InvalidArgument(
          "PartialTensorShape: Incompatible shapes during merge: ",
          DebugString(), " vs. ", shape.DebugString());
    }
    s.Update(result->AddDimWithStatus(dim0 >= 0 ? dim0 : dim1));
    if (!s.ok()) {
      return s;
    }
  }
  return OkStatus();
}

bool PartialTensorShape::AsTensorShape(TensorShape* shape) const {
  if (IsFullyDefined()) {
    const TensorShapeRep* rep = this;
    *shape = *static_cast<const TensorShape*>(rep);
    return true;
  }
  return false;
}

bool PartialTensorShape::IsIdenticalTo(const PartialTensorShape& shape) const {
  if (unknown_rank() || shape.unknown_rank()) {
    return unknown_rank() == shape.unknown_rank();
  }
  if (dims() != shape.dims()) return false;
  for (int i = 0; i < dims(); i++) {
    if (dim_size(i) != shape.dim_size(i)) return false;
  }
  return true;
}

bool PartialTensorShape::IsCompatibleWith(
    const PartialTensorShape& shape) const {
  if (unknown_rank() || shape.unknown_rank()) return true;
  if (dims() != shape.dims()) return false;
  for (int i = 0; i < dims(); i++) {
    const int64_t dim0 = dim_size(i);
    const int64_t dim1 = shape.dim_size(i);
    if (dim0 >= 0 && dim1 >= 0 && dim0 != dim1) return false;
  }
  return true;
}

string PartialTensorShapeUtils::PartialShapeListString(
    const gtl::ArraySlice<PartialTensorShape>& shapes) {
  string result = "[";
  bool first = true;
  for (const PartialTensorShape& shape : shapes) {
    strings::StrAppend(&result, (first ? "" : ", "), shape.DebugString());
    first = false;
  }
  strings::StrAppend(&result, "]");
  return result;
}

bool PartialTensorShapeUtils::AreCompatible(
    const gtl::ArraySlice<PartialTensorShape>& shapes0,
    const gtl::ArraySlice<PartialTensorShape>& shapes1) {
  if (shapes0.size() == shapes1.size()) {
    for (size_t i = 0; i < shapes0.size(); ++i) {
      if (!shapes0[i].IsCompatibleWith(shapes1[i])) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

bool PartialTensorShapeUtils::AreIdentical(
    const gtl::ArraySlice<PartialTensorShape>& shapes0,
    const gtl::ArraySlice<PartialTensorShape>& shapes1) {
  if (shapes0.size() == shapes1.size()) {
    for (size_t i = 0; i < shapes0.size(); ++i) {
      if (!shapes0[i].IsIdenticalTo(shapes1[i])) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

Status TensorShapeUtils::NumElements(gtl::ArraySlice<int64_t> shape,
                                     int64_t* num_elements) {
  int64_t n = 1;
  for (auto dim : shape) {
    n = MultiplyWithoutOverflow(n, dim);
    if (n < 0) {
      return errors::InvalidArgument("Can't compute total size of shape [",
                                     absl::StrJoin(shape, ","),
                                     "]; product would overflow int64");
    }
  }
  *num_elements = n;
  return OkStatus();
}

template class TensorShapeBase<TensorShape>;
template class TensorShapeBase<PartialTensorShape>;

}  // namespace tensorflow
