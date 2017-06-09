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

#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
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

void TensorShape::CheckDimsAtLeast(int NDIMS) const {
  CHECK_GE(NDIMS, dims()) << "Asking for tensor of at least " << NDIMS
                          << " dimensions from a tensor of " << dims()
                          << " dimensions";
}

template <class Shape>
bool TensorShapeBase<Shape>::IsValid(const TensorShapeProto& proto) {
  // NOTE(irving): Unfortunately, TensorShape allows parsing protos with
  // unknown_shape() set, and it seems hard to remove this without backwards
  // compatibility issues.
  if (kIsPartial && proto.unknown_rank()) return proto.dim_size() == 0;
  int64 num_elements = 1;
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
    return Status::OK();
  }
  int64 num_elements = 1;
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
  return Status::OK();
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
TensorShapeBase<Shape>::TensorShapeBase(gtl::ArraySlice<int64> dim_sizes) {
  set_tag(REP16);
  set_data_type(DT_INVALID);
  set_ndims_byte(0);
  set_num_elements(1);
  for (int64 s : dim_sizes) {
    AddDim(internal::SubtleMustCopy(s));
  }
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
    DCHECK_EQ(b.tag(), REP_OUT_OF_LINE);
    set_ndims_byte(b.ndims_byte());
    set_data_type(b.data_type());
    if (tag() == REP_OUT_OF_LINE) {
      // vector already allocated
      *(as64()->dims_) = *(b.as64()->dims_);
    } else {
      set_tag(REP_OUT_OF_LINE);
      as64()->dims_ = new gtl::InlinedVector<int64, 4>(*(b.as64()->dims_));
    }
  }
}

template <class Shape>
int64 TensorShapeBase<Shape>::dim_size(int d) const {
  if (unknown_rank()) return -1;
  DCHECK_GE(d, 0);
  DCHECK_LT(d, dims());
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
void TensorShapeBase<Shape>::RecomputeNumElements() {
  if (unknown_rank()) {
    set_num_elements(-1);
    return;
  }
  int64 n = 1;
  for (auto dim : *this) {
    if (kIsPartial && dim.size < 0) {
      n = -1;
      break;
    }
    n = MultiplyWithoutOverflow(n, dim.size);
    CHECK_LE(0, n);
  }
  set_num_elements(n);
}

template <class Shape>
void TensorShapeBase<Shape>::AddDim(int64 size) {
  if (!kIsPartial) CHECK_GE(size, 0);
  if (unknown_rank()) return;
  CHECK_LT(ndims_byte(), MaxDimensions()) << "Too many dimensions in tensor";
  int64 new_num_elements;
  if (kIsPartial && (num_elements() < 0 || size < 0)) {
    new_num_elements = -1;
  } else {
    new_num_elements = MultiplyWithoutOverflow(num_elements(), size);
    CHECK_LE(0, new_num_elements);
  }
  UnsafeAddDim(size, new_num_elements);
}

template <class Shape>
void TensorShapeBase<Shape>::UnsafeAddDim(int64 size, int64 new_num_elements) {
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
    gtl::InlinedVector<int64, 8> vals;
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
          new gtl::InlinedVector<int64, 4>(vals.begin(), vals.end());
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
void TensorShapeBase<Shape>::InsertDim(int d, int64 size) {
  CHECK_GE(d, 0);
  CHECK_LE(d, dims());
  if (!kIsPartial) CHECK_GE(size, 0);
  CHECK_LT(dims(), MaxDimensions());
  gtl::InlinedVector<int64, 8> vals;
  AppendTo(*this, &vals);
  vals.insert(vals.begin() + d, size);
  ClearAllButDataType();
  for (auto dval : vals) {
    AddDim(dval);
  }
}

template <class Shape>
gtl::InlinedVector<int64, 4> TensorShapeBase<Shape>::dim_sizes() const {
  gtl::InlinedVector<int64, 4> result;
  for (auto dim : *this) {
    result.push_back(dim.size);
  }
  return result;
}

template <class Shape>
void TensorShapeBase<Shape>::set_dim(int d, int64 size) {
  CHECK_GE(d, 0);
  CHECK_LT(d, dims());
  CHECK_GE(size, 0);
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
    gtl::InlinedVector<int64, 8> vals;
    AppendTo(*this, &vals);
    vals[d] = size;
    ClearAllButDataType();
    for (auto dval : vals) {
      AddDim(dval);
    }
  }
  RecomputeNumElements();
}

template <class Shape>
void TensorShapeBase<Shape>::RemoveDim(int d) {
  if (unknown_rank()) return;
  CHECK_GE(d, 0);
  CHECK_LT(d, dims());
  gtl::InlinedVector<int64, 8> vals;
  AppendTo(*this, &vals);
  vals.erase(vals.begin() + d);
  ClearAllButDataType();
  for (auto dval : vals) {
    AddDim(dval);
  }
  RecomputeNumElements();
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

void TensorShapeRep::DumpRep() const {
#if 0
  fprintf(stderr, "Rep: %d %d dims\n", tag(), dims());
  if (tag() == REP16) {
    fprintf(stderr, "REP16 NDIMS: %d\n", ndims_byte());
    for (int i = 0; i < ndims_byte(); i++) {
      fprintf(stderr, "dim %d: %d\n", i, as16()->dims_[i]);
    }
  } else if (tag_ == REP32) {
    fprintf(stderr, "REP32 NDIMS: %d\n", ndims_);
    for (int i = 0; i < ndims_byte(); i++) {
      fprintf(stderr, "dim %d: %d\n", i, as32()->dims_[i]);
    }
  } else if (tag_ == REP_OUT_OF_LINE) {
    fprintf(stderr, "REP_OUT_OF_LINE NDIMS: %d %p\n", ndims_, as16()->dims_);
    for (int i = 0; i < ndims_byte(); i++) {
      fprintf(stderr, "dim %d: %lld\n", i, (*as64()->dims_)[i]);
    }
  }
#endif
}

template <class Shape>
TensorShapeIter<Shape> TensorShapeBase<Shape>::begin() const {
  return TensorShapeIter<Shape>(static_cast<const Shape*>(this), 0);
}

template <class Shape>
TensorShapeIter<Shape> TensorShapeBase<Shape>::end() const {
  CHECK(!unknown_rank());
  return TensorShapeIter<Shape>(static_cast<const Shape*>(this), dims());
}

string TensorShapeRep::DebugString() const {
  const auto& shape = *static_cast<const PartialTensorShape*>(this);
  if (shape.unknown_rank()) return "<unknown>";
  string s = "[";
  for (int i = 0; i < shape.dims(); i++) {
    if (i > 0) strings::StrAppend(&s, ",");
    int64 dim = shape.dim_size(i);
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
Status MakeShapeHelper(const T* dims, int64 n, Shape* out) {
  out->Clear();
  if (n > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Too many dimensions");
  }
  if (n < 0) {
    return errors::InvalidArgument("Negative number of dimensions ", n);
  }
  for (int64 i = 0; i < n; ++i) {
    T dim = internal::SubtleMustCopy(dims[i]);
    int64 new_num_elements;
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
        for (int64 j = 0; j < n; ++j) {
          proto.add_dim()->set_size(dim);
        }
        return errors::InvalidArgument(
            "Shape ", TensorShape::DebugString(proto),
            " would have more than 2**63 - 1 elements");
      }
    }
    out->UnsafeAddDim(dim, new_num_elements);
  }
  return Status::OK();
}

#define MAKE_SHAPE(T, Shape)                                                 \
  Status TensorShapeUtils::MakeShape(const T* dims, int64 n, Shape* out) {   \
    return MakeShapeHelper(dims, n, out);                                    \
  }                                                                          \
  Status TensorShapeUtils::MakeShape(gtl::ArraySlice<T> shape, Shape* out) { \
    return MakeShapeHelper(shape.data(), shape.size(), out);                 \
  }
MAKE_SHAPE(int32, TensorShape)
MAKE_SHAPE(int64, TensorShape)
MAKE_SHAPE(int32, PartialTensorShape)
MAKE_SHAPE(int64, PartialTensorShape)
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

PartialTensorShape PartialTensorShape::Concatenate(int64 size) const {
  PartialTensorShape out = *this;
  out.AddDim(size);
  return out;
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

Status PartialTensorShape::MergeWith(const PartialTensorShape& shape,
                                     PartialTensorShape* result) const {
  if (unknown_rank()) {
    *result = shape;
    return Status::OK();
  }
  if (shape.unknown_rank()) {
    *result = *this;
    return Status::OK();
  }
  const int dims_ = dims();
  if (dims_ != shape.dims()) {
    return errors::InvalidArgument(
        "PartialTensorShape: Incompatible ranks during merge: ", dims_, " vs. ",
        shape.dims());
  }
  CHECK(result != this);
  result->Clear();
  for (int i = 0; i < dims_; ++i) {
    const int64 dim0 = dim_size(i);
    const int64 dim1 = shape.dim_size(i);
    if (dim0 >= 0 && dim1 >= 0 && dim0 != dim1) {
      return errors::InvalidArgument(
          "PartialTensorShape: Incompatible shapes during merge: ",
          DebugString(), " vs. ", shape.DebugString());
    }
    result->AddDim(dim0 >= 0 ? dim0 : dim1);
  }
  return Status::OK();
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
    const int64 dim0 = dim_size(i);
    const int64 dim1 = shape.dim_size(i);
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

Status TensorShapeUtils::NumElements(gtl::ArraySlice<int64> shape,
                                     int64* num_elements) {
  int64 n = 1;
  for (auto dim : shape) {
    n = MultiplyWithoutOverflow(n, dim);
    if (n < 0) {
      return errors::InvalidArgument("Can't compute total size of shape [",
                                     str_util::Join(shape, ","),
                                     "]; product would overflow int64");
    }
  }
  *num_elements = n;
  return Status::OK();
}

template class TensorShapeBase<TensorShape>;
template class TensorShapeBase<PartialTensorShape>;

}  // namespace tensorflow
