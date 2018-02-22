/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/literal_util.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::strings::Printf;
using tensorflow::strings::StrCat;

namespace xla {

namespace {

constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

// Converts between little and big endian, assuming elements in the array are 16
// bits long.
void ConvertEndianShort(char* bytes, int64 size) {
  CHECK_EQ(size / 2, 0);
  for (int64 i = 0; i < size; i += 2) {
    std::swap(bytes[i], bytes[i + 1]);
  }
}

}  // namespace

std::ostream& operator<<(std::ostream& out, const Literal& literal) {
  out << literal.ToString();
  return out;
}

Literal::StrideConfig::StrideConfig(
    const Shape& source_shape, const Shape& dest_shape,
    tensorflow::gtl::ArraySlice<int64> dimensions)
    : dimensions(dimensions),
      base(dimensions.size(), 0),
      step(dimensions.size(), 1) {
  if (!dimensions.empty()) {
    // Selects the shape with the largest minor dimension as the one upon
    // which to run the tight stride loop.
    if (dimensions[LayoutUtil::Minor(source_shape.layout(), 0)] >=
        dimensions[LayoutUtil::Minor(dest_shape.layout(), 0)]) {
      minor_dimension = LayoutUtil::Minor(source_shape.layout(), 0);
      dest_stride = IndexUtil::GetDimensionStride(dest_shape, minor_dimension);
    } else {
      minor_dimension = LayoutUtil::Minor(dest_shape.layout(), 0);
      source_stride =
          IndexUtil::GetDimensionStride(source_shape, minor_dimension);
    }
    minor_loop_size = dimensions[minor_dimension];
    step[minor_dimension] = minor_loop_size;
  }
}

Literal::Literal(const Shape& shape)
    : Literal(shape, /*allocate_arrays=*/true) {}

Literal::Literal(const Shape& shape, bool allocate_arrays)
    : shape_(shape), pieces_(shape), owns_buffers_(true) {
  CHECK(LayoutUtil::HasLayout(shape));
  for (auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;

    piece.set_subshape(&ShapeUtil::GetSubshape(shape_, index));
    const Shape& subshape = piece.subshape();
    if (ShapeUtil::IsArray(subshape)) {
      if (allocate_arrays) {
        piece.set_buffer(new char[piece.size_bytes()]);
        if (LayoutUtil::IsSparseArray(subshape)) {
          piece.set_sparse_indices(new SparseIndexArray(
              LayoutUtil::MaxSparseElements(subshape.layout()),
              ShapeUtil::Rank(subshape)));
        }
      } else {
        piece.set_buffer(nullptr);
      }
    }
  }
}

Literal::~Literal() { DeallocateBuffers(); }

void Literal::DeallocateBuffers() {
  if (owns_buffers_) {
    for (auto& pair : pieces_) {
      Piece& piece = pair.second;
      if (piece.buffer() != nullptr) {
        delete[] piece.buffer();
        delete piece.sparse_indices();
      }
    }
  }
}

Literal::Literal(Literal&& other) {
  shape_ = std::move(other.shape_);
  pieces_ = std::move(other.pieces_);
  // We need to iterate through the pieces to set the subshape pointer
  // properly. It must refer to subshapes within shape_.
  for (auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;
    piece.set_subshape(&ShapeUtil::GetSubshape(shape_, index));
  }
  owns_buffers_ = other.owns_buffers_;

  other.shape_ = ShapeUtil::MakeNil();
  other.pieces_ = ShapeTree<Piece>(other.shape_);
  other.piece({}).set_subshape(&other.shape_);
}

Literal& Literal::operator=(Literal&& other) {
  DeallocateBuffers();
  shape_ = std::move(other.shape_);
  pieces_ = std::move(other.pieces_);
  // We need to iterate through the pieces to set the subshape pointer
  // properly. It must refer to subshapes within shape_.
  for (auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;
    piece.set_subshape(&ShapeUtil::GetSubshape(shape_, index));
  }
  owns_buffers_ = other.owns_buffers_;

  other.shape_ = ShapeUtil::MakeNil();
  other.pieces_ = ShapeTree<Piece>(other.shape_);
  other.piece({}).set_subshape(&other.shape_);
  return *this;
}

std::unique_ptr<Literal> Literal::CreateFromShape(const Shape& shape) {
  auto literal = MakeUnique<Literal>(shape);
  for (auto& pair : literal->pieces_) {
    Piece& piece = pair.second;
    if (ShapeUtil::IsArray(piece.subshape())) {
      memset(piece.untyped_data(), 0, piece.size_bytes());
    }
  }
  return literal;
}

const SparseIndexArray* Literal::sparse_indices(
    const ShapeIndex& shape_index) const {
  return piece(shape_index).sparse_indices();
}

SparseIndexArray* Literal::sparse_indices(const ShapeIndex& shape_index) {
  return piece(shape_index).sparse_indices();
}

/* static */ std::unique_ptr<Literal> Literal::CreateFromDimensions(
    PrimitiveType primitive_type,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  return CreateFromShape(ShapeUtil::MakeShape(primitive_type, dimensions));
}

template <typename NativeT>
Status Literal::CopySliceFromInternal(
    const Literal& src_literal, tensorflow::gtl::ArraySlice<int64> src_base,
    tensorflow::gtl::ArraySlice<int64> dest_base,
    tensorflow::gtl::ArraySlice<int64> copy_size) {
  TF_RET_CHECK(ShapeUtil::Rank(src_literal.shape()) == src_base.size());
  TF_RET_CHECK(ShapeUtil::Rank(shape()) == dest_base.size());

  auto linear_index = [](const Shape& shape,
                         tensorflow::gtl::ArraySlice<int64> multi_index) {
    return IndexUtil::MultidimensionalIndexToLinearIndex(shape, multi_index);
  };

  if (ShapeUtil::Rank(src_literal.shape()) == 0 ||
      ShapeUtil::Rank(shape()) == 0) {
    // If any of the two shapes are scalars, we can just call the StridedCopy()
    // directly, and we know we will be copying only one value.
    TF_RET_CHECK(copy_size.empty());
    StridedCopy(data<NativeT>(), linear_index(shape(), dest_base), 0,
                src_literal.data<NativeT>(),
                linear_index(src_literal.shape(), src_base), 0, 1);
  } else if (!ShapeUtil::HasZeroElements(shape()) &&
             !ShapeUtil::HasZeroElements(src_literal.shape())) {
    // Perform copy if neither src nor dest has dimensions with zero element,
    // otherwise it's a no-op.
    TF_RET_CHECK(src_base.size() == dest_base.size());
    TF_RET_CHECK(src_base.size() == copy_size.size());

    // Scan the source from minor, stepping in copy size blocks, then within
    // the index enumaration functor, do a strided copy advancing source index
    // by one (walking through the minor dimension), and destination index by
    // proper stride size at the matching dimension.
    DimensionVector src_indexes(src_base.size(), 0);
    DimensionVector dest_indexes(dest_base.size(), 0);
    Literal::StrideConfig stride_config(src_literal.shape(), shape(),
                                        copy_size);

    auto copy_proc = [&](const std::vector<int64>& indexes) {
      // Map from multi-dimensional index, to source index.
      std::transform(indexes.begin(), indexes.end(), src_base.begin(),
                     src_indexes.begin(), std::plus<int64>());
      // Map from multi-dimensional index, to destination index.
      std::transform(indexes.begin(), indexes.end(), dest_base.begin(),
                     dest_indexes.begin(), std::plus<int64>());

      int64 src_index = linear_index(src_literal.shape(), src_indexes);
      int64 dest_index = linear_index(shape(), dest_indexes);

      // `this->` is needed to workaround MSVC bug: #16882
      StridedCopy(this->data<NativeT>(), dest_index, stride_config.dest_stride,
                  src_literal.data<NativeT>(), src_index,
                  stride_config.source_stride, stride_config.minor_loop_size);
      return true;
    };

    ShapeUtil::ForEachIndex(src_literal.shape(), stride_config.base,
                            stride_config.dimensions, stride_config.step,
                            copy_proc);
  }
  return Status::OK();
}

std::vector<Literal> Literal::DecomposeTuple() {
  CHECK(ShapeUtil::IsTuple(shape()));
  std::vector<Literal> elements;
  for (int i = 0; i < ShapeUtil::TupleElementCount(shape()); ++i) {
    elements.push_back(Literal(ShapeUtil::GetSubshape(shape(), {i}),
                               /*allocate_arrays=*/false));
    Literal& element = elements.back();
    for (auto& pair : element.pieces_) {
      const ShapeIndex& index = pair.first;
      Piece& dest_piece = pair.second;
      ShapeIndex src_index = {i};
      for (int64 j : index) {
        src_index.push_back(j);
      }
      Piece& src_piece = piece(src_index);

      // Move the respective buffer and sparse indices over to the element
      // Literal.
      dest_piece.set_buffer(src_piece.buffer());
      src_piece.set_buffer(nullptr);
      dest_piece.set_sparse_indices(src_piece.sparse_indices());
      src_piece.set_sparse_indices(nullptr);
    }
  }
  // Set this literal to be nil-shaped.
  *this = Literal();
  return elements;
}

/* static */ Literal Literal::MoveIntoTuple(
    tensorflow::gtl::MutableArraySlice<Literal> elements) {
  std::vector<Shape> element_shapes;
  for (const Literal& element : elements) {
    element_shapes.push_back(element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShape(element_shapes),
                  /*allocate_arrays=*/false);
  for (int i = 0; i < elements.size(); ++i) {
    TF_CHECK_OK(
        literal.MoveFrom(std::move(elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

namespace {

// Copies the elements in 'src' to 'dest'. The shape and layout of the data in
// the array slices are indicated by dest_shape and src_shape respectively.
template <typename NativeT>
void CopyElementsBetween(tensorflow::gtl::MutableArraySlice<NativeT> dest,
                         tensorflow::gtl::ArraySlice<NativeT> src,
                         const Shape& dest_shape, const Shape& src_shape) {
  CHECK(ShapeUtil::Compatible(dest_shape, src_shape));
  if (ShapeUtil::HasZeroElements(dest_shape)) {
    return;
  }
  std::vector<int64> index(ShapeUtil::Rank(dest_shape));
  do {
    dest[IndexUtil::MultidimensionalIndexToLinearIndex(dest_shape, index)] =
        src[IndexUtil::MultidimensionalIndexToLinearIndex(src_shape, index)];
  } while (IndexUtil::BumpIndices(dest_shape, &index));
}

}  // namespace

Status Literal::Piece::CopyFrom(const Literal::Piece& src) {
  if (ShapeUtil::Equal(subshape(), src.subshape())) {
    // If the layouts are equal it's faster just to memcpy.
    memcpy(buffer(), src.buffer(), src.size_bytes());
  } else {
    TF_RET_CHECK(ShapeUtil::Compatible(src.subshape(), subshape()));
    std::vector<int64> origin(ShapeUtil::Rank(subshape()), 0);
    switch (subshape().element_type()) {
#define COPY_ELEMENTS(XLA_T, NATIVE_T)                                    \
  case (XLA_T):                                                           \
    CopyElementsBetween<NATIVE_T>(data<NATIVE_T>(), src.data<NATIVE_T>(), \
                                  subshape(), src.subshape());            \
    break;
      COPY_ELEMENTS(U8, uint8);
      COPY_ELEMENTS(U16, uint16);
      COPY_ELEMENTS(U32, uint32);
      COPY_ELEMENTS(U64, uint64);
      COPY_ELEMENTS(S8, int8);
      COPY_ELEMENTS(S16, int16);
      COPY_ELEMENTS(S32, int32);
      COPY_ELEMENTS(S64, int64);
      COPY_ELEMENTS(F16, half);
      COPY_ELEMENTS(BF16, bfloat16);
      COPY_ELEMENTS(F32, float);
      COPY_ELEMENTS(F64, double);
      COPY_ELEMENTS(C64, complex64);
      COPY_ELEMENTS(PRED, bool);
#undef COPY_ELEMENTS
      default:
        return Unimplemented(
            "Unhandled primitive type %s",
            PrimitiveType_Name(subshape().element_type()).c_str());
    }
  }
  return Status::OK();
}

Status Literal::CopyFrom(const Literal& src_literal,
                         const ShapeIndex& dest_shape_index,
                         const ShapeIndex& src_shape_index) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  const Shape& src_subshape =
      ShapeUtil::GetSubshape(src_literal.shape(), src_shape_index);
  if (!ShapeUtil::Compatible(dest_subshape, src_subshape)) {
    return InvalidArgument(
        "Destination subshape incompatible with source subshape: %s vs %s",
        ShapeUtil::HumanString(dest_subshape).c_str(),
        ShapeUtil::HumanString(src_subshape).c_str());
  }

  for (auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;
    if (!ShapeUtil::IsArray(piece.subshape())) {
      continue;
    }

    // Determine if this index is in the part of this literal that we want to
    // copy over from src_literal.
    bool in_subtree_to_copy = true;
    for (int i = 0; i < dest_shape_index.size(); ++i) {
      if (index[i] != dest_shape_index[i]) {
        in_subtree_to_copy = false;
        break;
      }
    }
    if (!in_subtree_to_copy) {
      continue;
    }

    // Construct the index of the corresponding piece in the source literal.
    ShapeIndex src_piece_index = src_shape_index;
    for (int64 i = dest_shape_index.size(); i < index.size(); ++i) {
      src_piece_index.push_back(index[i]);
    }

    TF_RETURN_IF_ERROR(piece.CopyFrom(src_literal.piece(src_piece_index)));
  }
  return Status::OK();
}

Status Literal::MoveFrom(Literal&& src_literal,
                         const ShapeIndex& dest_shape_index) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  if (!ShapeUtil::Equal(dest_subshape, src_literal.shape())) {
    return InvalidArgument(
        "Destination subshape not equal to source shape: %s vs %s",
        ShapeUtil::HumanString(dest_subshape).c_str(),
        ShapeUtil::HumanString(src_literal.shape()).c_str());
  }

  if (!(owns_buffers_ && src_literal.owns_buffers_)) {
    return InvalidArgument(
        "Source and destination literals must both own their buffers (ie, not "
        "be views)");
  }

  for (auto& pair : src_literal.pieces_) {
    const ShapeIndex& src_index = pair.first;
    Piece& src_piece = pair.second;
    if (!ShapeUtil::IsArray(src_piece.subshape())) {
      continue;
    }

    ShapeIndex dest_index = dest_shape_index;
    for (int64 i : src_index) {
      dest_index.push_back(i);
    }
    Piece& dest_piece = piece(dest_index);
    delete[] dest_piece.buffer();
    dest_piece.set_buffer(src_piece.buffer());
    delete dest_piece.sparse_indices();
    dest_piece.set_sparse_indices(src_piece.sparse_indices());
  }

  src_literal.shape_ = ShapeUtil::MakeNil();
  src_literal.pieces_ = ShapeTree<Piece>(src_literal.shape_);
  src_literal.piece({}).set_subshape(&src_literal.shape_);
  return Status::OK();
}

Status Literal::CopySliceFrom(const Literal& src_literal,
                              tensorflow::gtl::ArraySlice<int64> src_base,
                              tensorflow::gtl::ArraySlice<int64> dest_base,
                              tensorflow::gtl::ArraySlice<int64> copy_size) {
  TF_RET_CHECK(ShapeUtil::IsArray(shape())) << ShapeUtil::HumanString(shape());
  TF_RET_CHECK(ShapeUtil::IsArray(src_literal.shape()))
      << ShapeUtil::HumanString(src_literal.shape());
  TF_RET_CHECK(ShapeUtil::SameElementType(src_literal.shape(), shape()));

  switch (shape().element_type()) {
    case U8:
      return CopySliceFromInternal<uint8>(src_literal, src_base, dest_base,
                                          copy_size);
    case U16:
      return CopySliceFromInternal<uint16>(src_literal, src_base, dest_base,
                                           copy_size);
    case U32:
      return CopySliceFromInternal<uint32>(src_literal, src_base, dest_base,
                                           copy_size);
    case U64:
      return CopySliceFromInternal<uint64>(src_literal, src_base, dest_base,
                                           copy_size);
    case S8:
      return CopySliceFromInternal<int8>(src_literal, src_base, dest_base,
                                         copy_size);
    case S16:
      return CopySliceFromInternal<int16>(src_literal, src_base, dest_base,
                                          copy_size);
    case S32:
      return CopySliceFromInternal<int32>(src_literal, src_base, dest_base,
                                          copy_size);
    case S64:
      return CopySliceFromInternal<int64>(src_literal, src_base, dest_base,
                                          copy_size);
    case F16:
      return CopySliceFromInternal<half>(src_literal, src_base, dest_base,
                                         copy_size);
    case BF16:
      return CopySliceFromInternal<bfloat16>(src_literal, src_base, dest_base,
                                             copy_size);
    case F32:
      return CopySliceFromInternal<float>(src_literal, src_base, dest_base,
                                          copy_size);
    case F64:
      return CopySliceFromInternal<double>(src_literal, src_base, dest_base,
                                           copy_size);
    case C64:
      return CopySliceFromInternal<complex64>(src_literal, src_base, dest_base,
                                              copy_size);
    case PRED:
      return CopySliceFromInternal<bool>(src_literal, src_base, dest_base,
                                         copy_size);
    default:
      break;
  }
  return Unimplemented("Unhandled primitive type %d", shape().element_type());
}

/* static */ Literal Literal::Zero(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return std::move(*Literal::CreateR0<uint8>(0));
    case U32:
      return std::move(*Literal::CreateR0<uint32>(0));
    case U64:
      return std::move(*Literal::CreateR0<uint64>(0));
    case S8:
      return std::move(*Literal::CreateR0<int8>(0));
    case S32:
      return std::move(*Literal::CreateR0<int32>(0));
    case S64:
      return std::move(*Literal::CreateR0<int64>(0));
    case F16:
      return std::move(*Literal::CreateR0<half>(static_cast<half>(0.0f)));
    case BF16:
      return std::move(
          *Literal::CreateR0<bfloat16>(static_cast<bfloat16>(0.0f)));
    case F32:
      return std::move(*Literal::CreateR0<float>(0));
    case F64:
      return std::move(*Literal::CreateR0<double>(0));
    case C64:
      return std::move(*Literal::CreateR0<complex64>(0));
    case PRED:
      return std::move(*Literal::CreateR0<bool>(false));
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type cannot take on value of 0";
    case OPAQUE:
      LOG(FATAL) << "opaque element type cannot take on value of 0";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal Literal::One(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return std::move(*Literal::CreateR0<uint8>(1));
    case U32:
      return std::move(*Literal::CreateR0<uint32>(1));
    case U64:
      return std::move(*Literal::CreateR0<uint64>(1));
    case S8:
      return std::move(*Literal::CreateR0<int8>(1));
    case S32:
      return std::move(*Literal::CreateR0<int32>(1));
    case S64:
      return std::move(*Literal::CreateR0<int64>(1));
    case F16:
      return std::move(*Literal::CreateR0<half>(static_cast<half>(1.0f)));
    case BF16:
      return std::move(
          *Literal::CreateR0<bfloat16>(static_cast<bfloat16>(1.0f)));
    case F32:
      return std::move(*Literal::CreateR0<float>(1));
    case F64:
      return std::move(*Literal::CreateR0<double>(1));
    case C64:
      return std::move(*Literal::CreateR0<complex64>(1));
    case PRED:
      return std::move(*Literal::CreateR0<bool>(true));
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case TUPLE:
      LOG(FATAL) << "tuple element type cannot take on value of 1";
    case OPAQUE:
      LOG(FATAL) << "opaque element type cannot take on value of 1";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal Literal::MinValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return std::move(
          *Literal::CreateR0<uint8>(std::numeric_limits<uint8>::min()));
    case U32:
      return std::move(
          *Literal::CreateR0<uint32>(std::numeric_limits<uint32>::min()));
    case U64:
      return std::move(
          *Literal::CreateR0<uint64>(std::numeric_limits<uint64>::min()));
    case S8:
      return std::move(
          *Literal::CreateR0<int8>(std::numeric_limits<int8>::min()));
    case S32:
      return std::move(
          *Literal::CreateR0<int32>(std::numeric_limits<int32>::min()));
    case S64:
      return std::move(
          *Literal::CreateR0<int64>(std::numeric_limits<int64>::min()));
    case F32:
      return std::move(
          *Literal::CreateR0<float>(-std::numeric_limits<float>::infinity()));
    case F64:
      return std::move(
          *Literal::CreateR0<double>(-std::numeric_limits<double>::infinity()));
    case C64:
      LOG(FATAL) << "C64 element type has no minimum value";
    case PRED:
      return std::move(*Literal::CreateR0<bool>(false));
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      return std::move(*Literal::CreateR0<half>(
          static_cast<half>(-std::numeric_limits<float>::infinity())));
    case BF16:
      return std::move(*Literal::CreateR0<bfloat16>(
          static_cast<bfloat16>(-std::numeric_limits<float>::infinity())));
    case TUPLE:
      LOG(FATAL) << "tuple element type has no minimum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no minimum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ Literal Literal::MaxValue(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case U8:
      return std::move(
          *Literal::CreateR0<uint8>(std::numeric_limits<uint8>::max()));
    case U32:
      return std::move(
          *Literal::CreateR0<uint32>(std::numeric_limits<uint32>::max()));
    case U64:
      return std::move(
          *Literal::CreateR0<uint64>(std::numeric_limits<uint64>::max()));
    case S8:
      return std::move(
          *Literal::CreateR0<int8>(std::numeric_limits<int8>::max()));
    case S32:
      return std::move(
          *Literal::CreateR0<int32>(std::numeric_limits<int32>::max()));
    case S64:
      return std::move(
          *Literal::CreateR0<int64>(std::numeric_limits<int64>::max()));
    case F32:
      return std::move(
          *Literal::CreateR0<float>(std::numeric_limits<float>::infinity()));
    case F64:
      return std::move(
          *Literal::CreateR0<double>(std::numeric_limits<double>::infinity()));
    case PRED:
      return std::move(*Literal::CreateR0<bool>(true));
    case S16:
    case U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case F16:
      return std::move(*Literal::CreateR0<half>(
          static_cast<half>(std::numeric_limits<float>::infinity())));
    case BF16:
      return std::move(*Literal::CreateR0<bfloat16>(
          static_cast<bfloat16>(std::numeric_limits<float>::infinity())));
    case TUPLE:
      LOG(FATAL) << "tuple element type has no maximum value";
    case OPAQUE:
      LOG(FATAL) << "opaque element type has no maximum value";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ std::unique_ptr<Literal> Literal::CreateR1(
    const tensorflow::core::Bitmap& values) {
  auto literal = MakeUnique<Literal>(
      ShapeUtil::MakeShape(PRED, {static_cast<int64>(values.bits())}));
  literal->PopulateR1(values);
  return literal;
}

void Literal::PopulateR1(const tensorflow::core::Bitmap& values) {
  CHECK(ShapeUtil::IsArray(shape()));
  CHECK_EQ(ShapeUtil::Rank(shape()), 1);
  CHECK_EQ(element_count(), values.bits());
  CHECK_EQ(shape().element_type(), PRED);
  for (int64 i = 0; i < static_cast<int64>(values.bits()); ++i) {
    Set({i}, values.get(i));
  }
}

/* static */ std::unique_ptr<Literal> Literal::CreateR1U8(
    tensorflow::StringPiece value) {
  auto literal = MakeUnique<Literal>(
      ShapeUtil::MakeShape(U8, {static_cast<int64>(value.size())}));
  for (int i = 0; i < value.size(); ++i) {
    literal->Set<uint8>({i}, value[i]);
  }
  return literal;
}

/* static */ std::unique_ptr<Literal> Literal::CreateR2F32Linspace(float from,
                                                                   float to,
                                                                   int64 rows,
                                                                   int64 cols) {
  auto value = MakeLinspaceArray2D(from, to, rows, cols);
  return CreateR2FromArray2D(*value);
}

std::unique_ptr<Literal> Literal::Relayout(
    const Layout& new_layout, const ShapeIndex& shape_index) const {
  // Create new shape with 'new_layout' set at the given shape index.
  Shape new_shape = shape();
  Shape* subshape = ShapeUtil::GetMutableSubshape(&new_shape, shape_index);
  TF_CHECK_OK(LayoutUtil::ValidateLayoutForShape(new_layout, *subshape));
  *subshape->mutable_layout() = new_layout;
  auto result = MakeUnique<Literal>(new_shape);
  TF_CHECK_OK(result->CopyFrom(*this));
  return result;
}

std::unique_ptr<Literal> Literal::Relayout(
    const Shape& shape_with_layout) const {
  CHECK(ShapeUtil::Compatible(shape_with_layout, shape()))
      << "Given shape_with_layout " << ShapeUtil::HumanString(shape_with_layout)
      << " not compatible with literal shape "
      << ShapeUtil::HumanString(shape());
  std::unique_ptr<Literal> result = CreateFromShape(shape_with_layout);
  ShapeUtil::ForEachSubshape(
      result->shape(),
      [this, &result](const Shape& subshape, const ShapeIndex& index) {
        if (ShapeUtil::IsArray(subshape)) {
          TF_CHECK_OK(result->CopyFrom(*this,
                                       /*dest_shape_index=*/index,
                                       /*src_shape_index=*/index));
        }
      });
  return result;
}

StatusOr<std::unique_ptr<Literal>> Literal::Reshape(
    tensorflow::gtl::ArraySlice<int64> dimensions) const {
  if (!ShapeUtil::IsArray(shape())) {
    return InvalidArgument("Reshape does not support tuples.");
  }
  std::unique_ptr<Literal> output;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape().layout())) {
    output =
        Relayout(LayoutUtil::GetDefaultLayoutForRank(ShapeUtil::Rank(shape())));
  } else {
    output = CloneToUnique();
  }
  // Because the layout is monotonic, we can simply reuse the same sequence of
  // values without changing their order.
  output->shape_ = ShapeUtil::MakeShape(shape().element_type(), dimensions);

  int64 elements_before = ShapeUtil::ElementsIn(shape());
  int64 elements_after = ShapeUtil::ElementsIn(output->shape());
  if (elements_before != elements_after) {
    return InvalidArgument(
        "Shapes before and after Literal::Reshape have different numbers "
        "of elements: %s vs %s.",
        ShapeUtil::HumanString(shape()).c_str(),
        ShapeUtil::HumanString(output->shape()).c_str());
  }
  return std::move(output);
}

std::unique_ptr<Literal> Literal::Transpose(
    tensorflow::gtl::ArraySlice<int64> permutation) const {
  CHECK(ShapeUtil::IsArray(shape())) << "Tuple is not supported for transpose";
  CHECK(IsPermutation(permutation, ShapeUtil::Rank(shape())))
      << "Given permutation is not a permutation of dimension numbers";
  // To transpose the array, we just permute the dimensions and layout, and
  // do a straight memory copy of the raw data set.
  // This is considerably faster than iterating over every array element using
  // the EachCell<>() and Set<>() APIs.
  std::vector<int64> inverse_permutation = InversePermutation(permutation);
  Shape permuted_shape =
      ShapeUtil::PermuteDimensions(inverse_permutation, shape());
  // Replace the layout with one affine to this shape, such that a
  // transpose operation can be performed by leaving the flat values
  // representation intact.
  // For example, consider the shape F32[11,8]{1,0} under a {1,0} permutation.
  // The shape with affine layout resulting from that operation will be
  // F32[8,11]{0,1}, since it leaves the original most minor (the 8 sized), the
  // most minor.
  //
  // Essentially, given MinMaj(Di) the position of the Di dimension within the
  // minor to major vector, and given T(Di) the index that the original Di
  // dimension has within the transposed array, a layout is affine if
  // MinMaj(Di) == TMinMaj(T(Di)), with TMinMaj() being the minor to major
  // vector of the affine layout.
  CHECK(LayoutUtil::IsDenseArray(permuted_shape));
  Layout* layout = permuted_shape.mutable_layout();
  layout->clear_minor_to_major();
  for (auto index : LayoutUtil::MinorToMajor(shape())) {
    layout->add_minor_to_major(inverse_permutation[index]);
  }
  std::unique_ptr<Literal> new_literal = CreateFromShape(permuted_shape);
  DCHECK_GE(ShapeUtil::ByteSizeOf(new_literal->shape()),
            ShapeUtil::ByteSizeOf(shape()));
  std::memcpy(new_literal->root_piece().buffer(), root_piece().buffer(),
              root_piece().size_bytes());
  return new_literal;
}

std::unique_ptr<Literal> Literal::Slice(
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices) const {
  CHECK(ShapeUtil::IsArray(shape())) << "tuple is not supported for slice";

  DimensionVector result_dimensions;
  for (int64 dnum = 0; dnum < ShapeUtil::Rank(shape()); ++dnum) {
    CHECK_GE(start_indices[dnum], 0);
    CHECK_LE(limit_indices[dnum], shape().dimensions(dnum));
    int64 dimension = limit_indices[dnum] - start_indices[dnum];
    CHECK_GT(dimension, 0);
    result_dimensions.push_back(dimension);
  }
  const auto result_shape =
      ShapeUtil::MakeShapeWithLayout(shape().element_type(), result_dimensions,
                                     LayoutUtil::MinorToMajor(shape()));

  auto result_literal = MakeUnique<Literal>(result_shape);

  DimensionVector new_indices(ShapeUtil::Rank(result_shape));
  switch (result_shape.element_type()) {
    case F32:
      result_literal->EachCell<float>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, float /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            float value = Get<float>(new_indices);
            result_literal->Set<float>(indices, value);
          });
      return result_literal;
    case C64:
      result_literal->EachCell<complex64>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, complex64 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            complex64 value = Get<complex64>(new_indices);
            result_literal->Set<complex64>(indices, value);
          });
      return result_literal;
    case S32:
      result_literal->EachCell<int32>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, int32 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            int32 value = Get<int32>(new_indices);
            result_literal->Set<int32>(indices, value);
          });
      return result_literal;
    case U32:
      result_literal->EachCell<uint32>(
          [&](tensorflow::gtl::ArraySlice<int64> indices, uint32 /*value*/) {
            for (int64 i = 0; i < ShapeUtil::Rank(result_shape); ++i) {
              new_indices[i] = indices[i] + start_indices[i];
            }
            uint32 value = Get<uint32>(new_indices);
            result_literal->Set<uint32>(indices, value);
          });
      return result_literal;
    default:
      LOG(FATAL) << "not yet implemented: "
                 << PrimitiveType_Name(result_shape.element_type());
  }
}

Literal Literal::Clone() const {
  Literal result(shape());
  TF_CHECK_OK(result.CopyFrom(*this));
  return result;
}

std::unique_ptr<Literal> Literal::CloneToUnique() const {
  auto result = MakeUnique<Literal>(shape());
  TF_CHECK_OK(result->CopyFrom(*this));
  return result;
}

string Literal::GetAsString(tensorflow::gtl::ArraySlice<int64> multi_index,
                            const ShapeIndex& shape_index) const {
  const Shape& subshape = ShapeUtil::GetSubshape(shape(), shape_index);
  CHECK(LayoutUtil::IsDenseArray(subshape));
  switch (subshape.element_type()) {
    case PRED:
      return Get<bool>(multi_index, shape_index) ? "true" : "false";
    case S8:
      return StrCat(Get<int8>(multi_index, shape_index));
    case S16:
      return StrCat(Get<int16>(multi_index, shape_index));
    case S32:
      return StrCat(Get<int32>(multi_index, shape_index));
    case S64:
      return StrCat(Get<int64>(multi_index, shape_index));
    case U8:
      return StrCat(Get<uint8>(multi_index, shape_index));
    case U16:
      return StrCat(Get<uint16>(multi_index, shape_index));
    case U32:
      return StrCat(Get<uint32>(multi_index, shape_index));
    case U64:
      return StrCat(Get<uint64>(multi_index, shape_index));
    case F16:
      return StrCat(Get<half>(multi_index, shape_index));
    case F32:
      return StrCat(Get<float>(multi_index, shape_index));
    case BF16:
      return StrCat(
          static_cast<float>(Get<bfloat16>(multi_index, shape_index)));
    case F64:
      return StrCat(Get<double>(multi_index, shape_index));
    case C64: {
      complex64 c = Get<complex64>(multi_index, shape_index);
      return StrCat("(", c.real(), ", ", c.imag(), ")");
    }
    default:
      LOG(FATAL) << PrimitiveType_Name(subshape.element_type());
  }
}

string Literal::GetSparseElementAsString(int64 sparse_element_number,
                                         const ShapeIndex& shape_index) const {
  const Shape& subshape = ShapeUtil::GetSubshape(shape(), shape_index);
  CHECK(LayoutUtil::IsSparseArray(subshape));
  switch (subshape.element_type()) {
    case PRED:
      return GetSparseElement<bool>(sparse_element_number, shape_index)
                 ? "true"
                 : "false";
    case S8:
      return StrCat(GetSparseElement<int8>(sparse_element_number, shape_index));
    case S16:
      return StrCat(
          GetSparseElement<int16>(sparse_element_number, shape_index));
    case S32:
      return StrCat(
          GetSparseElement<int32>(sparse_element_number, shape_index));
    case S64:
      return StrCat(
          GetSparseElement<int64>(sparse_element_number, shape_index));
    case U8:
      return StrCat(
          GetSparseElement<uint8>(sparse_element_number, shape_index));
    case U16:
      return StrCat(
          GetSparseElement<uint16>(sparse_element_number, shape_index));
    case U32:
      return StrCat(
          GetSparseElement<uint32>(sparse_element_number, shape_index));
    case U64:
      return StrCat(
          GetSparseElement<uint64>(sparse_element_number, shape_index));
    case F16:
      return StrCat(GetSparseElement<half>(sparse_element_number, shape_index));
    case F32:
      return StrCat(
          GetSparseElement<float>(sparse_element_number, shape_index));
    case BF16:
      return StrCat(static_cast<float>(
          GetSparseElement<bfloat16>(sparse_element_number, shape_index)));
    case F64:
      return StrCat(
          GetSparseElement<double>(sparse_element_number, shape_index));
    case C64: {
      complex64 c =
          GetSparseElement<complex64>(sparse_element_number, shape_index);
      return StrCat("(", c.real(), ", ", c.imag(), ")");
    }
    default:
      LOG(FATAL) << "Invalid element type for sparse arrays: "
                 << PrimitiveType_Name(subshape.element_type());
  }
}

StatusOr<int64> Literal::GetIntegralAsS64(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  CHECK(LayoutUtil::IsDenseArray(shape()));
  switch (shape().element_type()) {
    case PRED:
      return Get<bool>(multi_index);
    case U8:
      return Get<uint8>(multi_index);
    case S32:
      return Get<int32>(multi_index);
    case S64:
      return Get<int64>(multi_index);
    case U32:
      return Get<uint32>(multi_index);
    case U64:
      return Get<uint64>(multi_index);
    default:
      return FailedPrecondition(
          "Array element type is not integral: %s",
          PrimitiveType_Name(shape().element_type()).c_str());
  }
}

tensorflow::gtl::ArraySlice<int64> Literal::GetSparseIndex(
    int64 sparse_element_number, const ShapeIndex& shape_index) const {
  const Piece& p = piece(shape_index);
  CHECK_GE(sparse_element_number, 0);
  CHECK_LT(sparse_element_number, p.sparse_indices()->index_count());
  return p.sparse_indices()->At(sparse_element_number);
}

void Literal::SortSparseElements(const ShapeIndex& shape_index) {
  piece(shape_index).SortSparseElements();
}

Literal Literal::GetFirstScalarLiteral() const {
  CHECK(ShapeUtil::IsArray(shape_));
  CHECK_GT(ShapeUtil::ElementsIn(shape_), 0);
  switch (shape_.element_type()) {
    case PRED:
      return std::move(*Literal::CreateR0<bool>(GetFirstElement<bool>()));
    // 8 bit types.
    case S8:
      return std::move(*Literal::CreateR0<int8>(GetFirstElement<int8>()));
    case U8:
      return std::move(*Literal::CreateR0<uint8>(GetFirstElement<uint8>()));
    // 16 bit types.
    case BF16:
      return std::move(
          *Literal::CreateR0<bfloat16>(GetFirstElement<bfloat16>()));
    case F16:
      return std::move(*Literal::CreateR0<half>(GetFirstElement<half>()));
    case S16:
      return std::move(*Literal::CreateR0<int16>(GetFirstElement<int16>()));
    case U16:
      return std::move(*Literal::CreateR0<uint16>(GetFirstElement<uint16>()));
    // 32 bit types.
    case F32:
      return std::move(*Literal::CreateR0<float>(GetFirstElement<float>()));
    case S32:
      return std::move(*Literal::CreateR0<int32>(GetFirstElement<int32>()));
    case U32:
      return std::move(*Literal::CreateR0<uint32>(GetFirstElement<uint32>()));
    // 64 bit types.
    case C64:
      return std::move(
          *Literal::CreateR0<complex64>(GetFirstElement<complex64>()));
    case F64:
      return std::move(*Literal::CreateR0<double>(GetFirstElement<double>()));
    case S64:
      return std::move(*Literal::CreateR0<int64>(GetFirstElement<int64>()));
    case U64:
      return std::move(*Literal::CreateR0<uint64>(GetFirstElement<uint64>()));
    default:
      LOG(FATAL) << "Unhandled primitive type " << shape_.element_type();
  }
}

void Literal::Piece::SortSparseElements() {
  switch (subshape().element_type()) {
    case PRED:
      SortSparseElementsInternal<bool>();
      break;
    case S8:
      SortSparseElementsInternal<int8>();
      break;
    case U8:
      SortSparseElementsInternal<uint8>();
      break;
    case S16:
      SortSparseElementsInternal<int16>();
      break;
    case U16:
      SortSparseElementsInternal<uint16>();
      break;
    case S32:
      SortSparseElementsInternal<int32>();
      break;
    case U32:
      SortSparseElementsInternal<uint32>();
      break;
    case S64:
      SortSparseElementsInternal<int64>();
      break;
    case U64:
      SortSparseElementsInternal<uint64>();
      break;
    case F32:
      SortSparseElementsInternal<float>();
      break;
    case F64:
      SortSparseElementsInternal<double>();
      break;
    case C64:
      SortSparseElementsInternal<complex64>();
      break;
    case F16:
      SortSparseElementsInternal<half>();
      break;
    case BF16:
      SortSparseElementsInternal<bfloat16>();
      break;
    default:
      LOG(FATAL) << "Element type not valid for sparse array: "
                 << PrimitiveType_Name(subshape().element_type());
  }
}

template <typename NativeT>
void Literal::Piece::SortSparseElementsInternal() {
  CHECK(LayoutUtil::IsSparseArray(subshape()));
  int64 num_elements = sparse_indices()->index_count();
  auto values = data<NativeT>();
  CHECK_LE(num_elements, values.size());
  sparse_indices()->SortWithValues(
      tensorflow::gtl::MutableArraySlice<NativeT>(values.data(), num_elements));
}

namespace {

void ToStringHelper(const Literal& literal, const ShapeIndex& shape_index,
                    bool print_layout, std::vector<string>* pieces) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);

  auto shape_to_string = [print_layout](const Shape& shape) {
    if (print_layout) {
      return ShapeUtil::HumanStringWithLayout(shape);
    } else {
      return ShapeUtil::HumanString(shape);
    }
  };

  // TODO(b/32894291): refactor this code to reduce code duplication.
  if (ShapeUtil::IsTuple(subshape)) {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back(" (\n");
    std::vector<string> tuple_pieces;
    for (int i = 0; i < ShapeUtil::TupleElementCount(subshape); ++i) {
      ShapeIndex element_index = shape_index;
      element_index.push_back(i);
      std::vector<string> element_pieces;
      ToStringHelper(literal, element_index, print_layout, &element_pieces);
      tuple_pieces.push_back(tensorflow::str_util::Join(element_pieces, ""));
    }
    pieces->push_back(tensorflow::str_util::Join(tuple_pieces, ",\n"));
    pieces->push_back("\n)");
    return;
  }

  if (LayoutUtil::IsSparseArray(subshape)) {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back("{");
    int64 rank = ShapeUtil::Rank(subshape);
    int64 num_elements = literal.sparse_element_count();
    for (int64 i = 0; i < num_elements; ++i) {
      if (i > 0) {
        pieces->push_back(", ");
      }
      if (rank == 1) {
        pieces->push_back(StrCat(literal.GetSparseIndex(i)[0]));
        pieces->push_back(": ");
      } else {
        pieces->push_back("[");
        pieces->push_back(
            tensorflow::str_util::Join(literal.GetSparseIndex(i), ", "));
        pieces->push_back("]: ");
      }
      pieces->push_back(literal.GetSparseElementAsString(i));
    }
    pieces->push_back("}");
    return;
  }

  CHECK(LayoutUtil::IsDenseArray(subshape));

  auto element_to_string =
      [&](tensorflow::gtl::ArraySlice<int64> indices) -> string {
    PrimitiveType element_type = subshape.element_type();
    if (element_type == PRED) {
      // We display predicates in a densely packed form.
      return literal.Get<bool>(indices, shape_index) ? "1" : "0";
    }
    return ((!indices.empty() && indices.back() > 0) ? ", " : "") +
           literal.GetAsString(indices, shape_index);
  };

  if (ShapeUtil::Rank(subshape) == 0) {
    pieces->push_back(literal.GetAsString({}, shape_index));
  } else if (ShapeUtil::Rank(subshape) == 1) {
    pieces->push_back("{");
    for (int64 i0 = 0; i0 < subshape.dimensions(0); ++i0) {
      pieces->push_back(element_to_string({i0}));
    }
    pieces->push_back("}");
  } else if (ShapeUtil::Rank(subshape) == 2) {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back(" {\n");
    for (int64 i0 = 0; i0 < subshape.dimensions(0); ++i0) {
      pieces->push_back("  { ");
      for (int64 i1 = 0; i1 < subshape.dimensions(1); ++i1) {
        pieces->push_back(element_to_string({i0, i1}));
      }
      pieces->push_back(" ");
      pieces->push_back(i0 == subshape.dimensions(0) - 1 ? "}\n" : "},\n");
    }
    pieces->push_back("}");
  } else if (ShapeUtil::Rank(subshape) == 3) {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back(" {\n");
    for (int64 i0 = 0; i0 < subshape.dimensions(0); ++i0) {
      pieces->push_back(i0 > 0 ? ",\n{" : "{");
      for (int64 i1 = 0; i1 < subshape.dimensions(1); ++i1) {
        pieces->push_back(i1 > 0 ? ",\n  { " : " { ");
        for (int64 i2 = 0; i2 < subshape.dimensions(2); ++i2) {
          pieces->push_back(element_to_string({i0, i1, i2}));
        }
        pieces->push_back(" }");
      }
      pieces->push_back(" }");
    }
    pieces->push_back("\n}");
  } else if (ShapeUtil::Rank(subshape) == 4) {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back(" {\n");
    for (int64 i0 = 0; i0 < subshape.dimensions(0); ++i0) {
      pieces->push_back(Printf("  {  /*i0=%lld*/\n", i0));
      for (int64 i1 = 0; i1 < subshape.dimensions(1); ++i1) {
        pieces->push_back(Printf("    {  /*i1=%lld*/\n", i1));
        for (int64 i2 = 0; i2 < subshape.dimensions(2); ++i2) {
          pieces->push_back("      {");
          for (int64 i3 = 0; i3 < subshape.dimensions(3); ++i3) {
            pieces->push_back(element_to_string({i0, i1, i2, i3}));
          }
          pieces->push_back(i2 == subshape.dimensions(2) - 1 ? "}\n" : "},\n");
        }
        pieces->push_back(i1 == subshape.dimensions(1) - 1 ? "    }\n"
                                                           : "    },\n");
      }
      pieces->push_back(i0 == subshape.dimensions(0) - 1 ? "  }\n" : "  },\n");
    }
    pieces->push_back("}");
  } else if (ShapeUtil::Rank(subshape) == 5) {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back(" {\n");
    for (int64 i0 = 0; i0 < subshape.dimensions(0); ++i0) {
      pieces->push_back(Printf("  {  /*i0=%lld*/\n", i0));
      for (int64 i1 = 0; i1 < subshape.dimensions(1); ++i1) {
        pieces->push_back(Printf("    {  /*i1=%lld*/\n", i1));
        for (int64 i2 = 0; i2 < subshape.dimensions(2); ++i2) {
          pieces->push_back(Printf("      {  /*i2=%lld*/\n", i2));
          for (int64 i3 = 0; i3 < subshape.dimensions(3); ++i3) {
            pieces->push_back("        {");
            for (int64 i4 = 0; i4 < subshape.dimensions(4); ++i4) {
              pieces->push_back(element_to_string({i0, i1, i2, i3, i4}));
            }
            pieces->push_back(i3 == subshape.dimensions(3) - 1 ? "}\n"
                                                               : "},\n");
          }
          pieces->push_back(i2 == subshape.dimensions(2) - 1 ? "      }\n"
                                                             : "      },\n");
        }
        pieces->push_back(i1 == subshape.dimensions(1) - 1 ? "    }\n"
                                                           : "    },\n");
      }
      pieces->push_back(i0 == subshape.dimensions(0) - 1 ? "  }\n" : "  },\n");
    }
    pieces->push_back("}");
  } else {
    pieces->push_back(shape_to_string(subshape));
    pieces->push_back(" {");
    literal.EachCellAsString(
        [&](tensorflow::gtl::ArraySlice<int64> indices, const string& value) {
          pieces->push_back(" ");
          pieces->push_back(value);
        });
    pieces->push_back("}");
  }
}

}  // namespace

int64 Literal::sparse_element_count() const {
  CHECK(LayoutUtil::IsSparseArray(shape()));
  return sparse_indices()->index_count();
}

string Literal::ToString(bool print_layout) const {
  std::vector<string> pieces;
  ToStringHelper(*this, {}, print_layout, &pieces);
  return tensorflow::str_util::Join(pieces, "");
}

/* static */ std::unique_ptr<Literal> Literal::MakeTuple(
    tensorflow::gtl::ArraySlice<const Literal*> elements) {
  std::vector<Shape> element_shapes;
  for (const Literal* element : elements) {
    element_shapes.push_back(element->shape());
  }
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeTupleShape(element_shapes));
  for (int i = 0; i < elements.size(); ++i) {
    TF_CHECK_OK(literal->CopyFrom(*elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ std::unique_ptr<Literal> Literal::MakeTupleOwned(
    std::vector<std::unique_ptr<Literal>> elements) {
  std::vector<Shape> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(element->shape());
  }
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeTupleShape(element_shapes));
  for (int64 i = 0; i < elements.size(); ++i) {
    TF_CHECK_OK(
        literal->MoveFrom(std::move(*elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

void Literal::EachCellAsString(
    const std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                             const string& value)>& per_cell) const {
  if (ShapeUtil::HasZeroElements(shape())) {
    return;
  }
  std::vector<int64> indices = IndexUtil::LinearIndexToMultidimensionalIndex(
      shape(), /*linear_index=*/0);
  do {
    per_cell(indices, GetAsString(indices));
  } while (IndexUtil::BumpIndices(shape(), &indices));
}

namespace {
template <typename NativeSrcT, typename NativeDestT>
std::unique_ptr<Literal> ConvertBetweenNativeTypes(const Literal& src_literal) {
  CHECK(ShapeUtil::IsArray(src_literal.shape()));
  auto result_literal = MakeUnique<Literal>(ShapeUtil::ChangeElementType(
      src_literal.shape(),
      primitive_util::NativeToPrimitiveType<NativeDestT>()));
  auto src_data = src_literal.data<NativeSrcT>();
  auto dest_data = result_literal->template data<NativeDestT>();
  int64 num_elements = src_literal.element_count();

  for (int64 i = 0; i < num_elements; ++i) {
    dest_data[i] = static_cast<NativeDestT>(src_data[i]);
  }
  return result_literal;
}

template <PrimitiveType primitive_src_type>
std::unique_ptr<Literal> ConvertToC64(const Literal& src_literal) {
  CHECK(ShapeUtil::IsArray(src_literal.shape()));
  auto result_literal = MakeUnique<Literal>(
      ShapeUtil::ChangeElementType(src_literal.shape(), C64));
  using NativeSrcT =
      typename primitive_util::PrimitiveTypeToNative<primitive_src_type>::type;
  tensorflow::gtl::ArraySlice<NativeSrcT> src_data =
      src_literal.data<NativeSrcT>();
  tensorflow::gtl::MutableArraySlice<complex64> dest_data =
      result_literal->data<complex64>();
  int64 num_elements = src_literal.element_count();
  for (int64 i = 0; i < num_elements; ++i) {
    dest_data[i] = complex64(static_cast<float>(src_data[i]), 0);
  }
  return result_literal;
}

template <PrimitiveType primitive_src_type, PrimitiveType primitive_dest_type>
std::unique_ptr<Literal> ConvertIfTypesMatch(const Literal& src_literal) {
  CHECK_EQ(primitive_src_type, src_literal.shape().element_type());
  return ConvertBetweenNativeTypes<
      typename primitive_util::PrimitiveTypeToNative<primitive_src_type>::type,
      typename primitive_util::PrimitiveTypeToNative<
          primitive_dest_type>::type>(src_literal);
}

template <PrimitiveType primitive_src_type>
StatusOr<std::unique_ptr<Literal>> ConvertIfDestTypeMatches(
    const Literal& src_literal, PrimitiveType primitive_dest_type) {
  switch (primitive_dest_type) {
#define CONVERT_IF_TYPES_MATCH(type) \
  case (type):                       \
    return ConvertIfTypesMatch<primitive_src_type, (type)>(src_literal);
    CONVERT_IF_TYPES_MATCH(PRED)
    CONVERT_IF_TYPES_MATCH(S8)
    CONVERT_IF_TYPES_MATCH(S32)
    CONVERT_IF_TYPES_MATCH(S64)
    CONVERT_IF_TYPES_MATCH(U8)
    CONVERT_IF_TYPES_MATCH(U32)
    CONVERT_IF_TYPES_MATCH(U64)
    CONVERT_IF_TYPES_MATCH(F16)
    CONVERT_IF_TYPES_MATCH(F32)
    CONVERT_IF_TYPES_MATCH(F64)
    CONVERT_IF_TYPES_MATCH(BF16)
#undef CONVERT_IF_TYPES_MATCH
    case C64:
      return ConvertToC64<primitive_src_type>(src_literal);
    // Other types are not yet supported.
    default:
      return InvalidArgument(
          "Unimplemented: Convert from type %s to type %s",
          PrimitiveType_Name(src_literal.shape().element_type()).c_str(),
          PrimitiveType_Name(primitive_dest_type).c_str());
  }
}

}  // namespace

StatusOr<std::unique_ptr<Literal>> Literal::Convert(
    PrimitiveType primitive_dest_type) const {
  TF_RET_CHECK(ShapeUtil::IsArray(shape()));
  switch (shape().element_type()) {
#define CONVERT_IF_DEST_TYPE_MATCHES(type) \
  case (type):                             \
    return ConvertIfDestTypeMatches<(type)>(*this, primitive_dest_type);
    CONVERT_IF_DEST_TYPE_MATCHES(PRED)
    CONVERT_IF_DEST_TYPE_MATCHES(S8)
    CONVERT_IF_DEST_TYPE_MATCHES(S32)
    CONVERT_IF_DEST_TYPE_MATCHES(S64)
    CONVERT_IF_DEST_TYPE_MATCHES(U8)
    CONVERT_IF_DEST_TYPE_MATCHES(U32)
    CONVERT_IF_DEST_TYPE_MATCHES(U64)
    CONVERT_IF_DEST_TYPE_MATCHES(F16)
    CONVERT_IF_DEST_TYPE_MATCHES(F32)
    CONVERT_IF_DEST_TYPE_MATCHES(F64)
    CONVERT_IF_DEST_TYPE_MATCHES(BF16)
#undef CONVERT_IF_DEST_TYPE_MATCHES
      // Other types are not yet supported.
    default:
      return InvalidArgument("Unimplemented: Convert from type %s to type %s",
                             PrimitiveType_Name(shape().element_type()).c_str(),
                             PrimitiveType_Name(primitive_dest_type).c_str());
  }
}

template <typename NativeT>
bool Literal::Piece::EqualElementsInternal(
    const Literal::Piece& other, std::vector<int64>* multi_index) const {
  if (multi_index->size() == ShapeUtil::Rank(subshape())) {
    return (Get<NativeT>(*multi_index) == other.Get<NativeT>(*multi_index));
  }
  for (int64 i = 0; i < subshape().dimensions(multi_index->size()); ++i) {
    multi_index->push_back(i);
    if (!EqualElementsInternal<NativeT>(other, multi_index)) {
      return false;
    }
    multi_index->pop_back();
  }
  return true;
}

bool Literal::Piece::EqualElements(const Literal::Piece& other) const {
  DCHECK(ShapeUtil::Compatible(subshape(), other.subshape()));

  std::vector<int64> multi_index;
  switch (subshape().element_type()) {
    case PRED:
      return EqualElementsInternal<bool>(other, &multi_index);
    case U8:
      return EqualElementsInternal<uint8>(other, &multi_index);
    case S32:
      return EqualElementsInternal<int32>(other, &multi_index);
    case S64:
      return EqualElementsInternal<int64>(other, &multi_index);
    case U32:
      return EqualElementsInternal<uint32>(other, &multi_index);
    case U64:
      return EqualElementsInternal<uint64>(other, &multi_index);
    case F32:
      return EqualElementsInternal<float>(other, &multi_index);
    case F64:
      return EqualElementsInternal<double>(other, &multi_index);
    case F16:
      return EqualElementsInternal<half>(other, &multi_index);
    case BF16:
      return EqualElementsInternal<bfloat16>(other, &multi_index);
    case C64:
      return EqualElementsInternal<complex64>(other, &multi_index);
    default:
      LOG(FATAL) << "Unimplemented: Literal::Piece::EqualElements for type "
                 << PrimitiveType_Name(subshape().element_type());
  }
}

bool Literal::operator==(const Literal& other) const {
  if (!ShapeUtil::Compatible(shape(), other.shape())) {
    return false;
  }
  for (const auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    const Piece& piece = pair.second;
    if (!ShapeUtil::IsArray(piece.subshape())) {
      continue;
    }

    const Piece& other_piece = other.piece(index);
    if (!piece.EqualElements(other_piece)) {
      return false;
    }
  }
  return true;
}

namespace {

template <typename NativeT>
static bool AllElementsEqualValue(tensorflow::gtl::ArraySlice<NativeT> data,
                                  NativeT value) {
  for (int64 i = 0; i < data.size(); ++i) {
    if (data[i] != value) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool Literal::IsAll(int8 value) const {
  for (const auto& pair : pieces_) {
    const Piece& piece = pair.second;
    if (!ShapeUtil::IsArray(piece.subshape())) {
      continue;
    }

    auto piece_is_all = [&]() {
      switch (shape().element_type()) {
        case U8:
          if (value >= 0) {
            return AllElementsEqualValue<uint8>(piece.data<uint8>(), value);
          }
          return false;
        case U32:
          if (value >= 0) {
            return AllElementsEqualValue<uint32>(piece.data<uint32>(), value);
          }
          return false;
        case U64:
          if (value >= 0) {
            return AllElementsEqualValue<uint64>(piece.data<uint64>(), value);
          }
          return false;
        case S8:
          return AllElementsEqualValue<int8>(piece.data<int8>(), value);
        case S32:
          return AllElementsEqualValue<int32>(piece.data<int32>(), value);
        case S64:
          return AllElementsEqualValue<int64>(piece.data<int64>(), value);
        case F32:
          return AllElementsEqualValue<float>(piece.data<float>(), value);
        case F64:
          return AllElementsEqualValue<double>(piece.data<double>(), value);
        case F16:
          return AllElementsEqualValue<half>(piece.data<half>(),
                                             static_cast<half>(value));
        case BF16:
          return AllElementsEqualValue<bfloat16>(piece.data<bfloat16>(),
                                                 static_cast<bfloat16>(value));
        case PRED:
          if (value == 0) {
            return AllElementsEqualValue<bool>(piece.data<bool>(), false);
          }
          if (value == 1) {
            return AllElementsEqualValue<bool>(piece.data<bool>(), true);
          }
          return false;
        default:
          return false;
      }
      return false;
    };

    if (!piece_is_all()) {
      return false;
    }
  }
  return true;
}

bool Literal::IsAllFloat(float value) const {
  for (const auto& pair : pieces_) {
    const Piece& piece = pair.second;
    if (!ShapeUtil::IsArray(piece.subshape())) {
      continue;
    }

    auto piece_is_all = [&]() {
      switch (shape().element_type()) {
        case F32:
          return AllElementsEqualValue<float>(piece.data<float>(), value);
        case F64:
          return AllElementsEqualValue<double>(piece.data<double>(), value);
        case F16:
          return AllElementsEqualValue<half>(piece.data<half>(),
                                             static_cast<half>(value));
        case BF16:
          return AllElementsEqualValue<bfloat16>(piece.data<bfloat16>(),
                                                 static_cast<bfloat16>(value));
        default:
          return false;
      }
    };
    if (!piece_is_all()) {
      return false;
    }
  }
  return true;
}

bool Literal::IsAllComplex(complex64 value) const {
  switch (shape().element_type()) {
    case C64:
      return AllElementsEqualValue<complex64>(root_piece().data<complex64>(),
                                              value);
    default:
      return false;
  }
}

bool Literal::IsAllFirst() const {
  for (const auto& pair : pieces_) {
    const Piece& piece = pair.second;
    if (!ShapeUtil::IsArray(piece.subshape())) {
      continue;
    }

    // Empty shapes are not all the first element since there is no first
    // element.
    if (ShapeUtil::HasZeroElements(piece.subshape())) {
      return false;
    }
    auto piece_is_all = [&]() {
      switch (piece.subshape().element_type()) {
        case PRED: {
          auto data = piece.data<bool>();
          return AllElementsEqualValue<bool>(data, data[0]);
        }
        // 8 bit types
        case S8: {
          auto data = piece.data<int8>();
          return AllElementsEqualValue<int8>(data, data[0]);
        }
        case U8: {
          auto data = piece.data<uint8>();
          return AllElementsEqualValue<uint8>(data, data[0]);
        }
        // 16 bit types
        case BF16: {
          auto data = piece.data<bfloat16>();
          return AllElementsEqualValue<bfloat16>(data, data[0]);
        }
        case F16: {
          auto data = piece.data<half>();
          return AllElementsEqualValue<half>(data, data[0]);
        }
        case S16: {
          auto data = piece.data<int16>();
          return AllElementsEqualValue<int16>(data, data[0]);
        }
        case U16: {
          auto data = piece.data<uint16>();
          return AllElementsEqualValue<uint16>(data, data[0]);
        }
        // 32 bit types
        case F32: {
          auto data = piece.data<float>();
          return AllElementsEqualValue<float>(data, data[0]);
        }
        case U32: {
          auto data = piece.data<uint32>();
          return AllElementsEqualValue<uint32>(data, data[0]);
        }
        case S32: {
          auto data = piece.data<int32>();
          return AllElementsEqualValue<int32>(data, data[0]);
        }
        // 64 bit types
        case C64: {
          auto data = piece.data<complex64>();
          return AllElementsEqualValue<complex64>(data, data[0]);
        }
        case F64: {
          auto data = piece.data<double>();
          return AllElementsEqualValue<double>(data, data[0]);
        }
        case S64: {
          auto data = piece.data<int64>();
          return AllElementsEqualValue<int64>(data, data[0]);
        }
        case U64: {
          auto data = piece.data<uint64>();
          return AllElementsEqualValue<uint64>(data, data[0]);
        }
        default:
          return false;
      }
    };

    if (!piece_is_all()) {
      return false;
    }
  }
  return true;
}

bool Literal::IsZero(tensorflow::gtl::ArraySlice<int64> indices) const {
  CHECK(ShapeUtil::IsArray(shape()));
  switch (shape().element_type()) {
    case U8:
      return Get<uint8>(indices) == 0;
    case U32:
      return Get<uint32>(indices) == 0;
    case U64:
      return Get<uint64>(indices) == 0;
    case S8:
      return Get<int8>(indices) == 0;
    case S32:
      return Get<int32>(indices) == 0;
    case S64:
      return Get<int64>(indices) == 0;
    case F32:
      return Get<float>(indices) == 0.0f;
    case F64:
      return Get<double>(indices) == 0.0;
    case C64:
      return Get<complex64>(indices) == complex64(0.0f, 0.0f);
    case F16:
      return Get<half>(indices) == static_cast<half>(0.0f);
    case BF16:
      return Get<bfloat16>(indices) == static_cast<bfloat16>(0.0f);
    case PRED:
      return Get<bool>(indices) == false;
    default:
      LOG(FATAL) << "Input literal must be an array.";
  }
}

namespace {

template <typename RepeatedFieldT, typename NativeT>
void CopyToRepeatedField(RepeatedFieldT* dest,
                         const tensorflow::gtl::ArraySlice<NativeT> src) {
  *dest = RepeatedFieldT(src.begin(), src.end());
}

}  // namespace

void Literal::Piece::WriteToProto(LiteralProto* proto) const {
  *proto->mutable_shape() = subshape();
  switch (subshape().element_type()) {
    case PRED:
      CopyToRepeatedField(proto->mutable_preds(), data<bool>());
      break;
    case U8:
      proto->set_u8s(static_cast<const unsigned char*>(data<uint8>().data()),
                     element_count());
      break;
    case U32:
      CopyToRepeatedField(proto->mutable_u32s(), data<uint32>());
      break;
    case U64:
      CopyToRepeatedField(proto->mutable_u64s(), data<uint64>());
      break;
    case S32:
      CopyToRepeatedField(proto->mutable_s32s(), data<int32>());
      break;
    case S64:
      CopyToRepeatedField(proto->mutable_s64s(), data<int64>());
      break;
    case F16:
      *proto->mutable_f16s() = string(
          reinterpret_cast<const char*>(data<half>().data()), size_bytes());
      if (!kLittleEndian) {
        ConvertEndianShort(const_cast<char*>(proto->mutable_f16s()->data()),
                           proto->f16s().size());
      }
      break;
    case BF16:
      *proto->mutable_bf16s() = string(
          reinterpret_cast<const char*>(data<bfloat16>().data()), size_bytes());
      if (!kLittleEndian) {
        ConvertEndianShort(const_cast<char*>(proto->mutable_bf16s()->data()),
                           proto->bf16s().size());
      }
      break;
    case F32:
      CopyToRepeatedField(proto->mutable_f32s(), data<float>());
      break;
    case F64:
      CopyToRepeatedField(proto->mutable_f64s(), data<double>());
      break;
    case C64:
      for (complex64 value : data<complex64>()) {
        proto->add_c64s(value.real());
        proto->add_c64s(value.imag());
      }
      break;
    case TUPLE:
      // Nothing to do but assign the shape which is done above.
      return;
    default:
      LOG(FATAL) << "Unhandled primitive type " << subshape().element_type();
  }
}

const void* Literal::Piece::untyped_data() const {
  CHECK(ShapeUtil::IsArray(subshape())) << ShapeUtil::HumanString(subshape());
  return buffer();
}

void* Literal::Piece::untyped_data() {
  CHECK(ShapeUtil::IsArray(subshape())) << ShapeUtil::HumanString(subshape());
  return buffer();
}

namespace {

template <typename RepeatedFieldT, typename NativeT>
Status CopyFromRepeatedField(tensorflow::gtl::MutableArraySlice<NativeT> dest,
                             const RepeatedFieldT& src) {
  if (dest.size() != src.size()) {
    return InvalidArgument(
        "Expected %lu elements in LiteralProto repeated field, has %d",
        dest.size(), src.size());
  }
  std::copy(src.begin(), src.end(), dest.begin());
  return Status::OK();
}

}  // namespace

Status Literal::Piece::CopyFromProto(const LiteralProto& proto) {
  // These conditions should have been checked in Literal::CreateFromProto.
  TF_RET_CHECK(proto.has_shape());
  TF_RET_CHECK(LayoutUtil::HasLayout(proto.shape()));
  TF_RET_CHECK(ShapeUtil::Equal(proto.shape(), subshape()));

  switch (subshape().element_type()) {
    case PRED:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<bool>(), proto.preds()));
      break;
    case U8: {
      auto u8_data = data<uint8>();
      TF_RET_CHECK(proto.u8s().size() == u8_data.size());
      std::copy(proto.u8s().begin(), proto.u8s().end(), u8_data.begin());
    } break;
    case S32:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<int32>(), proto.s32s()));
      break;
    case S64:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<int64>(), proto.s64s()));
      break;
    case U32:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<uint32>(), proto.u32s()));
      break;
    case U64:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<uint64>(), proto.u64s()));
      break;
    case F16: {
      const string& s(proto.f16s());
      TF_RET_CHECK(data<half>().size() * sizeof(half) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(untyped_data()), s.size());
      }
    } break;

    case BF16: {
      const string& s(proto.bf16s());
      TF_RET_CHECK(data<bfloat16>().size() * sizeof(bfloat16) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(untyped_data()), s.size());
      }
    } break;
    case F32:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<float>(), proto.f32s()));
      break;
    case F64:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<double>(), proto.f64s()));
      break;
    case C64: {
      auto complex_data = data<complex64>();
      TF_RET_CHECK(proto.c64s_size() == complex_data.size() * 2);
      for (int64 i = 0; i < complex_data.size(); ++i) {
        complex_data[i] = complex64{proto.c64s(i * 2), proto.c64s(i * 2 + 1)};
      }
    } break;
    case TUPLE:
      LOG(FATAL) << "Should not be called on tuple shapes: "
                 << ShapeUtil::HumanString(subshape());
      break;
    default:
      LOG(FATAL) << "Unhandled primitive type " << subshape().element_type();
  }
  return Status::OK();
}

LiteralProto Literal::ToProto() const {
  LiteralProto proto;
  for (const auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    const Piece& piece = pair.second;

    LiteralProto* proto_piece = &proto;
    for (int64 i : index) {
      while (proto_piece->tuple_literals_size() <= i) {
        proto_piece->add_tuple_literals();
      }
      proto_piece = proto_piece->mutable_tuple_literals(i);
    }
    piece.WriteToProto(proto_piece);
  }

  if (LayoutUtil::IsSparseArray(shape())) {
    CopyToRepeatedField(proto.mutable_sparse_indices(),
                        sparse_indices()->data());
  }

  return proto;
}

/* static */
StatusOr<std::unique_ptr<Literal>> Literal::CreateFromProto(
    const LiteralProto& proto) {
  if (!proto.has_shape()) {
    return InvalidArgument("LiteralProto has no shape");
  }
  if (!LayoutUtil::HasLayout(proto.shape())) {
    return InvalidArgument("LiteralProto has no layout");
  }

  auto literal = MakeUnique<Literal>(proto.shape());

  for (auto& pair : literal->pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;
    const LiteralProto* proto_element = &proto;
    for (int64 i : index) {
      TF_RET_CHECK(i < proto_element->tuple_literals_size());
      proto_element = &proto_element->tuple_literals(i);
    }

    if (ShapeUtil::IsTuple(piece.subshape())) {
      if (proto_element->tuple_literals_size() !=
          ShapeUtil::TupleElementCount(piece.subshape())) {
        return InvalidArgument(
            "Expected %lld tuple elements in LiteralProto, has %d",
            ShapeUtil::TupleElementCount(piece.subshape()),
            proto_element->tuple_literals_size());
      }
      continue;
    }

    TF_RET_CHECK(ShapeUtil::IsArray(piece.subshape()));
    TF_RETURN_IF_ERROR(piece.CopyFromProto(*proto_element));
  }
  return std::move(literal);
}

const void* Literal::untyped_data(const ShapeIndex& shape_index) const {
  return piece(shape_index).untyped_data();
}

void* Literal::untyped_data(const ShapeIndex& shape_index) {
  return piece(shape_index).untyped_data();
}

int64 Literal::size_bytes(const ShapeIndex& shape_index) const {
  return piece(shape_index).size_bytes();
}

string Literal::GetR1U8AsString() const {
  CHECK(ShapeUtil::IsArray(shape()));
  CHECK_EQ(ShapeUtil::Rank(shape()), 1);
  CHECK_EQ(shape().element_type(), U8);
  return string(tensorflow::bit_cast<const char*>(data<uint8>().data()),
                ShapeUtil::ElementsIn(shape()));
}

/* static */ const LiteralView LiteralView::Create(
    const Literal& literal, const ShapeIndex& view_root) {
  return LiteralView(literal, view_root);
}

LiteralView::LiteralView(const Literal& literal, const ShapeIndex& view_root) {
  shape_ = ShapeUtil::GetSubshape(literal.shape(), view_root);
  pieces_ = ShapeTree<Piece>(shape_);
  owns_buffers_ = false;
  for (auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;

    ShapeIndex src_index = view_root;
    for (int64 i : index) {
      src_index.push_back(i);
    }
    const Piece& src_piece = literal.piece(src_index);
    piece.set_buffer(src_piece.buffer());
    piece.set_sparse_indices(src_piece.sparse_indices());
    piece.set_subshape(&ShapeUtil::GetSubshape(shape_, index));
  }
}

LiteralView::~LiteralView() {}

LiteralView::LiteralView(const LiteralView& other) { CopyFrom(other); }

LiteralView& LiteralView::operator=(const LiteralView& other) {
  CopyFrom(other);
  return *this;
}

void LiteralView::CopyFrom(const LiteralView& other) {
  // We can't use the default copy-constructor/copy-assignment because
  // Piece::subshape_ points to subshapes within the Shape of the owning
  // Literal/LiteralView.
  shape_ = other.shape();
  pieces_ = other.pieces_;
  for (auto& pair : pieces_) {
    const ShapeIndex& index = pair.first;
    Piece& piece = pair.second;
    piece.set_subshape(&ShapeUtil::GetSubshape(shape_, index));
  }
  owns_buffers_ = false;
}

}  // namespace xla
