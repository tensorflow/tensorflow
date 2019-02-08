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

#include "tensorflow/compiler/xla/literal.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using absl::StrCat;
using absl::StrFormat;

constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;

// Converts between little and big endian.
//
// Precondition: size % 2 == 0 (elements in the array are 16 bits long)
void ConvertEndianShort(string* bytes) {
  CHECK_EQ(bytes->size() / 2, 0);
  for (int64 i = 0; i < bytes->size(); i += 2) {
    std::swap((*bytes)[i], (*bytes)[i + 1]);
  }
}

void ConvertEndianShort(char* bytes, int64 size) {
  CHECK_EQ(size / 2, 0);
  for (int64 i = 0; i < size; i += 2) {
    std::swap(bytes[i], bytes[i + 1]);
  }
}

// Since Eigen::half doesn't satisfy the absl::bit_cast contract, we need to be
// able to transparently access the raw 16-bit value contained within.
template <typename T>
T GetRawValue(T val) {
  return val;
}
uint16 GetRawValue(Eigen::half val) { return val.x; }

}  // namespace

LiteralBase::~LiteralBase() {}

std::ostream& operator<<(std::ostream& out, const Literal& literal) {
  out << literal.ToString();
  return out;
}

MutableLiteralBase::StrideConfig::StrideConfig(
    const Shape& source_shape, const Shape& dest_shape,
    absl::Span<const int64> dimensions)
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

void Literal::SetPiece(const Shape& shape, Piece* piece, bool allocate_arrays) {
  if (shape.IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      const Shape& subshape = shape.tuple_shapes(i);

      auto child_piece = Piece();
      child_piece.set_subshape(&subshape);

      SetPiece(subshape, &child_piece, allocate_arrays);

      piece->emplace_back(std::move(child_piece));
    }
  } else if (shape.IsArray()) {
    if (allocate_arrays) {
      if (LayoutUtil::IsSparseArray(shape)) {
        // For sparse arrays, the buffer must be of the size of the maximum
        // number of sparse elements possible.
        const int64 max_sparse_elements =
            LayoutUtil::MaxSparseElements(shape.layout());
        piece->set_buffer(
            new char[max_sparse_elements *
                     ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type())]);
        piece->set_sparse_indices(
            new SparseIndexArray(max_sparse_elements, shape.rank()));
      } else {
        piece->set_buffer(new char[piece->size_bytes()]);
      }
    }
  } else {
    // If the shape is neither an array nor tuple, then it must be
    // zero-sized. Otherwise, some memory needs to be allocated for it.
    CHECK_EQ(piece->size_bytes(), 0);
  }
}

Literal::Literal(const Shape& shape, bool allocate_arrays)
    : MutableLiteralBase() {
  shape_ = absl::make_unique<Shape>(shape);
  CHECK(LayoutUtil::HasLayout(*shape_));
  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());
  CHECK(&root_piece_->subshape() == shape_.get());

  SetPiece(*shape_, root_piece_, allocate_arrays);
}

Literal::~Literal() {
  if (root_piece_ != nullptr) {
    DeallocateBuffers();
    delete root_piece_;
  }
}

void Literal::DeallocateBuffers() {
  root_piece_->ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        if (piece->buffer() != nullptr) {
          delete[] piece->buffer();
          delete piece->sparse_indices();
        }
      });
}

Literal::Literal(Literal&& other) : MutableLiteralBase() {
  *this = std::move(other);
}

Literal& Literal::operator=(Literal&& other) {
  DCHECK(&other.root_piece_->subshape() == other.shape_.get());
  using std::swap;
  swap(shape_, other.shape_);
  swap(root_piece_, other.root_piece_);
  DCHECK(&root_piece_->subshape() == shape_.get());

  return *this;
}

Literal LiteralBase::CreateFromShape(const Shape& shape) {
  Literal literal(shape);
  literal.root_piece_->ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        if (piece->subshape().IsArray()) {
          memset(piece->untyped_data(), 0, piece->size_bytes());
        }
      });
  return literal;
}

const SparseIndexArray* LiteralBase::sparse_indices(
    const ShapeIndex& shape_index) const {
  return piece(shape_index).sparse_indices();
}

SparseIndexArray* MutableLiteralBase::sparse_indices(
    const ShapeIndex& shape_index) {
  return piece(shape_index).sparse_indices();
}

template <typename NativeT>
Status MutableLiteralBase::CopySliceFromInternal(
    const LiteralBase& src_literal, absl::Span<const int64> src_base,
    absl::Span<const int64> dest_base, absl::Span<const int64> copy_size) {
  TF_RET_CHECK(src_literal.shape().rank() == src_base.size());
  TF_RET_CHECK(shape().rank() == dest_base.size());

  auto linear_index = [](const Shape& shape,
                         absl::Span<const int64> multi_index) {
    return IndexUtil::MultidimensionalIndexToLinearIndex(shape, multi_index);
  };

  if (src_literal.shape().rank() == 0 || shape().rank() == 0) {
    // If any of the two shapes are scalars, we can just call the StridedCopy()
    // directly, and we know we will be copying only one value.
    TF_RET_CHECK(copy_size.empty());
    StridedCopy(data<NativeT>(), linear_index(shape(), dest_base), 0,
                src_literal.data<NativeT>(),
                linear_index(src_literal.shape(), src_base), 0, 1);
  } else if (!ShapeUtil::IsZeroElementArray(shape()) &&
             !ShapeUtil::IsZeroElementArray(src_literal.shape())) {
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
    MutableLiteralBase::StrideConfig stride_config(src_literal.shape(), shape(),
                                                   copy_size);

    auto copy_proc = [&](absl::Span<const int64> indexes) {
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

Status MutableLiteralBase::CopyElementFrom(const LiteralSlice& src_literal,
                                           absl::Span<const int64> src_index,
                                           absl::Span<const int64> dest_index) {
  DCHECK_EQ(shape().element_type(), src_literal.shape().element_type());
  const int64 src_linear_index = IndexUtil::MultidimensionalIndexToLinearIndex(
      src_literal.shape(), src_index);
  const int64 dest_linear_index =
      IndexUtil::MultidimensionalIndexToLinearIndex(shape(), dest_index);
  const int64 primitive_size =
      ShapeUtil::ByteSizeOfPrimitiveType(shape().element_type());

  char* dest_address =
      static_cast<char*>(untyped_data()) + dest_linear_index * primitive_size;
  const char* source_address =
      static_cast<const char*>(src_literal.untyped_data()) +
      src_linear_index * primitive_size;
  if (dest_address != source_address) {
    memcpy(dest_address, source_address, primitive_size);
  }
  return Status::OK();
}

/* static */ StatusOr<Literal> MutableLiteralBase::CreateFromProto(
    const LiteralProto& proto) {
  if (!proto.has_shape()) {
    return InvalidArgument("LiteralProto has no shape");
  }
  Shape shape(proto.shape());
  if (ShapeUtil::HasPrimitiveType(shape, OPAQUE)) {
    return InvalidArgument("Literal shape cannot include OPAQUE sub-shape");
  }
  if (!LayoutUtil::HasLayout(shape)) {
    return InvalidArgument("LiteralProto has no layout");
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));

  Literal literal(shape);

  TF_RETURN_IF_ERROR(literal.root_piece_->ForEachMutableSubpieceWithStatus(
      [&](const ShapeIndex& index, Piece* piece) {
        const LiteralProto* proto_element = &proto;
        for (int64 i : index) {
          CHECK(i < proto_element->tuple_literals_size());
          proto_element = &proto_element->tuple_literals(i);
        }

        if (piece->subshape().IsTuple()) {
          if (proto_element->tuple_literals_size() !=
              ShapeUtil::TupleElementCount(piece->subshape())) {
            return InvalidArgument(
                "Expected %d tuple elements in LiteralProto, has %d",
                ShapeUtil::TupleElementCount(piece->subshape()),
                proto_element->tuple_literals_size());
          }
          return Status::OK();
        }
        if (piece->subshape().element_type() == TOKEN) {
          return Status::OK();
        }

        CHECK(piece->subshape().IsArray());
        TF_RETURN_IF_ERROR(piece->CopyFromProto(*proto_element));

        return Status::OK();
      }));

  return std::move(literal);
}

std::vector<Literal> Literal::DecomposeTuple() {
  CHECK(shape().IsTuple());
  std::vector<Literal> elements;
  for (int i = 0; i < ShapeUtil::TupleElementCount(shape()); ++i) {
    elements.push_back(Literal(ShapeUtil::GetSubshape(shape(), {i}),
                               /*allocate_arrays=*/false));
    Literal& element = elements.back();
    element.root_piece_->ForEachMutableSubpiece(
        [&](const ShapeIndex& index, Piece* dest_piece) {
          ShapeIndex src_index = {i};
          for (int64 j : index) {
            src_index.push_back(j);
          }
          Piece& src_piece = piece(src_index);

          // Move the respective buffer and sparse indices over to the element
          // Literal.
          dest_piece->set_buffer(src_piece.buffer());
          src_piece.set_buffer(nullptr);
          dest_piece->set_sparse_indices(src_piece.sparse_indices());
          src_piece.set_sparse_indices(nullptr);
        });
  }
  // Set this literal to be nil-shaped.
  *this = Literal();
  return elements;
}

namespace {

// Copies the elements in 'src' to 'dest'. The shape and layout of the data in
// the array slices are indicated by dest_shape and src_shape respectively.
template <typename NativeT>
void CopyElementsBetween(absl::Span<NativeT> dest,
                         absl::Span<const NativeT> src, const Shape& dest_shape,
                         const Shape& src_shape) {
  CHECK(ShapeUtil::Compatible(dest_shape, src_shape));
  if (ShapeUtil::IsZeroElementArray(dest_shape)) {
    return;
  }
  std::vector<int64> index(dest_shape.rank());
  do {
    dest[IndexUtil::MultidimensionalIndexToLinearIndex(dest_shape, index)] =
        src[IndexUtil::MultidimensionalIndexToLinearIndex(src_shape, index)];
  } while (IndexUtil::BumpIndices(dest_shape, absl::MakeSpan(index)));
}

}  // namespace

Status LiteralBase::Piece::CopyFrom(const LiteralBase::Piece& src) {
  CHECK(subshape_ != nullptr);
  CHECK(src.subshape_ != nullptr);
  if (ShapeUtil::Equal(subshape(), src.subshape())) {
    // If the layouts are equal it's faster just to memcpy.
    memcpy(buffer(), src.buffer(), src.size_bytes());
  } else {
    TF_RET_CHECK(ShapeUtil::Compatible(src.subshape(), subshape()));
    std::vector<int64> origin(subshape().rank(), 0);
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
      COPY_ELEMENTS(C128, complex128);
      COPY_ELEMENTS(PRED, bool);
#undef COPY_ELEMENTS
      default:
        return Unimplemented(
            "Copying a Literal object with element type %s is not implemented.",
            PrimitiveType_Name(subshape().element_type()));
    }
  }
  return Status::OK();
}

Status MutableLiteralBase::CopyFrom(const LiteralSlice& src_literal,
                                    const ShapeIndex& dest_shape_index,
                                    const ShapeIndex& src_shape_index) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  const Shape& src_subshape =
      ShapeUtil::GetSubshape(src_literal.shape(), src_shape_index);
  if (!ShapeUtil::Compatible(dest_subshape, src_subshape)) {
    return InvalidArgument(
        "Destination subshape incompatible with source subshape: %s vs %s",
        ShapeUtil::HumanString(dest_subshape),
        ShapeUtil::HumanString(src_subshape));
  }
  return root_piece_->ForEachMutableSubpieceWithStatus(
      [&](const ShapeIndex& index, Piece* piece) {
        if (!piece->subshape().IsArray()) {
          return Status::OK();
        }

        // Determine if this index is in the part of this literal that we want
        // to copy over from src_literal.
        bool in_subtree_to_copy = true;
        for (int i = 0; i < dest_shape_index.size(); ++i) {
          if (index[i] != dest_shape_index[i]) {
            in_subtree_to_copy = false;
            break;
          }
        }
        if (!in_subtree_to_copy) {
          return Status::OK();
        }
        // Construct the index of the corresponding piece in the source literal.
        ShapeIndex src_piece_index = src_shape_index;
        for (int64 i = dest_shape_index.size(); i < index.size(); ++i) {
          src_piece_index.push_back(index[i]);
        }
        TF_RETURN_IF_ERROR(piece->CopyFrom(src_literal.piece(src_piece_index)));
        return Status::OK();
      });
}

Status Literal::MoveFrom(Literal&& src_literal,
                         const ShapeIndex& dest_shape_index) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  if (!ShapeUtil::Equal(dest_subshape, src_literal.shape())) {
    return InvalidArgument(
        "Destination subshape not equal to source shape: %s vs %s",
        ShapeUtil::HumanString(dest_subshape),
        ShapeUtil::HumanString(src_literal.shape()));
  }

  src_literal.root_piece_->ForEachSubpiece(
      [&](const ShapeIndex& src_index, const Piece& src_piece) {
        if (!src_piece.subshape().IsArray()) {
          return;
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
      });

  src_literal.shape_ = absl::make_unique<Shape>(ShapeUtil::MakeNil());
  delete src_literal.root_piece_;
  src_literal.root_piece_ = new LiteralBase::Piece();
  src_literal.root_piece_->set_subshape(src_literal.shape_.get());

  return Status::OK();
}

Status MutableLiteralBase::CopySliceFrom(const LiteralSlice& src_literal,
                                         absl::Span<const int64> src_base,
                                         absl::Span<const int64> dest_base,
                                         absl::Span<const int64> copy_size) {
  TF_RET_CHECK(shape().IsArray()) << ShapeUtil::HumanString(shape());
  TF_RET_CHECK(src_literal.shape().IsArray())
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
    case C128:
      return CopySliceFromInternal<complex128>(src_literal, src_base, dest_base,
                                               copy_size);
    case PRED:
      return CopySliceFromInternal<bool>(src_literal, src_base, dest_base,
                                         copy_size);
    default:
      break;
  }
  return Unimplemented(
      "Copying a slice from a Literal object with element type %d is not "
      "implemented.",
      shape().element_type());
}

void MutableLiteralBase::PopulateR1(const tensorflow::core::Bitmap& values) {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().rank(), 1);
  CHECK_EQ(element_count(), values.bits());
  CHECK_EQ(shape().element_type(), PRED);
  for (int64 i = 0; i < static_cast<int64>(values.bits()); ++i) {
    Set({i}, values.get(i));
  }
}

Literal LiteralBase::Relayout(const Layout& new_layout,
                              const ShapeIndex& shape_index) const {
  // Create new shape with 'new_layout' set at the given shape index.
  Shape new_shape = shape();
  Shape* subshape = ShapeUtil::GetMutableSubshape(&new_shape, shape_index);
  TF_CHECK_OK(LayoutUtil::ValidateLayoutForShape(new_layout, *subshape));
  *subshape->mutable_layout() = new_layout;
  Literal result(new_shape);
  TF_CHECK_OK(result.CopyFrom(*this));
  return result;
}

Literal LiteralBase::Relayout(const Shape& shape_with_layout) const {
  CHECK(ShapeUtil::Compatible(shape_with_layout, shape()))
      << "Given shape_with_layout " << ShapeUtil::HumanString(shape_with_layout)
      << " not compatible with literal shape "
      << ShapeUtil::HumanString(shape());
  Literal result = CreateFromShape(shape_with_layout);
  ShapeUtil::ForEachSubshape(
      result.shape(),
      [this, &result](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsArray()) {
          TF_CHECK_OK(result.CopyFrom(*this,
                                      /*dest_shape_index=*/index,
                                      /*src_shape_index=*/index));
        }
      });
  return result;
}

StatusOr<Literal> LiteralBase::Broadcast(
    const Shape& result_shape, absl::Span<const int64> dimensions) const {
  if (!shape().IsArray()) {
    return InvalidArgument("Broadcast only supports arrays.");
  }

  for (int64 i = 0; i < dimensions.size(); i++) {
    TF_RET_CHECK(shape().dimensions(i) ==
                 result_shape.dimensions(dimensions[i]));
  }

  Literal result(result_shape);

  // scratch_source_index is temporary storage space for the computed index into
  // the input literal.  We put it here to avoid allocating an std::vector in
  // every iteration of ShapeUtil::ForEachIndex.
  std::vector<int64> scratch_source_index(shape().dimensions_size());

  char* dest_data = static_cast<char*>(result.untyped_data());
  const char* source_data = static_cast<const char*>(untyped_data());
  const int64 primitive_size =
      ShapeUtil::ByteSizeOfPrimitiveType(shape().element_type());

  ShapeUtil::ForEachIndex(
      result_shape, [&](absl::Span<const int64> output_index) {
        for (int64 i = 0; i < dimensions.size(); ++i) {
          scratch_source_index[i] = output_index[dimensions[i]];
        }
        int64 dest_index = IndexUtil::MultidimensionalIndexToLinearIndex(
            result_shape, output_index);
        int64 source_index = IndexUtil::MultidimensionalIndexToLinearIndex(
            shape(), scratch_source_index);
        memcpy(dest_data + primitive_size * dest_index,
               source_data + primitive_size * source_index, primitive_size);
        return true;
      });

  return std::move(result);
}

StatusOr<Literal> LiteralBase::Reshape(
    absl::Span<const int64> dimensions) const {
  if (!shape().IsArray()) {
    return InvalidArgument("Reshape does not support tuples.");
  }
  Literal output;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape().layout())) {
    output = Relayout(LayoutUtil::GetDefaultLayoutForRank(shape().rank()));
  } else {
    output = Clone();
  }
  // Because the layout is monotonic, we can simply reuse the same sequence of
  // values without changing their order.
  *output.mutable_shape_do_not_use() =
      ShapeUtil::MakeShape(shape().element_type(), dimensions);

  int64 elements_before = ShapeUtil::ElementsIn(shape());
  int64 elements_after = ShapeUtil::ElementsIn(output.shape());
  if (elements_before != elements_after) {
    return InvalidArgument(
        "Shapes before and after Literal::Reshape have different numbers "
        "of elements: %s vs %s.",
        ShapeUtil::HumanString(shape()),
        ShapeUtil::HumanString(output.shape()));
  }
  return std::move(output);
}

Literal LiteralBase::Transpose(absl::Span<const int64> permutation) const {
  CHECK(shape().IsArray()) << "Tuple is not supported for transpose";
  CHECK(IsPermutation(permutation, shape().rank()))
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
  Literal new_literal(permuted_shape);
  DCHECK_EQ(ShapeUtil::ByteSizeOf(new_literal.shape()),
            ShapeUtil::ByteSizeOf(shape()));
  std::memcpy(new_literal.untyped_data(), untyped_data(), size_bytes());
  return new_literal;
}

template <typename NativeT>
Literal LiteralBase::SliceInternal(
    const Shape& result_shape, absl::Span<const int64> start_indices) const {
  Literal result_literal(result_shape);
  DimensionVector new_indices(result_shape.rank());
  result_literal.EachCell<NativeT>(
      [&](absl::Span<const int64> indices, NativeT /*value*/) {
        for (int64 i = 0; i < result_shape.rank(); ++i) {
          new_indices[i] = indices[i] + start_indices[i];
        }
        NativeT value = Get<NativeT>(new_indices);
        result_literal.Set<NativeT>(indices, value);
      });
  return result_literal;
}

Literal LiteralBase::Slice(absl::Span<const int64> start_indices,
                           absl::Span<const int64> limit_indices) const {
  CHECK(shape().IsArray()) << "tuple is not supported for slice";

  DimensionVector result_dimensions;
  for (int64 dnum = 0; dnum < shape().rank(); ++dnum) {
    CHECK_GE(start_indices[dnum], 0);
    CHECK_LE(limit_indices[dnum], shape().dimensions(dnum))
        << "dnum = " << dnum;
    int64 dimension = limit_indices[dnum] - start_indices[dnum];
    CHECK_GE(dimension, 0) << "dnum = " << dnum;
    result_dimensions.push_back(dimension);
  }
  const auto result_shape =
      ShapeUtil::MakeShapeWithLayout(shape().element_type(), result_dimensions,
                                     LayoutUtil::MinorToMajor(shape()));
  switch (result_shape.element_type()) {
    case PRED:
      return SliceInternal<bool>(result_shape, start_indices);
    case U8:
      return SliceInternal<uint8>(result_shape, start_indices);
    case U16:
      return SliceInternal<uint16>(result_shape, start_indices);
    case U32:
      return SliceInternal<uint32>(result_shape, start_indices);
    case U64:
      return SliceInternal<uint64>(result_shape, start_indices);
    case S8:
      return SliceInternal<int8>(result_shape, start_indices);
    case S16:
      return SliceInternal<int16>(result_shape, start_indices);
    case S32:
      return SliceInternal<int32>(result_shape, start_indices);
    case S64:
      return SliceInternal<int64>(result_shape, start_indices);
    case F16:
      return SliceInternal<half>(result_shape, start_indices);
    case BF16:
      return SliceInternal<bfloat16>(result_shape, start_indices);
    case F32:
      return SliceInternal<float>(result_shape, start_indices);
    case F64:
      return SliceInternal<double>(result_shape, start_indices);
    case C64:
      return SliceInternal<complex64>(result_shape, start_indices);
    case C128:
      return SliceInternal<complex128>(result_shape, start_indices);
    default:
      LOG(FATAL) << "not yet implemented: "
                 << PrimitiveType_Name(result_shape.element_type());
  }
}

Literal LiteralBase::Clone() const {
  Literal result(shape());
  TF_CHECK_OK(result.CopyFrom(*this));
  return result;
}

string LiteralBase::GetAsString(absl::Span<const int64> multi_index,
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
      return StrCat(static_cast<float>(Get<half>(multi_index, shape_index)));
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
    case C128: {
      complex128 c = Get<complex128>(multi_index, shape_index);
      return StrCat("(", c.real(), ", ", c.imag(), ")");
    }
    default:
      LOG(FATAL) << PrimitiveType_Name(subshape.element_type());
  }
}

string LiteralBase::GetSparseElementAsString(
    int64 sparse_element_number, const ShapeIndex& shape_index) const {
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
      return StrCat(static_cast<float>(
          GetSparseElement<half>(sparse_element_number, shape_index)));
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
    case C128: {
      complex128 c =
          GetSparseElement<complex128>(sparse_element_number, shape_index);
      return StrCat("(", c.real(), ", ", c.imag(), ")");
    }
    default:
      LOG(FATAL) << "Invalid element type for sparse arrays: "
                 << PrimitiveType_Name(subshape.element_type());
  }
}

StatusOr<int64> LiteralBase::GetIntegralAsS64(
    absl::Span<const int64> multi_index) const {
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
      return FailedPrecondition("Array element type is not integral: %s",
                                PrimitiveType_Name(shape().element_type()));
  }
}

size_t LiteralBase::Hash() const {
  using tensorflow::Hash64;
  using tensorflow::Hash64Combine;

  size_t hash_value = ShapeUtil::Hash(shape());

  ShapeUtil::ForEachSubshape(
      shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }

        CHECK(LayoutUtil::IsDense(subshape.layout()));
        hash_value = Hash64Combine(
            hash_value, Hash64(static_cast<const char*>(untyped_data(index)),
                               size_bytes(index)));
      });

  return hash_value;
}

Status MutableLiteralBase::SetIntegralAsS64(absl::Span<const int64> multi_index,
                                            int64 value) {
  CHECK(LayoutUtil::IsDenseArray(shape()));
  switch (shape().element_type()) {
    case PRED:
      Set<bool>(multi_index, value);
      break;
    case U8:
      Set<uint8>(multi_index, value);
      break;
    case S32:
      Set<int32>(multi_index, value);
      break;
    case S64:
      Set<int64>(multi_index, value);
      break;
    case U32:
      Set<uint32>(multi_index, value);
      break;
    case U64:
      Set<uint64>(multi_index, value);
      break;
    default:
      return FailedPrecondition("Array element type is not integral: %s",
                                PrimitiveType_Name(shape().element_type()));
  }
  return Status::OK();
}

absl::Span<const int64> LiteralBase::GetSparseIndex(
    int64 sparse_element_number, const ShapeIndex& shape_index) const {
  const Piece& p = piece(shape_index);
  CHECK_GE(sparse_element_number, 0);
  CHECK_LT(sparse_element_number, p.sparse_indices()->index_count());
  return p.sparse_indices()->At(sparse_element_number);
}

void MutableLiteralBase::SortSparseElements(const ShapeIndex& shape_index) {
  piece(shape_index).SortSparseElements();
}

void LiteralBase::Piece::SortSparseElements() {
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
    case C128:
      SortSparseElementsInternal<complex128>();
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
void LiteralBase::Piece::SortSparseElementsInternal() {
  CHECK(LayoutUtil::IsSparseArray(subshape()));
  int64 num_elements = sparse_indices()->index_count();
  auto values = data<NativeT>();
  CHECK_LE(num_elements, values.size());
  sparse_indices()->SortWithValues(
      absl::Span<NativeT>(values.data(), num_elements));
}

namespace {

string ShapeToString(bool print_layout, const Shape& shape) {
  return print_layout ? ShapeUtil::HumanStringWithLayout(shape)
                      : ShapeUtil::HumanString(shape);
}

void ToStringHelper(const LiteralBase& literal, const ShapeIndex& shape_index,
                    bool print_shape, bool print_layout,
                    std::vector<string>* pieces);

void TupleToStringHelper(const LiteralBase& literal,
                         const ShapeIndex& shape_index, bool print_shape,
                         bool print_layout, std::vector<string>* pieces) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);
  pieces->push_back("(\n");
  std::vector<string> tuple_pieces;
  for (int i = 0; i < ShapeUtil::TupleElementCount(subshape); ++i) {
    ShapeIndex element_index = shape_index;
    element_index.push_back(i);
    std::vector<string> element_pieces;
    ToStringHelper(literal, element_index, print_shape, print_layout,
                   &element_pieces);
    tuple_pieces.push_back(absl::StrJoin(element_pieces, ""));
  }
  pieces->push_back(absl::StrJoin(tuple_pieces, ",\n"));
  pieces->push_back("\n)");
}

void SparseArrayToStringHelper(const LiteralBase& literal,
                               const Shape& subshape, bool print_shape,
                               bool print_layout, std::vector<string>* pieces) {
  if (print_shape) {
    pieces->push_back(ShapeToString(print_layout, subshape));
  }
  pieces->push_back("{");
  int64 rank = subshape.rank();
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
      pieces->push_back(absl::StrJoin(literal.GetSparseIndex(i), ", "));
      pieces->push_back("]: ");
    }
    pieces->push_back(literal.GetSparseElementAsString(i));
  }
  pieces->push_back("}");
}

void DenseArrayToStringHelper(const LiteralBase& literal,
                              const ShapeIndex& shape_index, bool print_shape,
                              bool print_layout, std::vector<string>* pieces) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);
  int64 rank = subshape.rank();

  std::function<void(absl::Span<const int64> dimensions, std::vector<int64>*)>
      to_string_recursive = [&](absl::Span<const int64> dimensions,
                                std::vector<int64>* accum_indices) {
        // dimensions.size() decreases by 1 at each recursive call,
        // and accum_indices->size() increases by 1.
        // Their sum is equal to the rank of the tensor.
        CHECK_EQ(rank, dimensions.size() + accum_indices->size());

        auto brace_to_string = [&](string brace) -> string {
          // Handle 1D tensor
          if (rank == 1) {
            return brace;
          }
          // Handle the innermost tensor of a 2D+ tensor.
          if (dimensions.size() == 1 && brace == "{") {
            return StrCat("  ", brace, dimensions[0] <= 1 ? "" : " ");
          }
          if (dimensions.size() == 1 && brace == "}") {
            return StrCat(dimensions[0] <= 1 ? "" : " ", brace);
          }
          // Handle the non-innermost tensors of a 2D+ tensor.
          if (brace == "{") {
            if (rank > 3 && !accum_indices->empty() &&
                accum_indices->size() < rank) {
              int index = accum_indices->size() - 1;
              int value = accum_indices->back();
              return StrCat(brace, " /*i", index, "=", value, "*/\n");
            }
            return StrCat(brace, "\n");
          }
          return StrCat("\n", brace);
        };

        if (dimensions.empty()) {
          // Display predicates as 0s and 1s so that the string is more dense.
          string elem;
          if (subshape.element_type() == PRED && rank > 0) {
            elem = literal.Get<bool>(*accum_indices, shape_index) ? "1" : "0";
          } else {
            elem = literal.GetAsString(*accum_indices, shape_index);
          }
          pieces->push_back(elem);
        } else {
          pieces->push_back(brace_to_string("{"));
          for (int i = 0; i < dimensions[0]; ++i) {
            std::vector<int64> cloned_indices(*accum_indices);
            cloned_indices.push_back(i);
            to_string_recursive(dimensions.subspan(1), &cloned_indices);
            if (i < dimensions[0] - 1) {
              pieces->push_back(",");
              pieces->push_back(dimensions.size() > 1 ? "\n" : " ");
            }
          }
          pieces->push_back(brace_to_string("}"));
        }
      };

  if (print_shape) {
    pieces->push_back(ShapeToString(print_layout, subshape));
    pieces->push_back(" ");
  }
  std::vector<int64> indices = {};
  std::vector<int64> dimensions(subshape.dimensions().begin(),
                                subshape.dimensions().end());
  to_string_recursive(dimensions, &indices);
}

void ToStringHelper(const LiteralBase& literal, const ShapeIndex& shape_index,
                    bool print_shape, bool print_layout,
                    std::vector<string>* pieces) {
  const Shape& subshape = ShapeUtil::GetSubshape(literal.shape(), shape_index);
  CHECK(LayoutUtil::HasLayout(literal.shape()));
  CHECK(LayoutUtil::HasLayout(subshape));
  if (subshape.IsTuple()) {
    TupleToStringHelper(literal, shape_index, print_shape, print_layout,
                        pieces);
  } else if (subshape.IsToken()) {
    pieces->push_back("token");
  } else if (LayoutUtil::IsSparseArray(subshape)) {
    SparseArrayToStringHelper(literal, subshape, print_shape, print_layout,
                              pieces);
  } else {
    CHECK(LayoutUtil::IsDenseArray(subshape));
    DenseArrayToStringHelper(literal, shape_index, print_shape, print_layout,
                             pieces);
  }
}

}  // namespace

int64 LiteralBase::sparse_element_count() const {
  CHECK(LayoutUtil::IsSparseArray(shape()));
  return sparse_indices()->index_count();
}

string LiteralBase::ToString() const {
  std::vector<string> pieces;
  CHECK(LayoutUtil::HasLayout(this->shape()));
  ToStringHelper(*this, {}, /*print_shape=*/true,
                 /*print_layout=*/false, &pieces);
  return absl::StrJoin(pieces, "");
}

string LiteralBase::ToStringWithoutShape() const {
  std::vector<string> pieces;
  CHECK(LayoutUtil::HasLayout(this->shape()));
  ToStringHelper(*this, {}, /*print_shape=*/false,
                 /*print_layout=*/false, &pieces);
  return absl::StrJoin(pieces, "");
}

string LiteralBase::ToStringWithLayout() const {
  std::vector<string> pieces;
  CHECK(LayoutUtil::HasLayout(this->shape()));
  ToStringHelper(*this, {}, /*print_shape=*/true,
                 /*print_layout=*/true, &pieces);
  return absl::StrJoin(pieces, "");
}

void LiteralBase::EachCellAsString(
    const std::function<void(absl::Span<const int64> indices,
                             const string& value)>& per_cell) const {
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64> indices = IndexUtil::LinearIndexToMultidimensionalIndex(
      shape(), /*linear_index=*/0);
  do {
    per_cell(indices, GetAsString(indices));
  } while (IndexUtil::BumpIndices(shape(), absl::MakeSpan(indices)));
}

namespace {
template <typename NativeSrcT, typename NativeDestT, typename ConverterType>
Literal ConvertBetweenNativeTypesWithConverter(const LiteralBase& src_literal,
                                               const ConverterType& converter) {
  CHECK(src_literal.shape().IsArray());
  Literal result_literal(ShapeUtil::ChangeElementType(
      src_literal.shape(),
      primitive_util::NativeToPrimitiveType<NativeDestT>()));
  auto src_data = src_literal.data<NativeSrcT>();
  auto dest_data = result_literal.template data<NativeDestT>();
  int64 num_elements = src_literal.element_count();

  for (int64 i = 0; i < num_elements; ++i) {
    dest_data[i] = converter(src_data[i]);
  }
  return result_literal;
}

template <typename NativeSrcT, typename NativeDestT>
typename std::enable_if<(std::is_same<NativeSrcT, Eigen::half>::value) &&
                            (std::is_same<NativeDestT, complex64>::value ||
                             std::is_same<NativeDestT, complex128>::value),
                        Literal>::type
ConvertBetweenNativeTypes(const LiteralBase& src_literal) {
  auto converter = [](NativeSrcT src) {
    return NativeDestT(static_cast<typename NativeDestT::value_type>(src));
  };
  return ConvertBetweenNativeTypesWithConverter<NativeSrcT, NativeDestT>(
      src_literal, converter);
}

template <typename NativeSrcT, typename NativeDestT>
typename std::enable_if<(!std::is_same<NativeSrcT, Eigen::half>::value) ||
                            (!std::is_same<NativeDestT, complex64>::value &&
                             !std::is_same<NativeDestT, complex128>::value),
                        Literal>::type
ConvertBetweenNativeTypes(const LiteralBase& src_literal) {
  auto converter = [](NativeSrcT src) { return static_cast<NativeDestT>(src); };
  return ConvertBetweenNativeTypesWithConverter<NativeSrcT, NativeDestT>(
      src_literal, converter);
}

template <typename NativeSrcT, typename NativeDestT>
typename std::enable_if<(sizeof(NativeSrcT) == sizeof(NativeDestT) &&
                         !std::is_same<NativeDestT, Eigen::half>::value),
                        Literal>::type
BitcastBetweenNativeTypes(const LiteralBase& src_literal) {
  auto converter = [](NativeSrcT src) {
    return absl::bit_cast<NativeDestT>(GetRawValue(src));
  };
  return ConvertBetweenNativeTypesWithConverter<NativeSrcT, NativeDestT>(
      src_literal, converter);
}

template <typename NativeSrcT, typename NativeDestT>
typename std::enable_if<(sizeof(NativeSrcT) == sizeof(Eigen::half) &&
                         std::is_same<NativeDestT, Eigen::half>::value),
                        Literal>::type
BitcastBetweenNativeTypes(const LiteralBase& src_literal) {
  // Eigen::half doesn't satisfy the absl::bit_cast contract, so explicitly
  // cast to unsigned short and then use raw_uint16_to_half.
  auto converter = [](NativeSrcT src) {
    return Eigen::half_impl::raw_uint16_to_half(
        absl::bit_cast<uint16>(GetRawValue(src)));
  };
  return ConvertBetweenNativeTypesWithConverter<NativeSrcT, Eigen::half>(
      src_literal, converter);
}

// This template specialization is here to make the compiler happy. bit_cast has
// a static check that the types are the same size. This specialization should
// never be used because the source and destination types are checked for
// identical sizes higher up.
template <typename NativeSrcT, typename NativeDestT>
typename std::enable_if<(sizeof(NativeSrcT) != sizeof(NativeDestT)),
                        Literal>::type
BitcastBetweenNativeTypes(const LiteralBase& src_literal) {
  LOG(FATAL) << "Invalid bitcast between types of different sizes.";
}

template <PrimitiveType primitive_src_type, PrimitiveType primitive_dest_type>
Literal ConvertIfTypesMatch(const LiteralBase& src_literal, bool bitcast) {
  CHECK_EQ(primitive_src_type, src_literal.shape().element_type());
  if (bitcast) {
    return BitcastBetweenNativeTypes<
        typename primitive_util::PrimitiveTypeToNative<
            primitive_src_type>::type,
        typename primitive_util::PrimitiveTypeToNative<
            primitive_dest_type>::type>(src_literal);
  } else {
    return ConvertBetweenNativeTypes<
        typename primitive_util::PrimitiveTypeToNative<
            primitive_src_type>::type,
        typename primitive_util::PrimitiveTypeToNative<
            primitive_dest_type>::type>(src_literal);
  }
}

template <PrimitiveType primitive_src_type>
StatusOr<Literal> ConvertIfDestTypeMatches(const LiteralBase& src_literal,
                                           PrimitiveType primitive_dest_type,
                                           bool bitcast) {
  switch (primitive_dest_type) {
#define CONVERT_IF_TYPES_MATCH(type)                                    \
  case (type):                                                          \
    return ConvertIfTypesMatch<primitive_src_type, (type)>(src_literal, \
                                                           bitcast);
    CONVERT_IF_TYPES_MATCH(PRED)
    CONVERT_IF_TYPES_MATCH(S8)
    CONVERT_IF_TYPES_MATCH(S16)
    CONVERT_IF_TYPES_MATCH(S32)
    CONVERT_IF_TYPES_MATCH(S64)
    CONVERT_IF_TYPES_MATCH(U8)
    CONVERT_IF_TYPES_MATCH(U16)
    CONVERT_IF_TYPES_MATCH(U32)
    CONVERT_IF_TYPES_MATCH(U64)
    CONVERT_IF_TYPES_MATCH(F16)
    CONVERT_IF_TYPES_MATCH(F32)
    CONVERT_IF_TYPES_MATCH(F64)
    CONVERT_IF_TYPES_MATCH(BF16)
#undef CONVERT_IF_TYPES_MATCH
    case C64:
      if (bitcast) {
        break;
      }
      return ConvertIfTypesMatch<primitive_src_type, C64>(src_literal, false);
    case C128:
      if (bitcast) {
        break;
      }
      return ConvertIfTypesMatch<primitive_src_type, C128>(src_literal, false);
    // Other types are not yet supported.
    default:
      break;
  }
  return Unimplemented("Converting from type %s to type %s is not implemented.",
                       PrimitiveType_Name(src_literal.shape().element_type()),
                       PrimitiveType_Name(primitive_dest_type));
}

StatusOr<Literal> ConvertSwitch(const LiteralBase& literal,
                                PrimitiveType primitive_dest_type,
                                bool bitcast) {
  TF_RET_CHECK(literal.shape().IsArray());
  if (literal.shape().element_type() == primitive_dest_type) {
    return literal.Clone();
  }
  switch (literal.shape().element_type()) {
#define CONVERT_IF_DEST_TYPE_MATCHES(type)                                \
  case (type):                                                            \
    return ConvertIfDestTypeMatches<(type)>(literal, primitive_dest_type, \
                                            bitcast);
    CONVERT_IF_DEST_TYPE_MATCHES(PRED)
    CONVERT_IF_DEST_TYPE_MATCHES(S8)
    CONVERT_IF_DEST_TYPE_MATCHES(S16)
    CONVERT_IF_DEST_TYPE_MATCHES(S32)
    CONVERT_IF_DEST_TYPE_MATCHES(S64)
    CONVERT_IF_DEST_TYPE_MATCHES(U8)
    CONVERT_IF_DEST_TYPE_MATCHES(U16)
    CONVERT_IF_DEST_TYPE_MATCHES(U32)
    CONVERT_IF_DEST_TYPE_MATCHES(U64)
    CONVERT_IF_DEST_TYPE_MATCHES(F16)
    CONVERT_IF_DEST_TYPE_MATCHES(F32)
    CONVERT_IF_DEST_TYPE_MATCHES(F64)
    CONVERT_IF_DEST_TYPE_MATCHES(BF16)
#undef CONVERT_IF_DEST_TYPE_MATCHES
      // Other types are not yet supported.
    default:
      return Unimplemented("%s from type %s to type %s is not implemented.",
                           (bitcast ? "Bitcast converting" : "Converting"),
                           PrimitiveType_Name(literal.shape().element_type()),
                           PrimitiveType_Name(primitive_dest_type));
  }
}

}  // namespace

StatusOr<Literal> LiteralBase::Convert(
    PrimitiveType primitive_dest_type) const {
  return ConvertSwitch(*this, primitive_dest_type, /*bitcast=*/false);
}

StatusOr<Literal> LiteralBase::BitcastConvert(
    PrimitiveType primitive_dest_type) const {
  if (primitive_util::BitWidth(shape().element_type()) !=
      primitive_util::BitWidth(primitive_dest_type)) {
    return InvalidArgument(
        "Cannot bitcast convert from %s to %s, bit widths are different: %d != "
        "%d",
        PrimitiveType_Name(shape().element_type()),
        PrimitiveType_Name(primitive_dest_type),
        primitive_util::BitWidth(shape().element_type()),
        primitive_util::BitWidth(primitive_dest_type));
  }
  return ConvertSwitch(*this, primitive_dest_type, /*bitcast=*/true);
}

StatusOr<Literal> LiteralBase::ConvertToShape(const Shape& dest_shape) const {
  if (!dest_shape.IsTuple()) {
    return Convert(dest_shape.element_type());
  }
  std::vector<Literal> elements;
  for (int i = 0; i < ShapeUtil::TupleElementCount(shape()); ++i) {
    auto element = LiteralSlice(*this, {i});
    TF_ASSIGN_OR_RETURN(
        auto new_element,
        element.ConvertToShape(ShapeUtil::GetSubshape(dest_shape, {i})));
    elements.push_back(std::move(new_element));
  }
  return MutableLiteralBase::MoveIntoTuple(absl::MakeSpan(elements));
}

/* static */ Literal MutableLiteralBase::MoveIntoTuple(
    absl::Span<Literal> elements) {
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

template <typename NativeT>
bool LiteralBase::Piece::EqualElementsInternal(
    const LiteralBase::Piece& other, std::vector<int64>* multi_index) const {
  if (multi_index->size() == subshape().rank()) {
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

bool LiteralBase::Piece::EqualElements(const LiteralBase::Piece& other) const {
  DCHECK(ShapeUtil::Compatible(subshape(), other.subshape()));

  if (ShapeUtil::Equal(subshape(), other.subshape()) &&
      LayoutUtil::IsDenseArray(subshape())) {
    CHECK_EQ(size_bytes(), other.size_bytes());
    return memcmp(buffer(), other.buffer(), size_bytes()) == 0;
  }

  std::vector<int64> multi_index;
  switch (subshape().element_type()) {
    case PRED:
      return EqualElementsInternal<bool>(other, &multi_index);
    case U8:
      return EqualElementsInternal<uint8>(other, &multi_index);
    case S16:
      return EqualElementsInternal<int16>(other, &multi_index);
    case S32:
      return EqualElementsInternal<int32>(other, &multi_index);
    case S64:
      return EqualElementsInternal<int64>(other, &multi_index);
    case U16:
      return EqualElementsInternal<uint16>(other, &multi_index);
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
    case C128:
      return EqualElementsInternal<complex128>(other, &multi_index);
    default:
      LOG(FATAL) << "Unimplemented: LiteralBase::Piece::EqualElements for type "
                 << PrimitiveType_Name(subshape().element_type());
  }
}

bool LiteralBase::operator==(const LiteralBase& other) const {
  if (!ShapeUtil::Compatible(shape(), other.shape())) {
    return false;
  }

  return root_piece().ForEachSubpieceWithBool(
      [&](const ShapeIndex& index, const Piece& piece) {
        if (!piece.subshape().IsArray()) {
          return true;
        }

        const Piece& other_piece = other.piece(index);
        if (!piece.EqualElements(other_piece)) {
          return false;
        }
        return true;
      });
}

namespace {

template <typename NativeT>
static bool AllElementsEqualValue(absl::Span<const NativeT> data,
                                  NativeT value) {
  for (int64 i = 0; i < data.size(); ++i) {
    if (data[i] != value) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool LiteralBase::IsAll(int8 value) const {
  return root_piece().ForEachSubpieceWithBool([&](const ShapeIndex& index,
                                                  const Piece& piece) {
    if (!piece.subshape().IsArray()) {
      return true;
    }

    auto piece_is_all = [&]() {
      switch (shape().element_type()) {
        case U8:
          if (value >= 0) {
            return AllElementsEqualValue<uint8>(piece.data<uint8>(), value);
          }
          return false;
        case U16:
          if (value >= 0) {
            return AllElementsEqualValue<uint16>(piece.data<uint16>(), value);
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
        case S16:
          return AllElementsEqualValue<int16>(piece.data<int16>(), value);
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
    return true;
  });
}

bool LiteralBase::IsAllFloat(float value) const {
  return root_piece().ForEachSubpieceWithBool(
      [&](const ShapeIndex& index, const Piece& piece) {
        if (!piece.subshape().IsArray()) {
          return true;
        }

        switch (shape().element_type()) {
          case F32:
            return AllElementsEqualValue<float>(piece.data<float>(), value);
          case F64:
            return AllElementsEqualValue<double>(piece.data<double>(), value);
          case F16:
            return AllElementsEqualValue<half>(piece.data<half>(),
                                               static_cast<half>(value));
          case BF16:
            return AllElementsEqualValue<bfloat16>(
                piece.data<bfloat16>(), static_cast<bfloat16>(value));
          default:
            return false;
        }
      });
}

bool LiteralBase::IsAllComplex(complex64 value) const {
  switch (shape().element_type()) {
    case C64:
      return AllElementsEqualValue<complex64>(root_piece().data<complex64>(),
                                              value);
    case C128:
      return AllElementsEqualValue<complex128>(root_piece().data<complex128>(),
                                               value);
    default:
      return false;
  }
}

bool LiteralBase::IsAllFirst() const {
  return root_piece().ForEachSubpieceWithBool(
      [&](const ShapeIndex& index, const Piece& piece) {
        if (!piece.subshape().IsArray()) {
          return true;
        }

        // Empty shapes are not all the first element since there is no first
        // element.
        if (ShapeUtil::IsZeroElementArray(piece.subshape())) {
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

            case C128: {
              auto data = piece.data<complex128>();
              return AllElementsEqualValue<complex128>(data, data[0]);
            }
            default:
              return false;
          }
        };

        if (!piece_is_all()) {
          return false;
        }
        return true;
      });
}

bool LiteralBase::IsR1Iota() const {
  if (!shape().IsArray()) {
    return false;
  }

  if (shape().rank() != 1) {
    return false;
  }

  auto is_iota_at_idx = [&](const int64 idx) {
    switch (shape().element_type()) {
      case U8:
        return Get<uint8>({idx}) == idx;
      case U16:
        return Get<uint16>({idx}) == idx;
      case U32:
        return Get<uint32>({idx}) == idx;
      case U64:
        return Get<uint64>({idx}) == idx;
      case S8:
        return Get<int8>({idx}) == idx;
      case S16:
        return Get<int16>({idx}) == idx;
      case S32:
        return Get<int32>({idx}) == idx;
      case S64:
        return Get<int64>({idx}) == idx;
      case F32:
        return Get<float>({idx}) == idx;
      case F64:
        return Get<double>({idx}) == idx;
      case F16:
        return Get<half>({idx}) == static_cast<half>(idx);
      case BF16:
        return Get<bfloat16>({idx}) == static_cast<bfloat16>(idx);
      case C64:
        return Get<complex64>({idx}) == complex64(idx, 0.0f);
      case C128:
        return Get<complex128>({idx}) == complex128(idx, 0.0f);
      case PRED:
        return Get<bool>({idx}) == idx;
      // token, opaque, tuple, etc. are all not iota.
      default:
        return false;
    }
  };

  const int64 elements = ShapeUtil::ElementsIn(shape());
  for (int64 idx = 0; idx < elements; ++idx) {
    if (!is_iota_at_idx(idx)) {
      return false;
    }
  }

  return true;
}

bool LiteralBase::IsZero(absl::Span<const int64> indices) const {
  CHECK(shape().IsArray());
  switch (shape().element_type()) {
    case U8:
      return Get<uint8>(indices) == 0;
    case U16:
      return Get<uint16>(indices) == 0;
    case U32:
      return Get<uint32>(indices) == 0;
    case U64:
      return Get<uint64>(indices) == 0;
    case S8:
      return Get<int8>(indices) == 0;
    case S16:
      return Get<int16>(indices) == 0;
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
    case C128:
      return Get<complex128>(indices) == complex128(0.0f, 0.0f);
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
                         const absl::Span<const NativeT> src) {
  *dest = RepeatedFieldT(src.begin(), src.end());
}

}  // namespace

void LiteralBase::Piece::WriteToProto(LiteralProto* proto) const {
  *proto->mutable_shape() = subshape().ToProto();
  switch (subshape().element_type()) {
    case PRED:
      CopyToRepeatedField(proto->mutable_preds(), data<bool>());
      break;
    case S8:
      proto->set_s8s(static_cast<const signed char*>(data<int8>().data()),
                     element_count());
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
    case U16:
      *proto->mutable_u16s() = string(
          reinterpret_cast<const char*>(data<uint16_t>().data()), size_bytes());
      if (!kLittleEndian) {
        ConvertEndianShort(proto->mutable_u16s());
      }
      break;
    case S16:
      *proto->mutable_s16s() = string(
          reinterpret_cast<const char*>(data<int16_t>().data()), size_bytes());
      if (!kLittleEndian) {
        ConvertEndianShort(proto->mutable_s16s());
      }
      break;
    case F16:
      *proto->mutable_f16s() = string(
          reinterpret_cast<const char*>(data<half>().data()), size_bytes());
      if (!kLittleEndian) {
        ConvertEndianShort(proto->mutable_f16s());
      }
      break;
    case BF16:
      *proto->mutable_bf16s() = string(
          reinterpret_cast<const char*>(data<bfloat16>().data()), size_bytes());
      if (!kLittleEndian) {
        ConvertEndianShort(proto->mutable_bf16s());
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
    case C128:
      for (complex128 value : data<complex128>()) {
        proto->add_c128s(value.real());
        proto->add_c128s(value.imag());
      }
      break;
    case TUPLE:
    case TOKEN:
      // Nothing to do but assign the shape which is done above.
      return;
    default:
      // TODO(b/111551621): Support serializing more PrimitiveTypes.
      LOG(FATAL) << "Unhandled primitive type "
                 << PrimitiveType_Name(subshape().element_type());
  }
}

const void* LiteralBase::Piece::untyped_data() const {
  CHECK(subshape().IsArray()) << ShapeUtil::HumanString(subshape());
  return buffer();
}

void* LiteralBase::Piece::untyped_data() {
  CHECK(subshape().IsArray()) << ShapeUtil::HumanString(subshape());
  return buffer();
}

namespace {

template <typename RepeatedFieldT, typename NativeT>
Status CopyFromRepeatedField(absl::Span<NativeT> dest,
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

Status LiteralBase::Piece::CopyFromProto(const LiteralProto& proto) {
  // These conditions should have been checked in
  // MutableLiteralBase::CreateFromProto.
  TF_RET_CHECK(proto.has_shape());
  Shape shape(proto.shape());
  TF_RET_CHECK(LayoutUtil::HasLayout(shape));
  TF_RET_CHECK(ShapeUtil::Equal(shape, subshape()));

  if (LayoutUtil::IsSparseArray(subshape())) {
    // Compute the number of elements (indices) in the sparse shape and reserve
    // the necessary space in spare_indices.
    TF_RET_CHECK(subshape().rank() != 0) << "Scalar shapes cannot be sparse";
    TF_RET_CHECK(proto.sparse_indices_size() % subshape().rank() == 0)
        << "Unexpected number of indices in proto ("
        << proto.sparse_indices_size() << ") for shape of rank "
        << subshape().rank();
    const int64 index_count = proto.sparse_indices_size() / subshape().rank();
    sparse_indices()->Resize(index_count);

    // Copy the indices from the proto into the SparseIndexArray object.
    TF_RETURN_IF_ERROR(CopyFromRepeatedField(sparse_indices()->mutable_data(),
                                             proto.sparse_indices()));
  }

  switch (subshape().element_type()) {
    case PRED:
      TF_RETURN_IF_ERROR(CopyFromRepeatedField(data<bool>(), proto.preds()));
      break;
    case S8: {
      auto s8_data = data<int8>();
      TF_RET_CHECK(proto.s8s().size() == s8_data.size());
      std::copy(proto.s8s().begin(), proto.s8s().end(), s8_data.begin());
    } break;
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
    case S16: {
      const string& s(proto.s16s());
      TF_RET_CHECK(data<int16_t>().size() * sizeof(int16_t) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(untyped_data()), s.size());
      }
    } break;
    case U16: {
      const string& s(proto.u16s());
      TF_RET_CHECK(data<uint16_t>().size() * sizeof(uint16_t) == s.size());
      memcpy(untyped_data(), s.data(), s.size());
      if (!kLittleEndian) {
        ConvertEndianShort(reinterpret_cast<char*>(untyped_data()), s.size());
      }
    } break;
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
      break;
    }
    case C128: {
      auto complex_data = data<complex128>();
      TF_RET_CHECK(proto.c128s_size() == complex_data.size() * 2);
      for (int64 i = 0; i < complex_data.size(); ++i) {
        complex_data[i] =
            complex128{proto.c128s(i * 2), proto.c128s(i * 2 + 1)};
      }
      break;
    }
    case TUPLE:
      return InvalidArgument("Should not be called on tuple shapes: %s",
                             ShapeUtil::HumanString(subshape()));
    default:
      return InvalidArgument("Is called on unsupported shape: %s",
                             ShapeUtil::HumanString(subshape()));
  }
  return Status::OK();
}

LiteralProto LiteralBase::ToProto() const {
  LiteralProto proto;
  root_piece().ForEachSubpiece(
      [&](const ShapeIndex& index, const Piece& piece) {
        LiteralProto* proto_piece = &proto;
        for (int64 i : index) {
          while (proto_piece->tuple_literals_size() <= i) {
            proto_piece->add_tuple_literals();
          }
          proto_piece = proto_piece->mutable_tuple_literals(i);
        }
        piece.WriteToProto(proto_piece);
      });

  if (LayoutUtil::IsSparseArray(shape())) {
    CopyToRepeatedField(proto.mutable_sparse_indices(),
                        sparse_indices()->data());
  }

  return proto;
}

const void* LiteralBase::untyped_data(const ShapeIndex& shape_index) const {
  return piece(shape_index).untyped_data();
}

void* MutableLiteralBase::untyped_data(const ShapeIndex& shape_index) {
  return piece(shape_index).untyped_data();
}

int64 LiteralBase::size_bytes(const ShapeIndex& shape_index) const {
  return piece(shape_index).size_bytes();
}

string LiteralBase::GetR1U8AsString() const {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().rank(), 1);
  CHECK_EQ(shape().element_type(), U8);
  return string(absl::bit_cast<const char*>(data<uint8>().data()),
                ShapeUtil::ElementsIn(shape()));
}

void MutableBorrowingLiteral::CopyPieceSubtree(const Shape& shape,
                                               Piece* src_piece,
                                               Piece* dest_piece) {
  DCHECK(ShapeUtil::Equal(src_piece->subshape(), dest_piece->subshape()))
      << "src_piece has shape: "
      << ShapeUtil::HumanString(src_piece->subshape())
      << "dest_piece has shape: "
      << ShapeUtil::HumanString(dest_piece->subshape());
  if (shape.IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      const Shape& subshape = shape.tuple_shapes(i);

      auto child_piece = Piece();
      child_piece.set_subshape(&subshape);

      CopyPieceSubtree(subshape, &src_piece->child(i), &child_piece);

      dest_piece->emplace_back(std::move(child_piece));
    }
  } else if (shape.IsArray()) {
    dest_piece->set_buffer(src_piece->buffer());
  } else {
    // If the shape is neither an array nor tuple, then it must be
    // zero-sized. Otherwise, some memory needs to be allocated for it.
    CHECK_EQ(dest_piece->size_bytes(), 0);
  }
}

MutableLiteralBase::~MutableLiteralBase() {}

MutableBorrowingLiteral::MutableBorrowingLiteral(
    const MutableBorrowingLiteral& literal)
    : MutableLiteralBase() {
  shape_ = absl::make_unique<Shape>(literal.shape());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.root_piece(), root_piece_);
}

MutableBorrowingLiteral& MutableBorrowingLiteral::operator=(
    const MutableBorrowingLiteral& literal) {
  shape_ = absl::make_unique<Shape>(literal.shape());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.root_piece(), root_piece_);

  return *this;
}

MutableBorrowingLiteral::MutableBorrowingLiteral(
    const MutableLiteralBase& literal)
    : MutableLiteralBase() {
  shape_ = absl::make_unique<Shape>(literal.shape());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.root_piece(), root_piece_);
}

MutableBorrowingLiteral::MutableBorrowingLiteral(MutableLiteralBase* literal)
    : MutableLiteralBase() {
  shape_ = absl::make_unique<Shape>(literal->shape());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal->root_piece(), root_piece_);
}

MutableBorrowingLiteral::MutableBorrowingLiteral(
    MutableBorrowingLiteral literal, const ShapeIndex& view_root)
    : MutableLiteralBase() {
  shape_ = absl::make_unique<Shape>(literal.piece(view_root).subshape());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = new Piece();
  root_piece_->set_subshape(shape_.get());

  CopyPieceSubtree(*shape_, &literal.piece(view_root), root_piece_);
}

MutableBorrowingLiteral::MutableBorrowingLiteral(const char* src_buf_ptr,
                                                 const Shape& shape)
    : MutableLiteralBase() {
  shape_ = absl::make_unique<Shape>(shape);
  CHECK(LayoutUtil::HasLayout(*shape_));
  CHECK(!shape_->IsTuple());

  root_piece_ = new Piece();
  root_piece_->set_buffer(const_cast<char*>(src_buf_ptr));
  root_piece_->set_subshape(shape_.get());
}

MutableBorrowingLiteral::~MutableBorrowingLiteral() {
  if (root_piece_ != nullptr) {
    root_piece_->ForEachMutableSubpiece(
        [&](const ShapeIndex& index, Piece* piece) {
          if (piece->buffer() != nullptr) {
            delete piece->sparse_indices();
          }
        });
    delete root_piece_;
  }
}

LiteralSlice::LiteralSlice(const LiteralBase& literal)
    : LiteralBase(), root_piece_(&literal.root_piece()) {}

LiteralSlice::LiteralSlice(const LiteralBase& literal,
                           const ShapeIndex& view_root)
    : LiteralBase(), root_piece_(&literal.piece(view_root)) {}

void BorrowingLiteral::BuildPieceSubtree(const Shape& shape, Piece* piece) {
  CHECK(shape.IsTuple());
  for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& subshape = shape.tuple_shapes(i);

    auto child_piece = Piece();
    child_piece.set_subshape(&subshape);

    if (subshape.IsTuple()) {
      BuildPieceSubtree(subshape, &child_piece);
    }

    piece->emplace_back(std::move(child_piece));
  }
}

BorrowingLiteral::BorrowingLiteral(const char* src_buf_ptr, const Shape& shape)
    : LiteralBase(), shape_(absl::make_unique<Shape>(shape)) {
  CHECK(shape_->IsArray());
  CHECK(LayoutUtil::HasLayout(*shape_));

  root_piece_ = Piece();
  root_piece_.set_buffer(const_cast<char*>(src_buf_ptr));
  root_piece_.set_subshape(shape_.get());
}

BorrowingLiteral::BorrowingLiteral(absl::Span<const char* const> src_buf_ptrs,
                                   const Shape& shape)
    : LiteralBase(), shape_(absl::make_unique<Shape>(shape)) {
  CHECK(shape_->IsTuple());
  CHECK(!ShapeUtil::IsNestedTuple(*shape_));
  CHECK_EQ(src_buf_ptrs.size(), ShapeUtil::TupleElementCount(*shape_));
  root_piece_ = Piece();
  root_piece_.set_subshape(shape_.get());
  BuildPieceSubtree(*shape_, &root_piece_);

  for (int i = 0; i < src_buf_ptrs.size(); ++i) {
    const auto& src_shape = shape_->tuple_shapes(i);
    CHECK(src_shape.IsArray());
    root_piece_.child(i).set_buffer(const_cast<char*>(src_buf_ptrs[i]));
  }
}

}  // namespace xla
