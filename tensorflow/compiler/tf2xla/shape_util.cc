/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/shape_util.h"

#include <numeric>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace {

absl::Status PopulateInfeedLayoutVector(const xla::Shape& shape,
                                        std::vector<int>* layouts) {
  if (shape.IsTuple()) {
    int64_t tuple_elements = xla::ShapeUtil::TupleElementCount(shape);
    for (int64_t i = 0; i < tuple_elements; ++i) {
      const xla::Shape& subshape =
          xla::ShapeUtil::GetTupleElementShape(shape, i);
      TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(subshape, layouts));
    }
  } else if (xla::LayoutUtil::HasLayout(shape)) {
    for (auto dim : xla::LayoutUtil::MinorToMajor(shape)) {
      layouts->push_back(dim);
    }
  } else {
    layouts->insert(layouts->end(), shape.dimensions().size(), -1);
  }
  return absl::OkStatus();
}

// Populate the output layout unless the minor_to_major array contains all -1
// value, in which case the layout is considered missing and the API returns
// false.
absl::StatusOr<bool> MakeLayout(absl::Span<const int64_t> minor_to_major,
                                xla::Layout* layout) {
  if (std::all_of(minor_to_major.begin(), minor_to_major.end(),
                  [](int64_t dim) { return dim == -1; })) {
    return false;
  }
  std::vector<bool> dim_present(minor_to_major.size(), false);
  for (auto dim : minor_to_major) {
    const int minor_to_major_size = minor_to_major.size();
    if (dim < 0 || dim >= minor_to_major_size) {
      return errors::InvalidArgument("Layout dimension out of range: dim=", dim,
                                     " rank=", minor_to_major.size());
    }
    if (dim_present[dim]) {
      return errors::InvalidArgument("Repeated layout dimension: dim=", dim);
    }
    dim_present[dim] = true;
  }
  *layout = xla::LayoutUtil::MakeLayout(minor_to_major);
  return true;
}

absl::Status AssignLayout(
    absl::Span<const int64_t> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* shape) {
  xla::Layout layout;
  TF_ASSIGN_OR_RETURN(bool has_layout, MakeLayout(minor_to_major, &layout));
  if (!has_layout && layout_func) {
    layout = layout_func(*shape);
  }
  *shape->mutable_layout() = layout;
  return absl::OkStatus();
}

}  // namespace

// Convert an XLA Shape into the equivalent TensorFlow shape.
absl::Status XLAShapeToTensorShape(const xla::Shape& shape,
                                   TensorShape* tensor_shape) {
  if (shape.IsTuple()) {
    return errors::InvalidArgument("XLA shape ",
                                   xla::ShapeUtil::HumanString(shape),
                                   " cannot be converted to a TensorShape");
  }
  *tensor_shape = TensorShape();
  for (int i = 0; i < shape.dimensions().size(); ++i) {
    TF_RETURN_IF_ERROR(tensor_shape->AddDimWithStatus(shape.dimensions(i)));
  }
  return absl::OkStatus();
}

// Convert a TensorShape into the equivalent XLA Shape proto.
absl::Status TensorShapeToXLAShape(DataType dtype,
                                   const PartialTensorShape& tensor_shape,
                                   xla::Shape* shape) {
  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return absl::OkStatus();
}

absl::Status TensorShapeToBoundedXLAShape(
    DataType dtype, const PartialTensorShape& tensor_shape,
    const TensorShape& bound, xla::Shape* shape) {
  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  if (tensor_shape.unknown_rank()) {
    // For unknown shape, create a rank 1 size 0 tensor.
    *shape = xla::ShapeUtil::MakeShapeWithDenseLayout(type, {0}, {0});
    return absl::OkStatus();
  }

  if (tensor_shape.dims() != bound.dims()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "`tensor_shape` and `bound` have different ranks. tensor_shape=",
        tensor_shape.dims(), "vs bound=", bound.dims()));
  }

  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    if (bound.dim_size(d) < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Bound dimension ", d, " has unknown size."));
    }
    if (tensor_shape.dim_size(d) > 0 &&
        bound.dim_size(d) != tensor_shape.dim_size(d)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Bounding shape does not match dynamic shape for known dimension ", d,
          tensor_shape.dim_size(d), " vs ", bound.dim_size(d)));
    }
    dimensions[d] = bound.dim_size(d);
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);
  xla::Shape result =
      xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimensions, layout);
  for (int d = 0; d < rank; ++d) {
    if (tensor_shape.dim_size(d) < 0) {
      result.set_dynamic_dimension(d, true);
    }
  }
  *shape = result;
  return absl::OkStatus();
}

xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const PartialTensorShape& tensor_shape) {
  if (tensor_shape.unknown_rank()) {
    // For unknown shape, create a rank 1 size 0 tensor.
    return xla::ShapeUtil::MakeShapeWithDenseLayout(type, {0}, {0});
  }
  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
    if (dimensions[d] < 0) {
      LOG(WARNING) << "Unable to convert TF shape with dynamic size to XLA "
                      "shape; returning unknown sentinel value";
      return xla::ShapeUtil::MakeShapeWithDenseLayout(type, {0}, {0});
    }
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);
  xla::Shape result =
      xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimensions, layout);
  return result;
}

// Convert a TensorShape into the equivalent XLA Shape proto.
absl::Status TensorShapeToXLAShape(DataType dtype,
                                   const TensorShape& tensor_shape,
                                   xla::Shape* shape) {
  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return absl::OkStatus();
}

absl::StatusOr<xla::Shape> TensorShapeToXLAShape(
    DataType dtype, const TensorShape& tensor_shape) {
  xla::Shape out;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, tensor_shape, &out));
  return out;
}

xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const TensorShape& tensor_shape) {
  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);

  return xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimensions, layout);
}

absl::StatusOr<std::vector<int>> GetShapeLayoutVector(const xla::Shape& shape) {
  std::vector<int> layouts;
  TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(shape, &layouts));
  return layouts;
}

absl::Status GetShapeWithLayout(
    const xla::Shape& input_shape, absl::Span<const int64_t> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* output_shape) {
  if (input_shape.IsTuple()) {
    int64_t tuple_elements = xla::ShapeUtil::TupleElementCount(input_shape);
    std::vector<xla::Shape> shapes;
    shapes.reserve(tuple_elements);
    size_t position = 0;
    for (int64_t i = 0; i < tuple_elements; ++i) {
      const xla::Shape& shape =
          xla::ShapeUtil::GetTupleElementShape(input_shape, i);
      if (shape.IsTuple()) {
        return errors::InvalidArgument(
            "Nested tuples not supported: ",
            xla::ShapeUtil::HumanString(input_shape));
      }
      int64_t rank = shape.dimensions().size();
      if (position + rank > minor_to_major.size()) {
        return errors::InvalidArgument(
            "Not enough layout attribute elements: position=", position,
            " rank=", rank, " elements=", minor_to_major.size());
      }
      shapes.push_back(shape);
      TF_RETURN_IF_ERROR(AssignLayout(
          absl::Span<const int64_t>(minor_to_major).subspan(position, rank),
          layout_func, &shapes.back()));
      position += rank;

      VLOG(4) << "Shape[" << i
              << "] = " << xla::ShapeUtil::HumanStringWithLayout(shapes.back());
    }
    if (position != minor_to_major.size()) {
      return errors::InvalidArgument(
          "Too many elements passed in the layout attribute: position=",
          position, " size=", minor_to_major.size());
    }
    *output_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  } else {
    int64_t rank = input_shape.dimensions().size();
    const int64_t minor_to_major_size = minor_to_major.size();
    if (rank != minor_to_major_size) {
      return errors::InvalidArgument(
          "Wrong number of layout attribute elements: rank=", rank,
          " elements=", minor_to_major.size());
    }
    *output_shape = input_shape;
    TF_RETURN_IF_ERROR(AssignLayout(minor_to_major, layout_func, output_shape));

    VLOG(4) << "Shape[] = "
            << xla::ShapeUtil::HumanStringWithLayout(*output_shape);
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
