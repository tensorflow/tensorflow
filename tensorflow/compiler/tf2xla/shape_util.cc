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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Convert an XLA Shape into the equivalent TensorFlow shape.
Status XLAShapeToTensorShape(const xla::Shape& shape,
                             TensorShape* tensor_shape) {
  if (xla::ShapeUtil::IsTuple(shape)) {
    return errors::InvalidArgument("XLA shape ",
                                   xla::ShapeUtil::HumanString(shape),
                                   " cannot be converted to a TensorShape");
  }
  *tensor_shape = TensorShape();
  for (int i = 0; i < xla::ShapeUtil::Rank(shape); ++i) {
    tensor_shape->AddDim(shape.dimensions(i));
  }
  return Status::OK();
}

// Convert a TensorShape into the equivalent XLA Shape proto.
Status TensorShapeToXLAShape(DataType dtype, const TensorShape& tensor_shape,
                             xla::Shape* shape) {
  int rank = tensor_shape.dims();
  std::vector<int64> dimensions(rank);
  std::vector<int64> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);

  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));

  *shape = xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
  return Status::OK();
}

}  // namespace tensorflow
