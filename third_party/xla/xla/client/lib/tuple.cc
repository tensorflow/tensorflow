/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/client/lib/tuple.h"

#include <utility>

#include "absl/container/inlined_vector.h"
#include "xla/client/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<ShapeTree<XlaOp>> DisassembleTuple(XlaOp tuple) {
  TF_ASSIGN_OR_RETURN(Shape shape, tuple.builder()->GetShape(tuple));
  ShapeTree<XlaOp> result(shape);
  result.ForEachMutableElement([&](ShapeIndexView index, XlaOp* element) {
    if (index.empty()) {
      *element = tuple;
    } else {
      ShapeIndexView parent_index = index.subspan(0, index.size() - 1);
      XlaOp parent = result.element(parent_index);
      *element = GetTupleElement(parent, index.back());
    }
  });
  return std::move(result);
}

XlaOp AssembleTuple(XlaBuilder* builder, ShapeTree<XlaOp> elements) {
  elements.ForEachMutableElementPostOrder(
      [&](const ShapeIndex& index, XlaOp* element) {
        const Shape& subshape = ShapeUtil::GetSubshape(elements.shape(), index);
        if (subshape.IsTuple()) {
          absl::InlinedVector<XlaOp, 2> children;
          ShapeIndex child_index = index;
          for (int i = 0; i < subshape.tuple_shapes_size(); ++i) {
            child_index.push_back(i);
            children.push_back(elements.element(child_index));
            child_index.pop_back();
          }
          *element = Tuple(builder, children);
        }
      });
  return elements.element({});
}

}  // namespace xla
