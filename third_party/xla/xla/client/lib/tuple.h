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

#ifndef XLA_CLIENT_LIB_TUPLE_H_
#define XLA_CLIENT_LIB_TUPLE_H_

#include "xla/client/xla_builder.h"
#include "xla/shape_tree.h"
#include "xla/statusor.h"

namespace xla {

// Returns a ShapeTree where each index is a GetTupleElement instruction for
// that subshape of the tuple.  The root index is the original argument.
absl::StatusOr<ShapeTree<XlaOp>> DisassembleTuple(XlaOp tuple);

// Assembles a tuple from a ShapeTree that contains the leaves of the tuple.
// Non-leaf elements of the ShapeTree are ignored.  DisassembleTuple and
// AssembleTuple are essentially inverse operations.
XlaOp AssembleTuple(XlaBuilder* builder, ShapeTree<XlaOp> elements);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_TUPLE_H_
