/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef CORE_FRAMEWORK_FULL_TYPE_UTIL_H_
#define CORE_FRAMEWORK_FULL_TYPE_UTIL_H_

#include <functional>
#include <string>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

namespace full_type {

// TODO(mdan): Specific helpers won't get too far. Use a parser instead.
// TODO(mdan): Move constructors into a separate file.

// Helpers that allow shorthand expression for the more common kinds of type
// constructors.
// Note: The arity below refers to the number of arguments of parametric types,
// not to the number of return values from a particular op.
// Note: Type constructors are meant to create static type definitions in the
// op definition (i.e. the OpDef proto).

// Helper for a no-op type constructor that indicates that the node's type
// should be set by external means (typically by the user).
OpTypeConstructor NoOp();

// Helper for a trivial type constructor that indicates a node has no
// outputs (that is, its output type is an empty TFT_PRODUCT).
OpTypeConstructor NoOutputs();

// Helper for a type constructor of <t>[] (with no parameters).
OpTypeConstructor Nullary(FullTypeId t);

// Helper for a type constructor of <t>[FT_VAR[<var_name>]].
OpTypeConstructor Unary(FullTypeId t, const string& var_name);

// Helper for a type constructor of <t>[FT_ANY].
OpTypeConstructor UnaryGeneric(FullTypeId t);

// Helper for a type constructor of <t>[FT_TENSOR[<dtype>]].
OpTypeConstructor UnaryTensorContainer(FullTypeId t, FullTypeId dtype);

// Helper for a type constructor of <t>[FT_VAR[<var_name>]].
OpTypeConstructor UnaryTensorContainer(FullTypeId t, const string& var_name);

// Helper for a type constructor of
// <t>[FT_FOR_EACH[
//     FT_PRODUCT,
//     FT_TENSOR[FT_VAR[<var_name>]],
//     FT_VAR[<var_name>]].
// Multi-valued type variables will expand the template (see full_type.proto).
OpTypeConstructor VariadicTensorContainer(FullTypeId t, const string& var_name);

// Type specialization and inference logic. This function narrows the type
// specified in an op definition. Such types are usually generic and dependent
// on input types. This function resolves the output types based on the input
// types specified in a given node def.
Status SpecializeType(const AttrSlice& attrs, const OpDef& op_def,
                      FullTypeDef& target);

const FullTypeDef& GetArgDefaultUnset(const FullTypeDef& t, int i);
const FullTypeDef& GetArgDefaultAny(const FullTypeDef& t, int i);

bool IsEqual(const FullTypeDef& lhs, const FullTypeDef& rhs);

bool IsSubtype(const FullTypeDef& lhs, const FullTypeDef& rhs,
               bool covariant = true);

uint64_t Hash(const FullTypeDef& arg);

// Determine if the given fulltype is a host memory type.
// While it is prefered that Placer (placer.cc and colocation_graph.cc) make
// all host memory type placement decisions, any decision made elsewhere
// should use this function (e.g. instead of assuming that all variants never
// contain host memory types).
inline bool IsHostMemoryType(const FullTypeDef& t) {
  switch (t.type_id()) {
    case TFT_TENSOR:
      return IsHostMemoryType(full_type::GetArgDefaultAny(t, 0));
    case TFT_ARRAY:
      return IsHostMemoryType(full_type::GetArgDefaultAny(t, 0));
    case TFT_DATASET:
      return true;
    case TFT_MUTEX_LOCK:
      return true;
    case TFT_RAGGED:
      return IsHostMemoryType(full_type::GetArgDefaultAny(t, 0));
    case TFT_STRING:
      return true;
    default:
      return false;
  }
}

}  // namespace full_type

}  // namespace tensorflow

#endif  // CORE_FRAMEWORK_FULL_TYPE_UTIL_H_
