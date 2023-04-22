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

// Helpers that allow shorthand expression for the more common kinds of type
// constructors.
// TODO(mdan): Specific helpers won't get too far. Use a parser instead.

// Helper for a type constructor of <t>[FT_VAR[<param_name>]].
// Note: Unary refers to a parametric type of a single argument, not to the
// number of an op's return values.
OpTypeConstructor Unary(FullTypeId t, const string& var_name);

// Helper for a type constructor of <t>[FT_UNKNOWN].
// Note: Unary refers to a parametric type of a single argument, not to the
// number of an op's return values.
OpTypeConstructor UnaryGeneric(FullTypeId t);

// Helper for a type constructor of <t>[FT_TENSOR[<dtype>]].
// Note: Unary refers to a parametric type of a single argument, not to the
// number of an op's return values.
OpTypeConstructor UnaryTensorContainer(FullTypeId t, FullTypeId dtype);

// Type specialization and inference logic. This function narrows the type
// specified in an op definition. Such types are usually generic and dependent
// on input types. This function resolves the output types based on the input
// types specified in a given node def.
StatusOr<FullTypeDef> SpecializeType(const AttrSlice& attrs,
                                     const OpDef& op_def);

}  // namespace full_type

}  // namespace tensorflow

#endif  // CORE_FRAMEWORK_FULL_TYPE_UTIL_H_
