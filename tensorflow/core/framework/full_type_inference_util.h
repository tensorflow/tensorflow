/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_FULL_TYPE_INFERENCE_UTIL1_H_
#define TENSORFLOW_CORE_FRAMEWORK_FULL_TYPE_INFERENCE_UTIL1_H_

#include <functional>
#include <string>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

namespace full_type {

// TODO(mdan): Specific helpers won't get too far. Use a parser instead.

// Helpers that allow shorthand expression for the more common kinds of type
// inference functions.
// TODO(mdan): Break into separate header if it grows.
// Note: The information contained in these functions is also expressed to some
// extent by opdef attributes of the kind "input: T, output T". But in that
// context, T has strong DType semantics (i.e. T is DT_VARIANT for most
// interesting cases). The logic here extends to the op's FullType, so it's best
// to keep them separate, even though it leads to some redundancy. The
// same can be said about the shape inference function.

// Note: Unlike type constructors, which describe op definitions, type inference
// functions are meant to modify the type information of specific nodes (i.e.
// NodeDef proto).

// Helper for a type inference function which has the same type as the i'th
// input.
// The n arg allows multiple outputs, e.g. (T -> Product[T, T]).
// TODO(mdan): Drop defaults for readability if more non-(0, 1) cases appear.
ForwardTypeInferenceFn ReplicateInput(int i = 0, int n = 1);

// Helper for a type inference function which has the same type as a variadic
// number of inputs, e.g. (T, T -> Product[T]), (T, T, T -> Product[T]), etc.
// Assumes all inputs are of identical type.
ForwardTypeInferenceFn ReplicateIdenticalInputs();

// Helper for the type inference counterpart of Unary, that is (U ->
// PRODUCT[<t>[U]]), where <t> is parameterized by this factory, and U is the
// type of the input specified by container_idx.
// Note: when we migrate to a more formal type definition of an op, these two
// functions will naturally merge.
ForwardTypeInferenceFn UnaryContainerCreate(FullTypeId t, int container_idx);

// Helper for ops with semantics of adding an element to a container (<t>[T]),
// that is (<t>[U], V -> PRODUCT[<t>[Union[U, V]]]), where <t> is parameterized
// by this factory, U is the type of the input specified by container_idx, and V
// is the type of the input specified by element_idx. The homogeneous arg allows
// for constraints which guarantee that U and V must have a subtyping
// relationship, case in which either V or U is selected, whichever is the
// supertype.
ForwardTypeInferenceFn UnaryContainerAdd(FullTypeId t, int container_idx,
                                         int element_idx, bool homogeneous);

}  // namespace full_type

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FULL_TYPE_INFERENCE_UTIL1_H_
