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

#ifndef TENSORFLOW_CORE_FRAMEWORK_FULL_TYPE_INFERENCE_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_FULL_TYPE_INFERENCE_UTIL_H_

#include <functional>
#include <string>
#include <vector>

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

// Helper for a no-op type inference function that indicates type inference
// should never alter the node's existing type.
// This is the same as not defining a type inference function at all, but
// explicitly communicates that intent.
TypeInferenceFn KeepExisting();

// A helper for a type inference function that indicates a single output that
// is a tensor of type t. This is the equivalent of a type construtor since it
// does not depend on inputs. This can be used with Tuple.
TypeInferenceFn Tensor(FullTypeId t);

// Helper for a type inference function which has the same type as the i'th
// input.
// The n arg allows multiple outputs, e.g. (T -> Product[T, T]).
// TODO(mdan): Drop defaults for readability if more non-(0, 1) cases appear.
// TODO(mdan): Rename to just Replicate.
TypeInferenceFn ReplicateInput(int i = 0, int n = 1);

// Helper for a type inference function which has the same type as a variadic
// number of inputs, e.g. (T, T -> Product[T]), (T, T, T -> Product[T]), etc.
// Infers the meet of the input types, in the sense of type meets (see
// https://en.wikipedia.org/wiki/Join_and_meet). This implementation is
// simplified to require the two inputs are a subtype of another.
TypeInferenceFn Merge();

// Helper for ops with semantics of encoding an input, that is,
// `T -> Encoded[T, <t>]`, where <t> is the encoded type.
TypeInferenceFn Encode(FullTypeId t, int i);

// Helper for ops with semantics of encoding an input, that is,
// `Encoded[T, <t>] -> T`, where <t> is the encoded type.
TypeInferenceFn Decode(FullTypeId t, int i);

// Helper for the type inference counterpart of Unary, that is (U ->
// PRODUCT[<t>[U]]), where <t> is parameterized by this factory, and U is the
// type of the input specified by element_idx.
// Note: when we migrate to a more formal type definition of an op, these two
// functions will naturally merge.
TypeInferenceFn UnaryContainerCreate(FullTypeId t, int element_idx);

// Helper for ops with semantics of adding an element to a container (<t>[T]),
// that is (<t>[U], V -> PRODUCT[<t>[Union[U, V]]]), where <t> is parameterized
// by this factory, U is the type of the input specified by container_idx, and V
// is the type of the input specified by element_idx. The homogeneous arg allows
// for constraints which guarantee that U and V must have a subtyping
// relationship, case in which either V or U is selected, whichever is the
// supertype.
TypeInferenceFn UnaryContainerAdd(FullTypeId t, int container_idx,
                                  int element_idx, bool homogeneous);

// Helper for ops with semantics of unstacking multiple inputs into a container
// `<t>[T1, ..., Tn]`, that is `T1, ..., Tn -> <t>[PRODUCT[U1, ..., Un]]`
// where Ui is obtained from an "unstack" mapping T -> U. Both <t> and the
// "unstack" mapping are parameterized by this factory.
// Note that when the "unstack" function is the identity function, this becomes
// equivalent to ContainerCreate.
TypeInferenceFn MultiaryUnstack(
    FullTypeId t, std::function<FullTypeDef(const FullTypeDef&)> unstack);

// Helper for ops with semantics of applying some transformation to the
// elements of a container:
// `<t>[PRODUCT[T1, ..., Tn]] -> <t>[PRODUCT[U1, ..., Un]]`,
// where Ui is obtained by applying a map T -> U. Both <t> and the "map"
// function are parameterized by this factory. See BatchTensor and ShardTensor
// for examples of "map".
TypeInferenceFn ContainerMap(
    FullTypeId t, int input_idx,
    std::function<FullTypeDef(const FullTypeDef&)> map);

// Helper for ops with semantics of repacking some element from a container to
// another `<t> -> <u>`, in a covariant way, that is, `<t>[T] -> <u>[T]`. <t>
// and <u> are parameterized by this factory. The input type is specified by
// element_idx.
TypeInferenceFn MapCovariant(FullTypeId t, FullTypeId u, int input_idx);

// Helper for ops with semantics of calling a function. The function is
// specified indirectly, as the name of an attribute that holds the actual
// function name.
TypeInferenceFn FunctionCall(const string& func_attr_name);

// Compose the type of a function by concatenating the outputs of multiple
// type inference functions. If func_list is {type inference function 1, type
// inference function 2} which return PRODUCT[T1], PRODUCT[T2] resprectively,
// the result is PRODUCT[T1, T2], This supports the Merge op that has an index
// output in addition to the result of the Merge type inference function.
TypeInferenceFn Tuple(const std::vector<TypeInferenceFn>& func_list);

// Auxiliary constructs to help creation of type inference functions.
// TODO(mdan): define these as type inference functions as well.

// Mapping function representing the type function for unstacking of
// Tensor (or Tensor-like) types. Note that this is a helper to use with
// other type inference functions; it's not a function itself.
// TODO(mdan): Replace with a trait, when available.
FullTypeDef UnstackTensor(const FullTypeDef& t);

// Mapping function representing the type function for an op that changes the
// batch size of dataset. Note that this is a helper to use with other type
// inference functions; it's not a function itself.
// TODO(mdan): Replace with a trait, when available.
FullTypeDef BatchTensor(const FullTypeDef& t);

// Mapping function representing the type function for an op that creates a
// fixed (given) number of tensors of a size calculated based on the input. Note
// that this is a helper to use with other type inference functions; it's not a
// function itself.
// TODO(mdan): Replace with a trait, when available.
FullTypeDef ShardTensor(const FullTypeDef& t);
}  // namespace full_type

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FULL_TYPE_INFERENCE_UTIL_H_
