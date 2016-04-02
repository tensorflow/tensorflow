/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_OPS_CONST_OP_H_
#define TENSORFLOW_CC_OPS_CONST_OP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

// If a shape is specified, you may either provide the same number of values,
// or a single value and that value will be duplicated to fill out the Tensor.
#define DECLARE_CONST(TYPE)                                                  \
  Node* Const(TYPE s, const GraphDefBuilder::Options& options); /* Scalar */ \
  Node* Const(gtl::ArraySlice<TYPE> v,                                       \
              const GraphDefBuilder::Options& options); /* Vector */         \
  Node* Const(gtl::ArraySlice<TYPE> t, const TensorShape& shape,             \
              const GraphDefBuilder::Options& options); /* Tensor */         \
  inline Node* Const(std::initializer_list<TYPE> v, /* Vector using {...} */ \
                     const GraphDefBuilder::Options& options) {              \
    return Const(gtl::ArraySlice<TYPE>(v), options);                         \
  }                                                                          \
  inline Node* Const(std::initializer_list<TYPE> t, /* Tensor using {...} */ \
                     const TensorShape& shape,                               \
                     const GraphDefBuilder::Options& options) {              \
    return Const(gtl::ArraySlice<TYPE>(t), shape, options);                  \
  }

DECLARE_CONST(Eigen::half);
DECLARE_CONST(float);
DECLARE_CONST(double);
DECLARE_CONST(int32);
DECLARE_CONST(uint8);
DECLARE_CONST(int16);
DECLARE_CONST(int8);
DECLARE_CONST(complex64);
DECLARE_CONST(complex128);
DECLARE_CONST(int64);
DECLARE_CONST(bool);

#undef DECLARE_CONST

// String
Node* Const(StringPiece s, const GraphDefBuilder::Options& options);
Node* Const(gtl::ArraySlice<string> v, const GraphDefBuilder::Options& options);
Node* Const(gtl::ArraySlice<string> t, const TensorShape& shape,
            const GraphDefBuilder::Options& options);
inline Node* Const(std::initializer_list<string> v,
                   const GraphDefBuilder::Options& options) {
  return Const(gtl::ArraySlice<string>(v), options);
}
inline Node* Const(std::initializer_list<string> t, const TensorShape& shape,
                   const GraphDefBuilder::Options& options) {
  return Const(gtl::ArraySlice<string>(t), shape, options);
}

// A Tensor of any type.
Node* Const(const Tensor& t, const GraphDefBuilder::Options& options);
Node* Const(const TensorProto& proto, const GraphDefBuilder::Options& options);

template <class T>
Node* EmptyConst(const GraphDefBuilder::Options& options) {
  return Const(gtl::ArraySlice<T>(), options);
}

// TODO(josh11b): Support other types (e.g. quantized ints, float16).

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CONST_OP_H_
