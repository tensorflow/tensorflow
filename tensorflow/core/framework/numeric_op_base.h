/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_NUMERIC_OP_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_NUMERIC_OP_BASE_H_

#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// One input and one output, both the same type.
template <class T, class OpKernelT, class OpKernelConstructionT>
class UnaryOpBase : public OpKernelT {
 public:
  explicit UnaryOpBase(OpKernelConstructionT* construction) :
      OpKernelT(construction) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(construction, construction->MatchSignature({dt}, {dt}));
  }
};

// Two inputs and one output, all the same type.
template <class T, class OpKernelT, class OpKernelConstructionT>
class BinaryOpBase : public OpKernelT {
 public:
  explicit BinaryOpBase(OpKernelConstructionT* construction) :
      OpKernelT(construction) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(construction, construction->MatchSignature({dt, dt}, {dt}));
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_NUMERIC_OP_BASE_H_
