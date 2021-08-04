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

// This file defines the attributes used in the TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_ATTRIBUTES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_ATTRIBUTES_H_

#include "tensorflow/core/ir/types/dialect.h"

namespace mlir {
namespace TF {

// This all moved under tensorflow/core/ir/types and these using declaration are
// to help with the transition.
using mlir::tf_type::FuncAttr;         // NOLINT
using mlir::tf_type::PlaceholderAttr;  // NOLINT
using mlir::tf_type::ShapeAttr;        // NOLINT

}  // end namespace TF
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_ATTRIBUTES_H_
