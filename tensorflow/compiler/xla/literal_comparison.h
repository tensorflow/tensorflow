/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Library for comparing literals without taking a dependency on testing
// libraries.

#ifndef TENSORFLOW_COMPILER_XLA_LITERAL_COMPARISON_H_
#define TENSORFLOW_COMPILER_XLA_LITERAL_COMPARISON_H_

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace literal_comparison {

// Returns ok if the given shapes have the same rank, dimension sizes, and
// primitive types.
Status EqualShapes(const Shape& expected, const Shape& actual);

// Returns ok if the expected and actual literals are (bitwise) equal for all
// elements in the literal. Also, asserts that the rank, dimensions sizes, and
// primitive type are equal.
Status Equal(const LiteralSlice& expected, const LiteralSlice& actual);

}  // namespace literal_comparison
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_COMPARISON_H_
