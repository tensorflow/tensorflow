/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_ARITHMETIC_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_ARITHMETIC_H_

#include <memory>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

using XlaOpGenerator = std::function<XlaOp(XlaOp, XlaOp)>;

// Creates a scalar computation based on a lambda and returns it.
XlaComputation CreateScalarComputation(const string& name, PrimitiveType type,
                                       XlaBuilder* builder,
                                       XlaOpGenerator generator);

// Creates a scalar add computation and returns it.
XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                          XlaBuilder* builder);

// Creates a scalar multiply computation and returns it.
XlaComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               XlaBuilder* builder);

// Creates a scalar ge computation and returns it.
XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                         XlaBuilder* builder);

// Creates a scalar max computation and returns it.
XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                          XlaBuilder* builder);

// Creates a scalar min computation and returns it.
XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                          XlaBuilder* builder);

// Creates a scalar logical AND computation and returns it.
XlaComputation CreateScalarAndComputation(PrimitiveType type,
                                          XlaBuilder* builder);

// Creates a scalar logical OR computation and returns it.
XlaComputation CreateScalarOrComputation(PrimitiveType type,
                                         XlaBuilder* builder);

// This is to be used for general purpose "identity" like reductions with zero
// for any type (ie. boolean operations for PRED and Add for real numbers).
// As an example, this operation can be used for a situation of:
// x_type = type(x)
// op = CreateScalarIdentityWithZeroComputation(x_type)
// ASSERT_TRUE(op(x, 0) == x)
//
// This functionality is used for operations that are similar to a slice,
// gather, or broadcast, but are created through a reduction.
XlaComputation CreateScalarIdentityWithZeroComputation(PrimitiveType type,
                                                       XlaBuilder* builder);

// Returns whether any predicate in "predicates" is set.
//
// Note: if predicates is zero-sized, Any() vacuously returns false.
XlaOp Any(XlaOp predicates);

// Returns the argmax of `input` along `axis`. `output_type` is the type to
// use for the output. The `tie_low` argument drives the index selection is case
// of same values. If `true` (default behavior) the lowest index will be
// returned, otherwise the higher. The tie_low argument only applies if `stable`
// is true.
XlaOp ArgMax(XlaOp input, PrimitiveType output_type, int axis,
             bool stable = false, bool tie_low = true);

// Returns the argmin of `input` along `axis`. `output_type` is the type to
// use for the output. The `tie_low` argument drives the index selection is case
// of same values. If `true` (default behavior) the lowest index will be
// returned, otherwise the higher. The tie_low argument only applies if `stable`
// is true.
XlaOp ArgMin(XlaOp input, PrimitiveType output_type, int axis,
             bool stable = false, bool tie_low = true);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_ARITHMETIC_H_
