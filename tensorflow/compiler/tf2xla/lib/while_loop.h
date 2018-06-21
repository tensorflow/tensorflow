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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_WHILE_LOOP_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_WHILE_LOOP_H_

#include <functional>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Function that builds a loop condition. Takes as input a sequence of input
// values, and returns a boolean value representing if the condition succeeds.
typedef std::function<xla::StatusOr<xla::XlaOp>(gtl::ArraySlice<xla::XlaOp>,
                                                xla::XlaBuilder*)>
    LoopConditionFunction;

// Function that builds a loop body. Takes as input a sequence of input values
// and returns a sequence of output values.
typedef std::function<xla::StatusOr<std::vector<xla::XlaOp>>(
    gtl::ArraySlice<xla::XlaOp>, xla::XlaBuilder*)>
    LoopBodyFunction;

// Helper function for building an XLA while loop, where the values carried by
// the loop are a tuple of values, e.g., (a, b, c):
// while(
//   condition: (a, b, c) -> bool,
//   body: (a, b, c) -> (a, b, c)
//   init: (a, b, c)
// )
// 'name' is a descriptive name for the loop.
xla::StatusOr<std::vector<xla::XlaOp>> XlaWhileLoop(
    const LoopConditionFunction& condition_function,
    const LoopBodyFunction& body_function,
    gtl::ArraySlice<xla::XlaOp> initial_values, StringPiece name,
    xla::XlaBuilder* builder);

// Builds an XLA loop that repeats a computation `num_iterations` times.
//
// The body function (ForEachIndexBodyFunction) takes as input a pair of
// (current iteration number, loop-carried values), and returns an updated
// vector of the loop-carried values.
typedef std::function<xla::StatusOr<std::vector<xla::XlaOp>>(
    xla::XlaOp, gtl::ArraySlice<xla::XlaOp>, xla::XlaBuilder*)>
    ForEachIndexBodyFunction;

xla::StatusOr<std::vector<xla::XlaOp>> XlaForEachIndex(
    int64 num_iterations, xla::PrimitiveType num_iterations_type,
    const ForEachIndexBodyFunction& body_function,
    gtl::ArraySlice<xla::XlaOp> initial_values, StringPiece name,
    xla::XlaBuilder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_WHILE_LOOP_H_
