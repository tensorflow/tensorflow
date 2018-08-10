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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_SCATTER_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_SCATTER_H_

#include <functional>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace tensorflow {

// Builds an XLA computation that performs a scatter operation on `buffer`,
// returning an updated buffer.
// For each i0, i1, ..., sets
// buffer[indices[i0, i1, ...], ...] := updates[i0, i1, ...]
//
// If `indices_are_vectors` is false, then each index in indices is a scalar,
// and the shape of `indices` must be a prefix of the shape of updates.
// Otherwise, `indices_are_vectors`, then indices are multidimensional and the
// minor dimension of `indices` represents a vector of indices.
//
// If any indices are negative, the corresponding update is discarded.
//
// If a `combiner` is provided, updates are combined with the existing values in
// the buffer using the combiner function. Otherwise, the updates replace the
// existing values. The order of updates is implementation-defined.
xla::StatusOr<xla::XlaOp> XlaScatter(
    const xla::XlaOp& buffer, const xla::XlaOp& updates,
    const xla::XlaOp& indices, bool indices_are_vectors,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>&
        combiner,
    xla::XlaBuilder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_SCATTER_H_
