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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SLICING_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SLICING_H_

namespace xla {

// Updates a slice of 'x', i.e.,
// x[start[0], ..., start[n]] = update
XlaOp UpdateSlice(XlaOp x, XlaOp update, absl::Span<const int64> start);

// Performs a slice in the minor dimensions of a tensor.
// x[..., start[0]:end[0], ..., start[n]:end[n]]
XlaOp SliceInMinorDims(XlaOp x, absl::Span<const int64> start,
                       absl::Span<const int64> end);

// Updates a slice of 'x', where 'start' contains a list of minor dimensions:
// x[..., start[0]:..., ..., start[n]:...] = update
XlaOp UpdateSliceInMinorDims(XlaOp x, XlaOp update,
                             absl::Span<const int64> start);

// Performs a dynamic slice in the minor dimensions of a tensor.
XlaOp DynamicSliceInMinorDims(XlaOp x, absl::Span<const XlaOp> starts,
                              absl::Span<const int64> sizes);

XlaOp DynamicUpdateSliceInMinorDims(XlaOp x, XlaOp update,
                                    absl::Span<const XlaOp> starts);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SLICING_H_
