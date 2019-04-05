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

// Gathers values along an axis specified by dim.
//
// For a 3-D tensor the output is specified by:
//
// out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
// out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
// out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
//
// If `input` is an n-dimensional tensor with size
// [X0,X1,X2,..XN] and dim = i `index` must be an n-dimensional tensor with size
// [X0,X1,...Y,Xi+1,...,X[N] where y >= 1 and `out` will have the same sizes as
// `index`.
XlaOp TorchGather(XlaOp input, XlaOp index, int64 dim);

// Returns a new tensor which indexes the input tensor along dimension dim using
// the entries in index.
//
// The returned tensor has the same number of dimensions as the original tensor
// (input). The dimth dimension has the same size as the length of index; other
// dimensions have the same size as in the original tensor.
XlaOp TorchIndexSelect(XlaOp input, XlaOp index, int64 dim);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SLICING_H_
