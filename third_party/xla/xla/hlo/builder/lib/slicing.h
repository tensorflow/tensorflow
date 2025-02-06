/* Copyright 2018 The OpenXLA Authors.

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

#include <cstdint>
#include <functional>

#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/types.h"

#ifndef XLA_HLO_BUILDER_LIB_SLICING_H_
#define XLA_HLO_BUILDER_LIB_SLICING_H_

namespace xla {

// Updates a slice of 'x', i.e.,
// x[start[0], ..., start[n]] = update
XlaOp UpdateSlice(XlaOp x, XlaOp update, absl::Span<const int64_t> start);

// Performs a slice in the minor dimensions of a tensor.
// x[..., start[0]:end[0], ..., start[n]:end[n]]
XlaOp SliceInMinorDims(XlaOp x, absl::Span<const int64_t> start,
                       absl::Span<const int64_t> end);

// Updates a slice of 'x', where 'start' contains a list of minor dimensions:
// x[..., start[0]:..., ..., start[n]:...] = update
XlaOp UpdateSliceInMinorDims(XlaOp x, XlaOp update,
                             absl::Span<const int64_t> start);

// Performs a dynamic slice in the minor dimensions of a tensor.
XlaOp DynamicSliceInMinorDims(XlaOp x, absl::Span<const XlaOp> starts,
                              absl::Span<const int64_t> sizes);

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
XlaOp TorchGather(XlaOp input, XlaOp index, int64_t dim, bool sparse = true);

// idx = index[i][j][k]
// output[idx][j][k] = combiner(input[idx][j][k], src[i][j][k])  # if dim == 0
// output[i][idx][k] = combiner(input[i][idx][k], src[i][j][k])  # if dim == 1
// output[i][j][idx] = combiner(input[i][j][idx], src[i][j][k])  # if dim == 2
XlaOp TorchScatterDense(XlaOp input, XlaOp index, XlaOp src, int64_t dim,
                        const std::function<XlaOp(XlaOp, XlaOp)>& combiner);

// Returns a new tensor which indexes the input tensor along dimension dim using
// the entries in index.
//
// The returned tensor has the same number of dimensions as the original tensor
// (input). The dimth dimension has the same size as the length of index; other
// dimensions have the same size as in the original tensor.
//
// This operation supports 0 or more major batch dimensions that act like a
// multidimensional loop over both the input and the index.
XlaOp TorchIndexSelect(XlaOp input, XlaOp index, int64_t dim,
                       int64_t batch_dims = 0);

}  // namespace xla

#endif  // XLA_HLO_BUILDER_LIB_SLICING_H_
