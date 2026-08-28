/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The sparse tensor manipulations: reshaping, reordering, slicing, splitting,
// concatenating and filling empty rows.
//
// Every one of them decides its output length from the coordinates it is
// given, and every one of them is index arithmetic rather than arithmetic on
// values. They run on the host, after waiting for the stream, which on Apple
// Silicon means waiting and nothing else: the coordinates are already in
// memory the CPU can read, and the values move by memcpy within the same
// unified pool.
//
// The alternative, a device kernel per op, would still have to learn the
// output length on the host first, so it would pay the same wait and then add
// a dispatch to it.

void WaitForStream(SP_Stream stream) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
}

struct ManipOp {
  int32_t num_split = 1;
  int32_t concat_dim = 0;
  int32_t inputs = 2;
};

void* ManipOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ManipOp();
  int32_t value = 1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_split", &value, status);
  if (TF_GetCode(status) == TF_OK) op->num_split = value;
  TF_SetStatus(status, TF_OK, "");
  value = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "concat_dim", &value, status);
  if (TF_GetCode(status) == TF_OK) op->concat_dim = value;
  TF_SetStatus(status, TF_OK, "");
  value = 2;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "N", &value, status);
  if (TF_GetCode(status) == TF_OK) op->inputs = value;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void ManipOp_Delete(void* kernel) { delete static_cast<ManipOp*>(kernel); }

// Reads a whole int64 tensor. Sparse coordinates are always int64.
bool ReadInt64s(TF_Tensor* t, std::vector<int64_t>* out, TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: sparse indices have no storage.");
    return false;
  }
  out->assign(static_cast<const int64_t*>(data),
              static_cast<const int64_t*>(data) + count);
  return true;
}

bool ReadFloats(TF_Tensor* t, std::vector<float>* out, TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: sparse values have no storage.");
    return false;
  }
  out->assign(static_cast<const float*>(data),
              static_cast<const float*>(data) + count);
  return true;
}

// Allocates an output and fills it from a host vector.
bool WriteInt64s(TF_OpKernelContext* ctx, int index,
                 const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& values, TF_Status* status) {
  int64_t count = 1;
  for (int64_t d : shape) count *= d;
  ScopedTensor out;
  out.reset(TF_AllocateOutput(ctx, index, TF_INT64, shape.data(),
                              static_cast<int>(shape.size()),
                              static_cast<size_t>(count) * sizeof(int64_t),
                              status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out.get());
  if (data != nullptr && !values.empty()) {
    std::memcpy(data, values.data(),
                std::min<size_t>(values.size(), static_cast<size_t>(count)) *
                    sizeof(int64_t));
  }
  return true;
}

bool WriteFloats(TF_OpKernelContext* ctx, int index,
                 const std::vector<int64_t>& shape,
                 const std::vector<float>& values, TF_Status* status) {
  int64_t count = 1;
  for (int64_t d : shape) count *= d;
  ScopedTensor out;
  out.reset(TF_AllocateOutput(ctx, index, TF_FLOAT, shape.data(),
                              static_cast<int>(shape.size()),
                              static_cast<size_t>(count) * sizeof(float),
                              status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out.get());
  if (data != nullptr && !values.empty()) {
    std::memcpy(data, values.data(),
                std::min<size_t>(values.size(), static_cast<size_t>(count)) *
                    sizeof(float));
  }
  return true;
}

// Row-major order over coordinates, which is the order a sparse tensor is
// defined to be in.
bool Precedes(const std::vector<int64_t>& indices, int64_t rank, int64_t a,
              int64_t b) {
  for (int64_t d = 0; d < rank; ++d) {
    const int64_t left = indices[a * rank + d];
    const int64_t right = indices[b * rank + d];
    if (left != right) return left < right;
  }
  return false;
}

/*** RESHAPE ***/

void SparseReshape_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor indices, in_shape, new_shape;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, in_shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, new_shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> coords, from, to;
  if (!ReadInt64s(indices.get(), &coords, status)) return;
  if (!ReadInt64s(in_shape.get(), &from, status)) return;
  if (!ReadInt64s(new_shape.get(), &to, status)) return;

  int64_t total = 1;
  for (int64_t d : from) total *= d;
  // At most one dimension may be left to work out, which is what -1 asks for.
  int unknown = -1;
  int64_t known = 1;
  for (size_t i = 0; i < to.size(); ++i) {
    if (to[i] == -1) {
      if (unknown >= 0) {
        TF_SetStatus(status, TF_INVALID_ARGUMENT,
                     "Metal: only one dimension of the new shape may be "
                     "unknown.");
        return;
      }
      unknown = static_cast<int>(i);
    } else {
      known *= to[i];
    }
  }
  if (unknown >= 0) {
    if (known == 0 || total % known != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: the new shape does not divide the old size.");
      return;
    }
    to[unknown] = total / known;
    known = total;
  }
  if (known != total) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the new shape has a different number of elements.");
    return;
  }

  const int64_t in_rank = static_cast<int64_t>(from.size());
  const int64_t out_rank = static_cast<int64_t>(to.size());
  const int64_t nnz = in_rank > 0 ? static_cast<int64_t>(coords.size()) / in_rank
                                  : 0;
  std::vector<int64_t> out(static_cast<size_t>(nnz * out_rank), 0);
  for (int64_t e = 0; e < nnz; ++e) {
    // Flatten under the old shape, then split under the new one, which is
    // exactly what reshaping a dense tensor does to its coordinates.
    int64_t flat = 0;
    for (int64_t d = 0; d < in_rank; ++d) {
      flat = flat * from[d] + coords[e * in_rank + d];
    }
    for (int64_t d = out_rank - 1; d >= 0; --d) {
      out[e * out_rank + d] = to[d] == 0 ? 0 : flat % to[d];
      flat = to[d] == 0 ? 0 : flat / to[d];
    }
  }
  if (!WriteInt64s(ctx, 0, {nnz, out_rank}, out, status)) return;
  WriteInt64s(ctx, 1, {out_rank}, to, status);
}

/*** REORDER ***/

void SparseReorder_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor indices, values, shape;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> coords;
  std::vector<float> data;
  if (!ReadInt64s(indices.get(), &coords, status)) return;
  if (!ReadFloats(values.get(), &data, status)) return;
  const std::vector<int64_t> index_shape = ShapeOf(indices.get());
  if (index_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: sparse indices must be a matrix.");
    return;
  }
  const int64_t nnz = index_shape[0];
  const int64_t rank = index_shape[1];

  std::vector<int64_t> order(static_cast<size_t>(nnz));
  for (int64_t i = 0; i < nnz; ++i) order[static_cast<size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](int64_t a, int64_t b) {
                     return Precedes(coords, rank, a, b);
                   });

  std::vector<int64_t> out_coords(static_cast<size_t>(nnz * rank), 0);
  std::vector<float> out_values(static_cast<size_t>(nnz), 0.0f);
  for (int64_t i = 0; i < nnz; ++i) {
    const int64_t from = order[static_cast<size_t>(i)];
    for (int64_t d = 0; d < rank; ++d) {
      out_coords[i * rank + d] = coords[from * rank + d];
    }
    out_values[static_cast<size_t>(i)] = data[static_cast<size_t>(from)];
  }
  if (!WriteInt64s(ctx, 0, {nnz, rank}, out_coords, status)) return;
  WriteFloats(ctx, 1, {nnz}, out_values, status);
}

/*** SLICE ***/

void SparseSlice_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor indices, values, shape, start, size;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, start.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 4, size.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> coords, dense, begin, extent;
  std::vector<float> data;
  if (!ReadInt64s(indices.get(), &coords, status)) return;
  if (!ReadFloats(values.get(), &data, status)) return;
  if (!ReadInt64s(shape.get(), &dense, status)) return;
  if (!ReadInt64s(start.get(), &begin, status)) return;
  if (!ReadInt64s(size.get(), &extent, status)) return;
  const int64_t rank = static_cast<int64_t>(dense.size());
  const int64_t nnz = rank > 0 ? static_cast<int64_t>(coords.size()) / rank : 0;

  std::vector<int64_t> out_shape(static_cast<size_t>(rank), 0);
  for (int64_t d = 0; d < rank; ++d) {
    // The slice is clipped to what the tensor actually has, as it is on the
    // dense side.
    out_shape[static_cast<size_t>(d)] =
        std::max<int64_t>(0, std::min(begin[d] + extent[d], dense[d]) -
                                 std::min(begin[d], dense[d]));
  }

  std::vector<int64_t> out_coords;
  std::vector<float> out_values;
  for (int64_t e = 0; e < nnz; ++e) {
    bool inside = true;
    for (int64_t d = 0; d < rank && inside; ++d) {
      const int64_t c = coords[e * rank + d];
      if (c < begin[d] || c >= begin[d] + extent[d]) inside = false;
    }
    if (!inside) continue;
    for (int64_t d = 0; d < rank; ++d) {
      out_coords.push_back(coords[e * rank + d] - begin[d]);
    }
    out_values.push_back(data[static_cast<size_t>(e)]);
  }
  const int64_t kept = static_cast<int64_t>(out_values.size());
  if (!WriteInt64s(ctx, 0, {kept, rank}, out_coords, status)) return;
  if (!WriteFloats(ctx, 1, {kept}, out_values, status)) return;
  WriteInt64s(ctx, 2, {rank}, out_shape, status);
}

void SparseSliceGrad_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                                 TF_Status* status) {
  ScopedTensor grad, indices, start, out_indices;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, start.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, out_indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> input_coords, begin, sliced_coords;
  std::vector<float> grad_values;
  if (!ReadFloats(grad.get(), &grad_values, status)) return;
  if (!ReadInt64s(indices.get(), &input_coords, status)) return;
  if (!ReadInt64s(start.get(), &begin, status)) return;
  if (!ReadInt64s(out_indices.get(), &sliced_coords, status)) return;
  const std::vector<int64_t> index_shape = ShapeOf(indices.get());
  if (index_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: sparse indices must be a matrix.");
    return;
  }
  const int64_t nnz = index_shape[0];
  const int64_t rank = index_shape[1];
  const int64_t sliced = rank > 0
                             ? static_cast<int64_t>(sliced_coords.size()) / rank
                             : 0;

  // The slice keeps the order of what it kept, so one walk down both lists
  // matches them up; an input coordinate the slice dropped simply gets zero.
  std::vector<float> out(static_cast<size_t>(nnz), 0.0f);
  int64_t j = 0;
  for (int64_t e = 0; e < nnz && j < sliced; ++e) {
    bool same = true;
    for (int64_t d = 0; d < rank && same; ++d) {
      if (input_coords[e * rank + d] - begin[d] != sliced_coords[j * rank + d]) {
        same = false;
      }
    }
    if (same) {
      out[static_cast<size_t>(e)] = grad_values[static_cast<size_t>(j)];
      ++j;
    }
  }
  WriteFloats(ctx, 0, {nnz}, out, status);
}

/*** SPLIT ***/

void SparseSplit_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor split_dim, indices, values, shape;
  TF_GetInput(ctx, 0, split_dim.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> axis_value, coords, dense;
  std::vector<float> data;
  if (!ReadInt64s(split_dim.get(), &axis_value, status)) return;
  if (!ReadInt64s(indices.get(), &coords, status)) return;
  if (!ReadFloats(values.get(), &data, status)) return;
  if (!ReadInt64s(shape.get(), &dense, status)) return;
  if (axis_value.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SparseSplit needs a split dimension.");
    return;
  }
  const int64_t axis = axis_value[0];
  const int64_t rank = static_cast<int64_t>(dense.size());
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the split dimension is out of range.");
    return;
  }
  const int64_t nnz = rank > 0 ? static_cast<int64_t>(coords.size()) / rank : 0;
  const int64_t pieces = std::max<int64_t>(op->num_split, 1);
  // The first few pieces take one extra row when the axis does not divide
  // evenly, which is what the dense split does.
  const int64_t base = dense[axis] / pieces;
  const int64_t extra = dense[axis] % pieces;

  std::vector<int64_t> offset(static_cast<size_t>(pieces + 1), 0);
  for (int64_t p = 0; p < pieces; ++p) {
    offset[static_cast<size_t>(p + 1)] =
        offset[static_cast<size_t>(p)] + base + (p < extra ? 1 : 0);
  }

  for (int64_t p = 0; p < pieces; ++p) {
    std::vector<int64_t> out_coords;
    std::vector<float> out_values;
    for (int64_t e = 0; e < nnz; ++e) {
      const int64_t c = coords[e * rank + axis];
      if (c < offset[static_cast<size_t>(p)] ||
          c >= offset[static_cast<size_t>(p + 1)]) {
        continue;
      }
      for (int64_t d = 0; d < rank; ++d) {
        out_coords.push_back(d == axis
                                 ? c - offset[static_cast<size_t>(p)]
                                 : coords[e * rank + d]);
      }
      out_values.push_back(data[static_cast<size_t>(e)]);
    }
    std::vector<int64_t> piece_shape = dense;
    piece_shape[static_cast<size_t>(axis)] =
        offset[static_cast<size_t>(p + 1)] - offset[static_cast<size_t>(p)];
    const int64_t kept = static_cast<int64_t>(out_values.size());
    if (!WriteInt64s(ctx, static_cast<int>(p), {kept, rank}, out_coords,
                     status)) {
      return;
    }
    if (!WriteFloats(ctx, static_cast<int>(pieces + p), {kept}, out_values,
                     status)) {
      return;
    }
    if (!WriteInt64s(ctx, static_cast<int>(2 * pieces + p), {rank},
                     piece_shape, status)) {
      return;
    }
  }
}

/*** CONCAT ***/

void SparseConcat_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                              TF_Status* status) {
  const int n = std::max(op->inputs, 1);
  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<std::vector<int64_t>> coords(n), shapes(n);
  std::vector<std::vector<float>> data(n);
  for (int i = 0; i < n; ++i) {
    ScopedTensor t;
    TF_GetInput(ctx, i, t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    if (!ReadInt64s(t.get(), &coords[i], status)) return;
    ScopedTensor v;
    TF_GetInput(ctx, n + i, v.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    if (!ReadFloats(v.get(), &data[i], status)) return;
    ScopedTensor s;
    TF_GetInput(ctx, 2 * n + i, s.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    if (!ReadInt64s(s.get(), &shapes[i], status)) return;
  }
  if (shapes[0].empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SparseConcat needs a shape of rank at least one.");
    return;
  }
  const int64_t rank = static_cast<int64_t>(shapes[0].size());
  int64_t axis = op->concat_dim;
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the concatenation dimension is out of range.");
    return;
  }

  std::vector<int64_t> out_shape = shapes[0];
  out_shape[static_cast<size_t>(axis)] = 0;
  for (int i = 0; i < n; ++i) {
    out_shape[static_cast<size_t>(axis)] += shapes[i][static_cast<size_t>(axis)];
  }

  std::vector<int64_t> out_coords;
  std::vector<float> out_values;
  int64_t offset = 0;
  for (int i = 0; i < n; ++i) {
    const int64_t nnz =
        rank > 0 ? static_cast<int64_t>(coords[i].size()) / rank : 0;
    for (int64_t e = 0; e < nnz; ++e) {
      for (int64_t d = 0; d < rank; ++d) {
        out_coords.push_back(coords[i][e * rank + d] +
                             (d == axis ? offset : 0));
      }
      out_values.push_back(data[i][static_cast<size_t>(e)]);
    }
    offset += shapes[i][static_cast<size_t>(axis)];
  }
  const int64_t total = static_cast<int64_t>(out_values.size());
  if (!WriteInt64s(ctx, 0, {total, rank}, out_coords, status)) return;
  if (!WriteFloats(ctx, 1, {total}, out_values, status)) return;
  WriteInt64s(ctx, 2, {rank}, out_shape, status);
}

/*** FILL EMPTY ROWS ***/

void SparseFillEmptyRows_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                                     TF_Status* status) {
  ScopedTensor indices, values, dense_shape, default_value;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, dense_shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, default_value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> coords, dense;
  std::vector<float> data, fill;
  if (!ReadInt64s(indices.get(), &coords, status)) return;
  if (!ReadFloats(values.get(), &data, status)) return;
  if (!ReadInt64s(dense_shape.get(), &dense, status)) return;
  if (!ReadFloats(default_value.get(), &fill, status)) return;
  if (dense.empty() || fill.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SparseFillEmptyRows needs a shape and a default.");
    return;
  }
  const int64_t rank = static_cast<int64_t>(dense.size());
  const int64_t nnz = rank > 0 ? static_cast<int64_t>(coords.size()) / rank : 0;
  const int64_t rows = dense[0];

  std::vector<char> occupied(static_cast<size_t>(std::max<int64_t>(rows, 0)),
                             0);
  for (int64_t e = 0; e < nnz; ++e) {
    const int64_t row = coords[e * rank];
    if (row >= 0 && row < rows) occupied[static_cast<size_t>(row)] = 1;
  }
  int64_t empty = 0;
  for (char c : occupied) {
    if (!c) ++empty;
  }

  // The result is ordered by row, with each filled-in row taking its place
  // among the entries that were already there.
  std::vector<std::vector<int64_t>> per_row(
      static_cast<size_t>(std::max<int64_t>(rows, 0)));
  for (int64_t e = 0; e < nnz; ++e) {
    const int64_t row = coords[e * rank];
    if (row >= 0 && row < rows) per_row[static_cast<size_t>(row)].push_back(e);
  }
  std::vector<int64_t> out_coords;
  std::vector<float> out_values;
  std::vector<int64_t> reverse(static_cast<size_t>(nnz), 0);
  for (int64_t row = 0; row < rows; ++row) {
    if (per_row[static_cast<size_t>(row)].empty()) {
      out_coords.push_back(row);
      for (int64_t d = 1; d < rank; ++d) out_coords.push_back(0);
      out_values.push_back(fill[0]);
      continue;
    }
    for (int64_t e : per_row[static_cast<size_t>(row)]) {
      reverse[static_cast<size_t>(e)] =
          static_cast<int64_t>(out_values.size());
      for (int64_t d = 0; d < rank; ++d) {
        out_coords.push_back(coords[e * rank + d]);
      }
      out_values.push_back(data[static_cast<size_t>(e)]);
    }
  }
  const int64_t total = static_cast<int64_t>(out_values.size());

  if (!WriteInt64s(ctx, 0, {total, rank}, out_coords, status)) return;
  if (!WriteFloats(ctx, 1, {total}, out_values, status)) return;
  {
    ScopedTensor indicator;
    const std::vector<int64_t> shape = {rows};
    indicator.reset(TF_AllocateOutput(ctx, 2, TF_BOOL, shape.data(), 1,
                                      static_cast<size_t>(rows), status));
    if (TF_GetCode(status) != TF_OK) return;
    char* out = static_cast<char*>(TF_TensorData(indicator.get()));
    if (out != nullptr) {
      for (int64_t row = 0; row < rows; ++row) {
        out[row] = occupied[static_cast<size_t>(row)] ? 0 : 1;
      }
    }
  }
  WriteInt64s(ctx, 3, {nnz}, reverse, status);
}

void SparseFillEmptyRowsGrad_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                                         TF_Status* status) {
  ScopedTensor reverse, grad;
  TF_GetInput(ctx, 0, reverse.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> map;
  std::vector<float> values;
  if (!ReadInt64s(reverse.get(), &map, status)) return;
  if (!ReadFloats(grad.get(), &values, status)) return;

  // Each original value takes back the gradient of wherever it ended up; the
  // rest of the gradient belongs to the default, which every filled-in row
  // shares.
  std::vector<float> d_values(map.size(), 0.0f);
  std::vector<char> taken(values.size(), 0);
  for (size_t i = 0; i < map.size(); ++i) {
    const int64_t j = map[i];
    if (j >= 0 && j < static_cast<int64_t>(values.size())) {
      d_values[i] = values[static_cast<size_t>(j)];
      taken[static_cast<size_t>(j)] = 1;
    }
  }
  float d_default = 0.0f;
  for (size_t j = 0; j < values.size(); ++j) {
    if (!taken[j]) d_default += values[j];
  }
  if (!WriteFloats(ctx, 0, {static_cast<int64_t>(map.size())}, d_values,
                   status)) {
    return;
  }
  WriteFloats(ctx, 1, {}, {d_default}, status);
}

/*** RAGGED FILL EMPTY ROWS ***/

// The ragged form of the same operation. A ragged tensor names each value's
// row directly instead of through a coordinate matrix, so the only difference
// is where the row comes from and that the number of rows is given rather
// than taken from a dense shape.
void RaggedFillEmptyRows_ComputeImpl(ManipOp* op, TF_OpKernelContext* ctx,
                                     TF_Status* status) {
  ScopedTensor rowids, values, nrows, default_value;
  TF_GetInput(ctx, 0, rowids.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, nrows.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, default_value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  WaitForStream(stream);

  std::vector<int64_t> rows, row_count;
  std::vector<float> data, fill;
  if (!ReadInt64s(rowids.get(), &rows, status)) return;
  if (!ReadFloats(values.get(), &data, status)) return;
  if (!ReadInt64s(nrows.get(), &row_count, status)) return;
  if (!ReadFloats(default_value.get(), &fill, status)) return;
  if (row_count.empty() || fill.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: RaggedFillEmptyRows needs a row count and a "
                 "default.");
    return;
  }
  const int64_t total_rows = row_count[0];
  const int64_t nnz = static_cast<int64_t>(data.size());

  std::vector<char> occupied(
      static_cast<size_t>(std::max<int64_t>(total_rows, 0)), 0);
  std::vector<std::vector<int64_t>> per_row(
      static_cast<size_t>(std::max<int64_t>(total_rows, 0)));
  for (int64_t e = 0; e < nnz; ++e) {
    const int64_t row = rows[static_cast<size_t>(e)];
    if (row < 0 || row >= total_rows) continue;
    occupied[static_cast<size_t>(row)] = 1;
    per_row[static_cast<size_t>(row)].push_back(e);
  }

  std::vector<int64_t> out_rows;
  std::vector<float> out_values;
  std::vector<int64_t> reverse(static_cast<size_t>(nnz), 0);
  for (int64_t row = 0; row < total_rows; ++row) {
    if (per_row[static_cast<size_t>(row)].empty()) {
      out_rows.push_back(row);
      out_values.push_back(fill[0]);
      continue;
    }
    for (int64_t e : per_row[static_cast<size_t>(row)]) {
      reverse[static_cast<size_t>(e)] =
          static_cast<int64_t>(out_values.size());
      out_rows.push_back(row);
      out_values.push_back(data[static_cast<size_t>(e)]);
    }
  }
  const int64_t total = static_cast<int64_t>(out_values.size());

  if (!WriteInt64s(ctx, 0, {total}, out_rows, status)) return;
  if (!WriteFloats(ctx, 1, {total}, out_values, status)) return;
  {
    ScopedTensor indicator;
    const std::vector<int64_t> shape = {total_rows};
    indicator.reset(TF_AllocateOutput(ctx, 2, TF_BOOL, shape.data(), 1,
                                      static_cast<size_t>(total_rows),
                                      status));
    if (TF_GetCode(status) != TF_OK) return;
    char* out = static_cast<char*>(TF_TensorData(indicator.get()));
    if (out != nullptr) {
      for (int64_t row = 0; row < total_rows; ++row) {
        out[row] = occupied[static_cast<size_t>(row)] ? 0 : 1;
      }
    }
  }
  WriteInt64s(ctx, 3, {nnz}, reverse, status);
}

#define METAL_MANIP_COMPUTE(NAME, IMPL)                                     \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<ManipOp*>(kernel);                               \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a sparse kernel has no state.");                 \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_MANIP_COMPUTE(SparseReshape_Compute, SparseReshape_ComputeImpl)
METAL_MANIP_COMPUTE(SparseReorder_Compute, SparseReorder_ComputeImpl)
METAL_MANIP_COMPUTE(SparseSlice_Compute, SparseSlice_ComputeImpl)
METAL_MANIP_COMPUTE(SparseSliceGrad_Compute, SparseSliceGrad_ComputeImpl)
METAL_MANIP_COMPUTE(SparseSplit_Compute, SparseSplit_ComputeImpl)
METAL_MANIP_COMPUTE(SparseConcat_Compute, SparseConcat_ComputeImpl)
METAL_MANIP_COMPUTE(SparseFillEmptyRows_Compute,
                    SparseFillEmptyRows_ComputeImpl)
METAL_MANIP_COMPUTE(SparseFillEmptyRowsGrad_Compute,
                    SparseFillEmptyRowsGrad_ComputeImpl)
METAL_MANIP_COMPUTE(RaggedFillEmptyRows_Compute,
                    RaggedFillEmptyRows_ComputeImpl)

#undef METAL_MANIP_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name, bool typed) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &ManipOp_Create, compute, &ManipOp_Delete);
  if (typed) TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalSparseManipKernels() {
  Register("SparseReshape", &SparseReshape_Compute, "MetalSparseReshape",
           false);
  Register("SparseReorder", &SparseReorder_Compute, "MetalSparseReorder",
           true);
  Register("SparseSlice", &SparseSlice_Compute, "MetalSparseSlice", true);
  Register("SparseSliceGrad", &SparseSliceGrad_Compute,
           "MetalSparseSliceGrad", true);
  Register("SparseSplit", &SparseSplit_Compute, "MetalSparseSplit", true);
  Register("SparseConcat", &SparseConcat_Compute, "MetalSparseConcat", true);
  Register("SparseFillEmptyRows", &SparseFillEmptyRows_Compute,
           "MetalSparseFillEmptyRows", true);
  Register("SparseFillEmptyRowsGrad", &SparseFillEmptyRowsGrad_Compute,
           "MetalSparseFillEmptyRowsGrad", true);
  Register("RaggedFillEmptyRows", &RaggedFillEmptyRows_Compute,
           "MetalRaggedFillEmptyRows", true);
  // The ragged gradient is the sparse one: both are handed the same reverse
  // index map and nothing else about them differs.
  Register("RaggedFillEmptyRowsGrad", &SparseFillEmptyRowsGrad_Compute,
           "MetalRaggedFillEmptyRowsGrad", true);
}

}  // namespace metal
}  // namespace tensorflow
