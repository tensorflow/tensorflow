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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The diagonal family, band extraction, the space/depth rearrangements,
// LinSpace and L2Loss.
//
// MPSGraph has bandPart but no diagonal operator. Rather than write a shader
// for each of these, the diagonal is expressed as a mask built from
// coordinateAlongAxis: comparing the row coordinate with the column
// coordinate yields exactly the diagonal, and multiplying or reducing against
// that mask gives both the "set a diagonal" and the "read a diagonal"
// directions. It stays inside the graph, so it fuses with whatever surrounds
// it instead of forcing a separate dispatch.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

bool ReadHostVector(TF_Tensor* t, std::vector<int64_t>* out,
                    TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const TF_DataType dtype = TF_TensorType(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a host-memory argument has no data.");
    return false;
  }
  out->clear();
  for (int64_t i = 0; i < count; ++i) {
    if (dtype == TF_INT32) out->push_back(static_cast<const int32_t*>(data)[i]);
    else if (dtype == TF_INT64) out->push_back(static_cast<const int64_t*>(data)[i]);
    else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an int32 or int64 argument.");
      return false;
    }
  }
  return true;
}

// A [.., rows, cols] mask that is 1 on the main diagonal and 0 elsewhere.
MPSGraphTensor* DiagonalMask(MPSGraph* g, const std::vector<int64_t>& shape,
                             MPSDataType dtype) {
  const NSInteger rank = static_cast<NSInteger>(shape.size());
  MPSGraphTensor* rows = [g coordinateAlongAxis:rank - 2
                                      withShape:MPSShape(shape)
                                           name:nil];
  MPSGraphTensor* cols = [g coordinateAlongAxis:rank - 1
                                      withShape:MPSShape(shape)
                                           name:nil];
  MPSGraphTensor* eq =
      [g equalWithPrimaryTensor:rows secondaryTensor:cols name:nil];
  return [g castTensor:eq toType:dtype name:nil];
}

struct MatrixOp {
  TF_DataType dtype = TF_FLOAT;
  int block_size = 2;
  bool nhwc = true;
};

void* MatrixOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new MatrixOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  int32_t block_size = 2;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "block_size", &block_size, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->block_size = block_size;

  char format[8] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->nhwc = true;
  } else {
    op->nhwc = std::strcmp(format, "NCHW") != 0;
  }
  TF_DeleteStatus(status);
  return op;
}

void MatrixOp_Delete(void* kernel) { delete static_cast<MatrixOp*>(kernel); }

/*** MATRIX BAND PART ***/

void MatrixBandPart_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor input, lower_t, upper_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, lower_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, upper_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MatrixBandPart needs a rank of at least 2.");
    return;
  }
  std::vector<int64_t> lo_v, up_v;
  if (!ReadHostVector(lower_t.get(), &lo_v, status)) return;
  if (!ReadHostVector(upper_t.get(), &up_v, status)) return;
  const int64_t num_lower = lo_v.empty() ? -1 : lo_v[0];
  const int64_t num_upper = up_v.empty() ? -1 : up_v[0];

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "MatrixBandPart";
  AppendShapeToKey(shape, &key);
  key.append("/l").append(std::to_string(num_lower));
  key.append("/u").append(std::to_string(num_upper));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        // MPSGraph uses the same -1 convention TensorFlow does for "keep the
        // whole triangle".
        [out->outputs
            addObject:[out->graph bandPartWithTensor:x
                                            numLower:static_cast<NSInteger>(
                                                         num_lower)
                                            numUpper:static_cast<NSInteger>(
                                                         num_upper)
                                                name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** MATRIX DIAG PART ***/

void MatrixDiagPart_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MatrixDiagPart needs a rank of at least 2.");
    return;
  }
  // V2 and V3 take a diagonal offset k and a padding value. Only the main
  // diagonal is handled; anything else is refused rather than silently
  // returning the main diagonal, which would look plausible and be wrong.
  if (TF_NumInputs(ctx) > 1) {
    ScopedTensor k_t;
    TF_GetInput(ctx, 1, k_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    std::vector<int64_t> k_v;
    if (!ReadHostVector(k_t.get(), &k_v, status)) return;
    for (int64_t k : k_v) {
      if (k != 0) {
        TF_SetStatus(status, TF_UNIMPLEMENTED,
                     "Metal: MatrixDiagPart supports only the main diagonal "
                     "(k = 0).");
        return;
      }
    }
  }

  const int rank = static_cast<int>(shape.size());
  const int64_t rows = shape[rank - 2];
  const int64_t cols = shape[rank - 1];
  const int64_t diag = std::min(rows, cols);

  std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
  out_shape.push_back(diag);

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "MatrixDiagPart";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const std::vector<int64_t> final_shape = out_shape;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        // Masking then summing the last axis leaves exactly the diagonal
        // entry of each row, since every other term is zero.
        MPSGraphTensor* masked =
            [g multiplicationWithPrimaryTensor:x
                               secondaryTensor:DiagonalMask(g, shape, mps_dtype)
                                          name:nil];
        MPSGraphTensor* rowwise = [g reductionSumWithTensor:masked
                                                       axis:-1
                                                       name:nil];
        // Rows past the shorter side hold no diagonal entry, so trim them.
        MPSGraphTensor* trimmed =
            [g sliceTensor:rowwise
                 dimension:static_cast<NSUInteger>(shape.size()) - 2
                     start:0
                    length:static_cast<NSInteger>(diag)
                      name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[g reshapeTensor:trimmed
                                       withShape:MPSShape(final_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** MATRIX DIAG ***/

void MatrixDiag_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  if (TF_NumInputs(ctx) > 1) {
    ScopedTensor k_t;
    TF_GetInput(ctx, 1, k_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    std::vector<int64_t> k_v;
    if (!ReadHostVector(k_t.get(), &k_v, status)) return;
    for (int64_t k : k_v) {
      if (k != 0) {
        TF_SetStatus(status, TF_UNIMPLEMENTED,
                     "Metal: MatrixDiag supports only the main diagonal "
                     "(k = 0).");
        return;
      }
    }
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MatrixDiag needs a rank of at least 1.");
    return;
  }
  const int64_t n = in_shape.back();
  std::vector<int64_t> out_shape = in_shape;
  out_shape.push_back(n);
  // The diagonal vector has to line up along the row axis before it is
  // masked, which means an extra trailing axis of size 1 to broadcast over.
  std::vector<int64_t> column_shape = in_shape;
  column_shape.push_back(1);

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "MatrixDiag";
  AppendShapeToKey(in_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* col = [g reshapeTensor:x
                                     withShape:MPSShape(column_shape)
                                          name:nil];
        [out->inputs addObject:x];
        [out->outputs
            addObject:[g multiplicationWithPrimaryTensor:col
                                         secondaryTensor:DiagonalMask(
                                                             g, out_shape,
                                                             mps_dtype)
                                                    name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** MATRIX SET DIAG ***/

void MatrixSetDiag_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor input, diagonal;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, diagonal.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MatrixSetDiag needs a rank of at least 2.");
    return;
  }
  const int rank = static_cast<int>(shape.size());
  if (shape[rank - 2] != shape[rank - 1]) {
    // A rectangular set-diagonal needs the replacement padded to the full
    // width before masking; that is left unimplemented rather than guessed.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: MatrixSetDiag supports square matrices only.");
    return;
  }
  const std::vector<int64_t> diag_shape = ShapeOf(diagonal.get());
  std::vector<int64_t> column_shape = diag_shape;
  column_shape.push_back(1);

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "MatrixSetDiag";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(diag_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* d = [g placeholderWithShape:MPSShape(diag_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* mask = DiagonalMask(g, shape, mps_dtype);
        MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:mps_dtype];
        MPSGraphTensor* off =
            [g subtractionWithPrimaryTensor:one secondaryTensor:mask name:nil];
        MPSGraphTensor* col = [g reshapeTensor:d
                                     withShape:MPSShape(column_shape)
                                          name:nil];
        // Keep everything off the diagonal, and take the diagonal from the
        // replacement.
        MPSGraphTensor* kept =
            [g multiplicationWithPrimaryTensor:x secondaryTensor:off name:nil];
        MPSGraphTensor* placed =
            [g multiplicationWithPrimaryTensor:col
                               secondaryTensor:mask
                                          name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:d];
        [out->outputs addObject:[g additionWithPrimaryTensor:kept
                                             secondaryTensor:placed
                                                        name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* d_data =
      TensorDataForTensor(diagonal.get(), op->dtype, device, status);
  if (d_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, d_data ], @[ o_data ], status);
}

/*** DIAG AND DIAG PART ***/

// Diag maps a rank-k tensor to rank 2k, with the input on the generalised
// diagonal. Flattening reduces it to the matrix case: build the [N, N]
// diagonal and reshape to shape+shape. DiagPart is the same in reverse.
template <bool kExtract>
void Diag_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Diag needs a rank of at least 1.");
    return;
  }

  std::vector<int64_t> out_shape;
  int64_t n = 1;
  if (kExtract) {
    if (in_shape.size() % 2 != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: DiagPart needs an even rank.");
      return;
    }
    const size_t half = in_shape.size() / 2;
    for (size_t i = 0; i < half; ++i) {
      if (in_shape[i] != in_shape[i + half]) {
        TF_SetStatus(status, TF_INVALID_ARGUMENT,
                     "Metal: DiagPart needs the two halves of the shape to "
                     "match.");
        return;
      }
      out_shape.push_back(in_shape[i]);
      n *= in_shape[i];
    }
  } else {
    for (int64_t d : in_shape) n *= d;
    out_shape = in_shape;
    for (int64_t d : in_shape) out_shape.push_back(d);
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = kExtract ? "DiagPart" : "Diag";
  AppendShapeToKey(in_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const std::vector<int64_t> square = {n, n};
  const std::vector<int64_t> flat = {n};
  const std::vector<int64_t> column = {n, 1};
  const std::vector<int64_t> final_shape = out_shape;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        [out->inputs addObject:x];
        MPSGraphTensor* r;
        if (kExtract) {
          MPSGraphTensor* sq =
              [g reshapeTensor:x withShape:MPSShape(square) name:nil];
          MPSGraphTensor* masked =
              [g multiplicationWithPrimaryTensor:sq
                                 secondaryTensor:DiagonalMask(g, square,
                                                              mps_dtype)
                                            name:nil];
          r = [g reductionSumWithTensor:masked axis:-1 name:nil];
        } else {
          MPSGraphTensor* col =
              [g reshapeTensor:x withShape:MPSShape(column) name:nil];
          r = [g multiplicationWithPrimaryTensor:col
                                 secondaryTensor:DiagonalMask(g, square,
                                                              mps_dtype)
                                            name:nil];
        }
        [out->outputs addObject:[g reshapeTensor:r
                                       withShape:MPSShape(final_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** LIN SPACE ***/

void LinSpace_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor start, stop, num_t;
  TF_GetInput(ctx, 0, start.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, stop.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, num_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> num_v;
  if (!ReadHostVector(num_t.get(), &num_v, status)) return;
  if (num_v.empty() || num_v[0] < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: LinSpace needs a positive count.");
    return;
  }
  const int64_t num = num_v[0];
  const std::vector<int64_t> out_shape = {num};

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 1,
      static_cast<size_t>(num) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "LinSpace";
  key.append("/n").append(std::to_string(num));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const std::vector<int64_t> scalar = {1};
  // With a single point TensorFlow returns start alone, so the step would
  // divide by zero; the scale is forced to 0 in that case.
  const double inv_span = num > 1 ? 1.0 / static_cast<double>(num - 1) : 0.0;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* a = [g placeholderWithShape:MPSShape(scalar)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* b = [g placeholderWithShape:MPSShape(scalar)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* idx = [g castTensor:[g coordinateAlongAxis:0
                                                         withShape:MPSShape(
                                                                       out_shape)
                                                              name:nil]
                                     toType:mps_dtype
                                       name:nil];
        MPSGraphTensor* span =
            [g subtractionWithPrimaryTensor:b secondaryTensor:a name:nil];
        MPSGraphTensor* step =
            [g multiplicationWithPrimaryTensor:span
                               secondaryTensor:[g constantWithScalar:inv_span
                                                            dataType:mps_dtype]
                                          name:nil];
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:a
                                   secondaryTensor:
                                       [g multiplicationWithPrimaryTensor:idx
                                                          secondaryTensor:step
                                                                     name:nil]
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice a_slice, b_slice;
  if (!SliceForTensor(start.get(), &a_slice, status)) return;
  if (!SliceForTensor(stop.get(), &b_slice, status)) return;
  MPSGraphTensorData* a_data =
      TensorDataFor(a_slice, scalar, op->dtype, device, status);
  if (a_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataFor(b_slice, scalar, op->dtype, device, status);
  if (b_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ a_data, b_data ], @[ o_data ], status);
}

/*** L2 LOSS ***/

void L2Loss_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, op->dtype, nullptr, 0,
                                 TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "L2Loss";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSMutableArray<NSNumber*>* all_axes = [NSMutableArray array];
  for (size_t i = 0; i < shape.size(); ++i) {
    [all_axes addObject:@(static_cast<NSInteger>(i))];
  }
  const std::vector<int64_t> scalar_shape = {};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        // TensorFlow defines L2Loss as sum(x^2)/2, not the norm.
        MPSGraphTensor* total =
            [g reductionSumWithTensor:[g squareWithTensor:x name:nil]
                                 axes:all_axes
                                 name:nil];
        MPSGraphTensor* half =
            [g multiplicationWithPrimaryTensor:total
                               secondaryTensor:[g constantWithScalar:0.5
                                                            dataType:mps_dtype]
                                          name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[g reshapeTensor:half
                                       withShape:MPSShape(scalar_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** SPACE AND DEPTH ***/

template <bool kToDepth>
void SpaceDepth_ComputeImpl(MatrixOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SpaceToDepth expects a rank-4 input.");
    return;
  }
  const int b = op->block_size;
  if (b < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: block_size must be at least 2.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int h = op->nhwc ? 1 : 2;
  const int w = op->nhwc ? 2 : 3;
  const int c = op->nhwc ? 3 : 1;

  std::vector<int64_t> out_shape = in_shape;
  if (kToDepth) {
    if (in_shape[h] % b != 0 || in_shape[w] % b != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: SpaceToDepth needs both spatial extents to divide "
                   "by block_size.");
      return;
    }
    out_shape[h] = in_shape[h] / b;
    out_shape[w] = in_shape[w] / b;
    out_shape[c] = in_shape[c] * b * b;
  } else {
    if (in_shape[c] % (b * b) != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: DepthToSpace needs the channel count to divide by "
                   "block_size squared.");
      return;
    }
    out_shape[h] = in_shape[h] * b;
    out_shape[w] = in_shape[w] * b;
    out_shape[c] = in_shape[c] / (b * b);
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = kToDepth ? "SpaceToDepth" : "DepthToSpace";
  AppendShapeToKey(in_shape, &key);
  key.append("/b").append(std::to_string(b));
  key.append(op->nhwc ? "/NHWC" : "/NCHW");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger wa = static_cast<NSUInteger>(w);
  const NSUInteger ha = static_cast<NSUInteger>(h);
  const NSUInteger ca = static_cast<NSUInteger>(c);
  const NSUInteger bs = static_cast<NSUInteger>(b);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        // TensorFlow interleaves depth in row-major block order, which is the
        // non-pixel-shuffle ordering.
        [out->outputs
            addObject:(kToDepth ? [out->graph spaceToDepth2DTensor:x
                                                         widthAxis:wa
                                                        heightAxis:ha
                                                         depthAxis:ca
                                                         blockSize:bs
                                              usePixelShuffleOrder:NO
                                                              name:nil]
                                : [out->graph depthToSpace2DTensor:x
                                                         widthAxis:wa
                                                        heightAxis:ha
                                                         depthAxis:ca
                                                         blockSize:bs
                                              usePixelShuffleOrder:NO
                                                              name:nil])];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<MatrixOp*>(kernel);                                \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(MatrixBandPart_Compute, MatrixBandPart_ComputeImpl)
METAL_COMPUTE(MatrixDiagPart_Compute, MatrixDiagPart_ComputeImpl)
METAL_COMPUTE(MatrixDiag_Compute, MatrixDiag_ComputeImpl)
METAL_COMPUTE(MatrixSetDiag_Compute, MatrixSetDiag_ComputeImpl)
METAL_COMPUTE(L2Loss_Compute, L2Loss_ComputeImpl)
METAL_COMPUTE(Diag_Compute, Diag_ComputeImpl<false>)
METAL_COMPUTE(DiagPart_Compute, Diag_ComputeImpl<true>)
METAL_COMPUTE(LinSpace_Compute, LinSpace_ComputeImpl)
METAL_COMPUTE(SpaceToDepth_Compute, SpaceDepth_ComputeImpl<true>)
METAL_COMPUTE(DepthToSpace_Compute, SpaceDepth_ComputeImpl<false>)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &MatrixOp_Create, compute, &MatrixOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  for (const char* a : host_args) TF_KernelBuilder_HostMemory(builder, a);
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

void RegisterMetalMatrixKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};

  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    // num_lower and num_upper are read on the host to key the graph.
    Register("MatrixBandPart", &MatrixBandPart_Compute, t,
             "MetalMatrixBandPart" + s, {"num_lower", "num_upper"});
    // TensorFlow's pre-1.0 spelling of the same ops; still present in old
    // graphs and identical in semantics, so they share the implementations
    // rather than falling back to the host.
    Register("BatchMatrixBandPart", &MatrixBandPart_Compute, t,
             "MetalBatchMatrixBandPart" + s, {"num_lower", "num_upper"});
    Register("BatchMatrixDiag", &MatrixDiag_Compute, t,
             "MetalBatchMatrixDiag" + s, {});
    Register("BatchMatrixDiagPart", &MatrixDiagPart_Compute, t,
             "MetalBatchMatrixDiagPart" + s, {});
    Register("BatchMatrixSetDiag", &MatrixSetDiag_Compute, t,
             "MetalBatchMatrixSetDiag" + s, {});
    Register("MatrixDiag", &MatrixDiag_Compute, t, "MetalMatrixDiag" + s, {});
    Register("MatrixDiagPart", &MatrixDiagPart_Compute, t,
             "MetalMatrixDiagPart" + s, {});
    Register("MatrixSetDiag", &MatrixSetDiag_Compute, t,
             "MetalMatrixSetDiag" + s, {});
    // V2 and V3 add k and an alignment attribute; only the main diagonal is
    // accepted, and the extra inputs stay on the host.
    Register("MatrixSetDiagV2", &MatrixSetDiag_Compute, t,
             "MetalMatrixSetDiagV2" + s, {"k"});
    Register("MatrixSetDiagV3", &MatrixSetDiag_Compute, t,
             "MetalMatrixSetDiagV3" + s, {"k"});
    // The V2 and V3 forms take k and a padding value in host memory; only the
    // main diagonal is accepted, and a non-zero k is refused at run time.
    Register("MatrixDiagV2", &MatrixDiag_Compute, t, "MetalMatrixDiagV2" + s,
             {"k", "num_rows", "num_cols", "padding_value"});
    Register("MatrixDiagV3", &MatrixDiag_Compute, t, "MetalMatrixDiagV3" + s,
             {"k", "num_rows", "num_cols", "padding_value"});
    Register("MatrixDiagPartV2", &MatrixDiagPart_Compute, t,
             "MetalMatrixDiagPartV2" + s, {"k", "padding_value"});
    Register("MatrixDiagPartV3", &MatrixDiagPart_Compute, t,
             "MetalMatrixDiagPartV3" + s, {"k", "padding_value"});
    Register("L2Loss", &L2Loss_Compute, t, "MetalL2Loss" + s, {});
    Register("Diag", &Diag_Compute, t, "MetalDiag" + s, {});
    Register("DiagPart", &DiagPart_Compute, t, "MetalDiagPart" + s, {});
    // The point count is read on the host to size the output.
    Register("LinSpace", &LinSpace_Compute, t, "MetalLinSpace" + s, {"num"});
    Register("SpaceToDepth", &SpaceToDepth_Compute, t, "MetalSpaceToDepth" + s,
             {});
    Register("DepthToSpace", &DepthToSpace_Compute, t, "MetalDepthToSpace" + s,
             {});
  }
}

}  // namespace metal
}  // namespace tensorflow
