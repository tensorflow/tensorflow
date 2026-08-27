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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// MatrixTriangularSolve and its deprecated alias.
//
// This is the one piece of dense linear algebra Metal Performance Shaders
// ships as a kernel, so it goes through MPSMatrixSolveTriangular rather than
// MPSGraph. Each matrix in a batch is encoded separately: the solve is
// sequential down the rows by nature, so batching it into one encode would buy
// nothing, and one encode per matrix keeps the offsets obvious.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct SolveOp {
  bool lower = true;
  bool adjoint = false;
  // Lu names the type of its permutation output.
  TF_DataType index_dtype = TF_INT32;
};

void* SolveOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new SolveOp();
  TF_Bool flag = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "lower", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->lower = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "adjoint", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->adjoint = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DataType index_dtype = TF_INT32;
  TF_OpKernelConstruction_GetAttrType(ctx, "output_idx_type", &index_dtype,
                                      status);
  if (TF_GetCode(status) == TF_OK) op->index_dtype = index_dtype;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void SolveOp_Delete(void* kernel) { delete static_cast<SolveOp*>(kernel); }

// Wraps one densely packed matrix inside a larger allocation.
//
// rowBytes is the exact packed stride: TensorFlow tensors have no row padding,
// and the padded stride MPS recommends would make it read the wrong elements.
MPSMatrix* MatrixAt(const BufferSlice& slice, size_t element_offset,
                    int64_t rows, int64_t columns, TF_Status* status) {
  const size_t offset = slice.offset + element_offset * sizeof(float);
  if (offset % 4 != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: a matrix lies at an offset MPS cannot address.");
    return nil;
  }
  MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
                       columns:static_cast<NSUInteger>(columns)
                      rowBytes:static_cast<NSUInteger>(columns) *
                               sizeof(float)
                      dataType:MPSDataTypeFloat32];
  return [[[MPSMatrix alloc] initWithBuffer:slice.buffer
                                     offset:offset
                                 descriptor:descriptor] autorelease];
}

void TriangularSolve_ComputeImpl(SolveOp* op, TF_OpKernelContext* ctx,
                                 TF_Status* status) {
  ScopedTensor matrix, rhs;
  TF_GetInput(ctx, 0, matrix.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> a_shape = ShapeOf(matrix.get());
  const std::vector<int64_t> b_shape = ShapeOf(rhs.get());
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MatrixTriangularSolve expects rank-2 or higher "
                 "inputs.");
    return;
  }
  const int64_t order = a_shape[a_shape.size() - 1];
  if (a_shape[a_shape.size() - 2] != order) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the coefficient matrix must be square.");
    return;
  }
  if (b_shape[b_shape.size() - 2] != order) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the right hand side does not match the coefficient "
                 "matrix.");
    return;
  }
  const int64_t rhs_count = b_shape[b_shape.size() - 1];

  int64_t a_batch = 1;
  for (size_t i = 0; i + 2 < a_shape.size(); ++i) a_batch *= a_shape[i];
  int64_t b_batch = 1;
  for (size_t i = 0; i + 2 < b_shape.size(); ++i) b_batch *= b_shape[i];
  // TensorFlow broadcasts the batch dimensions; the case that matters and the
  // only one handled here is a single matrix against many right hand sides,
  // or matching batches. Anything else is refused rather than guessed at.
  if (a_batch != b_batch && a_batch != 1) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: MatrixTriangularSolve broadcasts a single "
                 "coefficient matrix only.");
    return;
  }

  const std::vector<int64_t>& out_shape = b_shape;
  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  BufferSlice a_slice, b_slice, out_slice;
  if (!SliceForTensor(matrix.get(), &a_slice, status)) return;
  if (!SliceForTensor(rhs.get(), &b_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  MPSMatrixSolveTriangular* solve = [[[MPSMatrixSolveTriangular alloc]
             initWithDevice:device
                      right:NO
                      upper:op->lower ? NO : YES
                  transpose:op->adjoint ? YES : NO
                       unit:NO
                      order:static_cast<NSUInteger>(order)
     numberOfRightHandSides:static_cast<NSUInteger>(rhs_count)
                      alpha:1.0] autorelease];
  if (solve == nil) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: could not create a triangular solve.");
    return;
  }

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a triangular "
                 "solve.");
    return;
  }
  const size_t a_stride = static_cast<size_t>(order * order);
  const size_t b_stride = static_cast<size_t>(order * rhs_count);
  for (int64_t i = 0; i < b_batch; ++i) {
    MPSMatrix* a = MatrixAt(a_slice, (a_batch == 1 ? 0 : i) * a_stride, order,
                            order, status);
    if (a == nil) return;
    MPSMatrix* b = MatrixAt(b_slice, i * b_stride, order, rhs_count, status);
    if (b == nil) return;
    MPSMatrix* x = MatrixAt(out_slice, i * b_stride, order, rhs_count, status);
    if (x == nil) return;
    [solve encodeToCommandBuffer:command_buffer.get()
                    sourceMatrix:a
             rightHandSideMatrix:b
                  solutionMatrix:x];
  }
  command_buffer.Commit();
}

/*** LU FACTORISATION ***/

// MPS returns exactly the packed factor TensorFlow wants: the strictly lower
// triangle holds L without its unit diagonal, and the upper triangle holds U.
// What differs is the permutation. MPS reports row interchanges the way LAPACK
// does, entry by entry in order, while TensorFlow wants the permutation those
// interchanges produce, so the swaps are replayed by a shader rather than read
// back and replayed on the host.
void Lu_ComputeImpl(SolveOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Lu expects a rank-2 or higher input.");
    return;
  }
  const int64_t order = shape[shape.size() - 1];
  if (shape[shape.size() - 2] != order) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Lu expects square matrices.");
    return;
  }
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) batch *= shape[i];

  std::vector<int64_t> perm_shape(shape.begin(), shape.end() - 1);
  ScopedTensor lu, perm;
  lu.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(ElementCount(shape)) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  perm.reset(TF_AllocateOutput(
      ctx, 1, op->index_dtype, perm_shape.data(),
      static_cast<int>(perm_shape.size()),
      static_cast<size_t>(ElementCount(perm_shape)) *
          TF_DataTypeSize(op->index_dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (order == 0 || batch == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  // MPS wants the pivots as unsigned words; they never reach the host.
  const std::vector<int64_t> pivot_shape = {batch, order};
  ScopedTensor pivots;
  pivots.reset(TF_AllocateTemp(ctx, TF_UINT32, pivot_shape.data(), 2, nullptr,
                               status));
  if (TF_GetCode(status) != TF_OK) return;

  BufferSlice in_slice, lu_slice, pivot_slice, perm_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(lu.get(), &lu_slice, status)) return;
  if (!SliceForTensor(pivots.get(), &pivot_slice, status)) return;
  if (!SliceForTensor(perm.get(), &perm_slice, status)) return;

  MPSMatrixDecompositionLU* decomposition =
      [[[MPSMatrixDecompositionLU alloc]
          initWithDevice:device
                    rows:static_cast<NSUInteger>(order)
                 columns:static_cast<NSUInteger>(order)] autorelease];
  if (decomposition == nil) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: could not create an LU factorisation.");
    return;
  }

  {
    OrderedCommandBuffer command_buffer(stream);
    if (!command_buffer.ok()) {
      TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                   "Metal: could not create a command buffer for Lu.");
      return;
    }
    const size_t matrix_stride = static_cast<size_t>(order * order);
    for (int64_t i = 0; i < batch; ++i) {
      MPSMatrix* source =
          MatrixAt(in_slice, i * matrix_stride, order, order, status);
      if (source == nil) return;
      MPSMatrix* result =
          MatrixAt(lu_slice, i * matrix_stride, order, order, status);
      if (result == nil) return;
      MPSMatrixDescriptor* pivot_descriptor = [MPSMatrixDescriptor
          matrixDescriptorWithRows:1
                           columns:static_cast<NSUInteger>(order)
                          rowBytes:static_cast<NSUInteger>(order) *
                                   sizeof(uint32_t)
                          dataType:MPSDataTypeUInt32];
      MPSMatrix* pivot_matrix = [[[MPSMatrix alloc]
          initWithBuffer:pivot_slice.buffer
                  offset:pivot_slice.offset +
                         static_cast<size_t>(i * order) * sizeof(uint32_t)
              descriptor:pivot_descriptor] autorelease];
      // The status buffer is not requested: a singular matrix is reported by
      // TensorFlow through the values themselves, and reading a status would
      // mean synchronising on every call.
      [decomposition encodeToCommandBuffer:command_buffer.get()
                             sourceMatrix:source
                             resultMatrix:result
                             pivotIndices:pivot_matrix
                                   status:nil];
    }
    command_buffer.Commit();
  }

  id<MTLComputePipelineState> pipeline =
      PipelineFor(device,
                  op->index_dtype == TF_INT64
                      ? "tf_pivots_to_permutation_i64"
                      : "tf_pivots_to_permutation_i32",
                  status);
  if (pipeline == nil) return;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for the Lu "
                 "permutation.");
    return;
  }
  PivotParams params;
  params.batch = static_cast<uint32_t>(batch);
  params.order = static_cast<uint32_t>(order);
  params.padding0 = 0;
  params.padding1 = 0;
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:pivot_slice.buffer offset:pivot_slice.offset atIndex:0];
  [encoder setBuffer:perm_slice.buffer offset:perm_slice.offset atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  Dispatch1D(encoder, pipeline, params.batch);
  [encoder endEncoding];
  command_buffer.Commit();
}

void Lu_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<SolveOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: Lu has no state.");
  } else {
    Lu_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void TriangularSolve_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<SolveOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: MatrixTriangularSolve has no state.");
  } else {
    TriangularSolve_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &SolveOp_Create, compute, &SolveOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
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

void RegisterMetalLinalgKernels() {
  // Float32 only: the MPS solve does not take half, and a half triangular
  // solve would lose more precision than it would save time.
  Register("MatrixTriangularSolve", &TriangularSolve_Compute,
           "MetalMatrixTriangularSolve");
  Register("BatchMatrixTriangularSolve", &TriangularSolve_Compute,
           "MetalBatchMatrixTriangularSolve");
  // The permutation output can be either width, so both are registered.
  {
    static constexpr TF_DataType kIndex[] = {TF_INT32, TF_INT64};
    static constexpr const char* kSuffix[] = {"Int32", "Int64"};
    for (int i = 0; i < 2; ++i) {
      TF_Status* status = TF_NewStatus();
      TF_KernelBuilder* builder = TF_NewKernelBuilder(
          "Lu", kMetalDeviceType, &SolveOp_Create, &Lu_Compute,
          &SolveOp_Delete);
      TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
      if (TF_GetCode(status) == TF_OK) {
        TF_KernelBuilder_TypeConstraint(builder, "output_idx_type", kIndex[i],
                                        status);
      }
      const std::string name = std::string("MetalLu") + kSuffix[i];
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
  }
}

}  // namespace metal
}  // namespace tensorflow
