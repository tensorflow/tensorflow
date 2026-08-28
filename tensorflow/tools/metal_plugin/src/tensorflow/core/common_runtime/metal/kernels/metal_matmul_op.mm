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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
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

// MPSMatrixMultiplication rather than a hand-written tiled shader: MPS ships
// kernels tuned per GPU generation, which no reasonable amount of shader work
// would match.
//
// MPSMatrix rather than MPSGraph is now only a matter of directness. An
// earlier version of this file justified the choice by claiming MPSGraph could
// not address a tensor at an offset inside its allocation; that was wrong.
// MPSNDArray's initWithBuffer:offset:descriptor: does exactly that, and
// metal_mps_graph.h builds the whole MPSGraph path on it. A 2-D matrix
// multiply maps onto MPSMatrix with less machinery, so it stays here, but
// either would work.

struct MatMulOp {
  bool transpose_a = false;
  bool transpose_b = false;
  TF_DataType dtype = TF_FLOAT;
};

void* MatMulOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new MatMulOp();

  TF_Bool transpose_a = 0;
  TF_Bool transpose_b = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_a", &transpose_a, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_b", &transpose_b,
                                        status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  op->transpose_a = transpose_a != 0;
  op->transpose_b = transpose_b != 0;
  TF_DeleteStatus(status);
  return op;
}

void MatMulOp_Delete(void* kernel) { delete static_cast<MatMulOp*>(kernel); }

MPSDataType MPSTypeOf(TF_DataType dtype) {
  return dtype == TF_HALF ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
}

// Wraps a densely packed 2-D tensor slice as an MPSMatrix.
//
// rowBytes is the exact packed stride, not the padded stride
// rowBytesFromColumns: recommends: TF tensors have no row padding, and a
// padded stride would make MPS read the wrong elements.
MPSMatrix* MatrixFor(const BufferSlice& slice, int64_t rows, int64_t columns,
                     TF_DataType dtype, const char* what, TF_Status* status) {
  const size_t element_size = TF_DataTypeSize(dtype);
  const size_t row_bytes = static_cast<size_t>(columns) * element_size;

  // MPS requires the row stride and the buffer offset to be 4-byte aligned.
  // The offset always is, since core's allocator aligns well past that, but a
  // float16 matrix with an odd column count has a 2 mod 4 stride. Repacking
  // such a matrix is possible and belongs with the wider op coverage; until
  // then, say so rather than let MPS trap.
  if (row_bytes % 4 != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 (std::string("Metal: MatMul on ") + what + " with " +
                  std::to_string(columns) +
                  " columns of this dtype gives a row stride that MPS cannot "
                  "use; an even column count is required for float16.")
                     .c_str());
    return nil;
  }
  if (slice.offset % 4 != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 (std::string("Metal: MatMul ") + what +
                  " is at an unaligned offset within its allocation.")
                     .c_str());
    return nil;
  }

  MPSMatrixDescriptor* descriptor =
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                            columns:columns
                                           rowBytes:row_bytes
                                           dataType:MPSTypeOf(dtype)];
  return [[[MPSMatrix alloc] initWithBuffer:slice.buffer
                                     offset:slice.offset
                                 descriptor:descriptor] autorelease];
}

void MatMulOp_ComputeImpl(MatMulOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: MatMul kernel has no state; construction failed.");
    return;
  }

  ScopedTensor lhs;
  ScopedTensor rhs;
  TF_GetInput(ctx, 0, lhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  if (TF_NumDims(lhs.get()) != 2 || TF_NumDims(rhs.get()) != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MatMul expects two rank-2 tensors.");
    return;
  }

  // Logical dimensions after the transpose attributes are applied. MPS is told
  // about the transposes and reads the operands in their stored layout, so
  // nothing is transposed in memory.
  const int64_t m = op->transpose_a ? TF_Dim(lhs.get(), 1) : TF_Dim(lhs.get(), 0);
  const int64_t k = op->transpose_a ? TF_Dim(lhs.get(), 0) : TF_Dim(lhs.get(), 1);
  const int64_t k_rhs =
      op->transpose_b ? TF_Dim(rhs.get(), 1) : TF_Dim(rhs.get(), 0);
  const int64_t n = op->transpose_b ? TF_Dim(rhs.get(), 0) : TF_Dim(rhs.get(), 1);

  if (k != k_rhs) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Metal: MatMul inner dimensions do not match: " +
                  std::to_string(k) + " against " + std::to_string(k_rhs) + ".")
                     .c_str());
    return;
  }

  const int64_t out_dims[2] = {m, n};
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_dims, 2,
      static_cast<size_t>(m) * n * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (m == 0 || n == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  BufferSlice lhs_slice;
  BufferSlice rhs_slice;
  BufferSlice out_slice;
  if (!SliceForTensor(lhs.get(), &lhs_slice, status)) return;
  if (!SliceForTensor(rhs.get(), &rhs_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  // Rows and columns as stored, which is the transpose of the logical shape
  // when the corresponding transpose attribute is set.
  MPSMatrix* left =
      MatrixFor(lhs_slice, TF_Dim(lhs.get(), 0), TF_Dim(lhs.get(), 1),
                op->dtype, "the left operand", status);
  if (left == nil) return;
  MPSMatrix* right =
      MatrixFor(rhs_slice, TF_Dim(rhs.get(), 0), TF_Dim(rhs.get(), 1),
                op->dtype, "the right operand", status);
  if (right == nil) return;
  MPSMatrix* result =
      MatrixFor(out_slice, m, n, op->dtype, "the result", status);
  if (result == nil) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for MatMul.");
    return;
  }

  MPSMatrixMultiplication* multiply = [[[MPSMatrixMultiplication alloc]
       initWithDevice:DeviceForStream(stream)
        transposeLeft:op->transpose_a
       transposeRight:op->transpose_b
           resultRows:m
        resultColumns:n
      interiorColumns:k
                alpha:1.0
                 beta:0.0] autorelease];

  [multiply encodeToCommandBuffer:command_buffer.get()
                       leftMatrix:left
                      rightMatrix:right
                     resultMatrix:result];
  command_buffer.Commit();
}

void MatMulOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  MatMulOp_ComputeImpl(static_cast<MatMulOp*>(kernel), ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RegisterMatMul(TF_DataType dtype, const char* kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder("MatMul", kMetalDeviceType, &MatMulOp_Create,
                          &MatMulOp_Compute, &MatMulOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(kernel_name, builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << kernel_name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalMatMulKernels() {
  RegisterMatMul(TF_FLOAT, "MetalMatMulFloat");
  RegisterMatMul(TF_HALF, "MetalMatMulHalf");
}

}  // namespace metal
}  // namespace tensorflow
