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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNEL_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNEL_UTIL_H_

// Objective-C++ only.

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {

// Deletes a TF_Tensor on scope exit.
//
// The C kernel API hands out owned tensors, and a kernel has several
// early-return paths (shape rejection, allocation failure, pipeline
// compilation failure). Leaking one on any of them leaks device memory for the
// life of the process.
class ScopedTensor {
 public:
  ScopedTensor() = default;
  ~ScopedTensor() {
    if (tensor_ != nullptr) TF_DeleteTensor(tensor_);
  }

  ScopedTensor(const ScopedTensor&) = delete;
  ScopedTensor& operator=(const ScopedTensor&) = delete;

  // For handing to TF_GetInput, which writes through the pointer.
  TF_Tensor** address() { return &tensor_; }
  TF_Tensor* get() const { return tensor_; }
  void reset(TF_Tensor* tensor) {
    if (tensor_ != nullptr) TF_DeleteTensor(tensor_);
    tensor_ = tensor;
  }

 private:
  TF_Tensor* tensor_ = nullptr;
};

// A tensor's storage expressed the way a Metal encoder needs it.
//
// Device tensors do not start at the beginning of an MTLBuffer: core's BFC
// allocator sub-divides one allocation into many tensors, so the offset is the
// normal case rather than an edge case. Every Metal API this backend uses to
// reach tensor memory therefore has to accept an offset, which is why
// setBuffer:offset:atIndex: and MPSMatrix's initWithBuffer:offset:descriptor:
// are preferred over interfaces that assume a buffer starts at element zero.
struct BufferSlice {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  size_t length = 0;
};

// Attributes shared by the spatial ops (convolution, pooling).
//
// TensorFlow expresses strides and dilations as one entry per dimension of the
// data layout, with the batch and channel entries required to be 1; only the
// two spatial entries carry information, which is what MPSGraph wants.
struct SpatialParams {
  int stride_h = 1;
  int stride_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  // TensorFlow's SAME and VALID map exactly onto MPSGraph's TF_SAME and
  // TF_VALID padding styles. EXPLICIT is rejected by the reader.
  bool same_padding = false;
  // NHWC, TensorFlow's default, versus NCHW.
  bool nhwc = true;
  TF_DataType dtype = TF_FLOAT;
};

// Reads strides, padding, data_format, T, and optionally dilations, from a
// kernel's construction context. Fails `status` on anything unsupported,
// naming the attribute at fault.
bool ReadSpatialParams(TF_OpKernelConstruction* ctx, bool want_dilations,
                       SpatialParams* out, TF_Status* status);

// Index of the height and width entries in a 4-element per-dimension attribute
// list, given the data format.
inline int SpatialHeightIndex(bool nhwc) { return nhwc ? 1 : 2; }
inline int SpatialWidthIndex(bool nhwc) { return nhwc ? 2 : 3; }

// Resolves a device tensor's storage. Fails `status` with a diagnosable
// message if the tensor's data does not lie in a live Metal allocation, which
// would mean it was placed on the host rather than the device.
bool SliceForTensor(TF_Tensor* tensor, BufferSlice* slice, TF_Status* status);

// The stream this kernel must enqueue onto. Fails `status` if the context has
// no StreamExecutor stream, which happens when the kernel was somehow placed
// on a device this backend does not own.
SP_Stream StreamForContext(TF_OpKernelContext* ctx, TF_Status* status);

// The Metal device backing `stream`.
id<MTLDevice> DeviceForStream(SP_Stream stream);

// One dispatch of a one-dimensional shader over `count` elements.
//
// Uses dispatchThreadgroups: rather than dispatchThreads:, so the grid rounds
// up to whole threadgroups and the call works regardless of GPU family. Every
// shader in this backend bounds-checks against its element count, which is
// what makes the rounding safe.
void Dispatch1D(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline, uint32_t count);

// Number of elements described by a tensor's shape.
int64_t NumElements(TF_Tensor* tensor);

// Shape as a plain vector, for shape checks and for building descriptors.
std::vector<int64_t> ShapeOf(TF_Tensor* tensor);

// Whether this TensorFlow exports the kernel C API entry points a plugin needs
// to reach a resource variable.
//
// They are declared by tensorflow/c/kernels_experimental.h and, since 2.20.0,
// defined by no binary a release ships. Without them a plugin cannot implement
// the ops that read and write variables, so those fall back to TensorFlow's
// own kernels, which reach the tensor through its data pointer. On a unified
// memory device that pointer is host-addressable, so they read and write
// device memory from the host with no idea that GPU work is in flight against
// it.
bool ResourceVariableApiAvailable();

// Whether a kernel waits for the GPU before returning.
//
// True when the entry points above are missing, because then the host-side
// fallbacks race with anything still running. It costs the asynchrony and buys
// correctness; TF_METAL_SYNCHRONOUS forces it either way.
bool SynchronousMode();

// Waits for everything already enqueued on `stream`.
//
// A kernel that gives the GPU a temporary buffer must call this before it
// returns. TF_AllocateTemp memory goes back to the allocator the moment the
// tensor is destroyed, and the next kernel is handed the same block, so work
// still in flight reads what that next kernel has since written. Two inverse
// real transforms in a row were enough: whichever ran first was right and the
// other was not, in either order.
void WaitForStream(SP_Stream stream);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNEL_UTIL_H_
