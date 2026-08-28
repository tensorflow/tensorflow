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

#include <dlfcn.h>
#include <cstdlib>
#include "absl/log/log.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/metal/metal_buffer_registry.h"

namespace tensorflow {
namespace metal {

namespace {

// Reads a 4-element per-dimension attribute, checking that the batch and
// channel entries are 1. TensorFlow allows striding over batch or channels in
// principle; no GPU backend implements it, and silently ignoring those entries
// would compute the wrong thing.
bool ReadSpatialPair(TF_OpKernelConstruction* ctx, const char* attr_name,
                     bool nhwc, int* height, int* width, TF_Status* status) {
  int32_t values[4] = {1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, attr_name, values, 4, status);
  if (TF_GetCode(status) != TF_OK) return false;

  const int batch_index = 0;
  const int channel_index = nhwc ? 3 : 1;
  if (values[batch_index] != 1 || values[channel_index] != 1) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 (std::string("Metal: ") + attr_name +
                  " over the batch or channel dimension is not supported.")
                     .c_str());
    return false;
  }
  *height = values[SpatialHeightIndex(nhwc)];
  *width = values[SpatialWidthIndex(nhwc)];
  if (*height < 1 || *width < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 (std::string("Metal: ") + attr_name +
                  " entries must be at least 1.")
                     .c_str());
    return false;
  }
  return true;
}

}  // namespace

bool ReadSpatialParams(TF_OpKernelConstruction* ctx, bool want_dilations,
                       SpatialParams* out, TF_Status* status) {
  char format[8] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    // data_format is optional on some ops and defaults to NHWC.
    TF_SetStatus(status, TF_OK, "");
    out->nhwc = true;
  } else if (std::strcmp(format, "NHWC") == 0) {
    out->nhwc = true;
  } else if (std::strcmp(format, "NCHW") == 0) {
    out->nhwc = false;
  } else {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 (std::string("Metal: data_format '") + format +
                  "' is not supported; use NHWC or NCHW.")
                     .c_str());
    return false;
  }

  char padding[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding,
                                        sizeof(padding) - 1, status);
  if (TF_GetCode(status) != TF_OK) return false;
  if (std::strcmp(padding, "SAME") == 0) {
    out->same_padding = true;
  } else if (std::strcmp(padding, "VALID") == 0) {
    out->same_padding = false;
  } else {
    // EXPLICIT lands here. MPSGraph can express it, but the offsets have to be
    // read from a separate attribute and mapped per data format; that is left
    // until something needs it, rather than guessed at.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 (std::string("Metal: padding '") + padding +
                  "' is not supported; use SAME or VALID.")
                     .c_str());
    return false;
  }

  if (!ReadSpatialPair(ctx, "strides", out->nhwc, &out->stride_h,
                       &out->stride_w, status)) {
    return false;
  }

  if (want_dilations) {
    int height = 1;
    int width = 1;
    if (!ReadSpatialPair(ctx, "dilations", out->nhwc, &height, &width,
                         status)) {
      // dilations is optional and defaults to all ones.
      TF_SetStatus(status, TF_OK, "");
      height = 1;
      width = 1;
    }
    out->dilation_h = height;
    out->dilation_w = width;
  }

  TF_OpKernelConstruction_GetAttrType(ctx, "T", &out->dtype, status);
  if (TF_GetCode(status) != TF_OK) return false;
  return true;
}

bool SliceForTensor(TF_Tensor* tensor, BufferSlice* slice, TF_Status* status) {
  void* data = TF_TensorData(tensor);
  if (data == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: tensor has no backing data.");
    return false;
  }
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  if (!MetalBufferRegistry::Global().Lookup(data, &buffer, &offset)) {
    // Reaching this means the tensor is in host memory. Either the kernel was
    // registered with TF_KernelBuilder_HostMemory for this argument, or it ran
    // on a device this backend does not own.
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: tensor is not backed by a Metal allocation; it is "
                 "probably in host memory.");
    return false;
  }
  slice->buffer = buffer;
  slice->offset = offset;
  slice->length = TF_TensorByteSize(tensor);
  return true;
}

SP_Stream StreamForContext(TF_OpKernelContext* ctx, TF_Status* status) {
  SP_Stream stream = TF_GetStream(ctx, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  if (stream == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: op kernel context provided no stream.");
    return nullptr;
  }
  return stream;
}

id<MTLDevice> DeviceForStream(SP_Stream stream) { return stream->queue.device; }

void Dispatch1D(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline, uint32_t count) {
  const NSUInteger threads_per_group =
      std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, count);
  const NSUInteger groups = (count + threads_per_group - 1) / threads_per_group;
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

int64_t NumElements(TF_Tensor* tensor) {
  int64_t count = 1;
  const int rank = TF_NumDims(tensor);
  for (int i = 0; i < rank; ++i) count *= TF_Dim(tensor, i);
  return count;
}

bool ResourceVariableApiAvailable() {
  static const bool available = [] {
    static constexpr const char* kRequired[] = {
        "TF_AssignRefVariable",
        "TF_GetInputTensorFromVariable",
        "TF_MaybeLockVariableInputMutexesInOrder",
        "TF_ReleaseVariableInputLockHolder",
        "TF_OpKernelConstruction_GetAttrTensorShape",
        "TF_OpKernelContext_ForwardRefInputToRefOutput",
    };
    for (const char* name : kRequired) {
      if (dlsym(RTLD_DEFAULT, name) == nullptr) return false;
    }
    return true;
  }();
  return available;
}

bool SynchronousMode() {
  static const bool value = [] {
    const char* forced = std::getenv("TF_METAL_SYNCHRONOUS");
    if (forced != nullptr && forced[0] != '\0') {
      return std::strcmp(forced, "0") != 0;
    }
    if (!ResourceVariableApiAvailable()) {
      LOG(WARNING) << "Metal: this TensorFlow does not export the kernel C API "
                      "for resource variables, so its own kernels update "
                      "variables from the host. Every Metal kernel now waits "
                      "for the GPU before returning, which is slower and is "
                      "the only way those updates cannot race. See "
                      "https://github.com/tensorflow/tensorflow/issues/126374.";
      return true;
    }
    return false;
  }();
  return value;
}

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

std::vector<int64_t> ShapeOf(TF_Tensor* tensor) {
  const int rank = TF_NumDims(tensor);
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for (int i = 0; i < rank; ++i) shape.push_back(TF_Dim(tensor, i));
  return shape;
}

}  // namespace metal
}  // namespace tensorflow
