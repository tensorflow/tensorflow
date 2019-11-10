/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_H_

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_memory.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class Tensor {
 public:
  Tensor()
      : memory_(nullptr), image_buffer_memory_(nullptr), memory_owner_(true) {}
  Tensor(cl_mem memory, bool memory_owner, const BHWC& shape,
         const TensorDescriptor& descriptor);
  Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
         const BHWC& shape, const TensorDescriptor& descriptor);

  // Move only
  Tensor(Tensor&& tensor);
  Tensor& operator=(Tensor&& tensor);
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  virtual ~Tensor() { Release(); }

  int Width() const { return shape_.w; }
  int Height() const { return shape_.h; }
  int Channels() const { return shape_.c; }
  int Depth() const { return IntegralDivideRoundUp(shape_.c, 4); }
  int Batch() const { return shape_.b; }
  int4 GetSizeWithDepth() const {
    return int4(shape_.w, shape_.h, shape_.c, Depth());
  }

  // returns int4(width * batch, height, depth, batch)
  int4 GetWBatchedHDB() const {
    return int4(shape_.w * shape_.b, shape_.h, Depth(), shape_.b);
  }

  int4 GetWHDB() const { return int4(shape_.w, shape_.h, Depth(), shape_.b); }

  enum DataType DataType() const { return descriptor_.data_type; }
  TensorStorageType StorageType() const { return descriptor_.storage_type; }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const;

  cl_mem GetMemoryPtr() const;

  // This function returns buffer memory ptr for IMAGE_BUFFER instead of image
  // memory ptr.
  cl_mem GetMemoryPtrForWriting() const;

  Status WriteData(CLCommandQueue* queue, const TensorFloat32& src);
  Status ReadData(CLCommandQueue* queue, TensorFloat32* dst) const;

 private:
  Status IsValid(const BHWC& shape) const;

  int GetChannelsAlignment() const;
  int GetAlignedChannels() const;

  Status WriteDataBHWC(absl::Span<const float> in, CLCommandQueue* queue);
  Status ReadDataBHWC(absl::Span<float> out, CLCommandQueue* queue) const;

  template <typename T>
  void DataFromBHWC(absl::Span<const float> src, absl::Span<T> dst) const;
  template <typename T>
  void DataToBHWC(absl::Span<const T> src, absl::Span<float> dst) const;

  // TODO(sorokin) might be bad performance
  int GetLinearIndex(int b, int x, int y, int d, int sub_d) const {
    switch (descriptor_.storage_type) {
      case TensorStorageType::BUFFER:
      case TensorStorageType::IMAGE_BUFFER:
      case TensorStorageType::TEXTURE_ARRAY:
        return (((d * shape_.h + y) * shape_.w + x) * shape_.b + b) * 4 +
               sub_d;  // DHWBC4
      case TensorStorageType::TEXTURE_2D:
        return (((y * Depth() + d) * shape_.w + x) * shape_.b + b) * 4 +
               sub_d;  // HDWBC4
      case TensorStorageType::SINGLE_TEXTURE_2D:
        return ((y * shape_.w + x) * shape_.b + b) * shape_.c + sub_d;  // HWBC
      case TensorStorageType::UNKNOWN:
        return -1;
    }
  }

  int3 GetFullTensorRegion() const;
  void Release();

  cl_mem memory_;
  cl_mem image_buffer_memory_;  // for TensorStorageType::IMAGE_BUFFER only
  bool memory_owner_;
  BHWC shape_;
  TensorDescriptor descriptor_;
};

using TensorPtr = std::shared_ptr<Tensor>;

bool CanCreateTensorWithShape(const CLContext& context, const CLDevice& device,
                              const BHWC& shape,
                              const TensorDescriptor& descriptor);

Status AllocateTensorMemory(const CLContext& context, const CLDevice& device,
                            const BHWC& shape,
                            const TensorDescriptor& descriptor,
                            CLMemory* result);

Status CreateTensor(const CLContext& context, const CLDevice& device,
                    const BHWC& shape, const TensorDescriptor& descriptor,
                    Tensor* result);

Status CreateSharedTensor(const CLContext& context, const CLDevice& device,
                          cl_mem memory, const BHWC& shape,
                          const TensorDescriptor& descriptor, Tensor* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_H_
