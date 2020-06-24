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
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
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

class Tensor : public GPUObject {
 public:
  Tensor()
      : memory_(nullptr), image_buffer_memory_(nullptr), memory_owner_(true) {}
  Tensor(cl_mem memory, bool memory_owner, const BHWC& shape,
         const TensorDescriptor& descriptor);
  Tensor(cl_mem memory, bool memory_owner, const BHWDC& shape,
         const TensorDescriptor& descriptor);
  Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
         const BHWC& shape, const TensorDescriptor& descriptor);
  Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
         const BHWDC& shape, const TensorDescriptor& descriptor);

  // Move only
  Tensor(Tensor&& tensor);
  Tensor& operator=(Tensor&& tensor);
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  virtual ~Tensor() { Release(); }

  GPUResourcesWithValue GetGPUResources(AccessType access_type) const override;

  int Width() const { return shape_.w; }
  int Height() const { return shape_.h; }
  int Depth() const { return shape_.d; }
  int Channels() const { return shape_.c; }
  int Slices() const { return DivideRoundUp(shape_.c, 4); }
  int Batch() const { return shape_.b; }

  // returns int4(width * batch, height, slices, batch)
  int4 GetWBatchedHSB() const {
    return int4(shape_.w * shape_.b, shape_.h, Slices(), shape_.b);
  }
  int4 GetWBatchedHDS() const {
    return int4(shape_.w * shape_.b, shape_.h, shape_.d, Slices());
  }

  int4 GetWHSB() const { return int4(shape_.w, shape_.h, Slices(), shape_.b); }
  int4 GetWHDS() const { return int4(shape_.w, shape_.h, shape_.d, Slices()); }

  TensorDescriptor GetDescriptor() const { return descriptor_; }
  DataType GetDataType() const { return descriptor_.data_type; }
  TensorStorageType GetStorageType() const { return descriptor_.storage_type; }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const;

  cl_mem GetMemoryPtr() const;

  // This function returns buffer memory ptr for IMAGE_BUFFER instead of image
  // memory ptr.
  cl_mem GetMemoryPtrForWriting() const;

  absl::Status WriteData(CLCommandQueue* queue, const TensorFloat32& src);
  absl::Status WriteData(
      CLCommandQueue* queue,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);
  absl::Status WriteData(
      CLCommandQueue* queue,
      const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  absl::Status WriteData(CLCommandQueue* queue, const Tensor5DFloat32& src);
  absl::Status ReadData(CLCommandQueue* queue, TensorFloat32* dst) const;
  absl::Status ReadData(CLCommandQueue* queue, Tensor5DFloat32* dst) const;

 private:
  absl::Status IsValid(const BHWC& shape) const;
  absl::Status IsValid(const BHWDC& shape) const;

  int GetChannelsAlignment() const;
  int GetAlignedChannels() const;

  absl::Status WriteDataBHWDC(absl::Span<const float> in,
                              CLCommandQueue* queue);
  absl::Status ReadDataBHWDC(absl::Span<float> out,
                             CLCommandQueue* queue) const;

  template <typename T>
  void DataFromBHWDC(absl::Span<const float> src, absl::Span<T> dst) const;
  template <typename T>
  void DataToBHWDC(absl::Span<const T> src, absl::Span<float> dst) const;

  // TODO(sorokin) might be bad performance
  int GetLinearIndex(int b, int x, int y, int d, int s, int sub_c) const {
    switch (descriptor_.storage_type) {
      case TensorStorageType::BUFFER:
      case TensorStorageType::IMAGE_BUFFER:
      case TensorStorageType::TEXTURE_ARRAY:
      case TensorStorageType::TEXTURE_3D:
        return ((((d * Slices() + s) * shape_.h + y) * shape_.w + x) *
                    shape_.b +
                b) *
                   4 +
               sub_c;  // DSHWBC4
      case TensorStorageType::TEXTURE_2D:
        return ((((y * Slices() + s) * shape_.w + x) * shape_.b + b) *
                    shape_.d +
                d) *
                   4 +
               sub_c;  // HSWBDC4
      case TensorStorageType::SINGLE_TEXTURE_2D:
        return (((y * shape_.w + x) * shape_.b + b) * shape_.d + d) * shape_.c +
               sub_c;  // HWBDC
      case TensorStorageType::UNKNOWN:
        return -1;
    }
  }

  int3 GetFullTensorRegion() const;
  void Release();

  cl_mem memory_;
  cl_mem image_buffer_memory_;  // for TensorStorageType::IMAGE_BUFFER only
  bool memory_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
};

using TensorPtr = std::shared_ptr<Tensor>;

absl::Status AllocateTensorMemory(const CLContext& context,
                                  const CLDevice& device, const BHWC& shape,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result);

absl::Status AllocateTensorMemory(const CLContext& context,
                                  const CLDevice& device, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result);

absl::Status CreateTensor(const CLContext& context, const CLDevice& device,
                          const BHWC& shape, const TensorDescriptor& descriptor,
                          Tensor* result);

absl::Status CreateTensor(const CLContext& context, const CLDevice& device,
                          const BHWDC& shape,
                          const TensorDescriptor& descriptor, Tensor* result);

absl::Status CreateSharedTensor(const CLContext& context,
                                const CLDevice& device, cl_mem memory,
                                const BHWC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result);

absl::Status CreateSharedTensor(const CLContext& context,
                                const CLDevice& device, cl_mem memory,
                                const BHWDC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_H_
