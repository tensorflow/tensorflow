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

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_memory.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class Tensor : public GPUObject, public GpuSpatialTensor {
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

  ~Tensor() override { Release(); }

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  int Width() const override { return shape_.w; }
  int Height() const override { return shape_.h; }
  int Depth() const override { return shape_.d; }
  int Channels() const override { return shape_.c; }
  int Slices() const override { return DivideRoundUp(shape_.c, 4); }
  int Batch() const override { return shape_.b; }

  TensorDescriptor GetDescriptor() const override { return descriptor_; }
  DataType GetDataType() const { return descriptor_.GetDataType(); }
  TensorStorageType GetStorageType() const {
    return descriptor_.GetStorageType();
  }

  uint64_t GetMemorySizeInBytes() const;

  cl_mem GetMemoryPtr() const;

  // This function returns buffer memory ptr for IMAGE_BUFFER instead of image
  // memory ptr.
  cl_mem GetMemoryPtrForWriting() const;

  absl::Status WriteData(
      CLCommandQueue* queue,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);
  absl::Status WriteData(
      CLCommandQueue* queue,
      const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  template <DataType T>
  absl::Status WriteData(CLCommandQueue* queue,
                         const tflite::gpu::Tensor<BHWC, T>& src);
  template <DataType T>
  absl::Status WriteData(CLCommandQueue* queue,
                         const tflite::gpu::Tensor<BHWDC, T>& src);
  template <DataType T>
  absl::Status ReadData(CLCommandQueue* queue,
                        tflite::gpu::Tensor<BHWC, T>* dst) const;
  template <DataType T>
  absl::Status ReadData(CLCommandQueue* queue,
                        tflite::gpu::Tensor<BHWDC, T>* dst) const;

  absl::Status CreateFromDescriptor(const TensorDescriptor& desc,
                                    CLContext* context);
  absl::Status ToDescriptor(TensorDescriptor* desc,
                            CLCommandQueue* queue) const;

 private:
  friend absl::Status CreateTensorSharedImage2DBuffer(
      const CLContext& context, cl_mem memory,
      const TensorDescriptor& descriptor, int width_pixel_alignment,
      Tensor* result);
  absl::Status IsValid(const BHWC& shape) const;
  absl::Status IsValid(const BHWDC& shape) const;

  int GetChannelsAlignment() const;
  int GetAlignedChannels() const;

  template <typename T>
  absl::Status WriteDataBHWDC(const T* in, CLCommandQueue* queue);
  absl::Status WriteData(const void* ptr, CLCommandQueue* queue);
  template <typename T>
  absl::Status ReadDataBHWDC(T* out, CLCommandQueue* queue) const;
  absl::Status ReadData(void* ptr, CLCommandQueue* queue) const;

  int3 GetFullTensorRegion() const;
  void Release();

  cl_mem memory_;
  cl_mem image_buffer_memory_;  // for IMAGE_BUFFER/TEXTURE_2D/SINGLE_TEXTURE_2D
  bool memory_owner_;
  bool buffer_based_ = false;
  BHWDC shape_;
  TensorDescriptor descriptor_;
  // for use with TEXTURE_2D and when texture created from buffer.
  int aligned_texture_width_;
};

using TensorPtr = std::shared_ptr<Tensor>;

absl::Status AllocateTensorMemory(const CLContext& context,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result);

absl::Status CreateTensor(const CLContext& context,
                          const TensorDescriptor& descriptor, Tensor* result);

absl::Status CreateTensorShared(const CLContext& context, cl_mem memory,
                                const TensorDescriptor& descriptor,
                                Tensor* result);

absl::Status CreateSharedTensor(const CLContext& context, cl_mem memory,
                                const BHWC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result);

absl::Status CreateTensorSharedImage2DBuffer(const CLContext& context,
                                             cl_mem memory,
                                             const TensorDescriptor& descriptor,
                                             int width_pixel_alignment,
                                             Tensor* result);

template <DataType T>
absl::Status Tensor::WriteData(CLCommandQueue* queue,
                               const tflite::gpu::Tensor<BHWC, T>& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(src.data.data(), queue);
}

template <DataType T>
absl::Status Tensor::WriteData(CLCommandQueue* queue,
                               const tflite::gpu::Tensor<BHWDC, T>& src) {
  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(src.data.data(), queue);
}

template <DataType T>
absl::Status Tensor::ReadData(CLCommandQueue* queue,
                              tflite::gpu::Tensor<BHWC, T>* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(dst->data.data(), queue);
}

template <DataType T>
absl::Status Tensor::ReadData(CLCommandQueue* queue,
                              tflite::gpu::Tensor<BHWDC, T>* dst) const {
  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(dst->data.data(), queue);
}

template <typename T>
absl::Status Tensor::WriteDataBHWDC(const T* in, CLCommandQueue* queue) {
  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[GetMemorySizeInBytes()]);
  if (descriptor_.GetDataType() == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataFromBHWDC(reinterpret_cast<const float*>(in), shape_, descriptor_,
                  reinterpret_cast<half*>(data_copy.get()));
  } else {
    // rearrangement
    DataFromBHWDC(in, shape_, descriptor_,
                  reinterpret_cast<T*>(data_copy.get()));
  }

  return WriteData(data_copy.get(), queue);
}

template <typename T>
absl::Status Tensor::ReadDataBHWDC(T* out, CLCommandQueue* queue) const {
  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[GetMemorySizeInBytes()]);

  RETURN_IF_ERROR(ReadData(data_copy.get(), queue));

  if (descriptor_.GetDataType() == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataToBHWDC(reinterpret_cast<half*>(data_copy.get()), shape_, descriptor_,
                reinterpret_cast<float*>(out));
  } else {
    // rearrangement
    DataToBHWDC(reinterpret_cast<T*>(data_copy.get()), shape_, descriptor_,
                out);
  }

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_H_
