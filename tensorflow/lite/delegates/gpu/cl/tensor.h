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
  Tensor() : memory_(nullptr) {}
  Tensor(cl_mem memory, int width, int height, int channels, DataType data_type,
         TensorStorageType storage_type);

  // Move only
  Tensor(Tensor&& tensor);
  Tensor& operator=(Tensor&& tensor);
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  virtual ~Tensor() { Release(); }

  int Width() const { return width_; }
  int Height() const { return height_; }
  int Channels() const { return channels_; }
  enum DataType DataType() const { return data_type_; }
  TensorStorageType StorageType() const { return storage_type_; }

  int Depth() const { return IntegralDivideRoundUp(channels_, 4); }
  int4 GetSizeWithDepth() const {
    return int4(width_, height_, channels_,
                IntegralDivideRoundUp(channels_, 4));
  }
  cl_mem GetMemoryPtr() const { return memory_; }

  Status WriteDataBHWC(absl::Span<const float> in, CLCommandQueue* queue);

  Status ReadDataBHWC(absl::Span<float> out, CLCommandQueue* queue) const;

  Status WriteData(CLCommandQueue* queue, const TensorFloat32& src);
  Status ReadData(CLCommandQueue* queue, TensorFloat32* dst) const;

 protected:
  Status IsValid(const BHWC& shape) const;

  template <typename T>
  void DataFromBHWC(absl::Span<const float> src, absl::Span<T> dst) const;
  template <typename T>
  void DataToBHWC(absl::Span<const T> src, absl::Span<float> dst) const;

  // TODO(sorokin) might be bad performance
  int GetLinearIndex(int x, int y, int d, int sub_d) const {
    switch (storage_type_) {
      case TensorStorageType::BUFFER:
      case TensorStorageType::TEXTURE_ARRAY:
        return ((d * height_ + y) * width_ + x) * 4 + sub_d;  // DHWC4
      case TensorStorageType::TEXTURE_2D:
        return ((y * Depth() + d) * width_ + x) * 4 + sub_d;  // HDWC4
      case TensorStorageType::SINGLE_TEXTURE_2D:
        return (sub_d * height_ + y) * width_ + x;
      case TensorStorageType::UNKNOWN:
        return -1;
    }
  }

  int3 GetFullTensorRegion() const;
  void Release();

  cl_mem memory_;
  int width_;
  int height_;
  int channels_;
  enum DataType data_type_;
  TensorStorageType storage_type_;
};

class TensorBHWC : public Tensor {
 public:
  TensorBHWC() = default;
  TensorBHWC(cl_mem memory, int width, int height, int channels,
             enum DataType data_type, TensorStorageType storage_type)
      : Tensor(memory, width, height, channels, data_type, storage_type) {}

  // Move only
  TensorBHWC(TensorBHWC&& tensor);
  TensorBHWC& operator=(TensorBHWC&& tensor);
  TensorBHWC(const TensorBHWC&) = delete;
  TensorBHWC& operator=(const TensorBHWC&) = delete;

  Status WriteData(CLCommandQueue* queue, void* data_ptr) const {
    const size_t data_size =
        Width() * Height() * Channels() * SizeOf(DataType());
    RETURN_IF_ERROR(
        queue->EnqueueWriteBuffer(GetMemoryPtr(), data_size, data_ptr));
    return OkStatus();
  }

  Status ReadData(CLCommandQueue* queue, void* data_ptr) const {
    const size_t data_size =
        Width() * Height() * Channels() * SizeOf(DataType());
    RETURN_IF_ERROR(
        queue->EnqueueReadBuffer(GetMemoryPtr(), data_size, data_ptr));
    return OkStatus();
  }

  ~TensorBHWC() override { ReleaseBHWC(); }

 private:
  friend Status CreateTensorBHWCFromOpenGlObject(const CLContext& context,
                                                 cl_int ssbo_id,
                                                 const HWC& shape,
                                                 bool is_readonly,
                                                 TensorBHWC* tensor);

  void ReleaseBHWC();

  // When object created from GL object it isn't owner
  bool owner_ = true;
};

using TensorPtr = std::shared_ptr<Tensor>;

Status AllocateTensorMemory(const CLContext& context, const CLDevice& device,
                            int width, int height, int channels,
                            DataType data_type, TensorStorageType storage_type,
                            CLMemory* result);

Status CreateTensor(const CLContext& context, const CLDevice& device, int width,
                    int height, int channels, DataType data_type,
                    TensorStorageType storage_type, Tensor* result);

Status CreateTensorBHWC(const CLContext& context, const HWC& shape,
                        DataType data_type, void* data, Tensor* result);

Status CreateTensorBHWCFromOpenGlObject(const CLContext& context,
                                        cl_int ssbo_id, const HWC& shape,
                                        bool is_readonly, TensorBHWC* tensor);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_TENSOR_H_
