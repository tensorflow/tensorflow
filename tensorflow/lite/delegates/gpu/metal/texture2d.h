/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_TEXTURE2D_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_TEXTURE2D_H_

#import <Metal/Metal.h>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/gpu_object.h"

namespace tflite {
namespace gpu {
namespace metal {

// Texture2D represent formatted GPU data storage.
// Texture2D is moveable but not copyable.
class Texture2D : public GPUObject {
 public:
  Texture2D() {}  // just for using Texture2D as a class members
  Texture2D(id<MTLTexture> texture, int width, int height, MTLPixelFormat pixel_format);

  // Move only
  Texture2D(Texture2D&& texture);
  Texture2D& operator=(Texture2D&& texture);
  Texture2D(const Texture2D&) = delete;
  Texture2D& operator=(const Texture2D&) = delete;

  ~Texture2D() override { Release(); }

  // Writes data to a texture. Data should point to a region that
  // has exact width * height * sizeof(pixel) bytes.
  template <typename T>
  absl::Status WriteData(id<MTLDevice> device, const absl::Span<T> data);

  // Reads data from Texture2D into CPU memory.
  template <typename T>
  absl::Status ReadData(id<MTLDevice> device, std::vector<T>* result) const;

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  absl::Status CreateFromTexture2DDescriptor(const Texture2DDescriptor& desc, id<MTLDevice> device);

 private:
  void Release();

  id<MTLTexture> texture_ = nullptr;
  int width_;
  int height_;
  MTLPixelFormat pixel_format_;
};

// Creates new 4-channel 2D texture with f32 elements
absl::Status CreateTexture2DRGBA32F(int width, int height, id<MTLDevice> device, Texture2D* result);

// Creates new 4-channel 2D texture with f16 elements
absl::Status CreateTexture2DRGBA16F(int width, int height, id<MTLDevice> device, Texture2D* result);

absl::Status CreateTexture2DRGBA(DataType type, int width, int height, id<MTLDevice> device,
                                 Texture2D* result);

absl::Status CreateTexture2DRGBA(DataType type, int width, int height, void* data,
                                 id<MTLDevice> device, Texture2D* result);

template <typename T>
absl::Status Texture2D::WriteData(id<MTLDevice> device,
                                  const absl::Span<T> data) {
  const int pixel_size = PixelFormatToSizeInBytes(pixel_format_);
  if (width_ * height_ * pixel_size != data.size() * sizeof(T)) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is different from texture allocated size.");
  }

  WriteDataToTexture2D(texture_, device, data.data());

  return absl::OkStatus();
}

template <typename T>
absl::Status Texture2D::ReadData(id<MTLDevice> device,
                                 std::vector<T>* result) const {
  const int pixel_size = PixelFormatToSizeInBytes(pixel_format_);
  if (pixel_size % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Pixel format is different.");
  }
  result->resize(width_ * height_ * (pixel_size / sizeof(T)));

  ReadDataFromTexture2D(texture_, device, result->data());

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_TEXTURE2D_H_
