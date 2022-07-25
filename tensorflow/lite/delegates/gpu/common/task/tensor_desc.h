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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_

#include <cstddef>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

enum class TensorStorageType {
  UNKNOWN,
  BUFFER,
  IMAGE_BUFFER,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_ARRAY,
  SINGLE_TEXTURE_2D
};

class TensorDescriptor : public GPUObjectDescriptor {
 public:
  TensorDescriptor() = default;
  TensorDescriptor(DataType data_type, TensorStorageType storage_type,
                   Layout layout)
      : data_type_(data_type), storage_type_(storage_type), layout_(layout) {}

  TensorDescriptor(const TensorDescriptor&) = default;
  TensorDescriptor& operator=(const TensorDescriptor&) = default;
  TensorDescriptor(TensorDescriptor&& desc);
  TensorDescriptor& operator=(TensorDescriptor&& desc);

  void CopyWithoutData(TensorDescriptor* desc) const;

  bool operator==(const TensorDescriptor& d) const {
    return data_type_ == d.data_type_ && storage_type_ == d.storage_type_ &&
           layout_ == d.layout_;
  }

  bool operator!=(const TensorDescriptor& d) const { return !(*this == d); }

  void GetGpuResources(const BHWDC& tensor_shape,
                       GenericGPUResourcesWithValue* resources) const;

  absl::Status PerformConstExpr(const GpuInfo& gpu_info,
                                const std::string& const_expr,
                                std::string* result) const override;

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;

  void Release() override { data_.clear(); }
  uint64_t GetSizeInBytes() const override { return data_.size(); };
  size_t GetSizeInBytesForShape(const BHWDC& shape5d) const;

  bool HasAxis(Axis axis) const;
  int GetWidthSize(BHWDC shape) const;
  int GetSliceStrideSize(BHWDC shape) const;

  absl::Status GetLinkingContextFromWriteSelector(
      const std::vector<std::string>& args, std::string* value_name,
      std::string* x_coord, std::string* y_coord, std::string* s_coord) const;

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<BHWC, T>& src);
  template <DataType T>
  void DownloadData(tflite::gpu::Tensor<BHWC, T>* dst);
  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<BHWDC, T>& src);
  template <DataType T>
  void DownloadData(tflite::gpu::Tensor<BHWDC, T>* dst);

  void UploadData(const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  void UploadData(const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);

  int GetLinearIndex(const BHWDC& shape5d, int b, int x, int y, int d, int s,
                     int sub_c) const;

  bool SupportsZeroClamp(const Axis& axis, const GpuInfo& gpu_info) const;
  bool CanReadOutOfBorder(const Axis& axis) const;
  bool IsLinear() const;

  DataType GetDataType() const { return data_type_; }
  TensorStorageType GetStorageType() const { return storage_type_; }

  // applicable only for types that: IsLinear -> true.
  // In this case for address we have 1d component - addr (int)
  // If for addr == -1 this linear storage type returns zero value, this
  // function returns true, otherwise false
  bool ReturnsZeroForNegOneRead(const GpuInfo& gpu_info) const;

  absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                        const BHWDC& shape) const;

  absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                        const BHWC& shape) const;

  // Can udate storage type if in the current storage type this tensor can not
  // be allocated with shape on specified device(gpu_info)
  // Usual scenario is to create new tensor_desc on base of another and may be
  // update storage type for new tensor_desc shape because it can be unsuported
  // with old storage type
  absl::Status UpdateToSupportedStorageType(const GpuInfo& gpu_info,
                                            const BHWC& shape);

  void SetUseBufferForWriteOnlyTexture2d(bool value) {
    use_buffer_for_write_only_2d_texture_ = value;
  }
  bool GetUseBufferForWriteOnlyTexture2d() const {
    return use_buffer_for_write_only_2d_texture_;
  }

  void SetUseBufferForWriteOnlyImageBuffer(bool value) {
    use_buffer_for_write_only_image_buffer_ = value;
  }
  bool GetUseBufferForWriteOnlyImageBuffer() const {
    return use_buffer_for_write_only_image_buffer_;
  }

  void SetBHWCShape(const BHWC& new_shape) {
    shape_ = BHWDC(new_shape.b, new_shape.h, new_shape.w, 1, new_shape.c);
  }
  void SetBHWDCShape(const BHWDC& new_shape) { shape_ = new_shape; }
  BHWC GetBHWCShape() const {
    return BHWC(shape_.b, shape_.h, shape_.w, shape_.c);
  }
  BHWDC GetBHWDCShape() const { return shape_; }
  void SetData(std::vector<uint8_t>&& new_data) { data_ = new_data; }
  const std::vector<uint8_t>& GetData() const { return data_; }

 private:
  friend flatbuffers::Offset<data::TensorDescriptor> Encode(
      const TensorDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
  friend void Decode(const data::TensorDescriptor* fb_desc,
                     TensorDescriptor* desc);

  absl::Status PerformReadSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;
  absl::Status PerformReadNearestSelector(const GpuInfo& gpu_info,
                                          const std::vector<std::string>& args,
                                          std::string* result) const;
  absl::Status PerformReadBilinearSelector(const GpuInfo& gpu_info,
                                           const std::vector<std::string>& args,
                                           std::string* result) const;
  absl::Status PerformReadPerChannelSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformGetAddressSelector(const std::vector<std::string>& args,
                                         std::string* result) const;

  absl::Status PerformGetHandleSelector(const std::vector<std::string>& args,
                                        std::string* result) const;

  std::string StorageTypeToAddressType() const;

  absl::Status PerformWriteSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformWriteLinearSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformWrite2DSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  std::string Read(const GpuInfo& gpu_info, DataType read_as_type,
                   const std::vector<std::string>& coords) const;
  std::string Write(const GpuInfo& gpu_info, DataType write_type,
                    const std::string& var_name,
                    const std::vector<std::string>& coords) const;

  bool IsBatchedWidth() const;

  absl::Status MaybeGetDataTypeFromTemplateArgs(
      const std::vector<std::string>& template_args, DataType* result) const;

  std::string GetGlobalAddressNoDeclaration(const std::string& xc,
                                            const std::string& yc,
                                            const std::string& zc,
                                            const std::string& sc,
                                            const std::string& bc) const;

  std::vector<std::string> GetPhysicalCoordsWHS(const std::string& x,
                                                const std::string& y,
                                                const std::string& s) const;
  std::vector<std::string> GetPhysicalCoordsWHSB(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& s,
                                                 const std::string& b) const;
  std::vector<std::string> GetPhysicalCoordsWHDS(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& z,
                                                 const std::string& s) const;
  std::vector<std::string> GetPhysicalCoordsWHDSB(const std::string& x,
                                                  const std::string& y,
                                                  const std::string& z,
                                                  const std::string& s,
                                                  const std::string& b) const;
  std::vector<std::string> GetPhysicalCoords(const std::string& xc,
                                             const std::string& yc,
                                             const std::string& zc,
                                             const std::string& sc,
                                             const std::string& bc) const;

  bool ParseCoordsFromArgs(const std::vector<std::string>& args, int offset,
                           std::string* xc, std::string* yc, std::string* zc,
                           std::string* sc, std::string* bc) const;

  template <typename T>
  void UploadData(const T* src);
  template <typename T>
  void DownloadData(T* dst);

  DataType data_type_ = DataType::UNKNOWN;
  TensorStorageType storage_type_ = TensorStorageType::UNKNOWN;

  // This field describes logical layout, actual(physical) GPU layout can be
  // totally different.
  Layout layout_ =
      Layout::UNKNOWN;  // Supported layouts is HWC, BHWC, HWDC, BHWDC

  // applicable only for TEXTURE_2D.
  // When Texture 2d created from buffer, we can use it as texture or as buffer.
  // This option allows to use texture 2d as buffer when we use it as dst
  // tensor(write only).
  // Currently supported only for Metal/OpenCL.
  // By default false.
  bool use_buffer_for_write_only_2d_texture_ = false;

  // applicable only for IMAGE_BUFFER.
  // We can use image buffer as image or as buffer.
  // This option allows to use image buffer as buffer when we use it as dst
  // tensor(write only).
  // Currently supported only for Metal/OpenCL.
  // By default true.
  bool use_buffer_for_write_only_image_buffer_ = true;

  // optional
  BHWDC shape_;
  std::vector<uint8_t> data_;
};

TensorDescriptor CreateBhwcTensorDescriptor(DataType data_type,
                                            TensorStorageType storage_type,
                                            const BHWC& shape);
TensorDescriptor CreateHwcTensorDescriptor(DataType data_type,
                                           TensorStorageType storage_type,
                                           const HWC& shape);

template <DataType T>
void TensorDescriptor::UploadData(const tflite::gpu::Tensor<BHWC, T>& src) {
  shape_ = BHWDC(src.shape.b, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

template <DataType T>
void TensorDescriptor::DownloadData(tflite::gpu::Tensor<BHWC, T>* dst) {
  dst->shape = BHWC(shape_.b, shape_.h, shape_.w, shape_.c);
  dst->data.resize(dst->shape.DimensionsProduct(), 0.0f);
  DownloadData(dst->data.data());
}

template <DataType T>
void TensorDescriptor::UploadData(const tflite::gpu::Tensor<BHWDC, T>& src) {
  shape_ = src.shape;
  UploadData(src.data.data());
}

template <DataType T>
void TensorDescriptor::DownloadData(tflite::gpu::Tensor<BHWDC, T>* dst) {
  dst->shape = shape_;
  dst->data.resize(dst->shape.DimensionsProduct(), 0.0f);
  DownloadData(dst->data.data());
}

template <typename T>
void TensorDescriptor::UploadData(const T* src) {
  data_.resize(GetSizeInBytesForShape(shape_));
  if (data_type_ == DataType::FLOAT16) {
    half* gpu_data = reinterpret_cast<half*>(data_.data());
    DataFromBHWDC(src, shape_, *this, gpu_data);
  } else {
    T* gpu_data = reinterpret_cast<T*>(data_.data());
    DataFromBHWDC(src, shape_, *this, gpu_data);
  }
}

template <typename T>
void TensorDescriptor::DownloadData(T* dst) {
  data_.resize(GetSizeInBytesForShape(shape_));
  if (data_type_ == DataType::FLOAT16) {
    half* gpu_data = reinterpret_cast<half*>(data_.data());
    DataToBHWDC(gpu_data, shape_, *this, dst);
  } else {
    T* gpu_data = reinterpret_cast<T*>(data_.data());
    DataToBHWDC(gpu_data, shape_, *this, dst);
  }
}

template <typename FromType, typename ToType>
void DataFromBHWDC(const FromType* src, const BHWDC& shape,
                   const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment =
      desc.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                    : 4;
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              FromType value;
              if (s * 4 + c < shape.c) {
                const int cpu_index =
                    shape.LinearIndex({b, y, x, d, s * 4 + c});
                value = src[cpu_index];
              } else {
                value = 0;
              }
              int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
              dst[gpu_index] = value;
            }
          }
        }
      }
    }
  }
}

template <typename FromType, typename ToType>
void DataToBHWDC(const FromType* src, const BHWDC& shape,
                 const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment =
      desc.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                    : 4;
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              if (s * 4 + c >= shape.c) {
                continue;
              }
              int cpu_index = shape.LinearIndex({b, y, x, d, s * 4 + c});
              int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
              dst[cpu_index] = src[gpu_index];
            }
          }
        }
      }
    }
  }
}

std::string ToString(TensorStorageType type);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_
