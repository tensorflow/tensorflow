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

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

enum class AddressMode {
  kDontCare,
  kZero,
};

enum class TensorStorageType {
  UNKNOWN,
  BUFFER,
  IMAGE_BUFFER,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_ARRAY,
  SINGLE_TEXTURE_2D
};

struct TensorDescriptor : public GPUObjectDescriptor {
  TensorDescriptor() = default;
  TensorDescriptor(DataType dt, TensorStorageType st, Layout l)
      : data_type(dt), storage_type(st), layout(l) {}

  TensorDescriptor(const TensorDescriptor&) = default;
  TensorDescriptor& operator=(const TensorDescriptor&) = default;
  TensorDescriptor(TensorDescriptor&& desc);
  TensorDescriptor& operator=(TensorDescriptor&& desc);

  bool operator==(const TensorDescriptor& d) const {
    return data_type == d.data_type && storage_type == d.storage_type &&
           layout == d.layout;
  }

  bool operator!=(const TensorDescriptor& d) const { return !(*this == d); }

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources() const override;

  void Release() override { data.clear(); }

  bool HasAxis(Axis axis) const;
  void SetAddressMode(AddressMode mode);
  int GetWidthSize(BHWDC shape) const;
  int GetSliceStrideSize(BHWDC shape) const;

  absl::Status GetLinkingContextFromWriteSelector(
      const std::vector<std::string>& args, std::string* value_name,
      std::string* x_coord, std::string* y_coord, std::string* s_coord) const;

  void UploadData(const tflite::gpu::Tensor<BHWC, DataType::FLOAT32>& src);
  void UploadData(const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  void UploadData(const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);

  bool SupportsZeroClamp(const Axis& axis) const;
  bool CanReadOutOfBorder(const Axis& axis) const;
  bool IsLinear() const;

  // applicable only for types that: IsLinear -> true.
  // In this case for address we have 1d component - addr (int)
  // If for addr == -1 this linear storage type returns FLT4(0.0), this function
  // returns true, otherwise false
  bool ReturnsZeroForNegOneRead() const;

  DataType data_type = DataType::UNKNOWN;
  TensorStorageType storage_type = TensorStorageType::UNKNOWN;
  // This field describes logical layout, actual(physical) GPU layout can be
  // totally different.
  Layout layout =
      Layout::UNKNOWN;  // Supported layouts is HWC, BHWC, HWDC, BHWDC

  // optional
  BHWDC shape;
  std::vector<uint8_t> data;

 private:
  absl::Status PerformReadSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformGetAddressSelector(const std::vector<std::string>& args,
                                         std::string* result) const;

  absl::Status PerformGetPtrWithSliceOffsetSelector(
      const std::vector<std::string>& args, std::string* result) const;

  absl::Status PerformGetWHOffsetSelector(const std::vector<std::string>& args,
                                          std::string* result) const;

  absl::Status PerformGetHandleSelector(const std::vector<std::string>& args,
                                        std::string* result) const;

  std::string DeclareAddress(const std::string& var_name,
                             const std::string& address) const;

  std::string StorageTypeToAddressType() const;

  absl::Status PerformWriteSelector(const GpuInfo& gpu_info,
                                    const std::vector<std::string>& args,
                                    std::string* result) const;

  absl::Status PerformWriteLinearSelector(const GpuInfo& gpu_info,
                                          const std::vector<std::string>& args,
                                          std::string* result) const;

  absl::Status PerformWrite2DSelector(const GpuInfo& gpu_info,
                                      const std::vector<std::string>& args,
                                      std::string* result) const;

  std::string Read(const GpuInfo& gpu_info, DataType read_as_type,
                   const std::vector<std::string>& coords) const;
  std::string Write(const GpuInfo& gpu_info, const std::string& var_name,
                    const std::vector<std::string>& coords) const;

  bool IsBatchedWidth() const;

  std::string GetWidth() const;

  AddressMode AddressModeFromState() const;

  absl::Status GetDataTypeFromTemplateArgs(const std::string& template_arg,
                                           DataType* result) const;

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

  void UploadData(const float* src);
};

template <typename T>
void DataFromBHWDC(const float* src, const BHWDC& shape,
                   const TensorDescriptor& desc, T* dst);

template <typename T>
void DataToBHWDC(const T* src, const BHWDC& shape, const TensorDescriptor& desc,
                 float* dst);

std::string ToString(TensorStorageType type);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_
