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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_

#include <string>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

std::string GetCommonDefines(CalculationsPrecision precision);

std::string GetGlobalAddress(TensorStorageType storage_type,
                             const std::string& size_name,
                             const std::string& var_name, const std::string& x,
                             const std::string& y, const std::string& z);
std::string ReadGlobalFLT4(TensorStorageType storage_type,
                           const std::string& tensor_name,
                           const std::string& size_name, const std::string& x,
                           const std::string& y, const std::string& z);
std::string ReadGlobalFLT4(TensorStorageType storage_type,
                           const std::string& tensor_name,
                           const std::string& global_address);
std::string WriteGlobalFLT4(TensorStorageType storage_type,
                            const std::string& tensor_name,
                            const std::string& size_name,
                            const std::string& var_name, const std::string& x,
                            const std::string& y, const std::string& z);
std::string WriteGlobalFLT4(TensorStorageType storage_type,
                            const std::string& tensor_name,
                            const std::string& var_name,
                            const std::string& global_address);

std::string GetDataType(DataType type);
std::string GetDataType4(DataType type);

std::string GetTensorDeclaration(TensorStorageType storage_type,
                                 AccessType access, DataType data_type);

std::string GetTensorDeclaration(TensorStorageType storage_type,
                                 const std::string& tensor_name,
                                 AccessType access, DataType data_type);

std::string GenerateGlobal3DCoords(TensorStorageType storage_type);

enum class TextureAddressMode {
  DONT_CARE,  // translated to CLK_ADDRESS_NONE
  ZERO,       // translated to CLK_ADDRESS_CLAMP
};

class TensorCodeGenerator {
 public:
  TensorCodeGenerator(const std::string& name,
                      const std::string& uniform_size_name,
                      TensorStorageType storage_type, AccessType access);

  TensorCodeGenerator(const std::string& name,
                      const std::string& uniform_size_name,
                      const TensorDescriptor& descriptor);

  std::string GetDeclaration() const;

  std::string GetDeclaration(AccessType access) const;

  // This function (and functions below) accept TextureAddressMode, but this
  // argument applicable only for texture types. Buffer types ignore this
  // parameter.
  std::string Read3D(
      const std::string& x, const std::string& y, const std::string& z,
      TextureAddressMode address_mode = TextureAddressMode::ZERO) const;

  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsFloat3D(
      const std::string& x, const std::string& y, const std::string& z,
      TextureAddressMode address_mode = TextureAddressMode::ZERO) const;

  std::string Read3D(
      const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::ZERO) const;

  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsFloat3D(
      const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::ZERO) const;

  std::string GetAddress(const std::string& var_name, const std::string& x,
                         const std::string& y, const std::string& z) const;

  std::string Write3D(const std::string& var_name, const std::string& x,
                      const std::string& y, const std::string& z) const;

  std::string Write3D(const std::string& var_name,
                      const std::string& global_address) const;

 private:
  std::string name_;
  std::string uniform_size_name_;
  TensorStorageType storage_type_;
  AccessType access_;
  DataType data_type_ = DataType::UNKNOWN;
};

template <DataType S, typename T>
void RearrangeWeightsToOHWI4I4O(const ::tflite::gpu::Tensor<OHWI, S>& weights,
                                absl::Span<T> dst) {
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int s = 0; s < src_depth; ++s) {
          T filters[4];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index =
                    weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[j][i] = weights.data[f_index];
              } else {
                filters[j][i] = 0.0f;
              }
            }
          }
          dst[counter++] = filters[0];
          dst[counter++] = filters[1];
          dst[counter++] = filters[2];
          dst[counter++] = filters[3];
        }
      }
    }
  }
}

// returns float4 mask for last plane(batch of 4 channels)
// assumes that plane size is 4;
// for example we have 7 channels, in our data structures we align it to 8
// but 8s-channel will be empty, then last plane (batch of 4 channels) will
// have this mask (1, 1, 1, 0).
float4 GetMaskForLastPlane(int channels);
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_
