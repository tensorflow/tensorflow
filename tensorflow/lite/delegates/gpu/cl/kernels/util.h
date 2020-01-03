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
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
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

enum class TextureAddressMode {
  DONT_CARE,  // translated to CLK_ADDRESS_NONE
  ZERO,       // translated to CLK_ADDRESS_CLAMP
};

class TensorCodeGenerator {
 public:
  struct SizeVariablesNames {
    SizeVariablesNames() = default;
    SizeVariablesNames(const std::string& width_name,
                       const std::string& height_name,
                       const std::string& depth_name);
    SizeVariablesNames(const std::string& width_name,
                       const std::string& height_name,
                       const std::string& depth_name,
                       const std::string& batch_name);

    std::string width = "unknown";
    std::string height = "unknown";
    std::string depth = "unknown";
    std::string batch = "unknown";
  };
  TensorCodeGenerator() = default;
  TensorCodeGenerator(const std::string& name, const SizeVariablesNames& sizes,
                      const TensorDescriptor& descriptor);

  std::string GetDeclaration(AccessType access) const;

  std::string GetAddress(const std::string& var_name, const std::string& x,
                         const std::string& y, const std::string& z) const;

  std::string GetAddress(const std::string& var_name, const std::string& x,
                         const std::string& y, const std::string& z,
                         const std::string& b) const;

  // This function (and functions below) accept TextureAddressMode, but this
  // argument applicable only for texture types. Buffer types ignore this
  // parameter.
  std::string Read3D(
      const std::string& x, const std::string& y, const std::string& z,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string Read4D(
      const std::string& x, const std::string& y, const std::string& z,
      const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsFloat3D(
      const std::string& x, const std::string& y, const std::string& z,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsFloat4D(
      const std::string& x, const std::string& y, const std::string& z,
      const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string Write3D(const std::string& var_name, const std::string& x,
                      const std::string& y, const std::string& z) const;

  // Write4D supports BUFFER and IMAGE_BUFFER storage types.
  std::string Write4D(const std::string& var_name, const std::string& x,
                      const std::string& y, const std::string& z,
                      const std::string& b) const;

  std::string Read(
      const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;
  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsFloat(
      const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;
  std::string Write(const std::string& var_name,
                    const std::string& global_address) const;

 private:
  std::string GetGlobalAddressNoDeclaration(const std::string& x,
                                            const std::string& y,
                                            const std::string& z) const;
  std::string GetGlobalAddressNoDeclaration(const std::string& x,
                                            const std::string& y,
                                            const std::string& z,
                                            const std::string& b) const;
  std::string DeclareAddress(const std::string& var_name,
                             const std::string& address) const;

  std::string tensor_name_;
  SizeVariablesNames sizes_;
  TensorDescriptor descriptor_;
};

std::string GetTensorDeclaration(AccessType access,
                                 const std::string& tensor_name,
                                 const TensorDescriptor& descriptor);

// Calculates correct X coordinate when stride != 1 and batch != 1 for
// DHWBC4, HDWBC4, HWBC layouts
std::string GetXStrideCorrected(const std::string& src_x,
                                const std::string& batch_size,
                                const std::string& stride_x,
                                const std::string& padding_x);

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupI4O4(
    const ::tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int dst_groups = IntegralDivideRoundUp(dst_depth, out_group_size);

  int counter = 0;
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int s = 0; s < src_depth; ++s) {
          for (int d_group = 0; d_group < out_group_size; ++d_group) {
            for (int j = 0; j < 4; ++j) {
              T filter;
              for (int i = 0; i < 4; ++i) {
                const int s_ch = s * 4 + j;
                const int d_ch = (d * out_group_size + d_group) * 4 + i;
                if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                  const int f_index =
                      weights.shape.LinearIndex({d_ch, y, x, s_ch});
                  filter[i] = weights.data[f_index];
                } else {
                  filter[i] = 0.0f;
                }
              }
              dst[counter++] = filter;
            }
          }
        }
      }
    }
  }
}

// Returns fastest TextureAddressMode that return ZERO for out-of-range image
// coordinates.
//
// Unfortunately, CLK_ADDRESS_CLAMP is very slow on Adreno3xx and
// we can observe huge register overhead when compared to other modes.

// While using CLK_ADDRESS_NONE with out-of-range image coordinates is undefined
// in the OpenCL specification, we have observed that CLK_ADDRESS_NONE works
// like CLK_ADDRESS_CLAMP for out-of-range image coordinates for RGBA F16/F32
// textures on Adreno3xx devices. Using CLK_ADDRESS_NONE is significantly faster
// than CLK_ADDRESS_CLAMP on Adreno 3xx.
TextureAddressMode GetFastestZeroMode(const CLDevice& device);

// Returns float4 mask for last plane(batch of 4 channels)
// assumes that plane size is 4;
// for example we have 7 channels, in our data structures we align it to 8
// but 8s-channel will be empty, then last plane (batch of 4 channels) will
// have this mask (1, 1, 1, 0).
float4 GetMaskForLastPlane(int channels);
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_
