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

struct WHSPoint {
  std::string w_name;
  std::string h_name;
  std::string s_name;
};
struct WHSBPoint {
  std::string w_name;
  std::string h_name;
  std::string s_name;
  std::string b_name;
};
struct WHDSPoint {
  std::string w_name;
  std::string h_name;
  std::string d_name;
  std::string s_name;
};
struct WHDSBPoint {
  std::string w_name;
  std::string h_name;
  std::string d_name;
  std::string s_name;
  std::string b_name;
};

class TensorCodeGenerator {
 public:
  TensorCodeGenerator() = default;
  TensorCodeGenerator(const std::string& name, const WHSPoint& sizes,
                      const TensorDescriptor& descriptor);
  TensorCodeGenerator(const std::string& name, const WHSBPoint& sizes,
                      const TensorDescriptor& descriptor);
  TensorCodeGenerator(const std::string& name, const WHDSPoint& sizes,
                      const TensorDescriptor& descriptor);
  TensorCodeGenerator(const std::string& name, const WHDSBPoint& sizes,
                      const TensorDescriptor& descriptor);

  std::string GetDeclaration(AccessType access) const;

  std::string GetAddressWHS(const std::string& var_name, const std::string& x,
                            const std::string& y, const std::string& s) const;

  std::string GetAddressWHSB(const std::string& var_name, const std::string& x,
                             const std::string& y, const std::string& s,
                             const std::string& b) const;

  std::string GetAddressWHDS(const std::string& var_name, const std::string& x,
                             const std::string& y, const std::string& z,
                             const std::string& s) const;

  std::string GetAddressWHDSB(const std::string& var_name, const std::string& x,
                              const std::string& y, const std::string& z,
                              const std::string& s, const std::string& b) const;

  // This function (and functions below) accept TextureAddressMode, but this
  // argument applicable only for texture types. Buffer types ignore this
  // parameter.
  std::string ReadWHS(
      const std::string& x, const std::string& y, const std::string& s,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadWHSB(
      const std::string& x, const std::string& y, const std::string& s,
      const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadWHDS(
      const std::string& x, const std::string& y, const std::string& z,
      const std::string& s,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadWHDSB(
      const std::string& x, const std::string& y, const std::string& z,
      const std::string& s, const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsFloatWHS(
      const std::string& x, const std::string& y, const std::string& s,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsFloatWHSB(
      const std::string& x, const std::string& y, const std::string& s,
      const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsFloatWHDS(
      const std::string& x, const std::string& y, const std::string& z,
      const std::string& s,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsFloatWHDSB(
      const std::string& x, const std::string& y, const std::string& z,
      const std::string& s, const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsTypeWHS(
      DataType type, const std::string& x, const std::string& y,
      const std::string& s,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsTypeWHSB(
      DataType type, const std::string& x, const std::string& y,
      const std::string& s, const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsTypeWHDS(
      DataType type, const std::string& x, const std::string& y,
      const std::string& z, const std::string& s,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string ReadAsTypeWHDSB(
      DataType type, const std::string& x, const std::string& y,
      const std::string& z, const std::string& s, const std::string& b,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;

  std::string WriteWHS(const std::string& var_name, const std::string& x,
                       const std::string& y, const std::string& s) const;

  std::string WriteWHSB(const std::string& var_name, const std::string& x,
                        const std::string& y, const std::string& s,
                        const std::string& b) const;

  std::string WriteWHDS(const std::string& var_name, const std::string& x,
                        const std::string& y, const std::string& z,
                        const std::string& s) const;

  std::string WriteWHDSB(const std::string& var_name, const std::string& x,
                         const std::string& y, const std::string& z,
                         const std::string& s, const std::string& b) const;

  std::string Read(
      const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;
  // Optimization for textures, so as in opencl we can use read_imagef for any
  // texture type.
  std::string ReadAsFloat(
      const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;
  std::string ReadAsType(
      DataType type, const std::string& global_address,
      TextureAddressMode address_mode = TextureAddressMode::DONT_CARE) const;
  std::string Write(const std::string& var_name,
                    const std::string& global_address) const;

 private:
  std::string GetGlobalAddressNoDeclarationWHS(const std::string& x,
                                               const std::string& y,
                                               const std::string& s) const;
  std::string GetGlobalAddressNoDeclarationWHSB(const std::string& x,
                                                const std::string& y,
                                                const std::string& s,
                                                const std::string& b) const;
  std::string GetGlobalAddressNoDeclarationWHDS(const std::string& x,
                                                const std::string& y,
                                                const std::string& z,
                                                const std::string& s) const;
  std::string GetGlobalAddressNoDeclarationWHDSB(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& z,
                                                 const std::string& s,
                                                 const std::string& b) const;
  std::string DeclareAddress(const std::string& var_name,
                             const std::string& address) const;

  std::string tensor_name_;
  std::string width_name_ = "unknown";
  std::string height_name_ = "unknown";
  std::string depth_name_ = "unknown";
  std::string slices_name_ = "unknown";
  std::string batch_name_ = "unknown";
  TensorDescriptor descriptor_;
};

std::string GetTensorDeclaration(AccessType access,
                                 const std::string& tensor_name,
                                 const TensorDescriptor& descriptor);

// Calculates correct X coordinate when stride != 1 and batch != 1 for layouts
// with B after W (for example HWBC4) and WB stored in one axis of GPU
// resources.
std::string GetXStrideCorrected(const std::string& src_x,
                                const std::string& batch_size,
                                const std::string& stride_x,
                                const std::string& padding_x);

template <DataType S, typename T>
void RearrangeWeightsToOHWIOGroupI4O4(
    const tflite::gpu::Tensor<OHWI, S>& weights, int out_group_size,
    absl::Span<T> dst) {
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int dst_groups = DivideRoundUp(dst_slices, out_group_size);

  int counter = 0;
  for (int d = 0; d < dst_groups; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int s = 0; s < src_slices; ++s) {
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

// returns first work group from wgs that has size not bigger than max_wg_size
// if no suitable groups among wgs, returns {1, 1, 1}
int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size);

// task_size as amount of FLT4 processed elements.
int GetRecommendedBlockSizeForConv(const CLDevice& device,
                                   CalculationsPrecision precision,
                                   int task_size);
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UTIL_H_
