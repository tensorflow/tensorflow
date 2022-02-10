/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_ARGUMENTS_H_

#import <Metal/Metal.h>

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/metal/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"

namespace tflite {
namespace gpu {
namespace metal {

class MetalArguments : public ArgumentsBinder {
 public:
  MetalArguments() = default;

  absl::Status Init(bool use_arguments_buffer, MetalDevice* device,
                    Arguments* args, std::string* code);

  absl::Status Init(bool use_arguments_buffer, MetalDevice* device,
                    Arguments* args);

  // Move only
  MetalArguments(MetalArguments&& args) = default;
  MetalArguments& operator=(MetalArguments&& args) = default;
  MetalArguments(const MetalArguments&) = delete;
  MetalArguments& operator=(const MetalArguments&) = delete;

  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetFloat(const std::string& name, float value) override;
  absl::Status SetHalf(const std::string& name, half value) override;
  absl::Status SetObjectRef(const std::string& name, const GPUObject& object);

  void Encode(id<MTLComputeCommandEncoder> encoder, int buffer_offset,
              int texture_offset = 0) const;

  // For usage with Argument Buffers
  API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
  void AddResourcesToEncoder(id<MTLComputeCommandEncoder> encoder) const;
  API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
  void EncodeArguments(id<MTLArgumentEncoder> arguments_encoder);

 private:
  // creates structure with layout:
  // struct uniforms_buffer {
  //   int val_0;
  //   int val_1;
  //   float val_2;
  //   int dummy;  // for alignment
  // };
  std::string CopyScalarArgumentsToStructWithScalarFields(
      const Arguments& args, const std::string& call_prefix = "",
      std::string* code = nullptr);

  // creates structure with layout:
  // struct uniforms_buffer {
  //   int4 val_0_val_1_dummy_dummy;
  //   float4 val_2_dummy_dummy_dummy;
  // };
  std::string CopyScalarArgumentsToStructWithVec4Fields(
      const Arguments& args, const std::string& call_prefix = "",
      std::string* code = nullptr);

  absl::Status AllocateObjects(const Arguments& args, id<MTLDevice> device);
  absl::Status AddObjectArgs(const GpuInfo& gpu_info, const Arguments& args);

  void AddGPUResources(const std::string& name, const GPUResources& resources);

  std::string GetListOfArgs(int buffer_offset, int textures_offset = 0);

  std::string GetArgumentBufferStructDefinition(bool add_constants_struct);

  absl::Status SetGPUResources(const std::string& name,
                               const GPUResourcesWithValue& resources);

  void AddBuffer(const std::string& name, const GPUBufferDescriptor& desc);
  void AddImage2D(const std::string& name, const GPUImage2DDescriptor& desc);
  void AddImage2DArray(const std::string& name,
                       const GPUImage2DArrayDescriptor& desc);
  void AddImage3D(const std::string& name, const GPUImage3DDescriptor& desc);
  void AddImageBuffer(const std::string& name,
                      const GPUImageBufferDescriptor& desc);

  absl::Status SetBuffer(const std::string& name, id<MTLBuffer> handle,
                         uint64_t offset);
  absl::Status SetImage2D(const std::string& name, id<MTLTexture> handle);
  absl::Status SetImage2DArray(const std::string& name, id<MTLTexture> handle);
  absl::Status SetImage3D(const std::string& name, id<MTLTexture> handle);
  absl::Status SetImageBuffer(const std::string& name, id<MTLTexture> handle);

  absl::Status SetObjectsResources(const Arguments& args);

  static constexpr char kArgsPrefix[] = "args.";
  struct IntValue {
    int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, IntValue> int_values_;

  struct FloatValue {
    float value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<uint8_t> const_data_;

  struct MetalBufferDescriptor {
    GPUBufferDescriptor desc;
    id<MTLBuffer> handle;
    uint64_t offset;
  };
  struct MetalImage2DDescriptor {
    GPUImage2DDescriptor desc;
    id<MTLTexture> handle;
  };
  struct MetalImage2DArrayDescriptor {
    GPUImage2DArrayDescriptor desc;
    id<MTLTexture> handle;
  };
  struct MetalImage3DDescriptor {
    GPUImage3DDescriptor desc;
    id<MTLTexture> handle;
  };
  struct MetalImageBufferDescriptor {
    GPUImageBufferDescriptor desc;
    id<MTLTexture> handle;
  };

  std::map<std::string, MetalBufferDescriptor> buffers_;
  std::map<std::string, MetalImage2DDescriptor> images2d_;
  std::map<std::string, MetalImage2DArrayDescriptor> image2d_arrays_;
  std::map<std::string, MetalImage3DDescriptor> images3d_;
  std::map<std::string, MetalImageBufferDescriptor> image_buffers_;

  std::map<std::string, GPUObjectDescriptorPtr> object_refs_;
  std::vector<GPUObjectPtr> objects_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_ARGUMENTS_H_
