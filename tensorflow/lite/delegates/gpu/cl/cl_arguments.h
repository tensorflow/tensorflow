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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_ARGUMENTS_H_

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"

namespace tflite {
namespace gpu {
namespace cl {

class CLArguments : public ArgumentsBinder {
 public:
  CLArguments() = default;

  absl::Status Init(const GpuInfo& gpu_info,
                    const std::map<std::string, std::string>& linkables,
                    CLContext* context, Arguments* args, std::string* code);
  absl::Status Init(const GpuInfo& gpu_info, Arguments* args,
                    CLContext* context);

  // Temporary, will be resolved later
  void MoveObjectRefsIn(Arguments* args) {
    object_refs_ = std::move(args->object_refs_);
  }
  void MoveObjectRefsOut(Arguments* args) {
    args->object_refs_ = std::move(object_refs_);
  }
  void CopyScalarValues(Arguments* args) const;

  // Move only
  CLArguments(CLArguments&& args) = default;
  CLArguments& operator=(CLArguments&& args) = default;
  CLArguments(const CLArguments&) = delete;
  CLArguments& operator=(const CLArguments&) = delete;

  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetFloat(const std::string& name, float value) override;
  absl::Status SetHalf(const std::string& name, half value) override;
  absl::Status SetObjectRef(const std::string& name, const GPUObject* object);

  absl::Status Bind(cl_kernel kernel, int offset = 0);

 private:
  absl::Status AllocateObjects(const Arguments& args, CLContext* context);
  absl::Status AddObjectArgs(Arguments* args);

  absl::Status ResolveSelectorsPass(
      const GpuInfo& gpu_info, const Arguments& args,
      const std::map<std::string, std::string>& linkables, std::string* code);
  absl::Status ResolveSelector(
      const GpuInfo& gpu_info, const Arguments& args,
      const std::map<std::string, std::string>& linkables,
      const std::string& object_name, const std::string& selector,
      const std::vector<std::string>& function_args,
      const std::vector<std::string>& template_args, std::string* result);
  void ResolveObjectNames(const std::string& object_name,
                          const std::vector<std::string>& member_names,
                          std::string* code);
  void ResolveArgsPass(std::string* code);

  void CopyArguments(const Arguments& args, bool use_f32_for_halfs);
  void RenameArgumentsInCode(std::string* code);
  std::string GetListOfArgs();

  void AddBuffer(const std::string& name, const GPUBufferDescriptor& desc);
  void AddImage2D(const std::string& name, const GPUImage2DDescriptor& desc);
  void AddImage2DArray(const std::string& name,
                       const GPUImage2DArrayDescriptor& desc);
  void AddImage3D(const std::string& name, const GPUImage3DDescriptor& desc);
  void AddImageBuffer(const std::string& name,
                      const GPUImageBufferDescriptor& desc);
  void AddCustomMemory(const std::string& name,
                       const GPUCustomMemoryDescriptor& desc);
  void AddGPUResources(const std::string& name, const GPUResources& resources,
                       Arguments* args);
  absl::Status SetObjectsResources(const Arguments& args);
  absl::Status SetGPUResources(const std::string& name,
                               const GPUResourcesWithValue& resources);

  absl::Status SetImage2D(const std::string& name, cl_mem memory);
  absl::Status SetBuffer(const std::string& name, cl_mem memory);
  absl::Status SetImage2DArray(const std::string& name, cl_mem memory);
  absl::Status SetImage3D(const std::string& name, cl_mem memory);
  absl::Status SetImageBuffer(const std::string& name, cl_mem memory);
  absl::Status SetCustomMemory(const std::string& name, cl_mem memory);

  static constexpr char kArgsPrefix[] = "args.";
  struct IntValue {
    int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t offset = -1;
  };
  std::map<std::string, IntValue> int_values_;
  std::vector<int32_t> shared_int4s_data_;

  struct FloatValue {
    float value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<float> shared_float4s_data_;

  struct HalfValue {
    half value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // some devices have issues with half parameters.
    bool store_as_f32 = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, HalfValue> half_values_;
  std::vector<half> shared_half4s_data_;

  struct CLBufferDescriptor {
    GPUBufferDescriptor desc;
    cl_mem memory;
  };
  struct CLImage2DDescriptor {
    GPUImage2DDescriptor desc;
    cl_mem memory;
  };
  struct CLImage2DArrayDescriptor {
    GPUImage2DArrayDescriptor desc;
    cl_mem memory;
  };
  struct CLImage3DDescriptor {
    GPUImage3DDescriptor desc;
    cl_mem memory;
  };
  struct CLImageBufferDescriptor {
    GPUImageBufferDescriptor desc;
    cl_mem memory;
  };
  struct CLCustomMemoryDescriptor {
    GPUCustomMemoryDescriptor desc;
    cl_mem memory;
  };

  std::map<std::string, CLBufferDescriptor> buffers_;
  std::map<std::string, CLImage2DDescriptor> images2d_;
  std::map<std::string, CLImage2DArrayDescriptor> image2d_arrays_;
  std::map<std::string, CLImage3DDescriptor> images3d_;
  std::map<std::string, CLImageBufferDescriptor> image_buffers_;
  std::map<std::string, CLCustomMemoryDescriptor> custom_memories_;

  std::map<std::string, GPUObjectDescriptorPtr> object_refs_;
  std::vector<GPUObjectPtr> objects_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_ARGUMENTS_H_
