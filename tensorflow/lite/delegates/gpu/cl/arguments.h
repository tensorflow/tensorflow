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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_ARGUMENTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_ARGUMENTS_H_

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

class Arguments {
 public:
  Arguments() = default;
  void AddFloat(const std::string& name, float value = 0.0f);
  void AddHalf(const std::string& name, half value = half(0.0f));
  void AddInt(const std::string& name, int value = 0);
  void AddBuffer(const std::string& name, const GPUBufferDescriptor& desc);
  void AddImage2D(const std::string& name, const GPUImage2DDescriptor& desc);
  void AddImage2DArray(const std::string& name,
                       const GPUImage2DArrayDescriptor& desc);
  void AddImage3D(const std::string& name, const GPUImage3DDescriptor& desc);
  void AddImageBuffer(const std::string& name,
                      const GPUImageBufferDescriptor& desc);

  void AddObjectRef(const std::string& name, AccessType access_type,
                    GPUObjectDescriptorPtr&& descriptor_ptr);
  void AddObject(const std::string& name, AccessType access_type,
                 GPUObjectPtr&& object);

  absl::Status SetInt(const std::string& name, int value);
  absl::Status SetFloat(const std::string& name, float value);
  absl::Status SetHalf(const std::string& name, half value);
  absl::Status SetImage2D(const std::string& name, cl_mem memory);
  absl::Status SetBuffer(const std::string& name, cl_mem memory);
  absl::Status SetImage2DArray(const std::string& name, cl_mem memory);
  absl::Status SetImage3D(const std::string& name, cl_mem memory);
  absl::Status SetImageBuffer(const std::string& name, cl_mem memory);
  absl::Status SetObjectRef(const std::string& name, const GPUObject* object);

  std::string GetListOfArgs();

  absl::Status Bind(cl_kernel kernel, int offset);

  absl::Status TransformToCLCode(std::string* code);

  // Move only
  Arguments(Arguments&& args);
  Arguments& operator=(Arguments&& args);
  Arguments(const Arguments&) = delete;
  Arguments& operator=(const Arguments&) = delete;

 private:
  std::string AddActiveArgument(const std::string& arg_name);
  void AddGPUResources(const std::string& name, const GPUResources& resources);

  absl::Status SetGPUResources(const std::string& name,
                               const GPUResourcesWithValue& resources);

  absl::Status AddObjectArgs();

  void ResolveArgsPass(std::string* code);
  absl::Status ResolveSelectorsPass(std::string* code);

  absl::Status ResolveSelector(const std::string& object_name,
                               const std::string& selector,
                               const std::vector<std::string>& args,
                               std::string* result);

  void ResolveObjectNames(const std::string& object_name,
                          const std::vector<std::string>& member_names,
                          std::string* code);

  static constexpr char kArgsPrefix[] = "args.";

  struct IntValue {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, IntValue> int_values_;
  std::vector<int32_t> shared_int4s_data_;

  struct FloatValue {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<float> shared_float4s_data_;

  struct HalfValue {
    half value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, HalfValue> half_values_;
  std::vector<half> shared_half4s_data_;

  std::map<std::string, GPUBufferDescriptor> buffers_;
  std::map<std::string, GPUImage2DDescriptor> images2d_;
  std::map<std::string, GPUImage2DArrayDescriptor> image2d_arrays_;
  std::map<std::string, GPUImage3DDescriptor> images3d_;
  std::map<std::string, GPUImageBufferDescriptor> image_buffers_;

  struct ObjectRefArg {
    AccessType access_type;
    GPUObjectDescriptorPtr descriptor;
  };
  std::map<std::string, ObjectRefArg> object_refs_;

  struct ObjectArg {
    AccessType access_type;
    GPUObjectPtr obj_ptr;
  };
  std::map<std::string, ObjectArg> objects_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_ARGUMENTS_H_
