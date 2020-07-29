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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

struct GPUImage2DDescriptor {
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUImage3DDescriptor {
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUImage2DArrayDescriptor {
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUImageBufferDescriptor {
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUBufferDescriptor {
  DataType data_type;
  AccessType access_type;
  int element_size;
  cl_mem memory;
};

struct GPUResources {
  std::vector<std::string> ints;
  std::vector<std::string> floats;
  std::vector<std::pair<std::string, GPUBufferDescriptor>> buffers;
  std::vector<std::pair<std::string, GPUImage2DDescriptor>> images2d;
  std::vector<std::pair<std::string, GPUImage2DArrayDescriptor>> image2d_arrays;
  std::vector<std::pair<std::string, GPUImage3DDescriptor>> images3d;
  std::vector<std::pair<std::string, GPUImageBufferDescriptor>> image_buffers;

  std::vector<std::string> GetNames() const {
    std::vector<std::string> names = ints;
    names.insert(names.end(), floats.begin(), floats.end());
    for (const auto& obj : buffers) {
      names.push_back(obj.first);
    }
    for (const auto& obj : images2d) {
      names.push_back(obj.first);
    }
    for (const auto& obj : image2d_arrays) {
      names.push_back(obj.first);
    }
    for (const auto& obj : images3d) {
      names.push_back(obj.first);
    }
    for (const auto& obj : image_buffers) {
      names.push_back(obj.first);
    }
    return names;
  }
};

struct GPUResourcesWithValue {
  std::vector<std::pair<std::string, int>> ints;
  std::vector<std::pair<std::string, float>> floats;
  std::vector<std::pair<std::string, cl_mem>> buffers;
  std::vector<std::pair<std::string, cl_mem>> images2d;
  std::vector<std::pair<std::string, cl_mem>> image2d_arrays;
  std::vector<std::pair<std::string, cl_mem>> images3d;
  std::vector<std::pair<std::string, cl_mem>> image_buffers;
};

class GPUObjectDescriptor {
 public:
  GPUObjectDescriptor() = default;
  virtual ~GPUObjectDescriptor() = default;

  void SetStateVar(const std::string& key, const std::string& value) const {
    state_vars_[key] = value;
  }

  virtual std::string PerformConstExpr(const std::string& const_expr) const {
    return "";
  }

  virtual absl::Status PerformSelector(
      const std::string& selector, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args,
      std::string* result) const {
    *result = "";
    return absl::OkStatus();
  }
  virtual GPUResources GetGPUResources(AccessType access_type) const {
    return GPUResources();
  }

 protected:
  mutable std::map<std::string, std::string> state_vars_;
};

using GPUObjectDescriptorPtr = std::unique_ptr<GPUObjectDescriptor>;

class GPUObject {
 public:
  GPUObject() = default;
  // Move only
  GPUObject(GPUObject&& obj_desc) = default;
  GPUObject& operator=(GPUObject&& obj_desc) = default;
  GPUObject(const GPUObject&) = delete;
  GPUObject& operator=(const GPUObject&) = delete;
  virtual ~GPUObject() = default;
  virtual GPUResourcesWithValue GetGPUResources(
      AccessType access_type) const = 0;
};

using GPUObjectPtr = std::unique_ptr<GPUObject>;

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_
