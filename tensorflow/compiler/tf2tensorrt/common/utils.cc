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

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/errors.h"
#include "third_party/tensorrt/NvInferPlugin.h"
#endif

namespace tensorflow {
namespace tensorrt {

std::tuple<int, int, int> GetLinkedTensorRTVersion() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  return std::tuple<int, int, int>{NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                                   NV_TENSORRT_PATCH};
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

std::tuple<int, int, int> GetLoadedTensorRTVersion() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  int ver = getInferLibVersion();
  int major = ver / 1000;
  ver = ver - major * 1000;
  int minor = ver / 100;
  int patch = ver - minor * 100;
  return std::tuple<int, int, int>{major, minor, patch};
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {

Status GetTrtBindingIndex(const char* tensor_name, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
  // If the engine has been built for K profiles, the first getNbBindings() / K
  // bindings are used by profile number 0, the following getNbBindings() / K
  // bindings are used by profile number 1 etc.
  //
  // GetBindingIndex(tensor_name) returns the binding index for the progile 0.
  // We can also consider it as a "binding_index_within_profile".
  *binding_index = cuda_engine->getBindingIndex(tensor_name);
  if (*binding_index == -1) {
    const string msg = absl::StrCat("Input node ", tensor_name, " not found");
    return errors::NotFound(msg);
  }
  int n_profiles = cuda_engine->getNbOptimizationProfiles();
  // If we have more then one optimization profile, then we need to shift the
  // binding index according to the following formula:
  // binding_index_within_engine = binding_index_within_profile +
  //                               profile_index * bindings_per_profile
  const int bindings_per_profile = cuda_engine->getNbBindings() / n_profiles;
  *binding_index = *binding_index + profile_index * bindings_per_profile;
  return Status::OK();
}

Status GetTrtBindingIndex(int network_input_index, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
  const string input_name =
      absl::StrCat(IONamePrefixes::kInputPHName, network_input_index);
  return GetTrtBindingIndex(input_name.c_str(), profile_index, cuda_engine,
                            binding_index);
}

}  // namespace tensorrt
}  // namespace tensorflow
#endif
