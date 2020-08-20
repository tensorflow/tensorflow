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
#include "third_party/tensorrt/NvInfer.h"
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

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
