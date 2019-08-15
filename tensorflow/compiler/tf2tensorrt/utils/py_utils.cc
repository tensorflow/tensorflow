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

#include "tensorflow/compiler/tf2tensorrt/utils/py_utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "third_party/tensorrt/NvInfer.h"
#endif

namespace tensorflow {
namespace tensorrt {

bool IsGoogleTensorRTEnabled() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  auto handle_or = se::internal::DsoLoader::TryDlopenTensorRTLibraries();
  if (!handle_or.ok()) {
    LOG(WARNING) << "Cannot dlopen some TensorRT libraries. If you would like "
                    "to use Nvidia GPU with TensorRT, please make sure the "
                    "missing libraries mentioned above are installed properly.";
    return false;
  } else {
    return true;
  }
#else
  return false;
#endif
}

void GetLinkedTensorRTVersion(int* major, int* minor, int* patch) {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  *major = NV_TENSORRT_MAJOR;
  *minor = NV_TENSORRT_MINOR;
  *patch = NV_TENSORRT_PATCH;
#else
  *major = 0;
  *minor = 0;
  *patch = 0;
#endif
}

void GetLoadedTensorRTVersion(int* major, int* minor, int* patch) {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  int ver = getInferLibVersion();
  *major = ver / 1000;
  ver = ver - *major * 1000;
  *minor = ver / 100;
  *patch = ver - *minor * 100;
#else
  *major = 0;
  *minor = 0;
  *patch = 0;
#endif
}

}  // namespace tensorrt
}  // namespace tensorflow
