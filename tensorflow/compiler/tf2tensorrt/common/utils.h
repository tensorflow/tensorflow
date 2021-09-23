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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_COMMON_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_COMMON_UTILS_H_

#include <numeric>
#include <tuple>

#include "absl/strings/str_join.h"

namespace tensorflow {
namespace tensorrt {
// Returns the compile time TensorRT library version information
// {Maj, Min, Patch}.
std::tuple<int, int, int> GetLinkedTensorRTVersion();

// Returns the runtime time TensorRT library version information
// {Maj, Min, Patch}.
std::tuple<int, int, int> GetLoadedTensorRTVersion();
}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/core/platform/logging.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

#define IS_TRT_VERSION_GE(major, minor, patch, build)           \
  ((NV_TENSORRT_MAJOR > major) ||                               \
   (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR > minor) || \
   (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
    NV_TENSORRT_PATCH > patch) ||                               \
   (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
    NV_TENSORRT_PATCH == patch && NV_TENSORRT_BUILD >= build))

#define LOG_WARNING_WITH_PREFIX LOG(WARNING) << "TF-TRT Warning: "

// Initializes the TensorRT plugin registry if this hasn't been done yet.
void MaybeInitializeTrtPlugins(nvinfer1::ILogger* trt_logger);

}  // namespace tensorrt
}  // namespace tensorflow

namespace nvinfer1 {
// Prints nvinfer1::Dims or any drived type to the given ostream. Per GTest
// printing requirements, this must be in the nvinfer1 namespace.
inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& v) {
  os << "nvinfer1::Dims[";
  os << absl::StrJoin(std::vector<int>(v.d, v.d + v.nbDims), ",");
  os << "]";
  return os;
}  // namespace nvinfer1

// Returns true if any two derived nvinfer1::Dims type structs are equivalent.
inline bool operator==(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
  if (rhs.nbDims != lhs.nbDims) {
    return false;
  }
  for (int i = 0; i < lhs.nbDims; i++) {
    if (rhs.d[i] != lhs.d[i]) {
      return false;
    }
  }
  return true;
}

// Returns false if any 2 subclasses of nvinfer1::Dims are equivalent.
inline bool operator!=(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
  return !(rhs == lhs);
}

// Prints nvinfer1::INetworkDefinition* information to the given ostream.
inline std::ostream& operator<<(std::ostream& os,
                                nvinfer1::INetworkDefinition* n) {
  os << "nvinfer1::INetworkDefinition{\n";
  std::vector<int> layer_idxs(n->getNbLayers());
  std::iota(layer_idxs.begin(), layer_idxs.end(), 0);
  os << absl::StrJoin(layer_idxs, "\n ",
                      [n](std::string* out, const int layer_idx) {
                        out->append(n->getLayer(layer_idx)->getName());
                      });
  os << "}";
  return os;
}

}  // namespace nvinfer1

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_COMMON_UTILS_H_
