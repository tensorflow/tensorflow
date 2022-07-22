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
#include "tensorflow/core/lib/core/status.h"

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

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "third_party/tensorrt/NvInfer.h"

#define TFTRT_INTERNAL_ERROR_AT_NODE(node)                           \
  do {                                                               \
    return errors::Internal("TFTRT::", __FUNCTION__, ":", __LINE__,  \
                            " failed to add TRT layer, at: ", node); \
  } while (0)

#define TFTRT_RETURN_ERROR_IF_NULLPTR(ptr, node) \
  do {                                           \
    if (ptr == nullptr) {                        \
      TFTRT_INTERNAL_ERROR_AT_NODE(node);        \
    }                                            \
  } while (0)

// Use this macro within functions that return a Status or StatusOR<T> to check
// boolean conditions. If the condition fails, it returns an
// errors::Internal message with the file and line number.
#define TRT_ENSURE(x)                                                        \
  if (!(x)) {                                                                \
    return errors::Internal(__FILE__, ":", __LINE__, " TRT_ENSURE failure"); \
  }

// Checks that a Status or StatusOr<T> object does not carry an error message.
// If it does have an error, returns an errors::Internal instance
// containing the error message, along with the file and line number. For
// pointer-containing StatusOr<T*>, use the below TRT_ENSURE_PTR_OK macro.
#define TRT_ENSURE_OK(x)                                   \
  if (!x.ok()) {                                           \
    return errors::Internal(__FILE__, ":", __LINE__,       \
                            " TRT_ENSURE_OK failure:\n  ", \
                            x.status().ToString());        \
  }

// Checks that a StatusOr<T* >object does not carry an error, and that the
// contained T* is non-null. If it does have an error status, returns an
// errors::Internal instance containing the error message, along with the file
// and line number.
#define TRT_ENSURE_PTR_OK(x)                            \
  TRT_ENSURE_OK(x);                                     \
  if (*x == nullptr) {                                  \
    return errors::Internal(__FILE__, ":", __LINE__,    \
                            " pointer had null value"); \
  }

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

class IONamePrefixes {
 public:
  static constexpr const char* const kInputPHName = "TensorRTInputPH_";
  static constexpr const char* const kOutputPHName = "TensorRTOutputPH_";
};

// Gets the binding index of a tensor in an engine.
//
// The binding index is looked up using the tensor's name and the profile index.
// Profile index should be set to zero, if we do not have optimization profiles.
Status GetTrtBindingIndex(const char* tensor_name, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index);

// Gets the binding index of a tensor in an engine.
//
// Same as above, but uses the network input index to identify the tensor.
Status GetTrtBindingIndex(int network_input_idx, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index);
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
}

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

// Prints the TensorFormat enum name to the stream.
std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::TensorFormat& format);

// Prints the DataType enum name to the stream.
std::ostream& operator<<(std::ostream& os, const nvinfer1::DataType& data_type);

}  // namespace nvinfer1

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_COMMON_UTILS_H_
