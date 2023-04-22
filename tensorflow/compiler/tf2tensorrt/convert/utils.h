/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_

#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

static constexpr char kCastOutputTypeAttrName[] = "DstT";

class IONamePrefixes {
 public:
  static constexpr const char* const kInputPHName = "TensorRTInputPH_";
  static constexpr const char* const kOutputPHName = "TensorRTOutputPH_";
};

template <typename T>
struct TrtDestroyer {
  void operator()(T* t) {
    if (t) t->destroy();
  }
};

template <typename T>
using TrtUniquePtrType = std::unique_ptr<T, TrtDestroyer<T>>;

enum class TrtPrecisionMode { FP32, FP16, INT8 };

Status TrtPrecisionModeToName(const TrtPrecisionMode mode, string* name);

Status TrtPrecisionModeFromName(const string& name, TrtPrecisionMode* mode);

// Define a hash function for vector<TensorShape> because it is used as the key
// for the engine cache.
struct VectorTensorShapeHasher {
  std::size_t operator()(const std::vector<TensorShape>& key) const {
    return std::hash<std::string>()(TensorShapeUtils::ShapeListString(key));
  }
};

#if GOOGLE_CUDA && GOOGLE_TENSORRT

using absl::StrAppend;
using absl::StrCat;

// This utility template converts an arithmetic type to a string. This function
// is necessary to allow the following function to behave recursively:
// `string DebugString(const std::vector<CType>&)`.
template <typename CType, typename = typename std::enable_if<
                              std::is_arithmetic<CType>::value, CType>::type>
string DebugString(const CType& el) {
  string el_str = std::to_string(el);
  // Prettify std::to_string which can sometimes returns 1.50000 instead of 1.5.
  // In short it removes trailing 0s in a string-formatted number.
  el_str.erase(el_str.find_last_not_of('0') + 1, std::string::npos);
  return el_str;
}
// This utility template converts nested vectors to a string for debug purposes.
template <typename CType>
string DebugString(const std::vector<CType>& vector) {
  string tmp_s = "";
  for (const auto el : vector) {
    StrAppend(&tmp_s, StrCat(DebugString(el), ", "));
  }
  return StrCat("{", tmp_s.substr(0, tmp_s.length() - 2), "}");
}
string DebugString(const nvinfer1::Dims& dims);
string DebugString(const nvinfer1::DataType trt_dtype);
string DebugString(const TrtPrecisionMode mode);
string DebugString(const DataType tf_type);
string DebugString(const nvinfer1::Permutation& permutation, int len);
string DebugString(const ITensorProxyPtr& tensor);
string DebugString(const nvinfer1::ITensor& tensor);
string DebugString(const std::vector<nvinfer1::Dims>& dimvec);
string DebugString(const std::vector<TensorShape>& shapes);
string DebugString(const std::vector<PartialTensorShape>& shapes);

inline bool HasStaticShape(const nvinfer1::Dims& dims) {
  if (dims.nbDims < 0) return false;
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] < 0) return false;
  }
  return true;
}

template <typename T>
bool HasStaticShape(const T& dims) {
  return !absl::c_any_of(dims, [](int i) { return i < 0; });
}

// Returns whether a shape is compatible with a TRT shape tensor.
template <typename TensorShapeType>
inline bool IsTrtShapeTensorCompatible(const TensorShapeType& shape) {
  return (
      shape.dims() == 0 ||
      (shape.dims() == 1 && shape.num_elements() <= nvinfer1::Dims::MAX_DIMS));
}

// Returns whether a TF tensor could be interpreted as a TRT shape tensor.
inline bool IsTrtShapeTensorCompatible(const Tensor& tensor) {
  return tensor.dtype() == DT_INT32 &&
         IsTrtShapeTensorCompatible(tensor.shape());
}

template <typename Container>
Status ContainerToTrtDims(const Container& shape, nvinfer1::Dims* trt_dims,
                          bool ignore_first_dim = false) {
  if (shape.size() == 0) {
    // scalar
    if (ignore_first_dim) {
      return errors::Internal(
          "Scalars cannot be represented in implicit batch mode");
    }
    *trt_dims = {0, {1}};
  } else {
    const int offset = (ignore_first_dim ? 1 : 0);
    for (int i = offset; i < shape.size(); i++) {
      trt_dims->d[i - offset] = shape.at(i);
    }
    trt_dims->nbDims = shape.size() - offset;
  }
  return Status::OK();
}

template <typename TensorShapeType>
Status TensorShapeToTrtDims(const TensorShapeType& shape, bool ignore_first_dim,
                            nvinfer1::Dims* trt_dims) {
  if (shape.dims() == -1) {
    trt_dims->nbDims = -1;
    return Status::OK();
  }
  return ContainerToTrtDims(shape.dim_sizes(), trt_dims, ignore_first_dim);
}

Status GetNetworkInputShapes(const nvinfer1::INetworkDefinition* network,
                             std::vector<PartialTensorShape>* input_shapes);

Status TrtDimsToTensorShape(const std::vector<int>& trt_dims,
                            TensorShape* shape,
                            absl::optional<int> batch_size = absl::nullopt);

template <typename TensorShapeType>
Status TrtDimsToTensorShape(const nvinfer1::Dims trt_dims,
                            TensorShapeType* shape,
                            absl::optional<int> batch_size = absl::nullopt) {
  TF_RETURN_IF_ERROR(
      TensorShapeUtils::MakeShape(trt_dims.d, trt_dims.nbDims, shape));
  if (batch_size) {
    shape->InsertDim(0, batch_size.value());
  }
  return Status::OK();
}

Status TfTypeToTrtType(DataType tf_type, nvinfer1::DataType* trt_type);
Status TrtTypeToTfType(nvinfer1::DataType trt_type, DataType* tf_type);

// Returns true if an engine built for cached_shapes can also run actual_shapes.
bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes);

// Returns the number of inputs for the engine, which also correspends to the
// number of input tensors for the network. This can differ from the number of
// input bindings, because the number of total input bindings equals the number
// of profiles times the number of engine inputs.
int GetNumberOfEngineInputs(const nvinfer1::ICudaEngine* engine);

// Returns the string representation for the assigned device or the requested
// device of the given node.
absl::string_view GetDeviceName(const Node* node);

// Returns the ParsedName representation for the assigned device or the
// requested device string of the given node. If the device string is invalid,
// returns absl::nullopt.
absl::optional<DeviceNameUtils::ParsedName> GetDeviceParsedName(
    const Node* node);

// If the given two device assignments as compatible, returns the merge of the
// two assignments. Otherwise, returns absl::nullopt.
absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, const DeviceNameUtils::ParsedName& b);
// Similar to the above, except that the second device assignment is represented
// by a string_view.
absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, absl::string_view b);

// Optimization profile generation strategies.
enum class ProfileStrategy {
  kRange,
  kOptimal,
  kRangeOptimal,
  kImplicitBatchModeCompatible,
};

string ProfileStrategyToName(const ProfileStrategy strategy);
Status ProfileStrategyFromName(const string& name, ProfileStrategy* strategy);

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
