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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

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

Status TrtPrecisionModeToName(TrtPrecisionMode mode, string* name);

Status TrtPrecisionModeFromName(const string& name, TrtPrecisionMode* mode);

// Define a hash function for vector<TensorShape> because it is used as the key
// for the engine cache.
struct VectorTensorShapeHasher {
  std::size_t operator()(const std::vector<TensorShape>& key) const {
    return std::hash<std::string>()(TensorShapeUtils::ShapeListString(key));
  }
};

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#define IS_TRT_VERSION_GE(major, minor, patch, build)           \
  ((NV_TENSORRT_MAJOR > major) ||                               \
   (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR > minor) || \
   (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
    NV_TENSORRT_PATCH > patch) ||                               \
   (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && \
    NV_TENSORRT_PATCH == patch && NV_TENSORRT_BUILD >= build))

string DebugString(const nvinfer1::DimensionType type);
string DebugString(const nvinfer1::Dims& dims);
string DebugString(const nvinfer1::DataType trt_dtype);
string DebugString(const nvinfer1::Permutation& permutation, int len);
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

inline bool HasStaticShape(std::vector<int> dims) {
  return !absl::c_any_of(dims, [](int i) { return i < 0; });
}

template <typename TensorShapeType>
inline nvinfer1::Dims TensorShapeToTrtDims(const TensorShapeType& shape,
                                           bool ignore_first_dim) {
  nvinfer1::Dims trt_dims;
  const int offset = (ignore_first_dim ? 1 : 0);
  for (int i = offset; i < shape.dims(); i++) {
    trt_dims.d[i - offset] = shape.dim_size(i);
  }
  trt_dims.nbDims = shape.dims() - offset;
  return trt_dims;
}

Status TrtDimsToTensorShape(const std::vector<int>& trt_dims,
                            bool use_implicit_batch, int batch_size,
                            TensorShape& shape);

Status TrtDimsToTensorShape(const nvinfer1::Dims trt_dims,
                            bool use_implicit_batch, int batch_size,
                            TensorShape& shape);

Status TfTypeToTrtType(DataType tf_type, nvinfer1::DataType* trt_type);
Status TrtTypeToTfType(nvinfer1::DataType trt_type, DataType* tf_type);

// Returns a string that includes compile time TensorRT library version
// information {Maj, Min, Patch}.
string GetLinkedTensorRTVersion();

// Returns a string that includes runtime time TensorRT library version
// information {Maj, Min, Patch}.
string GetLoadedTensorRTVersion();

// Returns true if an engine built for cached_shapes can also run actual_shapes.
bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes);

// Returns the number of inputs for the engine, which also correspends to the
// number of input tensors for the network. This can differ from the number of
// input bindings, because the number of total input bindings equals the number
// of profiles times the number of engine inputs.
int GetNumberOfEngineInputs(const nvinfer1::ICudaEngine* engine);
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
