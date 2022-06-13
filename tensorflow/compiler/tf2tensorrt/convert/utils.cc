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

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "absl/strings/ascii.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tensorrt {

string DebugString(const nvinfer1::Dims& dims) {
  string out = StrCat("nvinfer1::Dims(nbDims=", dims.nbDims, ", d=");
  for (int i = 0; i < std::max(dims.nbDims, 0); ++i) {
    StrAppend(&out, dims.d[i]);
    StrAppend(&out, ",");
  }
  StrAppend(&out, ")");
  return out;
}

string DebugString(const DataType tf_type) {
  switch (tf_type) {
    case DT_FLOAT:
      return "DT_FLOAT";
    case DT_HALF:
      return "DT_HALF";
    case DT_INT32:
      return "DT_INT32";
    case DT_INT8:
      return "DT_INT8";
    case DT_BOOL:
      return "DT_BOOL";
    default:
      return "Unknow TF DataType";
  }
}

string DebugString(const nvinfer1::DataType trt_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "kFLOAT";
    case nvinfer1::DataType::kHALF:
      return "kHALF";
    case nvinfer1::DataType::kINT8:
      return "kINT8";
    case nvinfer1::DataType::kINT32:
      return "kINT32";
    case nvinfer1::DataType::kBOOL:
      return "kBOOL";
    default:
      return "Invalid TRT data type";
  }
}

string DebugString(const nvinfer1::Permutation& permutation, int len) {
  string out = "nvinfer1::Permutation(";
  for (int i = 0; i < len; ++i) {
    StrAppend(&out, permutation.order[i], ",");
  }
  StrAppend(&out, ")");
  return out;
}

string DebugString(const ITensorProxyPtr& tensor) {
  return StrCat(
      tensor->is_trt_tensor() ? "nvinfer1::ITensor(@" : "SimpleItensor(@",
      reinterpret_cast<uintptr_t>(&tensor), ", name=", tensor->getName(),
      ", dtype=", DebugString(tensor->getType()),
      ", dims=", DebugString(tensor->getDimensions()), ")");
}

string DebugString(const nvinfer1::ITensor& tensor) {
  return StrCat("nvinfer1::ITensor(@", reinterpret_cast<uintptr_t>(&tensor),
                ", name=", tensor.getName(),
                ", dtype=", DebugString(tensor.getType()),
                ", dims=", DebugString(tensor.getDimensions()), ")");
}

string DebugString(const std::vector<nvinfer1::Dims>& dimvec) {
  return absl::StrCat("[",
                      absl::StrJoin(dimvec, ",",
                                    [](std::string* out, nvinfer1::Dims in) {
                                      out->append(DebugString(in));
                                    }),
                      "]");
}

string DebugString(const std::vector<TensorShape>& shapes) {
  return TensorShapeUtils::ShapeListString(shapes);
}

string DebugString(const std::vector<PartialTensorShape>& shapes) {
  return PartialTensorShapeUtils::PartialShapeListString(shapes);
}

// Checks whether actual_shapes are compatible with cached_shapes. This should
// only be used in implicit batch mode (in explicit batch mode one needs to
// check the profile ranges). Therefore implicit batch mode is assumed.
// It is also assumed that both actual_shapes and cached_shapes have been
// verified by TRTEngineOp::VerifyInputShapes, which ensures that the batch size
// for all tensors are the same.
bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes) {
  auto match_shape = [](const TensorShape& actual_shape,
                        const TensorShape& cached_shape) {
    // Match the rank.
    if (actual_shape.dims() != cached_shape.dims()) return false;
    // Match the batch size. In implicit batch mode cached_shape.dim_size(0) is
    // the max batch size, which can be larger than the actual batch size.
    if (actual_shape.dim_size(0) > cached_shape.dim_size(0)) return false;
    // Match remaining dimensions.
    for (int i = 1; i < actual_shape.dims(); ++i) {
      if (actual_shape.dim_size(i) != cached_shape.dim_size(i)) return false;
    }
    return true;
  };
  for (int i = 0; i < actual_shapes.size(); ++i) {
    if (!match_shape(actual_shapes[i], cached_shapes[i])) {
      return false;
    }
  }
  return true;
}
Status GetNetworkInputShapes(const nvinfer1::INetworkDefinition* network,
                             std::vector<PartialTensorShape>* input_shapes) {
  const int n_inputs = network->getNbInputs();
  input_shapes->resize(n_inputs);
  for (int i = 0; i < n_inputs; i++) {
    const ITensorProxyPtr input = network->getInput(i);
    TF_RETURN_IF_ERROR(DimsAdapter(input->getDimensions())
                           .PartialTensorShape(&input_shapes->at(i)));
  }
  return Status::OK();
}

Status TfTypeToTrtType(DataType tf_type, nvinfer1::DataType* trt_type) {
  switch (tf_type) {
    case DT_FLOAT:
      *trt_type = nvinfer1::DataType::kFLOAT;
      break;
    case DT_HALF:
      *trt_type = nvinfer1::DataType::kHALF;
      break;
    case DT_INT32:
      *trt_type = nvinfer1::DataType::kINT32;
      break;
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    case DT_BOOL:
      *trt_type = nvinfer1::DataType::kBOOL;
      break;
#endif
    default:
      return errors::InvalidArgument("Unsupported tensorflow data type ",
                                     DataTypeString(tf_type));
  }
  return Status::OK();
}

Status TrtTypeToTfType(nvinfer1::DataType trt_type, DataType* tf_type) {
  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      *tf_type = DT_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *tf_type = DT_HALF;
      break;
    case nvinfer1::DataType::kINT32:
      *tf_type = DT_INT32;
      break;
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    case nvinfer1::DataType::kBOOL:
      *tf_type = DT_BOOL;
      break;
#endif
    default:
      return errors::InvalidArgument("Invalid TRT data type");
  }
  return Status::OK();
}

int GetNumberOfEngineInputs(const nvinfer1::ICudaEngine* engine) {
  int n_bindings = engine->getNbBindings();
  int n_input = 0;
  for (int i = 0; i < n_bindings; i++) {
    if (engine->bindingIsInput(i)) n_input++;
  }
  // According to TensorRT 7 doc: "If the engine has been built for K profiles,
  // the first getNbBindings() / K bindings are used by profile number 0, the
  // following getNbBindings() / K bindings are used by profile number 1 etc."
  // Therefore, to get the number of input tensors, we need to divide by the
  // the number of profiles.
  int n_profiles = engine->getNbOptimizationProfiles();
  return n_input / n_profiles;
}

absl::string_view GetDeviceName(const Node* node) {
  if (node->has_assigned_device_name()) {
    return node->assigned_device_name();
  }
  return node->requested_device();
}

std::optional<DeviceNameUtils::ParsedName> GetDeviceParsedName(
    const Node* node) {
  absl::string_view device_name = GetDeviceName(node);
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
    return std::nullopt;
  }
  return parsed_name;
}

std::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a,
    const DeviceNameUtils::ParsedName& b) {
  DeviceNameUtils::ParsedName merged_name = a;
  if (!DeviceNameUtils::MergeDevNames(&merged_name, b,
                                      /*allow_soft_placement=*/false)
           .ok()) {
    return std::nullopt;
  }
  return merged_name;
}

std::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, absl::string_view b) {
  DeviceNameUtils::ParsedName b_parsed_name;
  if (!DeviceNameUtils::ParseFullName(b, &b_parsed_name)) {
    return std::nullopt;
  }

  return MergeIfCompatible(a, b_parsed_name);
}

bool isExperimentalFeatureActivated(string feature_name) {
  string envvar_str;
  TF_CHECK_OK(
      ReadStringFromEnvVar("TF_TRT_EXPERIMENTAL_FEATURES", "", &envvar_str));
  return envvar_str.find(feature_name) != string::npos;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
