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

#include "absl/strings/ascii.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tensorrt {

Status TrtPrecisionModeToName(const TrtPrecisionMode mode, string* name) {
  switch (mode) {
    case TrtPrecisionMode::FP32:
      *name = "FP32";
      break;
    case TrtPrecisionMode::FP16:
      *name = "FP16";
      break;
    case TrtPrecisionMode::INT8:
      *name = "INT8";
      break;
    default:
      *name = "UNKNOWN";
      return errors::OutOfRange("Unknown precision mode");
  }
  return Status::OK();
}

Status TrtPrecisionModeFromName(const string& name, TrtPrecisionMode* mode) {
  if (name == "FP32") {
    *mode = TrtPrecisionMode::FP32;
  } else if (name == "FP16") {
    *mode = TrtPrecisionMode::FP16;
  } else if (name == "INT8") {
    *mode = TrtPrecisionMode::INT8;
  } else {
    return errors::InvalidArgument("Invalid precision mode name: ", name);
  }
  return Status::OK();
}

#if GOOGLE_CUDA && GOOGLE_TENSORRT

string DebugString(const nvinfer1::DimensionType type) {
  switch (type) {
    case nvinfer1::DimensionType::kSPATIAL:
      return "kSPATIAL";
    case nvinfer1::DimensionType::kCHANNEL:
      return "kCHANNEL";
    case nvinfer1::DimensionType::kINDEX:
      return "kINDEX";
    case nvinfer1::DimensionType::kSEQUENCE:
      return "kSEQUENCE";
    default:
      return StrCat(static_cast<int>(type), "=unknown");
  }
}

string DebugString(const nvinfer1::Dims& dims) {
  string out = StrCat("nvinfer1::Dims(nbDims=", dims.nbDims, ", d=");
  for (int i = 0; i < dims.nbDims; ++i) {
    StrAppend(&out, dims.d[i]);
    if (VLOG_IS_ON(2)) {
      StrAppend(&out, "[", DebugString(dims.type[i]), "],");
    } else {
      StrAppend(&out, ",");
    }
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
    default:
      return "Invalid TRT data type";
  }
}

string DebugString(const TrtPrecisionMode mode) {
  string mode_str;
  TF_CHECK_OK(TrtPrecisionModeToName(mode, &mode_str));
  return StrCat("TrtPrecisionMode::", mode_str);
}

string DebugString(const nvinfer1::Permutation& permutation, int len) {
  string out = "nvinfer1::Permutation(";
  for (int i = 0; i < len; ++i) {
    StrAppend(&out, permutation.order[i], ",");
  }
  StrAppend(&out, ")");
  return out;
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
    const nvinfer1::ITensor* input = network->getInput(i);
    const nvinfer1::Dims input_dim = input->getDimensions();
    TF_RETURN_IF_ERROR(TrtDimsToTensorShape(input_dim, &input_shapes->at(i)));
  }
  return Status::OK();
}
Status TrtDimsToTensorShape(const std::vector<int>& trt_dims,
                            TensorShape* shape,
                            absl::optional<int> batch_size) {
  TF_RETURN_IF_ERROR(
      TensorShapeUtils::MakeShape(trt_dims.data(), trt_dims.size(), shape));
  if (batch_size) {
    shape->InsertDim(0, batch_size.value());
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
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  int n_profiles = engine->getNbOptimizationProfiles();
#else
  int n_profiles = 1;
#endif
  return n_input / n_profiles;
}

string ProfileStrategyToName(const ProfileStrategy strategy) {
  switch (strategy) {
    case ProfileStrategy::kRange:
      return "Range";
    case ProfileStrategy::kOptimal:
      return "Optimal";
    case ProfileStrategy::kRangeOptimal:
      return "Range+Optimal";
    case ProfileStrategy::kImplicitBatchModeCompatible:
      return "ImplicitBatchModeCompatible";
  }
  return "Unknown";
}

Status ProfileStrategyFromName(const string& name, ProfileStrategy* strategy) {
  if (name == "range") {
    *strategy = ProfileStrategy::kRange;
  } else if (name == "optimal") {
    *strategy = ProfileStrategy::kOptimal;
  } else if (name == "range+optimal") {
    *strategy = ProfileStrategy::kRangeOptimal;
  } else if (name == "implicitbatchmodecompatible") {
    *strategy = ProfileStrategy::kImplicitBatchModeCompatible;
  } else {
    return errors::InvalidArgument("Invalid profile strategy: ", name);
  }
  return Status::OK();
}

#endif

absl::string_view GetDeviceName(const Node* node) {
  if (node->has_assigned_device_name()) {
    return node->assigned_device_name();
  }
  return node->requested_device();
}

absl::optional<DeviceNameUtils::ParsedName> GetDeviceParsedName(
    const Node* node) {
  absl::string_view device_name = GetDeviceName(node);
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
    return absl::nullopt;
  }
  return parsed_name;
}

absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a,
    const DeviceNameUtils::ParsedName& b) {
  DeviceNameUtils::ParsedName merged_name = a;
  if (!DeviceNameUtils::MergeDevNames(&merged_name, b,
                                      /*allow_soft_placement=*/false)
           .ok()) {
    return absl::nullopt;
  }
  return merged_name;
}

absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, absl::string_view b) {
  DeviceNameUtils::ParsedName b_parsed_name;
  if (!DeviceNameUtils::ParseFullName(b, &b_parsed_name)) {
    return absl::nullopt;
  }

  return MergeIfCompatible(a, b_parsed_name);
}

}  // namespace tensorrt
}  // namespace tensorflow
