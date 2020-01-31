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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace tensorrt {

Status TrtPrecisionModeToName(TrtPrecisionMode mode, string* name) {
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
using absl::StrAppend;
using absl::StrCat;

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
  string out = "[";
  for (auto dims: dimvec) {
    StrAppend(&out, DebugString(dims));
  }
  StrAppend(&out, "]");
  return out;
}

string DebugString(const std::vector<TensorShape>& shapes) {
  string out = "[";
  for (auto shape: shapes) {
    StrAppend(&out, shape.DebugString());
  }
  StrAppend(&out, "]");
  return out;
}

string DebugString(const std::vector<PartialTensorShape>& shapes) {
  string out = "[";
  for (auto shape: shapes) {
    StrAppend(&out, shape.DebugString());
  }
  StrAppend(&out, "]");
  return out;
}
#endif

string GetLinkedTensorRTVersion() {
  int major, minor, patch;
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  major = NV_TENSORRT_MAJOR;
  minor = NV_TENSORRT_MINOR;
  patch = NV_TENSORRT_PATCH;
#else
  major = 0;
  minor = 0;
  patch = 0;
#endif
  return absl::StrCat(major, ".", minor, ".", patch);
}

string GetLoadedTensorRTVersion() {
  int major, minor, patch;
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  int ver = getInferLibVersion();
  major = ver / 1000;
  ver = ver - major * 1000;
  minor = ver / 100;
  patch = ver - minor * 100;
#else
  major = 0;
  minor = 0;
  patch = 0;
#endif
  return absl::StrCat(major, ".", minor, ".", patch);
}

}  // namespace tensorrt
}  // namespace tensorflow
