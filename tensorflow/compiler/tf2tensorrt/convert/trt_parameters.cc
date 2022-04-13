/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <algorithm>
#include <cctype>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tensorrt {

Status TrtPrecisionModeToName(const TrtPrecisionMode mode, string* name) {
  const char* kUnknown = "UNKNOWN";
  *name = *kUnknown;
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
  }
  if (name->compare(kUnknown) == 0)
    return errors::OutOfRange("Unknown precision mode");
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

string DebugString(const TrtPrecisionMode mode) {
  string mode_str;
  TF_CHECK_OK(TrtPrecisionModeToName(mode, &mode_str));
  return absl::StrCat("TrtPrecisionMode::", mode_str);
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
  string name_lowercase(name);
  std::transform(name.begin(), name.end(), name_lowercase.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (name_lowercase == "range") {
    *strategy = ProfileStrategy::kRange;
  } else if (name_lowercase == "optimal") {
    *strategy = ProfileStrategy::kOptimal;
  } else if (name_lowercase == "range+optimal") {
    *strategy = ProfileStrategy::kRangeOptimal;
  } else if (name_lowercase == "implicitbatchmodecompatible") {
    *strategy = ProfileStrategy::kImplicitBatchModeCompatible;
  } else {
    return errors::InvalidArgument("Invalid profile strategy: ", name);
  }
  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
