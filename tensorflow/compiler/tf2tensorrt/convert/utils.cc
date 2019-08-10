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

}  // namespace tensorrt
}  // namespace tensorflow
