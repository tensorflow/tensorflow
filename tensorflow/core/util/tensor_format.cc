/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

string GetConvnetDataFormatAttrString() {
  return "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ";
}

string ToString(TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
      return "NHWC";
    case FORMAT_NCHW:
      return "NCHW";
    default:
      LOG(FATAL) << "Invalid Format: " << static_cast<int32>(format);
      return "INVALID_FORMAT";
  }
}

bool FormatFromString(const string& format_str, TensorFormat* format) {
  if (format_str == "NHWC") {
    *format = FORMAT_NHWC;
    return true;
  } else if (format_str == "NCHW") {
    *format = FORMAT_NCHW;
    return true;
  }
  return false;
}

}  // namespace tensorflow
