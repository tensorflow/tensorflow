/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

string GetConvnet3dDataFormatAttrString() {
  return "data_format: { 'NDHWC', 'NCDHW' } = 'NDHWC' ";
}

string GetConvnetDataFormat2D3DAttrString() {
  return "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ";
}

string GetConvnetFilterFormatAttrString() {
  return "filter_format: { 'HWIO', 'OIHW' } = 'HWIO' ";
}

string GetConvnet3dFilterFormatAttrString() {
  return "filter_format: { 'DHWIO', 'OIDHW' } = 'DHWIO' ";
}

string ToString(TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
      return "NHWC";
    case FORMAT_NCHW:
      return "NCHW";
    case FORMAT_NCHW_VECT_C:
      return "NCHW_VECT_C";
    case FORMAT_NHWC_VECT_W:
      return "NHWC_VECT_W";
    case FORMAT_HWNC:
      return "HWNC";
    case FORMAT_HWCN:
      return "HWCN";
    default:
      LOG(FATAL) << "Invalid Format: " << static_cast<int32>(format);
      return "INVALID_FORMAT";
  }
}

string ToString(FilterTensorFormat format) {
  switch (format) {
    case FORMAT_HWIO:
      return "HWIO";
    case FORMAT_OIHW:
      return "OIHW";
    case FORMAT_OHWI:
      return "OHWI";
    case FORMAT_OIHW_VECT_I:
      return "OIHW_VECT_I";
    default:
      LOG(FATAL) << "Invalid Filter Format: " << static_cast<int32>(format);
      return "INVALID_FORMAT";
  }
}

bool FormatFromString(absl::string_view format_str, TensorFormat* format) {
  if (format_str == "NHWC" || format_str == "NDHWC") {
    *format = FORMAT_NHWC;
    return true;
  }
  if (format_str == "NCHW" || format_str == "NCDHW") {
    *format = FORMAT_NCHW;
    return true;
  }
  if (format_str == "NCHW_VECT_C") {
    *format = FORMAT_NCHW_VECT_C;
    return true;
  }
  if (format_str == "NHWC_VECT_W") {
    *format = FORMAT_NHWC_VECT_W;
    return true;
  }
  if (format_str == "HWNC") {
    *format = FORMAT_HWNC;
    return true;
  }
  if (format_str == "HWCN") {
    *format = FORMAT_HWCN;
    return true;
  }
  return false;
}

bool FilterFormatFromString(absl::string_view format_str,
                            FilterTensorFormat* format) {
  if (format_str == "HWIO" || format_str == "DHWIO") {
    *format = FORMAT_HWIO;
    return true;
  }
  if (format_str == "OIHW" || format_str == "OIDHW") {
    *format = FORMAT_OIHW;
    return true;
  }
  if (format_str == "OIHW_VECT_I") {
    *format = FORMAT_OIHW_VECT_I;
    return true;
  }
  return false;
}

}  // namespace tensorflow
