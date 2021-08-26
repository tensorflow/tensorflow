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
#include "tensorflow/c/experimental/pluggable_device_utils/pluggable_device_utils.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
tensorflow::Status ValidateDeviceType(tensorflow::StringPiece type) {
  // Validate device type. Device type must start with a capital letter and
  // consist of capital letters and underscores. Reasoning behind this decision:
  // * At the minimum we want to disallow '/' and ':' since
  //   these characters are used in device spec, for e.g.
  //   /job:foo/replica:12/device:GPU:1.
  // * Underscores seem useful, for e.g. XLA_GPU uses underscores.
  // * Allowing lowercase might get confusing. For example, say someone
  //   registers a new type called "Gpu". It might be confusing for users that
  //   "Gpu" is not the same device type as "GPU".
  //   Note that lowercase "cpu" and "gpu" are currently supported only for
  //   legacy reasons:
  //   https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/python/framework/device_spec.py;l=46;drc=d3a378f9665d8eee827c74cb9ecbee81e4c288dd
  static const LazyRE2 kTfDeviceTypeRegEx = {"[A-Z][A-Z_]*"};
  bool matches = RE2::FullMatch(type, *kTfDeviceTypeRegEx);
  if (!matches) {
    return tensorflow::Status(
        tensorflow::error::FAILED_PRECONDITION,
        tensorflow::strings::StrCat("Device name/type '", type, "' must match ",
                                    kTfDeviceTypeRegEx->pattern(), "."));
  }
  return tensorflow::Status::OK();
}
