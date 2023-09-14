/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/utils.h"

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/c/c_api_decl.h"

namespace tensorflow {

XLA_LayoutPreference ConvertToCXlaLayoutPreference(XlaLayoutPreference input) {
  switch (input) {
    case XlaLayoutPreference::kNoPreference:
      return XLA_LayoutPreference::XLA_LayoutPreference_kNoPreference;
    case XlaLayoutPreference::kTpuPreferCompactChunkPaddedLayout:
      return XLA_LayoutPreference::
          XLA_LayoutPreference_kTpuPreferCompactChunkPaddedLayout;
    case XlaLayoutPreference::kTpuPreferLinearLayout:
      return XLA_LayoutPreference::XLA_LayoutPreference_kTpuPreferLinearLayout;
  }
  LOG(ERROR) << "Unexpected value for XlaLayoutPreference: "
             << static_cast<int>(input);
  return XLA_LayoutPreference::XLA_LayoutPreference_kNoPreference;
}

XlaLayoutPreference ConvertFromCXlaLayoutPreference(
    XLA_LayoutPreference input) {
  switch (input) {
    case XLA_LayoutPreference::XLA_LayoutPreference_kNoPreference:
      return XlaLayoutPreference::kNoPreference;
    case XLA_LayoutPreference::
        XLA_LayoutPreference_kTpuPreferCompactChunkPaddedLayout:
      return XlaLayoutPreference::kTpuPreferCompactChunkPaddedLayout;
    case XLA_LayoutPreference::XLA_LayoutPreference_kTpuPreferLinearLayout:
      return XlaLayoutPreference::kTpuPreferLinearLayout;
  }
  LOG(ERROR) << "Unexpected value for XLA_LayoutPreference: "
             << static_cast<int>(input);
  return XlaLayoutPreference::kNoPreference;
}

}  // namespace tensorflow
