/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_ACTIVATION_MODE_H_
#define TENSORFLOW_CORE_UTIL_ACTIVATION_MODE_H_

// This file contains helper routines to deal with activation mode in various
// ops and kernels.

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// ActivationMode: the activation function we apply to the input tensor:
enum ActivationMode {
  NONE = 0,
  SIGMOID = 1,
  RELU = 2,
  RELU6 = 3,
  RELUX = 4,
  TANH = 5,
  BANDPASS = 6,
};

// Specialization to parse an attribute directly into a ActivationMode enum.
Status GetActivationModeFromString(const string& str_value,
                                   ActivationMode* value);

inline absl::string_view ToString(ActivationMode mode) {
  switch (mode) {
    case NONE:
      return "NONE";
    case SIGMOID:
      return "SIGMOID";
    case RELU:
      return "RELU";
    case RELU6:
      return "RELU6";
    case RELUX:
      return "RELUX";
    case TANH:
      return "TANH";
    case BANDPASS:
      return "BANDPASS";
  }
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_ACTIVATION_MODE_H_
