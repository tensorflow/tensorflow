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

#include "tensorflow/core/util/activation_mode.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status GetActivationModeFromString(const string& str_value,
                                   ActivationMode* value) {
  if (str_value == "None") {
    *value = NONE;
  } else if (str_value == "Sigmoid") {
    *value = SIGMOID;
  } else if (str_value == "Relu") {
    *value = RELU;
  } else if (str_value == "Relu6") {
    *value = RELU6;
  } else if (str_value == "ReluX") {
    *value = RELUX;
  } else if (str_value == "Tanh") {
    *value = TANH;
  } else if (str_value == "BandPass") {
    *value = BANDPASS;
  } else {
    return errors::NotFound(str_value, " is not an allowed activation mode");
  }
  return Status::OK();
}

}  // end namespace tensorflow
