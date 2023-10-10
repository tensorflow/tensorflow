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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_OFFSET_COUNTER_HELPER_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_OFFSET_COUNTER_HELPER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/python/framework/op_reg_offset.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"

namespace tensorflow {
tsl::Status FindOpRegistationFromFile(absl::string_view filename,
                                      OpRegOffsets& op_reg_offsets);
}

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_OFFSET_COUNTER_HELPER_H_
