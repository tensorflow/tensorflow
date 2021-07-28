/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/custom_parsers.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/unimplemented_operation_parser.h"

namespace tflite {
namespace gpu {

std::unique_ptr<TFLiteOperationParser> NewCustomOperationParser(
    absl::string_view op_name) {
  return absl::make_unique<UnimplementedOperationParser>(op_name);
}

}  // namespace gpu
}  // namespace tflite
