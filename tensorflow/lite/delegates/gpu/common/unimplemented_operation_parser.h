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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_UNIMPLEMENTED_OPERATION_PARSER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_UNIMPLEMENTED_OPERATION_PARSER_H_

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

class UnimplementedOperationParser : public TFLiteOperationParser {
 public:
  explicit UnimplementedOperationParser(absl::string_view op_name)
      : op_name_(op_name) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return absl::UnimplementedError(op_name_);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    return absl::UnimplementedError(op_name_);
  }

 private:
  std::string op_name_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_UNIMPLEMENTED_OPERATION_PARSER_H_
