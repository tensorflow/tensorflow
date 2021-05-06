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

#include <cstdint>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

class UnimplementedCustomOperationParser : public TFLiteOperationParser {
 public:
  explicit UnimplementedCustomOperationParser(absl::string_view op_name)
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

}  // namespace

std::unique_ptr<TFLiteOperationParser> NewCustomOperationParser(
    absl::string_view op_name) {
  return absl::make_unique<UnimplementedCustomOperationParser>(op_name);
}

absl::Status ParseCustomAttributes(absl::string_view op_name, int version,
                                   const void* data, uint32_t data_size,
                                   absl::any* attr, BHWC* output_shape) {
  return absl::UnimplementedError(absl::StrCat(
      "Attributes parsing is not enabled for ", op_name, " operation."));
}

}  // namespace gpu
}  // namespace tflite
