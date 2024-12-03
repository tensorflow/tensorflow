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
#ifndef TENSORFLOW_LITE_TOCO_TOCO_TOOLING_H_
#define TENSORFLOW_LITE_TOCO_TOCO_TOOLING_H_

#include <memory>
#include <string>

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"

namespace toco {

// Imports the input file into a Model object.
std::unique_ptr<Model> Import(const TocoFlags& toco_flags,
                              const ModelFlags& model_flags,
                              const std::string& input_file_contents);

// Transforms a Model. The resulting Model is ready to be passed
// to Export with the exact same toco_flags.
absl::Status TransformWithStatus(const TocoFlags& toco_flags, Model* model);
inline void Transform(const TocoFlags& toco_flags, Model* model) {
  auto s = TransformWithStatus(toco_flags, model);
  CHECK(s.ok()) << s.message();
}

// Exports the Model, which must be of the 'lowered' form returned by
// Transform, to a file of the format given by
// toco_flags.output_format().
absl::Status Export(const TocoFlags& toco_flags, const Model& model,
                    bool allow_custom_ops, std::string* output_file_contents);

// This if for backward-compatibility with internal tools.
inline void Export(const TocoFlags& toco_flags, const Model& model,
                   std::string* output_file_contents) {
  auto status = Export(toco_flags, model, true, output_file_contents);
  if (!status.ok()) {
    LOG(QFATAL) << status.message();
  }
}

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TOCO_TOOLING_H_
