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
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_TOOLING_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_TOOLING_H_

#include <memory>
#include <string>

#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_flags.pb.h"

namespace toco {

// Imports the input file into a Model object.
std::unique_ptr<Model> Import(const TocoFlags& toco_flags,
                              const ModelFlags& model_flags,
                              const string& input_file_contents);

// Transforms a Model. The resulting Model is ready to be passed
// to Export with the exact same toco_flags.
void Transform(const TocoFlags& toco_flags, Model* model);

// Exports the Model, which must be of the 'lowered' form returned by
// Transform, to a file of the format given by
// toco_flags.output_format().
void Export(const TocoFlags& toco_flags, const Model& model,
            bool allow_custom_ops, string* output_file_contents);

// This if for backward-compatibility with internal tools.
inline void Export(const TocoFlags& toco_flags, const Model& model,
                   string* output_file_contents) {
  Export(toco_flags, model, true, output_file_contents);
}

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_TOOLING_H_
