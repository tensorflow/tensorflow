/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_SAVED_MODEL_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_SAVED_MODEL_H_

#include <string>
#include <vector>

#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/contrib/lite/toco/args.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_flags.pb.h"
#include "tensorflow/contrib/lite/toco/types.pb.h"

namespace toco {

// Parses metadata into `toco_flags` and `model_flags`.
//
// Stores `inputs` as input_arrays and `outputs` as output_arrays in
// `model_flags`. Infers input_shapes from the GraphDef and stores it in
// `model_flags` as part of the input_arrays. Assumes inference_type is FLOAT
// and stores it in `toco_flags`.
void ParseMetaData(const tensorflow::GraphDef& graph_def,
                   const std::unordered_set<string>& inputs,
                   const std::unordered_set<string>& outputs,
                   const ParsedTocoFlags& parsed_toco_flags,
                   const ParsedModelFlags& parsed_model_flags,
                   TocoFlags* toco_flags, ModelFlags* model_flags);

// Generates a frozen graph from the SavedModel in the directory specified in
// `toco_flags`. Reads frozen graph contents into `graph_def_contents`. Parses
// metadata relating to the GraphDef into `toco_flags` and `model_flags`.
void GetSavedModelContents(const ParsedTocoFlags& parsed_toco_flags,
                           const ParsedModelFlags& parsed_model_flags,
                           TocoFlags* toco_flags, ModelFlags* model_flags,
                           string* graph_def_contents);

}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_TOCO_SAVED_MODEL_H_
