/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/python/analyzer_wrapper/model_analyzer.h"

PYBIND11_MODULE(_pywrap_analyzer_wrapper, m) {
  m.def(
      "ModelAnalyzer",
      [](const std::string& model_path, bool input_is_filepath,
         bool gpu_compatibility) {
        return ::tflite::model_analyzer(model_path, input_is_filepath,
                                        gpu_compatibility);
      },
      R"pbdoc(
    Returns txt dump of the given TFLite file.
  )pbdoc");
  m.def(
      "ModelAnalyzer",
      [](const std::vector<std::string>& checked_delegates,
         const std::unordered_map<std::string, std::string>& delegate_configs,
         const std::string& model_path, bool input_is_filepath) {
        return ::tflite::model_analyzer(checked_delegates, delegate_configs,
                                        model_path, input_is_filepath);
      },
      R"pbdoc(
    Returns txt dump of the given TFLite file.
  )pbdoc");
}
