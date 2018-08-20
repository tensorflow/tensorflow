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

%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%typemap(in) const tensorflow::MetaGraphDef& (tensorflow::MetaGraphDef temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The MetaGraphDef could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

%{
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/python/grappler/model_analyzer.h"
%}

%{
string GenerateModelReport(const tensorflow::MetaGraphDef& metagraph,
                           bool assume_valid_feeds, bool debug) {
  tensorflow::grappler::ItemConfig cfg;
  cfg.apply_optimizations = false;
  std::unique_ptr<tensorflow::grappler::GrapplerItem> item =
      tensorflow::grappler::GrapplerItemFromMetaGraphDef("metagraph", metagraph, cfg);
  if (!item) {
    return "Error: failed to preprocess metagraph: check your log file for errors";
  }

  string suffix;
  tensorflow::grappler::ModelAnalyzer analyzer(*item);

  std::stringstream os;
  analyzer.GenerateReport(debug, assume_valid_feeds, os);
  return os.str();
}

%}

string GenerateModelReport(const tensorflow::MetaGraphDef& metagraph,
                           bool assume_valid_feeds, bool debug);
