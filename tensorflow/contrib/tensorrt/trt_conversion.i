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

/* Wrap trt_conversion */
%{
#define SWIG_FILE_WITH_INIT
%}
%include "std_pair.i"
%include "tensorflow/python/platform/base.i"

%{
PyObject* pair_helper(std::pair<string, string>* in) {
  PyObject *first(nullptr), *second(nullptr), *tuple(nullptr);
  first = PyBytes_FromStringAndSize(in->first.data(), in->first.length());
  if (!first) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "Pair conversion first argument failed");
    }
    return NULL;
  }
  second = PyBytes_FromStringAndSize(in->second.data(), in->second.length());
  if (!second) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError,
                      "Pair conversion second argument failed");
    }
    return NULL;
  }
  tuple = Py_BuildValue("(OO)", first, second);
  if (!tuple) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError,
                      "Tuple creation from pair<string,string> failed!");
    }
    return NULL;
  }
  return tuple;
}

struct version_struct{
  int vmajor;
  int vminor;
  int vpatch;
};

PyObject* version_helper(version_struct* in) {
  PyObject *tuple(nullptr);
  tuple = Py_BuildValue("(iii)", in->vmajor, in->vminor, in->vpatch);
  if (!tuple) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError,
                      "Tuple creation from version structure failed!");
    }
    return NULL;
  }
  return tuple;
}
/* Define converters for vector<int> */
template<>
bool _PyObjAs(PyObject *pyobj, int* dest) {
  *dest = PyLong_AsLong(pyobj);
  return true;
}

template<>
PyObject *_PyObjFrom(const int& src) {
  return PyLong_FromLong(src);
}

%}

_LIST_OUTPUT_TYPEMAP(int, PyLong_FromLong);

%typemap(out) std::pair<string, string> {
  PyObject *tuple = pair_helper(&$1);
  if (!tuple) SWIG_fail;
  $result = tuple;
}

%typemap(out) version_struct {
  PyObject *tuple = version_helper(&$1);
  if (!tuple) SWIG_fail;
  $result = tuple;
}

%{
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/contrib/tensorrt/convert/convert_graph.h"
%}

%ignoreall
%unignore tensorflow;
%unignore trt_convert;
%unignore calib_convert;
%unignore get_linked_tensorrt_version;
%unignore get_loaded_tensorrt_version;

%{

std::pair<string, string> trt_convert(
    string graph_def_string,  // The serialized GraphDef string.
    std::vector<string> output_names,
    size_t max_batch_size,
    size_t max_workspace_size_bytes,
    int precision_mode,
    int minimum_segment_size,
    bool is_dyn_op,
    int max_cached_engines,
    std::vector<int> cached_engine_batches
    // Unfortunately we can't use TF_Status here since it
    // is in c/c_api and brings in a lot of other libraries
    // which in turn declare ops. These ops are included
    // statically in our library and cause an abort when
    // module is loaded due to double registration
    // until Tensorflow properly exposes these headers
    // we have to work around this by returning a string
    // and converting it to exception on python side.
    //,TF_Status* out_status) {
) {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  string out_status;

  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    out_status = "InvalidArgument;Couldn't interpret input as a GraphDef";
    return std::pair<string, string>{out_status, ""};
  }

  if(precision_mode < 0 || precision_mode > 2){
    out_status = "InvalidArgument;Invalid precision_mode";
    return std::pair<string, string>{out_status, ""};
  }
  if (!output_names.size()) {
    out_status = "InvalidArgument;Size of the output_names vector is 0";
    return std::pair<string, string>{out_status, ""};
  }
  tensorflow::GraphDef out_graph;
  tensorflow::Status conversion_status =
      tensorflow::tensorrt::convert::ConvertGraphDefToTensorRT(
          graph_def, output_names, max_batch_size, max_workspace_size_bytes,
          &out_graph, precision_mode, minimum_segment_size,
          is_dyn_op, max_cached_engines, cached_engine_batches);
  if (!conversion_status.ok()) {
    auto retCode = (int)conversion_status.code();
    char buff[2000];
    snprintf(buff, 2000, "%d;%s", retCode,
             conversion_status.error_message().c_str());
    out_status = buff;
    return std::pair<string, string>{out_status, ""};
  }
  string result;
  if (!out_graph.SerializeToString(&result)) {
    out_status = "InvalidArgument;Couldn't serialize output as a GraphDef";
    return std::pair<string, string>{out_status, ""};
  }
  out_status = "OK;All good!";
  return std::pair<string, string>{out_status, result};
#else
  // Returns FAILED_PRECONDITION.
  return std::pair<string, string>{"9;TensorRT is not enabled!", ""};
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
}

std::pair<string, string> calib_convert(
    string graph_def_string, bool is_dyn_op
    // unfortunately we can't use TF_Status here since it
    // is in c/c_api and brings in a lot of other libraries
    // which in turn declare ops. These ops are included
    // statically in our library and cause an abort when
    // module is loaded due to double registration
    // until Tensorflow properly exposes these headers
    // we have to work around this by returning a string
    // and converting it to exception on python side.
    //,TF_Status* out_status) {
) {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  string out_status;

  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    out_status = "InvalidArgument;Couldn't interpret input as a GraphDef";
    return std::pair<string, string>{out_status, ""};
  }
  graph_def_string.resize(0);
  tensorflow::GraphDef out_graph;
  tensorflow::Status conversion_status =
      tensorflow::tensorrt::convert::ConvertCalibGraphToInferGraph(
          graph_def, &out_graph, is_dyn_op);
  if (!conversion_status.ok()) {
    auto retCode = (int)conversion_status.code();
    char buff[2000];
    snprintf(buff, 2000, "%d;%s", retCode,
             conversion_status.error_message().c_str());
    out_status = buff;
    return std::pair<string, string>{out_status, ""};
  }
  string result;
  if (!out_graph.SerializeToString(&result)) {
    out_status = "InvalidArgument;Couldn't serialize output as a GraphDef";
    return std::pair<string, string>{out_status, ""};
  }
  out_status = "OK;All good!";
  return std::pair<string, string>{out_status, result};
#else
  // Returns FAILED_PRECONDITION.
  return std::pair<string, string>{"9;TensorRT is not enabled!", ""};
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
}

version_struct get_linked_tensorrt_version(){
  // Return the version at the link time.
  const auto &lv = tensorflow::tensorrt::convert::GetLinkedTensorRTVersion();
  version_struct s;
  s.vmajor = lv[0];
  s.vminor = lv[1];
  s.vpatch = lv[2];
  return s;
}
version_struct get_loaded_tensorrt_version(){
  // Return the version from the loaded library.
  const auto &lv = tensorflow::tensorrt::convert::GetLoadedTensorRTVersion();
  version_struct s;
  s.vmajor = lv[0];
  s.vminor = lv[1];
  s.vpatch = lv[2];
  return s;
}

%}

std::pair<string, string> calib_convert(string graph_def_string, bool is_dyn_op);

std::pair<string, string> trt_convert(string graph_def_string,
                                      std::vector<string> output_names,
                                      size_t max_batch_size,
                                      size_t max_workspace_size_bytes,
                                      int precision_mode, int minimum_segment_size,
                                      bool is_dyn_op,
                                      int max_cached_engines,
                                      std::vector<int> cached_engine_batches);
version_struct get_linked_tensorrt_version();
version_struct get_loaded_tensorrt_version();

%unignoreall
