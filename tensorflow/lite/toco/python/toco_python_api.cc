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
#include "tensorflow/lite/toco/python/toco_python_api.h"

#include <Python.h>

#include <fstream>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/logging/conversion_log_util.h"
#include "tensorflow/lite/toco/logging/toco_conversion_log.pb.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_convert.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/toco_graphviz_dump_options.h"
#include "tensorflow/lite/toco/toco_tooling.h"
#include "tensorflow/lite/toco/toco_types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/lite/toco/types.pb.h"

namespace toco {

void PopulateConversionLogHelper(const toco::ModelFlags& model_flags,
                                 toco::TocoFlags* toco_flags,
                                 const std::string& input_contents_txt,
                                 const std::string& output_file_contents_txt,
                                 absl::string_view error_message,
                                 GraphVizDumpOptions* dump_options) {
  // Make sure the graphviz file will be dumped under the same folder.
  dump_options->dump_graphviz = toco_flags->conversion_summary_dir();
  // Here we construct the `toco::Model` class based on the input graph def,
  // it will then be used to populate the conversion log.
  // TODO(haoliang): Don't depend on `toco::Model`.
  std::unique_ptr<toco::Model> imported_model =
      toco::Import(*toco_flags, model_flags, input_contents_txt);
  // Dump pre-conversion toco logs.
  TocoConversionLog toco_log_before;
  PopulateConversionLog(*imported_model, &toco_log_before);
  std::ofstream osstream_before(toco_flags->conversion_summary_dir() +
                                "/toco_log_before.pb");
  toco_log_before.SerializeToOstream(&osstream_before);
  osstream_before.close();
  toco::LogDump(toco::kLogLevelModelChanged, "tf_graph", *imported_model);

  // Populate the post-conversion log, for convenient initiate the
  // `toco::Model` class from the generated flatbuffer.
  toco_flags->set_input_format(toco::FileFormat::TFLITE);
  std::unique_ptr<toco::Model> flatbuffer_model =
      toco::Import(*toco_flags, model_flags, output_file_contents_txt);
  // Dump post-conversion toco logs.
  TocoConversionLog toco_log_after;
  PopulateConversionLog(*flatbuffer_model, &toco_log_after);
  // Make sure we sanitize the error message.
  toco_log_after.set_toco_err_logs(SanitizeErrorMessage(error_message));
  std::ofstream ostream_after(toco_flags->conversion_summary_dir() +
                              "/toco_log_after.pb");
  toco_log_after.SerializeToOstream(&ostream_after);
  ostream_after.close();
  toco::LogDump(toco::kLogLevelModelChanged, "tflite_graph", *flatbuffer_model);
}

// NOTE(aselle): We are using raw PyObject's here because we want to make
// sure we input and output bytes rather than unicode strings for Python3.
PyObject* TocoConvert(PyObject* model_flags_proto_txt_raw,
                      PyObject* toco_flags_proto_txt_raw,
                      PyObject* input_contents_txt_raw, bool extended_return) {
  // Use Python C API to validate and convert arguments. In py3 (bytes),
  // in py2 (str).
  auto ConvertArg = [&](PyObject* obj, bool* error) {
    char* buf;
    Py_ssize_t len;
    if (::tflite::python_utils::ConvertFromPyString(obj, &buf, &len) == -1) {
      *error = true;
      return std::string();
    } else {
      *error = false;
      return std::string(buf, len);
    }
  };

  bool error;
  std::string model_flags_proto_txt =
      ConvertArg(model_flags_proto_txt_raw, &error);
  if (error) {
    PyErr_SetString(PyExc_ValueError, "Model flags are invalid.");
    return nullptr;
  }
  std::string toco_flags_proto_txt =
      ConvertArg(toco_flags_proto_txt_raw, &error);
  if (error) {
    PyErr_SetString(PyExc_ValueError, "Toco flags are invalid.");
    return nullptr;
  }

  // Use TOCO to produce new outputs.
  toco::ModelFlags model_flags;
  if (!model_flags.ParseFromString(model_flags_proto_txt)) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert Model to Python String.");
    return nullptr;
  }
  toco::TocoFlags toco_flags;
  if (!toco_flags.ParseFromString(toco_flags_proto_txt)) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert Toco to Python String.");
    return nullptr;
  }

  std::string input_contents_txt;
  if (model_flags.saved_model_dir().empty()) {
    input_contents_txt = ConvertArg(input_contents_txt_raw, &error);
    if (error) {
      PyErr_SetString(PyExc_ValueError, "Input GraphDef is invalid.");
      return nullptr;
    }
  }

  auto& dump_options = *GraphVizDumpOptions::singleton();
  if (toco_flags.has_dump_graphviz_dir()) {
    dump_options.dump_graphviz = toco_flags.dump_graphviz_dir();
  }
  if (toco_flags.has_dump_graphviz_include_video()) {
    dump_options.dump_graphviz_video = toco_flags.dump_graphviz_include_video();
  }

  std::string output_file_contents_txt;
  int64_t arithmetic_ops_count;

  // Convert model.
  absl::Status status =
      Convert(input_contents_txt, toco_flags, model_flags,
              &output_file_contents_txt, &arithmetic_ops_count);

  if (!status.ok()) {
    PyErr_SetString(PyExc_Exception, absl::StatusMessageAsCStr(status));
    return nullptr;
  }
  if (extended_return) {
    PyObject* dict = PyDict_New();
    PyDict_SetItemString(
        dict, "flatbuffer",
        ::tflite::python_utils::ConvertToPyString(
            output_file_contents_txt.data(), output_file_contents_txt.size()));
    PyDict_SetItemString(dict, "arithmetic_ops",
                         PyLong_FromLong(arithmetic_ops_count));
    return dict;
  }
  // Convert arguments back to byte (py3) or str (py2)
  return ::tflite::python_utils::ConvertToPyString(
      output_file_contents_txt.data(), output_file_contents_txt.size());
}

tflite::TensorType FromTocoDataTypeToTflitToTensorType(int inference_type) {
  switch (inference_type) {
    case toco::IODataType::QUANTIZED_INT16:
      return tflite::TensorType_INT16;
    case toco::IODataType::QUANTIZED_UINT8:
      return tflite::TensorType_UINT8;
    case toco::IODataType::UINT8:
      return tflite::TensorType_UINT8;
    case toco::IODataType::QUANTIZED_INT8:
      return tflite::TensorType_INT8;
    case toco::IODataType::INT8:
      return tflite::TensorType_INT8;
    default:
      return tflite::TensorType_FLOAT32;
  }
}

int ToStringSet(PyObject* py_denylist,
                absl::flat_hash_set<std::string>* string_set) {
  using tflite::python_utils::ConvertFromPyString;
  // Ensure op_denylist is non null
  if (!py_denylist) {
    return 0;
  }
  if (PyList_Check(py_denylist)) {
    for (int i = 0; i < PyList_GET_SIZE(py_denylist); ++i) {
      PyObject* value = PyList_GetItem(py_denylist, i);
      char* str_buf;
      Py_ssize_t length;
      if (ConvertFromPyString(value, &str_buf, &length) == -1) {
        return -1;
      }
      string_set->emplace(str_buf, length);
    }
  }
  if (PySet_Check(py_denylist)) {
    auto* tmp = PySet_New(py_denylist);
    while (PySet_GET_SIZE(tmp)) {
      PyObject* value = PySet_Pop(tmp);
      char* str_buf;
      Py_ssize_t length;
      if (ConvertFromPyString(value, &str_buf, &length) == -1) {
        return -1;
      }
      string_set->emplace(str_buf, length);
    }
  }
  return 0;
}

PyObject* RegisterCustomOpdefs(PyObject* list) {
  if (!PyList_Check(list)) {
    PyErr_SetString(PyExc_TypeError, "Expected list in argument");
    return nullptr;
  }

  int64_t size = PyList_Size(list);
  for (int i = 0; i < size; ++i) {
    // Get character array from Python object.
    char* tf_opdefs;
    Py_ssize_t len;
    if (tflite::python_utils::ConvertFromPyString(PyList_GetItem(list, i),
                                                  &tf_opdefs, &len) == -1) {
      PyErr_Format(PyExc_ValueError,
                   "Failed to convert Python string at index %d of custom op "
                   "defs argument",
                   i);
      return nullptr;
    }

    // Parse op def from character array.
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs, &opdef)) {
      PyErr_Format(
          PyExc_ValueError,
          "Failed to parse opdefs at index %d of custom op defs argument: %s",
          i, tf_opdefs);
      return nullptr;
    }

    // Register extra opdefs to TensorFlow global op registry.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData* op_reg_data) -> absl::Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return absl::OkStatus();
        });

    // Register the corresponding fake op kernel.
    const char* node_name = opdef.name().c_str();
    const char* op_name = opdef.name().c_str();
    const char* device_name = "CPU";
    static auto fake_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    };

    TF_KernelBuilder* builder =
        TF_NewKernelBuilder(op_name, device_name, /*create_func=*/nullptr,
                            fake_compute_func, /*delete_func=*/nullptr);

    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_DeleteStatus(status);
      PyErr_Format(PyExc_ValueError,
                   "Failed to register fake op kernel at index %d of custom op "
                   "defs argument",
                   i);
      return nullptr;
    }
    TF_DeleteStatus(status);
  }

  Py_RETURN_TRUE;
}

}  // namespace toco
