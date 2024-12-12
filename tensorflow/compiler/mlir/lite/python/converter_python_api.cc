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
#include "tensorflow/compiler/mlir/lite/python/converter_python_api.h"

#include <Python.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "google/protobuf/text_format.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector.h"
#include "tensorflow/compiler/mlir/lite/model_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/python/flatbuffer_to_mlir.h"
#include "tensorflow/compiler/mlir/lite/python/graphdef_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/compiler/mlir/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/python/saved_model_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/sparsity/sparsify_model.h"
#include "tensorflow/compiler/mlir/lite/types.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/status.h"

namespace tflite {

// NOTE(aselle): We are using raw PyObject's here because we want to make
// sure we input and output bytes rather than unicode strings for Python3.
PyObject* Convert(PyObject* model_flags_proto_txt_raw,
                  PyObject* converter_flags_proto_txt_raw,
                  PyObject* input_contents_txt_raw, bool extended_return,
                  PyObject* debug_info_txt_raw,
                  const tensorflow::quantization::PyFunctionLibrary*
                      quantization_py_function_library) {
  // Use Python C API to validate and convert arguments. In py3 (bytes),
  // in py2 (str).
  auto ConvertArg = [&](PyObject* obj, bool* error) {
    char* buf;
    Py_ssize_t len;
    if (mlirlite::python_utils::ConvertFromPyString(obj, &buf, &len) == -1) {
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
  std::string converter_flags_proto_txt =
      ConvertArg(converter_flags_proto_txt_raw, &error);
  if (error) {
    PyErr_SetString(PyExc_ValueError, "Converter flags are invalid.");
    return nullptr;
  }

  // Produce new outputs.
  tflite::ModelFlags model_flags;
  if (!model_flags.ParseFromString(model_flags_proto_txt)) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert Model to Python String.");
    return nullptr;
  }
  tflite::ConverterFlags converter_flags;
  if (!converter_flags.ParseFromString(converter_flags_proto_txt)) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert ConverterFlags to Python String.");
    return nullptr;
  }

  tensorflow::GraphDebugInfo debug_info;
  if (debug_info_txt_raw && debug_info_txt_raw != Py_None) {
    std::string debug_info_txt = ConvertArg(debug_info_txt_raw, &error);
    if (error) {
      PyErr_SetString(PyExc_ValueError, "Input DebugInfo is invalid.");
      return nullptr;
    }
    if (!debug_info.ParseFromString(debug_info_txt)) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to convert DebugInfo to Python String.");
      return nullptr;
    }
  }

  tensorflow::GraphDef graph_def;
  std::string input_contents_txt;
  if (model_flags.saved_model_dir().empty()) {
    input_contents_txt = ConvertArg(input_contents_txt_raw, &error);
    if (error) {
      PyErr_SetString(PyExc_ValueError, "Input GraphDef is invalid.");
      return nullptr;
    }
    if (!model_flags.use_hlo_import() &&
        !graph_def.ParseFromString(input_contents_txt)) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to convert GraphDef to Python String.");
      return nullptr;
    }
  }

  std::string output_file_contents_txt;
  absl::Status status;

  // Convert model.
  if (model_flags.use_hlo_import() && model_flags.has_saved_model_dir()) {
    PyErr_SetString(PyExc_ValueError,
                    "Cannot specify both saved_model and hlo import.");
    return nullptr;
  }

  if (model_flags.use_hlo_import()) {
    status = tensorflow::ConvertJaxToTFLiteFlatBuffer(
        input_contents_txt, model_flags, converter_flags,
        &output_file_contents_txt);
  } else if (!model_flags.saved_model_dir().empty()) {
    status = tensorflow::ConvertSavedModelToTFLiteFlatBuffer(
        model_flags, converter_flags, &output_file_contents_txt,
        quantization_py_function_library);
  } else {
    tensorflow::GraphDef graph_def;
    if (!graph_def.ParseFromString(input_contents_txt)) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to convert GraphDef to Python String.");
      return nullptr;
    }

    status = tensorflow::ConvertGraphDefToTFLiteFlatBuffer(
        model_flags, converter_flags, debug_info, graph_def,
        &output_file_contents_txt);
  }

  if (!status.ok()) {
    PyErr_SetString(PyExc_Exception, absl::StatusMessageAsCStr(status));
    return nullptr;
  }
  // Convert arguments back to byte (py3) or str (py2)
  return mlirlite::python_utils::ConvertToPyString(
      output_file_contents_txt.data(), output_file_contents_txt.size());
}

tflite::TensorType FromConverterFlagsToTfLiteDType(int inference_type) {
  switch (inference_type) {
    case tflite::IODataType::QUANTIZED_INT16:
      return tflite::TensorType_INT16;
    case tflite::IODataType::QUANTIZED_UINT8:
      return tflite::TensorType_UINT8;
    case tflite::IODataType::UINT8:
      return tflite::TensorType_UINT8;
    case tflite::IODataType::QUANTIZED_INT8:
      return tflite::TensorType_INT8;
    case tflite::IODataType::INT8:
      return tflite::TensorType_INT8;
    default:
      return tflite::TensorType_FLOAT32;
  }
}

int ToStringSet(PyObject* py_denylist,
                absl::flat_hash_set<std::string>* string_set) {
  using mlirlite::python_utils::ConvertFromPyString;
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

PyObject* MlirQuantizeModel(PyObject* data, bool disable_per_channel,
                            bool fully_quantize, int inference_type,
                            int input_data_type, int output_data_type,
                            bool enable_numeric_verify,
                            bool enable_whole_model_verify,
                            PyObject* op_denylist, PyObject* node_denylist,
                            bool enable_variable_quantization,
                            bool disable_per_channel_for_dense_layers,
                            PyObject* debug_options_proto_txt_raw) {
  using tflite_migration::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);

  if (mlirlite::python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert input PyObject");
    return nullptr;
  }

  std::optional<tensorflow::converter::DebugOptions> debug_options =
      tensorflow::converter::DebugOptions();
  if (debug_options_proto_txt_raw != nullptr) {
    auto ConvertArg = [&](PyObject* obj, bool* error) {
      char* buf;
      Py_ssize_t len;
      if (mlirlite::python_utils::ConvertFromPyString(obj, &buf, &len) == -1) {
        *error = true;
        return std::string();
      } else {
        *error = false;
        return std::string(buf, len);
      }
    };

    bool error;
    std::string debug_options_proto_txt =
        ConvertArg(debug_options_proto_txt_raw, &error);
    if (error) {
      PyErr_SetString(PyExc_ValueError, "Converter flags are invalid.");
      return nullptr;
    }

    if (!debug_options->ParseFromString(debug_options_proto_txt)) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to convert ConverterFlags to Python String.");
      return nullptr;
    }
  } else {
    debug_options = std::nullopt;
  }

  absl::flat_hash_set<std::string> denylisted_ops;
  absl::flat_hash_set<std::string> denylisted_nodes;
  if (ToStringSet(op_denylist, &denylisted_ops) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert op denylist PyObject");
    return nullptr;
  }
  if (ToStringSet(node_denylist, &denylisted_nodes) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert node denylist PyObject");
    return nullptr;
  }

  std::unique_ptr<mlir::TFL::FlatBufferModelAbslError> model =
      mlir::TFL::FlatBufferModelAbslError::BuildFromBuffer(
          buf, length, error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }
  auto tflite_model = std::make_unique<tflite::ModelT>();
  model->GetModel()->UnPackTo(tflite_model.get(), nullptr);

  const tflite::TensorType inference_tensor_type =
      FromConverterFlagsToTfLiteDType(inference_type);
  const tflite::TensorType input_type =
      FromConverterFlagsToTfLiteDType(input_data_type);
  const tflite::TensorType output_type =
      FromConverterFlagsToTfLiteDType(output_data_type);

  std::string output_model;
  const absl::string_view input_model_buffer(buf, length);
  auto status = mlir::lite::QuantizeModel(
      input_model_buffer, input_type, output_type, inference_tensor_type,
      /*operator_names=*/{}, disable_per_channel, fully_quantize, output_model,
      enable_numeric_verify, enable_whole_model_verify,
      /*legacy_float_scale=*/true, denylisted_ops, denylisted_nodes,
      enable_variable_quantization, disable_per_channel_for_dense_layers,
      debug_options);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to quantize model: " << status;
    error_reporter->exception();
    return nullptr;
  }

  return mlirlite::python_utils::ConvertToPyString(output_model.data(),
                                                   output_model.size());
}

PyObject* MlirSparsifyModel(PyObject* data) {
  using tflite_migration::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);

  if (mlirlite::python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    PyErr_Format(PyExc_ValueError, "Failed to convert input PyObject");
    return nullptr;
  }
  std::unique_ptr<mlir::TFL::FlatBufferModelAbslError> model =
      mlir::TFL::FlatBufferModelAbslError::BuildFromBuffer(
          buf, length, error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }
  auto tflite_model = std::make_unique<tflite::ModelT>();
  model->GetModel()->UnPackTo(tflite_model.get(), nullptr);

  flatbuffers::FlatBufferBuilder builder;
  auto status = mlir::lite::SparsifyModel(*tflite_model, &builder);

  if (!status.ok()) {
    error_reporter->exception();
    return nullptr;
  }
  return mlirlite::python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
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
    if (mlirlite::python_utils::ConvertFromPyString(PyList_GetItem(list, i),
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

std::vector<std::string> RetrieveCollectedErrors() {
  mlir::TFL::ErrorCollector* collector =
      mlir::TFL::ErrorCollector::GetErrorCollector();
  std::vector<std::string> collected_errors;
  for (const auto& error_data : collector->CollectedErrors()) {
    collected_errors.push_back(error_data.SerializeAsString());
  }
  collector->Clear();
  return collected_errors;
}

std::string FlatBufferFileToMlir(const std::string& model,
                                 bool input_is_filepath) {
  return ::tensorflow::FlatBufferFileToMlir(model, input_is_filepath);
}

}  // namespace tflite
