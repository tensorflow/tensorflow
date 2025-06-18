/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// clang-format off
// This #include needs to precede the inclusion of any other TF Lite header
// file that might depend on the non-mutable schema_generated.h, directly,
// e.g. core/api/op_resolver.h, or indirectly, e.g. core/subgraph.h.
// That's because "tensorflow/lite/schema/mutable/schema_generated.h"
// and "tensorflow/lite/schema/schema_generated.h" both use the same
// header guard macro (FLATBUFFERS_GENERATED_SCHEMA_TFLITE_H_), but have
// different contents (the former is a superset of the latter). In particular
// the one in mutable/ is built with the "--gen-mutable" and "--gen-object-api"
// flags to the flatbuffer schema compiler which cause some additional
// (non-virtual) accessor methods and API functions to be declared.
// The code here uses those methods, so we need to make sure that we get
// the mutable variant of this header.
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"

#include "tensorflow/lite/python/optimize/calibration_wrapper.h"
// clang-format on

// NOLINTBEGIN
// Nolint disables warnings about header file ordering caused by
// `mutable/schema_generated.h`.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
// NOLINTEND

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/offset_buffer.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/shared_library.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"
#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace calibration_wrapper {
namespace {

using ::tflite::interpreter_wrapper::PythonErrorReporter;
using ::tflite::ops::builtin::BuiltinOpResolver;
using ::tflite::optimize::AddIntermediateTensorsToFusedOp;
using ::tflite::optimize::QuantizeModelAllOperators;
using ::tflite::optimize::calibration::BuildLoggingInterpreter;
using ::tflite::optimize::calibration::CalibrationReader;
using ::tflite::python::ImportNumpy;
using ::tflite::python_utils::ConvertFromPyString;
using ::tflite::python_utils::ConvertToPyString;
using ::tflite::python_utils::PyDecrefDeleter;
using ::tflite::python_utils::TfLiteTypeFromPyArray;
using ::tflite::python_utils::TfLiteTypeFromPyType;

std::unique_ptr<ModelT> CreateMutableModel(const Model& model) {
  auto copied_model = std::make_unique<ModelT>();
  model.UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

bool NoOpModel(const FlatBufferModel& model) {
  return model->subgraphs()->size() == 1 &&
         (!model->subgraphs()->begin()->operators() ||
          model->subgraphs()->begin()->operators()->size() == 0);
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(b/129336260): No schema type for none.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteBFloat16:
      return TensorType_BFLOAT16;
    case kTfLiteFloat64:
      return TensorType_FLOAT64;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt32:
      return TensorType_UINT32;
    case kTfLiteInt4:
      return TensorType_INT4;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteUInt64:
      return TensorType_UINT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteUInt16:
      return TensorType_UINT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
    case kTfLiteComplex128:
      return TensorType_COMPLEX128;
    case kTfLiteResource:
      return TensorType_RESOURCE;
    case kTfLiteVariant:
      return TensorType_VARIANT;
  }
  // No default to get compiler error when new type is introduced.
}

bool RegisterCustomOpByName(const char* registerer_name,
                            MutableOpResolver* resolver) {
  // Registerer functions take a pointer to a BuiltinOpResolver as an input
  // parameter and return void.
  // TODO(b/137576229): We should implement this functionality in a more
  // principled way.
  typedef void (*RegistererFunctionType)(MutableOpResolver*);

  // Look for the Registerer function by name.
  RegistererFunctionType registerer = reinterpret_cast<RegistererFunctionType>(
      SharedLibrary::GetSymbol(registerer_name));

  // Fail in an informative way if the function was not found.
  if (registerer == nullptr) {
    PyErr_Format(PyExc_ValueError,
                 "Looking up symbol '%s' failed with error '%s'.",
                 registerer_name, SharedLibrary::GetError());
    return false;
  }

  // Call the registerer with the resolver.
  registerer(resolver);
  return true;
}

// Returns the dimension from the stored list in the PyObject. If the given
// PyObject is not a list, it will return absl::optional and set the Python
// error message to notify users.
std::optional<std::vector<int>> ConvertInputShapeToVector(
    PyObject* input_shapes, size_t index) {
  PyObject* shape = PyList_GetItem(input_shapes, index);
  if (!shape || !PyList_Check(shape)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid %ld input shape: expected to be a list.", index);
    return std::nullopt;
  }
  size_t size = PyList_Size(shape);
  std::vector<int> dims(size);
  for (size_t dim_index = 0; dim_index < size; ++dim_index) {
    PyObject* dim = PyList_GetItem(shape, dim_index);
    dims[dim_index] = PyLong_AsLong(dim);
  }
  return dims;
}

// Finds the starting position of the offset buffer within `model_buffer` if the
// `model_buffer` can be split into base buffer and offset buffer. Returns
// `std::nullopt` iff offset buffer is not used or there were no buffers with
// valid offset. Assumes `model_buffer` is valid.
std::optional<int64_t> GetOffsetBufferStartPosition(
    const absl::string_view model_buffer) {
  const Model& model = *GetModel(model_buffer.data());

  if (!FlatBufferModel::CheckBufferOutsideModel(&model)) {
    // Means the offset buffer is not used, e.g.
    // `_experimental_use_buffer_offset` is not set.
    return std::nullopt;
  }

  const int64_t int64_max = std::numeric_limits<int64_t>::max();
  const int64_t min_offset = absl::c_accumulate(
      *model.buffers(), /*init=*/int64_max,
      /*binary_op=*/[](const int64_t acc, const Buffer* buffer) -> int64_t {
        const int64_t buffer_offset = buffer->offset();
        return IsValidBufferOffset(buffer_offset) ? std::min(acc, buffer_offset)
                                                  : acc;
      });
  if (min_offset == int64_max) {
    // Means there were no buffers with valid offset.
    return std::nullopt;
  }
  return min_offset;
}

// Splits the model buffer into base buffer and offset buffer. Offset buffer may
// exist when `_experimental_use_buffer_offset` is set.
std::pair<absl::string_view, absl::string_view> SplitOffsetBuffer(
    const absl::string_view model_buffer) {
  const std::optional<int64_t> offset_buffer_pos =
      GetOffsetBufferStartPosition(model_buffer);
  if (offset_buffer_pos == std::nullopt) {
    return {model_buffer, absl::string_view(model_buffer.data(), /*len=*/0)};
  }

  const absl::string_view base_buffer(model_buffer.data(), *offset_buffer_pos);

  const int64_t offset_buffer_length = model_buffer.size() - *offset_buffer_pos;
  const absl::string_view offset_buffer(
      model_buffer.data() + *offset_buffer_pos, offset_buffer_length);

  return {base_buffer, offset_buffer};
}

// Merges `base_buffer` with the `offset_buffer` that contains the actual tensor
// buffer data.
std::string MergeOffsetBuffer(const absl::string_view base_buffer,
                              const absl::string_view offset_buffer) {
  return absl::StrCat(base_buffer, offset_buffer);
}

// Updates buffer offsets in `base_buffer` by `offset_diff`.
std::string UpdateBufferOffsets(const absl::string_view base_buffer,
                                const int64_t offset_diff) {
  std::string result_buffer(base_buffer);

  Model* mutable_model = GetMutableModel(result_buffer.data());
  for (Buffer* buffer : *mutable_model->mutable_buffers()) {
    if (const int64_t offset = buffer->offset(); IsValidBufferOffset(offset)) {
      buffer->mutate_offset(offset + offset_diff);
    }
  }

  return result_buffer;
}

}  // namespace

PyObject* AddIntermediateTensors(PyObject* data) {
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ImportNumpy();

  if (ConvertFromPyString(data, &buf, &length) == -1) {
    return nullptr;
  }

  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromBuffer(buf, length, error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }

  const auto [base_buffer, offset_buffer] =
      SplitOffsetBuffer(/*model_buffer=*/absl::string_view(buf, length));

  flatbuffers::FlatBufferBuilder builder;
  auto tflite_model = CreateMutableModel(*model->GetModel());
  if (AddIntermediateTensorsToFusedOp(&builder, tflite_model.get()) !=
      kTfLiteOk) {
    error_reporter->exception();
    return nullptr;
  }

  const int64_t result_base_buffer_size = builder.GetSize();
  if (result_base_buffer_size == 0) {
    // When AddIntermediateTensorsToFusedOp early returns, return the model as
    // it is.
    return ConvertToPyString(buf, length);
  }

  const int64_t offset_diff =
      result_base_buffer_size - static_cast<int64_t>(base_buffer.size());
  const std::string updated_result_base_buffer = UpdateBufferOffsets(
      /*base_buffer=*/absl::string_view(
          reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
          builder.GetSize()),
      offset_diff);

  const std::string result_buffer =
      MergeOffsetBuffer(updated_result_base_buffer, offset_buffer);

  return ConvertToPyString(result_buffer.data(), result_buffer.size());
}

CalibrationWrapper::CalibrationWrapper(
    std::unique_ptr<Interpreter> interpreter,
    std::unique_ptr<BuiltinOpResolver> resolver,
    std::unique_ptr<PythonErrorReporter> error_reporter,
    std::unique_ptr<FlatBufferModel> model,
    std::unique_ptr<CalibrationReader> reader,
    std::unique_ptr<std::string> model_str)
    : interpreter_(std::move(interpreter)),
      error_reporter_(std::move(error_reporter)),
      resolver_(std::move(resolver)),
      model_(std::move(model)),
      reader_(std::move(reader)),
      model_str_(std::move(model_str)) {}

CalibrationWrapper::~CalibrationWrapper() = default;

PyObject* CalibrationWrapper::Prepare() {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Prepare(std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  impl::SignatureRunner* runner =
      interpreter_->GetSignatureRunner(signature_key.c_str());
  if (runner == nullptr) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  TFLITE_PY_CHECK(runner->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Prepare(PyObject* input_shapes,
                                      std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_shapes)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected shapes to be a list.");
    return nullptr;
  }
  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  auto* subgraph = interpreter_->subgraph(subgraph_index);

  const size_t inputs_size = PyList_Size(input_shapes);
  if (inputs_size != subgraph->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected %ld items got %ld items.",
                 subgraph->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    std::optional<std::vector<int>> dims =
        ConvertInputShapeToVector(input_shapes, i);
    if (!dims.has_value()) {
      return nullptr;
    }
    int input_tensor_idx = subgraph->inputs()[i];
    if (subgraph->ResizeInputTensor(input_tensor_idx, *dims) != kTfLiteOk) {
      PyErr_Format(PyExc_ValueError, "Failed to resize %ld input tensor.", i);
      return nullptr;
    }
  }

  return Prepare(signature_key);
}

PyObject* CalibrationWrapper::Prepare(PyObject* input_shapes) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_shapes)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected shapes to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_shapes);
  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    std::optional<std::vector<int>> dims =
        ConvertInputShapeToVector(input_shapes, i);
    if (!dims.has_value()) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (interpreter_->ResizeInputTensor(input_tensor_idx, *dims) != kTfLiteOk) {
      PyErr_Format(PyExc_ValueError, "Failed to resize %ld input tensor.", i);
      return nullptr;
    }
  }

  return Prepare();
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value,
                                         std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }
  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  const size_t inputs_size = PyList_Size(input_value);

  auto* subgraph = interpreter_->subgraph(subgraph_index);
  if (inputs_size != subgraph->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 subgraph->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = subgraph->inputs()[i];
    if (!SetTensor(input_tensor_idx, input, signature_key)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(subgraph->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_value);

  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (!SetTensor(input_tensor_idx, input)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(interpreter_->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value,
                                        std::string signature_key) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  auto* subgraph = interpreter_->subgraph(subgraph_index);
  const TfLiteTensor* tensor = subgraph->tensor(index);

  if (TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: "
                 "Got value of type %s "
                 "but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension count mismatch, expected %d "
                 "but found %d",
                 tensor->dims->size, PyArray_NDIM(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_NDIM(array));
  bool has_unknown_dims = false;
  for (int j = 0; j < PyArray_NDIM(array); ++j) {
    // Ensure the calibration data input shape is the same as the model input
    // shape unless the dimension is unknown.
    if (tensor->dims_signature != nullptr &&
        tensor->dims_signature->size == tensor->dims->size &&
        tensor->dims_signature->data[j] == -1) {
      has_unknown_dims = true;
    } else if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Size mismatch, expected %d for dim "
                   "%d but found %ld",
                   tensor->dims->data[j], j, PyArray_SHAPE(array)[j]);
      return nullptr;
    }
    dims[j] = PyArray_SHAPE(array)[j];
  }

  // Resize the input tensor if there are unknown dimensions.
  if (has_unknown_dims) {
    // Does strict checking on the `ResizeInputTensor` call.
    TFLITE_PY_CHECK(subgraph->ResizeInputTensorStrict(index, dims));
    TFLITE_PY_CHECK(subgraph->AllocateTensors());
  }

  // Re-read the updated tensor after the allocation is done.
  tensor = subgraph->tensor(index);

  size_t size = PyArray_NBYTES(array);

  if (tensor->type == kTfLiteString) {
    DynamicBuffer buffer;
    buffer.AddString(reinterpret_cast<const char*>(PyArray_BYTES(array)), size);
    buffer.WriteToTensor(subgraph->tensor(index), /*new_shape=*/nullptr);
    Py_RETURN_NONE;
  }

  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value) {
  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  const TfLiteTensor* tensor = interpreter_->tensor(index);

  if (TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: "
                 "Got value of type %s "
                 "but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(
        PyExc_ValueError,
        "Cannot set tensor: Dimension count mismatch, expected %d but found %d",
        tensor->dims->size, PyArray_NDIM(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_NDIM(array));
  bool has_unknown_dims = false;
  for (int j = 0; j < PyArray_NDIM(array); ++j) {
    // Ensure the calibration data input shape is the same as the model input
    // shape unless the dimension is unknown.
    if (tensor->dims_signature != nullptr &&
        tensor->dims_signature->size == tensor->dims->size &&
        tensor->dims_signature->data[j] == -1) {
      has_unknown_dims = true;
    } else if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Size mismatch, expected %d for dim "
                   "%d but found %ld",
                   tensor->dims->data[j], j, PyArray_SHAPE(array)[j]);
      return nullptr;
    }
    dims[j] = PyArray_SHAPE(array)[j];
  }

  // Resize the input tensor if there are unknown dimensions.
  if (has_unknown_dims) {
    // Does strict checking on the `ResizeInputTensor` call.
    TFLITE_PY_CHECK(interpreter_->ResizeInputTensorStrict(index, dims));
    TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  }

  // Re-read the updated tensor after the allocation is done.
  tensor = interpreter_->tensor(index);

  size_t size = PyArray_NBYTES(array);

  if (tensor->type == kTfLiteString) {
    DynamicBuffer buffer;
    buffer.AddString(reinterpret_cast<const char*>(PyArray_BYTES(array)), size);
    buffer.WriteToTensor(interpreter_->tensor(index), /*new_shape=*/nullptr);
    Py_RETURN_NONE;
  }

  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Calibrate() {
  const auto [base_buffer, offset_buffer] =
      SplitOffsetBuffer(/*model_buffer=*/absl::string_view(
          reinterpret_cast<const char*>(model_->allocation()->base()),
          model_->allocation()->bytes()));

  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto loc = Model::Pack(builder, tflite_model.get());
  FinishModelBuffer(builder, loc);

  const int64_t result_base_buffer_size = builder.GetSize();
  const int64_t offset_diff =
      result_base_buffer_size - static_cast<int64_t>(base_buffer.size());
  const std::string updated_result_base_buffer = UpdateBufferOffsets(
      /*base_buffer=*/absl::string_view(
          reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
          result_base_buffer_size),
      offset_diff);
  const std::string result_buffer =
      MergeOffsetBuffer(updated_result_base_buffer, offset_buffer);

  return ConvertToPyString(result_buffer.data(), result_buffer.size());
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            int activations_py_type,
                                            int bias_py_type) {
  return QuantizeModel(
      input_py_type, output_py_type, allow_float, activations_py_type,
      bias_py_type,
      /*disable_per_channel=*/false,
      /*disable_per_channel_quantization_for_dense_layers=*/false);
}

PyObject* CalibrationWrapper::QuantizeModel(
    int input_py_type, int output_py_type, bool allow_float,
    int activations_py_type, int bias_py_type, bool disable_per_channel,
    bool disable_per_channel_quantization_for_dense_layers) {
  if (NoOpModel(*model_)) {
    return ConvertToPyString(model_str_->data(), model_str_->size());
  }

  TfLiteType input_type = TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = TfLiteTypeFromPyType(output_py_type);
  TfLiteType activations_type = TfLiteTypeFromPyType(activations_py_type);
  TfLiteType bias_type = TfLiteTypeFromPyType(bias_py_type);

  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = kTfLiteOk;

  status = QuantizeModelAllOperators(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float,
      TfLiteTypeToSchemaType(activations_type),
      TfLiteTypeToSchemaType(bias_type), disable_per_channel,
      disable_per_channel_quantization_for_dense_layers, error_reporter_.get());

  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            const char* operator_output_name) {
  string op_name = std::string(operator_output_name);

  TfLiteType input_type = TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = TfLiteTypeFromPyType(output_py_type);
  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = optimize::QuantizeModel(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float, {op_name},
      /*activations_type=*/TensorType_INT8, /*bias_type=*/TensorType_INT32,
      error_reporter_.get());
  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

/*static*/ CalibrationWrapper* CalibrationWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, const std::vector<std::string>& registerers_by_name,
    const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
    std::string* error_msg) {
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ImportNumpy();

  if (ConvertFromPyString(data, &buf, &length) == -1) {
    *error_msg = "Failed to convert from python string";
    return nullptr;
  }
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::BuildFromBuffer(buf, length, error_reporter.get());
  if (!model) {
    *error_msg = "Invalid model";
    return nullptr;
  }

  auto resolver = std::make_unique<BuiltinOpResolver>();
  for (const auto& registerer : registerers_by_name) {
    if (!RegisterCustomOpByName(registerer.c_str(), resolver.get())) {
      *error_msg =
          absl::StrFormat("Looking up symbol '%s' failed with error '%s'.",
                          registerer.c_str(), SharedLibrary::GetError());
      return nullptr;
    }
  }
  for (const auto& registerer : registerers_by_func) {
    registerer(reinterpret_cast<uintptr_t>(resolver.get()));
  }
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status =
      BuildLoggingInterpreter(*model, *resolver, &interpreter, &reader);
  if (status != kTfLiteOk) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  auto model_str = std::make_unique<std::string>(buf, length);
  // If we are not going to use this string during quantization, reset the
  // pointer and release the memory.
  if (!NoOpModel(*model)) {
    model_str.reset();
  }

  auto wrapper = new CalibrationWrapper(
      std::move(interpreter), std::move(resolver), std::move(error_reporter),
      std::move(model), std::move(reader), std::move(model_str));
  return wrapper;
}

}  // namespace calibration_wrapper
}  // namespace tflite
