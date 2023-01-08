/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "Python.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/dlpack.h"
#include "tensorflow/c/eager/tfe_cancellation_manager_internal.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/get_compiler_ir.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(TFE_Executor);
PYBIND11_MAKE_OPAQUE(TFE_ContextOptions);
PYBIND11_MAKE_OPAQUE(tensorflow::CancellationManager);

PYBIND11_MAKE_OPAQUE(TFE_MonitoringCounter0);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringCounter1);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringCounter2);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringStringGauge0);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringStringGauge1);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringStringGauge2);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringStringGauge3);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringStringGauge4);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringIntGauge0);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringIntGauge1);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringIntGauge2);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringBoolGauge0);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringBoolGauge1);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringBoolGauge2);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringSampler0);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringSampler1);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringSampler2);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringCounterCell);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringIntGaugeCell);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringStringGaugeCell);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringBoolGaugeCell);
PYBIND11_MAKE_OPAQUE(TFE_MonitoringSamplerCell);

PYBIND11_MAKE_OPAQUE(TF_DeviceList);
PYBIND11_MAKE_OPAQUE(TF_Function);
PYBIND11_MAKE_OPAQUE(TF_Buffer);

// Eager helper functions migrated from pywrap_tfe.i.

namespace tensorflow {

// We cannot use Context as an opaque type. SWIG also had
// difficult directly passing the pointer around. These
// typemaps are migrated over from pywrap_tfe.i. I tried
// using a custom type caster, but we get segfaults periodically.

// TODO(amitpatankar): Move input and output logic of Context into a
// pybind11 custom type caster.

TFE_Context* InputTFE_Context(const py::handle& ctx) {
  return static_cast<TFE_Context*>(PyCapsule_GetPointer(ctx.ptr(), nullptr));
}

PyObject* OutputTFE_Context(TFE_Context* context) {
  return PyCapsule_New(context, nullptr, TFE_DeleteContextCapsule);
}

TF_Buffer* ProtoStringToTFBuffer(PyObject* input) {
  // Convert a Python string object to TF_Buffer.
  char* c_string;
  Py_ssize_t py_size;
  // PyBytes_AsStringAndSize() does not copy but simply interprets the input
  if (PyBytes_AsStringAndSize(input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    throw py::error_already_set();
  }
  return TF_NewBufferFromString(static_cast<void*>(c_string),
                                static_cast<size_t>(py_size));
}

// These functions are typemaps from the Python side. I did not use
// a custom type caster since the logic is slightly harder to follow. This
// converter is also only used once in `TFE_Py_ExecuteCancelable_wrapper`.
TFE_InputTensorHandles InputTFE_InputTensorHandles(
    const py::handle& input_tensors) {
  TFE_InputTensorHandles input_tensor_handles;
  if (input_tensors.ptr() != Py_None) {
    if (!PyList_Check(input_tensors.ptr())) {
      tensorflow::ThrowTypeError("must provide a list of Tensors as inputs");
    }
    Py_ssize_t len = PyList_Size(input_tensors.ptr());
    input_tensor_handles.resize(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PyList_GetItem(input_tensors.ptr(), i);
      if (!elem) {
        tensorflow::ThrowTypeError("Input Tensor does not exist.");
      }
      if (EagerTensor_CheckExact(elem)) {
        (input_tensor_handles)[i] = EagerTensor_Handle(elem);
      } else if (tensorflow::swig::IsEagerTensorSlow(elem)) {
        // Use equivalent of object.__getattribute__ to get the underlying
        // tf wrapped EagerTensor (if there is one).
        tensorflow::Safe_PyObjectPtr tf_should_use_attr(
#if PY_MAJOR_VERSION < 3
            PyString_InternFromString("_tf_should_use_wrapped_value")
#else
            PyUnicode_InternFromString("_tf_should_use_wrapped_value")
#endif
        );
        tensorflow::Safe_PyObjectPtr value_attr(
            PyObject_GenericGetAttr(elem, tf_should_use_attr.get()));
        if (value_attr) {
          // This is an EagerTensor wrapped inside a TFShouldUse wrapped object.
          (input_tensor_handles)[i] = EagerTensor_Handle(value_attr.get());
        } else {
          // This is a subclass of EagerTensor that we don't support.
          PyErr_Clear();
          tensorflow::ThrowTypeError(
              tensorflow::strings::StrCat(
                  "Saw an object that is an instance of a strict subclass of "
                  "EagerTensor, which is not supported.  Item ",
                  i, " is type: ", elem->ob_type->tp_name)
                  .c_str());
        }
      } else if (tensorflow::swig::IsTensor(elem)) {
        // If it isnt an EagerTensor, but is still a Tensor, it must be a graph
        // tensor.
        tensorflow::Safe_PyObjectPtr py_tensor_repr(PyObject_Repr(elem));
        std::string tensor_repr =
            py_tensor_repr ? TFE_GetPythonString(py_tensor_repr.get())
                           : "<unknown>";
        tensorflow::Safe_PyObjectPtr py_op(PyObject_GetAttrString(elem, "op"));
        tensorflow::Safe_PyObjectPtr py_defined_graph(
            PyObject_GetAttrString(py_op.get(), "graph"));
        tensorflow::Safe_PyObjectPtr py_defined_graph_str(
            PyObject_Str(py_defined_graph.get()));
        std::string defined_graph_str =
            py_defined_graph_str
                ? TFE_GetPythonString(py_defined_graph_str.get())
                : "<unknown>";
        tensorflow::Safe_PyObjectPtr c_op(
            PyObject_GetAttrString(py_op.get(), "_c_op"));
        auto& node = py::cast<TF_Operation*>(c_op.get())->node;
        auto node_name_str = node.name();
        std::string frame_str, traceback_str;
        if (auto stack_trace = node.GetStackTrace()) {
          auto frame = stack_trace->LastUserFrame();
          frame_str =
              absl::StrFormat("File \"%s\", line %d, in %s", frame.file_name,
                              frame.line_number, frame.function_name);
          auto stack_trace_list =
              absl::StrSplit(stack_trace->ToString({true}), '\n');
          traceback_str = absl::StrJoin(
              stack_trace_list, "", [&](std::string* out, const auto line) {
                absl::StrAppend(out, "    ", line, "\n");
              });
        } else {
          frame_str = "<unknown>";
          traceback_str = "<unknown>\n";
        }
        // Keep in sync with func_graph.py.
        // TODO(b/200991648): Unify those two paths.
        tensorflow::ThrowTypeError(
            tensorflow::strings::StrCat(
                tensor_repr,
                " is out of scope and cannot be used here. "
                "Use return values, explicit Python locals or TensorFlow "
                "collections to access it.\n"
                "Please see https://www.tensorflow.org/guide/"
                "function#all_outputs_of_a_tffunction_must_be_return_values "
                "for more information.\n\n",
                tensor_repr, " was defined here:\n", traceback_str,
                "\nThe tensor ", tensor_repr,
                " cannot be accessed from here, because it was "
                "defined in ",
                defined_graph_str, ", which is out of scope.")
                .c_str());
      } else {
        tensorflow::ThrowTypeError(
            tensorflow::strings::StrCat(
                "provided list of inputs contains objects other "
                "than 'EagerTensor'. Item ",
                i, " is type: ", elem->ob_type->tp_name)
                .c_str());
      }
    }
  }
  return input_tensor_handles;
}

// These functions are typemaps from the Python side. I did not use
// a custom type caster since the logic is slightly harder to follow. This
// converter is also only used once in `TFE_Py_ExecuteCancelable_wrapper`.
// This function actually takes a number rather than an output Tensor holder.
TFE_OutputTensorHandles InputTFE_OutputTensorHandles(
    const py::handle& num_outputs) {
  TFE_OutputTensorHandles output_tensor_handles;
#if PY_MAJOR_VERSION < 3
  if (!PyInt_Check(num_outputs.ptr())) {
#else
  if (!PyLong_Check(num_outputs.ptr())) {
#endif
    PyErr_SetString(PyExc_TypeError,
                    "expected an integer value (size of the number of "
                    "outputs of the operation)");
    throw py::error_already_set();
  }
#if PY_MAJOR_VERSION < 3
  long sz = PyInt_AsLong(num_outputs.ptr());  // NOLINT
#else
  long sz = PyLong_AsLong(num_outputs.ptr());  // NOLINT
#endif
  // PyLong_AsLong might throw an error if an overflow occurs.
  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, tensorflow::strings::StrCat(
                                          "Number of outputs is too big: ", sz)
                                          .c_str());
    throw py::error_already_set();
  }
  // We can't handle more than int32 sizes for number of outputs.
  if (static_cast<long>(static_cast<int32_t>(sz)) != sz) {  // NOLINT
    PyErr_SetString(PyExc_ValueError, tensorflow::strings::StrCat(
                                          "Number of outputs is too big: ", sz)
                                          .c_str());
    throw py::error_already_set();
  }
  if (sz > 0) {
#if PY_MAJOR_VERSION < 3
    output_tensor_handles.resize(PyInt_AsLong(num_outputs.ptr()), nullptr);
#else
    output_tensor_handles.resize(PyLong_AsLong(num_outputs.ptr()), nullptr);
#endif
  }
  return output_tensor_handles;
}

tensorflow::Device* GetMatchedDevice(py::handle& ctx, const char* device_name) {
  auto* context = reinterpret_cast<tensorflow::ImmediateExecutionContext*>(
      tensorflow::InputTFE_Context(ctx));

  tensorflow::DeviceNameUtils::ParsedName input_device_name;
  if (!tensorflow::DeviceNameUtils::ParseFullOrLocalName(device_name,
                                                         &input_device_name)) {
    tensorflow::ThrowValueError(
        absl::StrFormat("Failed parsing device name: '%s'. Note a valid device "
                        "string should at least contain a device type and a "
                        "device index, like \"GPU:0\".",
                        device_name)
            .c_str());
  }

  std::vector<tensorflow::Device*> devices = context->ListLocalTfDevices();

  tensorflow::Device* matched_device = nullptr;
  for (int device_idx = 0; device_idx < devices.size(); device_idx++) {
    tensorflow::Device* device = devices[device_idx];

    if (tensorflow::DeviceNameUtils::AreCompatibleDevNames(
            input_device_name, device->parsed_name())) {
      if (matched_device != nullptr) {
        tensorflow::ThrowValueError(
            absl::StrFormat("Multiple devices match the provided string "
                            "'%s': '%s' and '%s'.",
                            device_name, matched_device->name(), device->name())
                .c_str());
      }
      matched_device = device;
    }
  }

  if (matched_device == nullptr) {
    tensorflow::ThrowValueError(
        absl::StrFormat("No matching devices found for '%s'", device_name)
            .c_str());
  }

  return matched_device;
}

// Packs multiple `EagerTensor`s of the same dtype and shape into one
// `EagerTensor`.
py::object TFE_Py_PackEagerTensors_wrapper(const py::handle& context,
                                           const py::handle& tensors) {
  TFE_Context* ctx = tensorflow::InputTFE_Context(context);
  TFE_InputTensorHandles handles = InputTFE_InputTensorHandles(tensors);
  tensorflow::Safe_TF_StatusPtr status = tensorflow::make_safe(TF_NewStatus());
  int size = handles.size();
  TFE_TensorHandle* packed_handle =
      TFE_CreatePackedTensorHandle(ctx, handles.data(), &size, status.get());
  tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  PyObject* packed_tensor =
      EagerTensorFromHandle(packed_handle, /*is_packed=*/true);
  return tensorflow::PyoOrThrow(packed_tensor);
}

// This function was created from fusing the typemap logic in platform/base.i.
py::object TFE_Py_ExecuteCancelable_wrapper(
    const py::handle& context, const char* device_name, const char* op_name,
    const py::handle& inputs, const py::handle& attrs,
    tensorflow::CancellationManager* cancellation_manager,
    const py::handle& num_outputs) {
  TFE_Context* ctx = tensorflow::InputTFE_Context(context);
  TFE_InputTensorHandles input_tensor_handles =
      InputTFE_InputTensorHandles(inputs);
  TFE_OutputTensorHandles output_tensor_handles =
      InputTFE_OutputTensorHandles(num_outputs);
  tensorflow::Safe_TF_StatusPtr status = tensorflow::make_safe(TF_NewStatus());
  TFE_Py_ExecuteCancelable(ctx, device_name, op_name, &input_tensor_handles,
                           attrs.ptr(), tensorflow::wrap(cancellation_manager),
                           &output_tensor_handles, status.get());

  int output_len = output_tensor_handles.size();
  PyObject* output_list = PyList_New(output_len);
  for (int i = 0; i < output_len; ++i) {
    PyObject* output;
    output = EagerTensorFromHandle(output_tensor_handles.at(i));
    PyList_SetItem(output_list, i, output);
  }
  tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  return tensorflow::PyoOrThrow(output_list);
}

static py::object TF_ListPhysicalDevices() {
  std::vector<string> devices;
  tensorflow::Status s =
      tensorflow::DeviceFactory::ListAllPhysicalDevices(&devices);
  MaybeRaiseRegisteredFromStatus(s);
  PyObject* result = PyList_New(devices.size());
  int i = 0;
  for (auto& dev : devices) {
    PyObject* dev_obj = PyBytes_FromStringAndSize(dev.data(), dev.size());
    PyList_SetItem(result, i, dev_obj);
    ++i;
  }
  return tensorflow::PyoOrThrow(result);
}

static py::object TF_ListPluggablePhysicalDevices() {
  std::vector<string> devices;
  tensorflow::Status s =
      tensorflow::DeviceFactory::ListPluggablePhysicalDevices(&devices);
  MaybeRaiseRegisteredFromStatus(s);
  Safe_PyObjectPtr result(PyList_New(devices.size()));
  int i = 0;
  for (auto& dev : devices) {
    PyObject* dev_obj = PyBytes_FromStringAndSize(dev.data(), dev.size());
    PyList_SetItem(result.get(), i, dev_obj);
    ++i;
  }
  return tensorflow::PyoOrThrow(result.release());
}

static std::unordered_map<string, string> TF_GetDeviceDetails(int index) {
  tensorflow::Safe_TF_StatusPtr status = tensorflow::make_safe(TF_NewStatus());
  std::unordered_map<string, string> device_details;
  tensorflow::Status s =
      tensorflow::DeviceFactory::GetAnyDeviceDetails(index, &device_details);
  tensorflow::Set_TF_Status_from_Status(status.get(), s);
  MaybeRaiseRegisteredFromTFStatus(status.get());
  return device_details;
}

static py::object TFE_ClearScalarCache() {
  tensorflow::TFE_TensorHandleCache::Get()->Clear();
  return py::none();
}

// Returns compiler IR for a given function.
static py::bytes TFE_GetCompilerIr(py::handle& ctx,
                                   const char* concrete_function_name,
                                   const char* stage, const char* device_name,
                                   py::handle& inputs) {
  EagerContext* context = ContextFromInterface(
      reinterpret_cast<ImmediateExecutionContext*>(InputTFE_Context(ctx)));

  std::string s_stage(stage);
  IrExportStage selected_stage = [&] {
    if (s_stage == "hlo") {
      return IrExportStage::HLO;
    } else if (s_stage == "hlo_no_metadata") {
      return IrExportStage::HLO_NO_METADATA;
    } else if (s_stage == "hlo_serialized") {
      return IrExportStage::HLO_SERIALIZED;
    } else if (s_stage == "optimized_hlo") {
      return IrExportStage::OPTIMIZED_HLO;
    } else if (s_stage == "optimized_hlo_serialized") {
      return IrExportStage::OPTIMIZED_HLO_SERIALIZED;
    } else if (s_stage == "optimized_hlo_proto_serialized") {
      return IrExportStage::OPTIMIZED_HLO_PROTO_SERIALIZED;
    } else if (s_stage == "optimized_hlo_dot") {
      return IrExportStage::OPTIMIZED_HLO_DOT;
    } else {
      ThrowValueError(
          absl::StrFormat("Invalid stage selected: '%s'. Valid values are: "
                          "'hlo', 'hlo_serialized', 'optimized_hlo', "
                          "'optimized_hlo_serialized', 'optimized_hlo_dot'",
                          s_stage)
              .c_str());
    }
  }();

  TFE_InputTensorHandles handles = InputTFE_InputTensorHandles(inputs);

  std::vector<const TensorHandle*> input_handles;
  for (TFE_TensorHandle* tensor_handle : handles) {
    AbstractTensorHandle* abstract_tensor_handle = unwrap(tensor_handle);
    input_handles.push_back(TensorHandleFromInterface(abstract_tensor_handle));
  }

  DeviceNameUtils::ParsedName input_device_name;
  if (!DeviceNameUtils::ParseFullOrLocalName(device_name, &input_device_name)) {
    ThrowValueError(
        absl::StrFormat("Failed parsing device name: '%s'", device_name)
            .c_str());
  }

  std::vector<Device*> devices = context->local_device_mgr()->ListDevices();
  auto selected_device = absl::c_find_if(devices, [&](const Device* d) {
    return DeviceNameUtils::AreCompatibleDevNames(input_device_name,
                                                  d->parsed_name());
  });
  if (selected_device == devices.end()) {
    ThrowValueError(
        absl::StrFormat("No matching device found for '%s'", device_name)
            .c_str());
  }

  StatusOr<std::string> hlo_str =
      GetCompilerIr(selected_stage, context->pflr(), concrete_function_name,
                    *selected_device, context, input_handles);

  if (!hlo_str.ok()) {
    ThrowValueError(absl::StrFormat("Failed getting HLO text: '%s'",
                                    hlo_str.status().error_message())
                        .c_str());
  }
  return py::bytes(*hlo_str);
}

}  // namespace tensorflow

namespace {

// Wrapper around the EagerContextThreadLocalData struct (defined in
// pywrap_tfe.h), so it can be accessed from Python.
//
// For PyObject* fields, the get_*() methods return a new reference; and the
// set_*() methods create a new reference (i.e., they do not steal a reference).
class EagerContextThreadLocalDataWrapper {
 public:
  explicit EagerContextThreadLocalDataWrapper(py::handle py_eager_context,
                                              py::handle is_eager,
                                              py::handle device_spec)
      : py_eager_context_(py_eager_context.ptr()) {
    tensorflow::MakeEagerContextThreadLocalData(
        py_eager_context.ptr(), is_eager.ptr(), device_spec.ptr());
  }

  ~EagerContextThreadLocalDataWrapper() {
    tensorflow::DestroyEagerContextThreadLocalData(py_eager_context_);
  }

  bool get_is_eager() const { return GetData()->is_eager; }
  void set_is_eager(bool v) { GetData()->is_eager = v; }

  bool get_invoking_op_callbacks() const {
    return GetData()->invoking_op_callbacks;
  }
  void set_invoking_op_callbacks(bool v) {
    GetData()->invoking_op_callbacks = v;
  }

  py::object get_device_name() const {
    return GetPyObject(&GetData()->device_name);
  }
  void set_device_name(py::handle v) {
    SetPyObject(v, &GetData()->device_name);
  }

  py::object get_scope_name() const {
    return GetPyObject(&GetData()->scope_name);
  }
  void set_scope_name(py::handle v) { SetPyObject(v, &GetData()->scope_name); }

  py::object get_device_spec() const {
    return GetPyObject(&GetData()->device_spec);
  }
  void set_device_spec(py::handle v) {
    SetPyObject(v, &GetData()->device_spec);
  }

  py::object get_function_call_options() const {
    return GetPyObject(&GetData()->function_call_options);
  }
  void set_function_call_options(py::handle v) {
    SetPyObject(v, &GetData()->function_call_options);
  }

  py::handle get_executor() const { return GetPyObject(&GetData()->executor); }
  void set_executor(py::handle v) { SetPyObject(v, &GetData()->executor); }

  py::object get_op_callbacks() const {
    return GetPyObject(&GetData()->op_callbacks);
  }
  void set_op_callbacks(py::handle v) {
    SetPyObject(v, &GetData()->op_callbacks);
  }

 private:
  tensorflow::EagerContextThreadLocalData* GetData() const {
    auto* result =
        tensorflow::GetEagerContextThreadLocalData(py_eager_context_);
    if (!result) {
      throw py::error_already_set();
    }
    return result;
  }

  py::object GetPyObject(tensorflow::Safe_PyObjectPtr* obj) const {
    return pybind11::reinterpret_borrow<py::object>(obj->get());
  }

  void SetPyObject(py::handle value, tensorflow::Safe_PyObjectPtr* ptr) {
    Py_INCREF(value.ptr());
    ptr->reset(value.ptr());
  }

  PyObject* py_eager_context_;  // not owned (borrowed reference).
};

}  // namespace

// py::return_value_policy::reference is defined as specified by the
// pybind11 documents listed here.
// https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies
// This means that C++ maintains ownership of the object. We
// are only assigning this to functions that return opaque types.

PYBIND11_MODULE(_pywrap_tfe, m) {
  py::class_<TFE_Executor> TFE_Executor_class(m, "TFE_Executor");
  py::class_<TFE_ContextOptions> TFE_ContextOptions_class(m,
                                                          "TFE_ContextOptions");
  py::class_<TFE_MonitoringCounter0> TFE_MonitoringCounter0_class(
      m, "TFE_MonitoringCounter0");
  py::class_<TFE_MonitoringCounter1> TFE_MonitoringCounter1_class(
      m, "TFE_MonitoringCounter1");
  py::class_<TFE_MonitoringCounter2> TFE_MonitoringCounter2_class(
      m, "TFE_MonitoringCounter2");
  py::class_<TFE_MonitoringStringGauge0> TFE_MonitoringStringGauge0_class(
      m, "TFE_MonitoringStringGauge0");
  py::class_<TFE_MonitoringStringGauge1> TFE_MonitoringStringGauge1_class(
      m, "TFE_MonitoringStringGauge1");
  py::class_<TFE_MonitoringStringGauge2> TFE_MonitoringStringGauge2_class(
      m, "TFE_MonitoringStringGauge2");
  py::class_<TFE_MonitoringStringGauge3> TFE_MonitoringStringGauge3_class(
      m, "TFE_MonitoringStringGauge3");
  py::class_<TFE_MonitoringStringGauge4> TFE_MonitoringStringGauge4_class(
      m, "TFE_MonitoringStringGauge4");
  py::class_<TFE_MonitoringIntGauge0> TFE_MonitoringIntGauge0_class(
      m, "TFE_MonitoringIntGauge0");
  py::class_<TFE_MonitoringIntGauge1> TFE_MonitoringIntGauge1_class(
      m, "TFE_MonitoringIntGauge1");
  py::class_<TFE_MonitoringIntGauge2> TFE_MonitoringIntGauge2_class(
      m, "TFE_MonitoringIntGauge2");
  py::class_<TFE_MonitoringBoolGauge0> TFE_MonitoringBoolGauge0_class(
      m, "TFE_MonitoringBoolGauge0");
  py::class_<TFE_MonitoringBoolGauge1> TFE_MonitoringBoolGauge1_class(
      m, "TFE_MonitoringBoolGauge1");
  py::class_<TFE_MonitoringBoolGauge2> TFE_MonitoringBoolGauge2_class(
      m, "TFE_MonitoringBoolGauge2");
  py::class_<TFE_MonitoringCounterCell> TFE_MonitoringCounterCell_class(
      m, "TFE_MonitoringCounterCell");
  py::class_<TFE_MonitoringIntGaugeCell> TFE_MonitoringIntGaugeCell_class(
      m, "TFE_MonitoringIntGaugeCell");
  py::class_<TFE_MonitoringStringGaugeCell> TFE_MonitoringStringGaugeCell_class(
      m, "TFE_MonitoringStringGaugeCell");
  py::class_<TFE_MonitoringBoolGaugeCell> TFE_MonitoringBoolGaugeCell_class(
      m, "TFE_MonitoringBoolGaugeCell");
  py::class_<TFE_MonitoringSamplerCell> TFE_MonitoringSamplerCell_class(
      m, "TFE_MonitoringSamplerCell");
  py::class_<TFE_MonitoringBuckets> TFE_MonitoringBuckets_class(
      m, "TFE_MonitoringBuckets");
  py::class_<TFE_MonitoringSampler0> TFE_MonitoringSampler0_class(
      m, "TFE_MonitoringSampler0");
  py::class_<TFE_MonitoringSampler1> TFE_MonitoringSampler1_class(
      m, "TFE_MonitoringSampler1");
  py::class_<TFE_MonitoringSampler2> TFE_MonitoringSampler2_class(
      m, "TFE_MonitoringSampler2");
  py::class_<tensorflow::CancellationManager> TFE_CancellationManager_class(
      m, "TFE_CancellationManager");

  py::class_<TF_DeviceList> TF_DeviceList_class(m, "TF_DeviceList");
  py::class_<TF_Function> TF_Function_class(m, "TF_Function");

  m.def("TFE_Py_RegisterExceptionClass", [](const py::handle& e) {
    return tensorflow::PyoOrThrow(TFE_Py_RegisterExceptionClass(e.ptr()));
  });
  m.def("TFE_Py_RegisterFallbackExceptionClass", [](const py::handle& e) {
    return tensorflow::PyoOrThrow(
        TFE_Py_RegisterFallbackExceptionClass(e.ptr()));
  });

  m.def("TFE_GetMemoryInfo", [](py::handle& ctx, const char* device_name) {
    tensorflow::Device* matched_device =
        tensorflow::GetMatchedDevice(ctx, device_name);

    tensorflow::AllocatorAttributes attrs;
    tensorflow::Allocator* allocator = matched_device->GetAllocator(attrs);

    if (absl::optional<tensorflow::AllocatorStats> stats =
            allocator->GetStats()) {
      return std::map<std::string, int64_t>{{"current", stats->bytes_in_use},
                                            {"peak", stats->peak_bytes_in_use}};
    }

    tensorflow::ThrowValueError(
        absl::StrFormat("Allocator stats not available for device '%s'",
                        device_name)
            .c_str());
  });

  m.def("TFE_ResetMemoryStats", [](py::handle& ctx, const char* device_name) {
    tensorflow::Device* matched_device =
        tensorflow::GetMatchedDevice(ctx, device_name);

    tensorflow::AllocatorAttributes attrs;
    tensorflow::Allocator* allocator = matched_device->GetAllocator(attrs);

    if (!allocator->ClearStats()) {
      tensorflow::ThrowValueError(
          absl::StrFormat("Cannot reset memory stats for device '%s'",
                          device_name)
              .c_str());
    }
  });

  // XLA Eager Logic
  m.def("TF_SetXlaEnableLazyCompilation", &TF_SetXlaEnableLazyCompilation);
  m.def("TF_SetTfXlaCpuGlobalJit", &TF_SetTfXlaCpuGlobalJit);
  m.def("TF_SetXlaAutoJitMode", &TF_SetXlaAutoJitMode);
  m.def("TF_SetXlaConstantFoldingDisabled", &TF_SetXlaConstantFoldingDisabled);
  m.def("TF_GetXlaConstantFoldingDisabled", &TF_GetXlaConstantFoldingDisabled);
  m.def("TF_SetXlaMinClusterSize", &TF_SetXlaMinClusterSize);
  m.def("TF_GetCompilerIr", &tensorflow::TFE_GetCompilerIr);

  // MLIR Logic
  m.def("TF_IsMlirBridgeEnabled", [] {
    // Since python protobuf enums are integers, cast to an integer before
    // returning the enum to python.
    return static_cast<int32_t>(
        tensorflow::GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge);
  });
  m.def("TF_EnableMlirBridge", [](bool enabled) {
    tensorflow::GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge =
        enabled
            ? tensorflow::ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED
            : tensorflow::ConfigProto::Experimental::
                  MLIR_BRIDGE_ROLLOUT_DISABLED;
  });
  m.def("TF_EnableXlaDevices", [] {
    tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  });
  m.def("TF_ResetJitCompilerFlags",
        [] { tensorflow::ResetJitCompilerFlags(); });

  // TFE_Context Logic
  m.def(
      "TFE_NewContext",
      [](const TFE_ContextOptions* opts) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        TFE_Context* context = TFE_NewContext(opts, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return tensorflow::PyoOrThrow(tensorflow::OutputTFE_Context(context));
      },
      py::return_value_policy::reference);
  m.def("TFE_DeleteContext", [](py::handle& o) {
    TFE_DeleteContext(tensorflow::InputTFE_Context(o));
  });
  m.def(
      "TFE_ContextListDevices",
      [](py::handle& o) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_ContextListDevices(tensorflow::InputTFE_Context(o),
                                             status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def(
      "TFE_SetLogicalCpuDevices",
      [](py::handle& ctx, int num_cpus, const char* prefix) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        TFE_SetLogicalCpuDevices(tensorflow::InputTFE_Context(ctx), num_cpus,
                                 prefix, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
      },
      py::return_value_policy::reference);
  m.def("TFE_HostAddressSpace", [](py::handle& o, TF_Buffer& buf) {
    TFE_HostAddressSpace(tensorflow::InputTFE_Context(o), &buf);
  });
  m.def("TFE_ContextAddFunction", [](py::handle& ctx, TF_Function* func) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextAddFunction(tensorflow::InputTFE_Context(ctx), func,
                           status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextAddFunctionDef",
        [](py::handle& ctx, const char* serialized_function_def, size_t size) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TFE_ContextAddFunctionDef(tensorflow::InputTFE_Context(ctx),
                                    serialized_function_def, size,
                                    status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });
  m.def("TFE_ContextGetFunctionDef",
        [](py::handle& ctx, const char* function_name, TF_Buffer& buf) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TFE_ContextGetFunctionDef(tensorflow::InputTFE_Context(ctx),
                                    function_name, &buf, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });
  m.def("TFE_ContextRemoveFunction", [](py::handle& ctx, const char* name) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextRemoveFunction(tensorflow::InputTFE_Context(ctx), name,
                              status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextHasFunction", [](py::handle& ctx, const char* name) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    auto output =
        TFE_ContextHasFunction(tensorflow::InputTFE_Context(ctx), name);
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });
  m.def("TFE_ContextListFunctionNames", [](py::handle& ctx) {
    return tensorflow::unwrap(tensorflow::InputTFE_Context(ctx))
        ->ListFunctionNames();
  });
  m.def("TFE_ContextEnableRunMetadata", [](py::handle& ctx) {
    TFE_ContextEnableRunMetadata(tensorflow::InputTFE_Context(ctx));
  });
  m.def("TFE_ContextDisableRunMetadata", [](py::handle& ctx) {
    TFE_ContextEnableRunMetadata(tensorflow::InputTFE_Context(ctx));
  });
  m.def("TFE_ContextEnableGraphCollection", [](py::handle& ctx) {
    TFE_ContextEnableGraphCollection(tensorflow::InputTFE_Context(ctx));
  });
  m.def("TFE_ContextDisableGraphCollection", [](py::handle& ctx) {
    TFE_ContextDisableGraphCollection(tensorflow::InputTFE_Context(ctx));
  });
  m.def("TFE_ContextExportRunMetadata", [](py::handle& ctx, TF_Buffer& buf) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextExportRunMetadata(tensorflow::InputTFE_Context(ctx), &buf,
                                 status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextClearCaches", [](py::handle& o) {
    TFE_ContextClearCaches(tensorflow::InputTFE_Context(o));
  });
  m.def("TFE_GetContextId", [](py::handle& ctx) {
    return TFE_GetContextId(tensorflow::InputTFE_Context(ctx));
  });
  m.def("TFE_ContextGetDevicePlacementPolicy", [](py::handle& ctx) {
    return TFE_ContextGetDevicePlacementPolicy(
        tensorflow::InputTFE_Context(ctx));
  });
  m.def("TFE_ContextSetThreadLocalDevicePlacementPolicy",
        [](py::handle& ctx, TFE_ContextDevicePlacementPolicy policy) {
          TFE_ContextSetThreadLocalDevicePlacementPolicy(
              tensorflow::InputTFE_Context(ctx), policy);
        });
  m.def("TFE_ContextSetServerDef", [](py::handle& ctx, int keep_alive_secs,
                                      py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(tensorflow::ProtoStringToTFBuffer(proto.ptr()));
    TFE_ContextSetServerDef(tensorflow::InputTFE_Context(ctx), keep_alive_secs,
                            buf.get()->data, buf.get()->length, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextUpdateServerDef", [](py::handle& ctx, int keep_alive_secs,
                                         py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(tensorflow::ProtoStringToTFBuffer(proto.ptr()));
    Py_BEGIN_ALLOW_THREADS;
    TFE_ContextUpdateServerDef(tensorflow::InputTFE_Context(ctx),
                               keep_alive_secs, buf.get()->data,
                               buf.get()->length, status.get());
    Py_END_ALLOW_THREADS;
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextCheckAlive", [](py::handle& ctx, const char* worker_name) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    bool output = TFE_ContextCheckAlive(tensorflow::InputTFE_Context(ctx),
                                        worker_name, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });
  m.def("TFE_ContextSyncExecutors", [](py::handle& ctx) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // NOTE: release Python GIL for pending PyFunc ops to be executed properly.
    Py_BEGIN_ALLOW_THREADS;
    TFE_ContextAsyncWait(tensorflow::InputTFE_Context(ctx), status.get());
    Py_END_ALLOW_THREADS;
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextClearExecutors", [](py::handle& ctx) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // NOTE: release Python GIL for pending PyFunc ops to be executed properly.
    Py_BEGIN_ALLOW_THREADS;
    TFE_ContextAsyncWait(tensorflow::InputTFE_Context(ctx), status.get());
    Py_END_ALLOW_THREADS;
    // NOTE: different from TFE_ContextSyncExecutors that raises potential
    // errors, deliberately ignore executor statuses in cleanup.
  });
  m.def(
      "TFE_InsertConfigKeyValue",
      [](py::handle& ctx, const char* config_key, const char* config_value) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        Py_BEGIN_ALLOW_THREADS;
        TFE_InsertConfigKeyValue(tensorflow::InputTFE_Context(ctx), config_key,
                                 config_value, status.get());
        Py_END_ALLOW_THREADS;
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
      },
      py::return_value_policy::reference);
  m.def(
      "TFE_GetConfigKeyValue",
      [](py::handle& ctx, const char* config_key, TF_Buffer& config_value) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        Py_BEGIN_ALLOW_THREADS;
        TFE_GetConfigKeyValue(tensorflow::InputTFE_Context(ctx), config_key,
                              &config_value, status.get());
        Py_END_ALLOW_THREADS;
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
      },
      py::return_value_policy::reference);
  m.def(
      "TFE_DeleteConfigKeyValue",
      [](py::handle& ctx, const char* config_key) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        Py_BEGIN_ALLOW_THREADS;
        TFE_DeleteConfigKeyValue(tensorflow::InputTFE_Context(ctx), config_key,
                                 status.get());
        Py_END_ALLOW_THREADS;
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
      },
      py::return_value_policy::reference);
  m.def(
      "TFE_ReportErrorToCluster",
      [](py::handle& ctx, int error_code, const char* error_message) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        TFE_ReportErrorToCluster(tensorflow::InputTFE_Context(ctx), error_code,
                                 error_message, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
      },
      py::return_value_policy::reference);
  m.def("TFE_ContextSetSoftDevicePlacement", [](py::handle& ctx, bool enable) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextSetSoftDevicePlacement(tensorflow::InputTFE_Context(ctx), enable,
                                      status.get());
  });
  m.def("TFE_ContextSetLogDevicePlacement", [](py::handle& ctx, bool enable) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextSetSoftDevicePlacement(tensorflow::InputTFE_Context(ctx), enable,
                                      status.get());
  });
  m.def("TFE_ContextSetRunEagerOpAsFunction", [](py::handle& ctx, bool enable) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextSetRunEagerOpAsFunction(tensorflow::InputTFE_Context(ctx),
                                       enable, status.get());
  });
  m.def("TFE_ContextSetJitCompileRewrite", [](py::handle& ctx, bool enable) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TFE_ContextSetJitCompileRewrite(tensorflow::InputTFE_Context(ctx), enable,
                                    status.get());
  });
  m.def("TFE_GetTaskStates", [](py::handle& ctx,
                                const std::vector<std::string>& job_names,
                                const std::vector<int>& task_nums) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    if (job_names.size() != task_nums.size()) {
      status->status = tensorflow::errors::InvalidArgument(
          "The size of job names is not equal to the size of task nums.");
      tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    }
    std::vector<tensorflow::CoordinatedTask> coordinated_tasks;
    for (size_t i = 0; i < job_names.size(); ++i) {
      for (size_t j = 0; j < task_nums[i]; ++j) {
        auto& coordinated_task = coordinated_tasks.emplace_back();
        coordinated_task.set_job_name(job_names[i]);
        coordinated_task.set_task_id(j);
      }
    }
    size_t task_len = coordinated_tasks.size();
    auto state = std::make_unique<TF_Status[]>(task_len);
    TF_Buffer tasks;
    tasks.data = coordinated_tasks.data();
    tasks.length = task_len;
    TFE_GetTaskStates(tensorflow::InputTFE_Context(ctx), tasks, state.get(),
                      status.get());
    py::list output(task_len);
    for (size_t i = 0; i < task_len; ++i) {
      auto code = TF_GetCode(&state[i]);
      if (code != TF_Code::TF_OK) {
        py::dict payloads;
        for (const auto& payload :
             tensorflow::errors::GetPayloads(state[i].status)) {
          payloads[payload.first.c_str()] = payload.second;
        }
        auto exception_class = py::reinterpret_steal<py::object>(
            tensorflow::PyExceptionRegistry::Lookup(code));
        if (!exception_class) {
          status->status = tensorflow::errors::Internal(absl::StrCat(
              "Fail to find the corresponding exception class for ", code));
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        }
        output[i] = exception_class(py::none(), py::none(),
                                    TF_Message(&state[i]), payloads);
      } else {
        output[i] = py::none();
      }
    }
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return tensorflow::PyoOrThrow(output.release().ptr());
  });

  m.def("TFE_WaitAtBarrier",
        [](py::handle& ctx, const char* barrier_id, int64_t timeout_in_ms) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());

          TFE_WaitAtBarrier(tensorflow::InputTFE_Context(ctx), barrier_id,
                            timeout_in_ms, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  // TFE_Executor logic
  m.def(
      "TFE_NewExecutor",
      [](const bool is_async, const bool enable_streaming_enqueue,
         const int in_flight_nodes_limit) {
        TFE_Executor* exc = TFE_NewExecutor(is_async, enable_streaming_enqueue,
                                            in_flight_nodes_limit);
        return exc;
      },
      py::return_value_policy::reference);
  m.def("TFE_DeleteExecutor", &TFE_DeleteExecutor);
  m.def("TFE_ExecutorIsAsync", &TFE_ExecutorIsAsync);
  m.def("TFE_ExecutorWaitForAllPendingNodes", [](TFE_Executor& exc) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // NOTE: release Python GIL for pending PyFunc ops to be executed properly.
    Py_BEGIN_ALLOW_THREADS;
    TFE_ExecutorWaitForAllPendingNodes(&exc, status.get());
    Py_END_ALLOW_THREADS;
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ExecutorClearError", &TFE_ExecutorClearError);
  m.def("TFE_ContextSetExecutorForThread", [](py::handle& ctx,
                                              TFE_Executor& exc) {
    TFE_ContextSetExecutorForThread(tensorflow::InputTFE_Context(ctx), &exc);
  });
  m.def(
      "TFE_ContextGetExecutorForThread",
      [](py::handle& o) {
        return TFE_ContextGetExecutorForThread(tensorflow::InputTFE_Context(o));
      },
      py::return_value_policy::reference);

  m.def("TFE_OpNameGetAttrType",
        [](py::handle& ctx, const char* op_or_function_name,
           const char* attr_name) {
          int temp = 0;
          unsigned char* is_list = reinterpret_cast<unsigned char*>(&temp);
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          auto output = TFE_OpNameGetAttrType(tensorflow::InputTFE_Context(ctx),
                                              op_or_function_name, attr_name,
                                              is_list, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
#if PY_MAJOR_VERSION < 3
          PyObject* output_pyo = PyInt_FromLong(output);
#else
          PyObject* output_pyo = PyLong_FromLong(output);
#endif
          if (*is_list == 1) {
            PyObject* list = PyList_New(1);
            PyList_SetItem(list, 0, output_pyo);
            return tensorflow::PyoOrThrow(list);
          }
          return tensorflow::PyoOrThrow(output_pyo);
        });
  m.def("TFE_Py_InitEagerTensor", [](const py::handle& o) {
    return tensorflow::PyoOrThrow(TFE_Py_InitEagerTensor(o.ptr()));
  });
  m.def("TFE_Py_PackEagerTensors",
        [](const py::handle& context, const py::handle& handles) {
          return tensorflow::TFE_Py_PackEagerTensors_wrapper(context, handles);
        });
  m.def("TFE_Py_SetEagerTensorProfiler", &TFE_Py_SetEagerTensorProfiler);
  m.def("TFE_Py_RegisterJVPFunction", [](const py::handle& o) {
    return tensorflow::PyoOrThrow(TFE_Py_RegisterJVPFunction(o.ptr()));
  });
  m.def("TFE_Py_RegisterGradientFunction", [](const py::handle& o) {
    return tensorflow::PyoOrThrow(TFE_Py_RegisterGradientFunction(o.ptr()));
  });
  m.def("TFE_Py_Execute",
        [](const py::handle& context, const char* device_name,
           const char* op_name, const py::handle& inputs,
           const py::handle& attrs, const py::handle& num_outputs) {
          return tensorflow::TFE_Py_ExecuteCancelable_wrapper(
              context, device_name, op_name, inputs, attrs.ptr(), nullptr,
              num_outputs);
        });
  m.def(
      "TFE_Py_ExecuteCancelable",
      [](const py::handle& context, const char* device_name,
         const char* op_name, const py::handle& inputs, const py::handle& attrs,
         tensorflow::CancellationManager& cancellation_manager,
         const py::handle& num_outputs) {
        return tensorflow::TFE_Py_ExecuteCancelable_wrapper(
            context, device_name, op_name, inputs, attrs.ptr(),
            &cancellation_manager, num_outputs);
      });
  m.def("TFE_Py_FastPathExecute", [](const py::args args) {
    // TFE_Py_FastPathExecute requires error checking prior to returning.
    return tensorflow::PyoOrThrow(TFE_Py_FastPathExecute_C(args.ptr()));
  });
  m.def("TFE_Py_RecordGradient",
        [](const py::handle& op_name, const py::handle& inputs,
           const py::handle& attrs, const py::handle& results,
           const py::handle& forward_pass_name_scope) {
          return tensorflow::PyoOrThrow(TFE_Py_RecordGradient(
              op_name.ptr(), inputs.ptr(), attrs.ptr(), results.ptr(),
              forward_pass_name_scope.ptr()));
        });
  m.def("TFE_Py_UID", []() { return tensorflow::PyoOrThrow(TFE_Py_UID()); });

  // TFE_Py_Tape Logic
  m.def("TFE_Py_TapeSetNew", [](const py::handle& persistent,
                                const py::handle& watch_accessed_variables) {
    return tensorflow::PyoOrThrow(
        TFE_Py_TapeSetNew(persistent.ptr(), watch_accessed_variables.ptr()));
  });
  m.def("TFE_Py_TapeSetAdd",
        [](const py::handle& tape) { TFE_Py_TapeSetAdd(tape.ptr()); });
  m.def("TFE_Py_TapeSetRemove",
        [](const py::handle& tape) { TFE_Py_TapeSetRemove(tape.ptr()); });
  m.def("TFE_Py_TapeSetStopOnThread", &TFE_Py_TapeSetStopOnThread);
  m.def("TFE_Py_TapeSetRestartOnThread", &TFE_Py_TapeSetRestartOnThread);
  m.def("TFE_Py_TapeSetIsStopped",
        []() { return tensorflow::PyoOrThrow(TFE_Py_TapeSetIsStopped()); });
  m.def("TFE_Py_TapeSetIsEmpty",
        []() { return tensorflow::PyoOrThrow(TFE_Py_TapeSetIsEmpty()); });
  m.def("TFE_Py_TapeSetShouldRecordBackprop", [](const py::handle& tensors) {
    return tensorflow::PyoOrThrow(
        TFE_Py_TapeSetShouldRecordBackprop(tensors.ptr()));
  });
  m.def("TFE_Py_TapeSetPossibleGradientTypes", [](const py::handle& tensors) {
    return tensorflow::PyoOrThrow(
        TFE_Py_TapeSetPossibleGradientTypes(tensors.ptr()));
  });
  m.def("TFE_Py_TapeSetDeleteTrace", &TFE_Py_TapeSetDeleteTrace);
  m.def("TFE_Py_TapeSetRecordOperation",
        [](const py::handle& op_type, const py::handle& output_tensors,
           const py::handle& input_tensors, const py::handle& backward_function,
           const py::handle& forward_function) {
          return tensorflow::PyoOrThrow(TFE_Py_TapeSetRecordOperation(
              op_type.ptr(), output_tensors.ptr(), input_tensors.ptr(),
              backward_function.ptr(), forward_function.ptr()));
        });
  m.def(
      "TFE_Py_TapeSetRecordOperationBackprop",
      [](const py::handle& op_type, const py::handle& output_tensors,
         const py::handle& input_tensors, const py::handle& backward_function) {
        return tensorflow::PyoOrThrow(TFE_Py_TapeSetRecordOperationBackprop(
            op_type.ptr(), output_tensors.ptr(), input_tensors.ptr(),
            backward_function.ptr()));
      });
  m.def(
      "TFE_Py_TapeSetRecordOperationForwardprop",
      [](const py::handle& op_type, const py::handle& output_tensors,
         const py::handle& input_tensors, const py::handle& backward_function,
         const py::handle& forwardprop_output_indices) {
        return tensorflow::PyoOrThrow(TFE_Py_TapeSetRecordOperationForwardprop(
            op_type.ptr(), output_tensors.ptr(), input_tensors.ptr(),
            backward_function.ptr(), forwardprop_output_indices.ptr()));
      });
  m.def("TFE_Py_TapeGradient",
        [](const py::handle& tape, const py::handle& target,
           const py::handle& sources, const py::handle& output_gradients,
           const py::handle& sources_raw,
           const py::handle& unconnected_gradients) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          PyObject* output = TFE_Py_TapeGradient(
              tape.ptr(), target.ptr(), sources.ptr(), output_gradients.ptr(),
              sources_raw.ptr(), unconnected_gradients.ptr(), status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return tensorflow::PyoOrThrow(output);
        });

  m.def("TFE_Py_TapeVariableAccessed", [](const py::handle& variable) {
    TFE_Py_TapeVariableAccessed(variable.ptr());
  });
  m.def("TFE_Py_TapeWatch",
        [](const py::handle& tape, const py::handle& tensor) {
          TFE_Py_TapeWatch(tape.ptr(), tensor.ptr());
        });
  m.def("TFE_Py_TapeWatchVariable",
        [](const py::handle& tape, const py::handle& variable) {
          TFE_Py_TapeWatchVariable(tape.ptr(), variable.ptr());
        });
  m.def("TFE_Py_TapeWatchedVariables", [](const py::handle& tape) {
    return tensorflow::PyoOrThrow(TFE_Py_TapeWatchedVariables(tape.ptr()));
  });

  // TFE_Py_VariableWatcher logic.
  m.def("TFE_Py_VariableWatcherNew",
        []() { return tensorflow::PyoOrThrow(TFE_Py_VariableWatcherNew()); });
  m.def("TFE_Py_VariableWatcherRemove", [](const py::handle& variable_watcher) {
    TFE_Py_VariableWatcherRemove(variable_watcher.ptr());
  });
  m.def("TFE_Py_VariableWatcherVariableAccessed",
        [](const py::handle& variable) {
          TFE_Py_VariableWatcherVariableAccessed(variable.ptr());
        });
  m.def("TFE_Py_VariableWatcherWatchedVariables",
        [](const py::handle& variable_watcher) {
          return tensorflow::PyoOrThrow(
              TFE_Py_VariableWatcherWatchedVariables(variable_watcher.ptr()));
        });

  // TFE_Py_ForwardAccumulator logic.
  m.def("TFE_Py_ForwardAccumulatorNew", [](bool use_batch) {
    return tensorflow::PyoOrThrow(TFE_Py_ForwardAccumulatorNew(use_batch));
  });

  m.def("TFE_Py_ForwardAccumulatorSetAdd", [](const py::handle& accumulator) {
    return tensorflow::PyoOrThrow(
        TFE_Py_ForwardAccumulatorSetAdd(accumulator.ptr()));
  });
  m.def("TFE_Py_ForwardAccumulatorSetRemove",
        [](const py::handle& accumulator) {
          TFE_Py_ForwardAccumulatorSetRemove(accumulator.ptr());
        });

  m.def("TFE_Py_ForwardAccumulatorWatch",
        [](const py::handle& accumulator, const py::handle& tensor,
           const py::handle& tangent) {
          TFE_Py_ForwardAccumulatorWatch(accumulator.ptr(), tensor.ptr(),
                                         tangent.ptr());
        });
  m.def("TFE_Py_ForwardAccumulatorJVP",
        [](const py::handle& accumulator, const py::handle& tensor) {
          return tensorflow::PyoOrThrow(
              TFE_Py_ForwardAccumulatorJVP(accumulator.ptr(), tensor.ptr()));
        });
  m.def("TFE_Py_ForwardAccumulatorPushState", []() {
    return tensorflow::PyoOrThrow(TFE_Py_ForwardAccumulatorPushState());
  });
  m.def("TFE_Py_ForwardAccumulatorPopState", []() {
    return tensorflow::PyoOrThrow(TFE_Py_ForwardAccumulatorPopState());
  });
  m.def("TFE_Py_PackJVPs", [](const py::handle& tensors) {
    return tensorflow::PyoOrThrow(TFE_Py_PackJVPs(tensors.ptr()));
  });

  // TFE_ContextOptions Logic
  m.def("TFE_NewContextOptions", &TFE_NewContextOptions,
        py::return_value_policy::reference);
  m.def("TFE_ContextOptionsSetConfig", [](TFE_ContextOptions* options,
                                          py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(tensorflow::ProtoStringToTFBuffer(proto.ptr()));
    TFE_ContextOptionsSetConfig(options, buf.get()->data, buf.get()->length,
                                status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_ContextOptionsSetDevicePlacementPolicy",
        &TFE_ContextOptionsSetDevicePlacementPolicy);
  m.def("TFE_ContextOptionsSetTfrt", &TFE_ContextOptionsSetTfrt);
  m.def("TFE_ContextOptionsSetTfrtDistributedRuntime",
        &TFE_ContextOptionsSetTfrtDistributedRuntime);
  // Experimental feature, intentionally not exposed as a C API yet.
  m.def("TFE_ContextOptionsSetRunEagerOpAsFunction",
        [](TFE_ContextOptions* options, bool run_eager_op_as_function) {
          options->run_eager_op_as_function = run_eager_op_as_function;
        });
  m.def("TFE_ContextOptionsSetJitCompileRewrite",
        [](TFE_ContextOptions* options, bool jit_compile_rewrite) {
          options->jit_compile_rewrite = jit_compile_rewrite;
        });
  m.def("TFE_ContextOptionsSetAsync", &TFE_ContextOptionsSetAsync);
  m.def("TFE_DeleteContextOptions", &TFE_DeleteContextOptions,
        py::return_value_policy::reference);

  // TFE_Py_TensorShape Logic
  m.def("TFE_Py_TensorShapeSlice",
        [](const py::handle& tensors, int slice_dim) {
          return tensorflow::PyoOrThrow(
              TFE_Py_TensorShapeSlice(tensors.ptr(), slice_dim));
        });
  m.def("TFE_Py_TensorShapeOnDevice", [](const py::handle& tensors,
                                         int slice_dim) {
    return tensorflow::PyoOrThrow(TFE_Py_TensorShapeOnDevice(tensors.ptr()));
  });
  m.def("TFE_Py_EnableInteractivePythonLogging",
        &TFE_Py_EnableInteractivePythonLogging);

  // Additional Context Logic
  m.def("TFE_Py_SetEagerContext", [](const py::handle& o) {
    return tensorflow::PyoOrThrow(TFE_Py_SetEagerContext(o.ptr()));
  });
  m.def("TFE_Py_SetCEagerContext", [](const py::handle& ctx) {
    // TODO(mdan): This cast might need rewriting to ImmediateExecutionContext.
    tensorflow::SetCEagerContext(reinterpret_cast<tensorflow::EagerContext*>(
        tensorflow::InputTFE_Context(ctx)));
  });
  m.def("TFE_Py_RegisterVSpace", [](const py::handle& o) {
    return tensorflow::PyoOrThrow(TFE_Py_RegisterVSpace(o.ptr()));
  });
  m.def("TFE_EnableCollectiveOps", [](const py::handle& ctx, py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(tensorflow::ProtoStringToTFBuffer(proto.ptr()));
    TFE_EnableCollectiveOps(tensorflow::InputTFE_Context(ctx), buf.get()->data,
                            buf.get()->length, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("TFE_AbortCollectiveOps", [](const py::handle& ctx, int code,
                                     const char* message) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TF_SetStatus(status.get(), static_cast<TF_Code>(code), message);
    TFE_AbortCollectiveOps(tensorflow::InputTFE_Context(ctx), status.get());
  });
  m.def("TFE_CollectiveOpsCheckPeerHealth",
        [](const py::handle& ctx, const char* task, int64_t timeout_in_ms) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TFE_CollectiveOpsCheckPeerHealth(tensorflow::InputTFE_Context(ctx),
                                           task, timeout_in_ms, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });
  m.def("TF_ListPhysicalDevices", &tensorflow::TF_ListPhysicalDevices);
  m.def("TF_ListPluggablePhysicalDevices",
        &tensorflow::TF_ListPluggablePhysicalDevices);
  m.def("TF_GetDeviceDetails", &tensorflow::TF_GetDeviceDetails);
  m.def("TF_DeleteDeviceList", &TF_DeleteDeviceList,
        py::return_value_policy::reference);
  m.def("TF_DeviceListCount", &TF_DeviceListCount);
  m.def("TF_DeviceListName", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    auto output = TF_DeviceListName(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });
  m.def("TF_DeviceListType", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    auto output = TF_DeviceListType(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_PickUnusedPortOrDie", &TF_PickUnusedPortOrDie);

  // TFE_MonitoringCounter Logic
  m.def("TFE_MonitoringCounterCellIncrementBy",
        &TFE_MonitoringCounterCellIncrementBy);
  m.def("TFE_MonitoringCounterCellValue", &TFE_MonitoringCounterCellValue);
  m.def(
      "TFE_MonitoringNewCounter0",
      [](const char* name, const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewCounter0(name, status.get(), description);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteCounter0", &TFE_MonitoringDeleteCounter0,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellCounter0", &TFE_MonitoringGetCellCounter0,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewCounter1",
      [](const char* name, const char* description, const char* label1) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewCounter1(name, status.get(), description, label1);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteCounter1", &TFE_MonitoringDeleteCounter1,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellCounter1", &TFE_MonitoringGetCellCounter1,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewCounter2",
      [](const char* name, const char* description, const char* label1,
         const char* label2) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewCounter2(name, status.get(), description,
                                                label1, label2);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteCounter2", &TFE_MonitoringDeleteCounter2,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellCounter2", &TFE_MonitoringGetCellCounter2,
        py::return_value_policy::reference);

  // TFE_MonitoringIntGauge Logic
  m.def("TFE_MonitoringIntGaugeCellSet", &TFE_MonitoringIntGaugeCellSet);
  m.def("TFE_MonitoringIntGaugeCellValue", &TFE_MonitoringIntGaugeCellValue);
  m.def(
      "TFE_MonitoringNewIntGauge0",
      [](const char* name, const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewIntGauge0(name, status.get(), description);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteIntGauge0", &TFE_MonitoringDeleteIntGauge0,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellIntGauge0", &TFE_MonitoringGetCellIntGauge0,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewIntGauge1",
      [](const char* name, const char* description, const char* label1) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewIntGauge1(name, status.get(), description, label1);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteIntGauge1", &TFE_MonitoringDeleteIntGauge1,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellIntGauge1", &TFE_MonitoringGetCellIntGauge1,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewIntGauge2",
      [](const char* name, const char* description, const char* label1,
         const char* label2) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewIntGauge2(name, status.get(),
                                                 description, label1, label2);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteIntGauge2", &TFE_MonitoringDeleteIntGauge2,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellIntGauge2", &TFE_MonitoringGetCellIntGauge2,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringStringGaugeCellSet", &TFE_MonitoringStringGaugeCellSet);
  m.def("TFE_MonitoringStringGaugeCellValue",
        &TFE_MonitoringStringGaugeCellValue);
  m.def(
      "TFE_MonitoringNewStringGauge0",
      [](const char* name, const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewStringGauge0(name, status.get(), description);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  // TFE_MonitoringStringGauge Logic
  m.def("TFE_MonitoringDeleteStringGauge0", &TFE_MonitoringDeleteStringGauge0);
  m.def("TFE_MonitoringGetCellStringGauge0", &TFE_MonitoringGetCellStringGauge0,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewStringGauge1",
      [](const char* name, const char* description, const char* label1) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewStringGauge1(name, status.get(),
                                                    description, label1);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteStringGauge1", &TFE_MonitoringDeleteStringGauge1);
  m.def("TFE_MonitoringGetCellStringGauge1", &TFE_MonitoringGetCellStringGauge1,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewStringGauge2",
      [](const char* name, const char* description, const char* label1,
         const char* label2) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewStringGauge2(
            name, status.get(), description, label1, label2);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteStringGauge2", &TFE_MonitoringDeleteStringGauge2);
  m.def("TFE_MonitoringGetCellStringGauge2", &TFE_MonitoringGetCellStringGauge2,
        py::return_value_policy::reference);

  m.def(
      "TFE_MonitoringNewStringGauge3",
      [](const char* name, const char* description, const char* label1,
         const char* label2, const char* label3) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewStringGauge3(
            name, status.get(), description, label1, label2, label3);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteStringGauge3", &TFE_MonitoringDeleteStringGauge3);
  m.def("TFE_MonitoringGetCellStringGauge3", &TFE_MonitoringGetCellStringGauge3,
        py::return_value_policy::reference);

  m.def(
      "TFE_MonitoringNewStringGauge4",
      [](const char* name, const char* description, const char* label1,
         const char* label2, const char* label3, const char* label4) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewStringGauge4(
            name, status.get(), description, label1, label2, label3, label4);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteStringGauge4", &TFE_MonitoringDeleteStringGauge4);
  m.def("TFE_MonitoringGetCellStringGauge4", &TFE_MonitoringGetCellStringGauge4,
        py::return_value_policy::reference);

  // TFE_MonitoringBoolGauge Logic
  m.def("TFE_MonitoringBoolGaugeCellSet", &TFE_MonitoringBoolGaugeCellSet);
  m.def("TFE_MonitoringBoolGaugeCellValue", &TFE_MonitoringBoolGaugeCellValue);
  m.def(
      "TFE_MonitoringNewBoolGauge0",
      [](const char* name, const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewBoolGauge0(name, status.get(), description);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteBoolGauge0", &TFE_MonitoringDeleteBoolGauge0,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellBoolGauge0", &TFE_MonitoringGetCellBoolGauge0,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewBoolGauge1",
      [](const char* name, const char* description, const char* label1) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewBoolGauge1(name, status.get(),
                                                  description, label1);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteBoolGauge1", &TFE_MonitoringDeleteBoolGauge1,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellBoolGauge1", &TFE_MonitoringGetCellBoolGauge1,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewBoolGauge2",
      [](const char* name, const char* description, const char* label1,
         const char* label2) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewBoolGauge2(name, status.get(),
                                                  description, label1, label2);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteBoolGauge2", &TFE_MonitoringDeleteBoolGauge2,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellBoolGauge2", &TFE_MonitoringGetCellBoolGauge2,
        py::return_value_policy::reference);

  // TFE_MonitoringSampler Logic
  m.def("TFE_MonitoringSamplerCellAdd", &TFE_MonitoringSamplerCellAdd);
  m.def("TFE_MonitoringSamplerCellValue", &TFE_MonitoringSamplerCellValue);
  m.def("TFE_MonitoringNewExponentialBuckets",
        &TFE_MonitoringNewExponentialBuckets,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteBuckets", &TFE_MonitoringDeleteBuckets,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewSampler0",
      [](const char* name, TFE_MonitoringBuckets* buckets,
         const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TFE_MonitoringNewSampler0(name, buckets, status.get(), description);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteSampler0", &TFE_MonitoringDeleteSampler0,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellSampler0", &TFE_MonitoringGetCellSampler0,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewSampler1",
      [](const char* name, TFE_MonitoringBuckets* buckets,
         const char* description, const char* label1) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewSampler1(name, buckets, status.get(),
                                                description, label1);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteSampler1", &TFE_MonitoringDeleteSampler1,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellSampler1", &TFE_MonitoringGetCellSampler1,
        py::return_value_policy::reference);
  m.def(
      "TFE_MonitoringNewSampler2",
      [](const char* name, TFE_MonitoringBuckets* buckets,
         const char* description, const char* label1, const char* label2) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TFE_MonitoringNewSampler2(name, buckets, status.get(),
                                                description, label1, label2);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);
  m.def("TFE_MonitoringDeleteSampler2", &TFE_MonitoringDeleteSampler2,
        py::return_value_policy::reference);
  m.def("TFE_MonitoringGetCellSampler2", &TFE_MonitoringGetCellSampler2,
        py::return_value_policy::reference);

  // TFE_CancellationManager Logic
  m.def("TFE_NewCancellationManager",
        []() { return new tensorflow::CancellationManager(); });
  m.def("TFE_CancellationManagerIsCancelled",
        &tensorflow::CancellationManager::IsCancelled);
  m.def("TFE_CancellationManagerStartCancel",
        &tensorflow::CancellationManager::StartCancel);

  m.def("TFE_ClearScalarCache", &tensorflow::TFE_ClearScalarCache);

  // Util buffer helper functions
  m.def("TF_NewBufferFromString", &TF_NewBufferFromString,
        py::return_value_policy::reference);

  // DLPack functions
  m.def("TFE_ToDlpackCapsule", [](py::handle& o) {
    PyObject* eager_tensor_pyobject_ptr = o.ptr();
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());

    if (!EagerTensor_CheckExact(eager_tensor_pyobject_ptr)) {
      status->status = tensorflow::errors::InvalidArgument(
          "The argument to `to_dlpack` must be a TF tensor, not Python object");
      tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    }

    TFE_TensorHandle* thandle = EagerTensor_Handle(eager_tensor_pyobject_ptr);
    void* dlm_ptr = tensorflow::TFE_HandleToDLPack(thandle, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    py::capsule capsule(
        dlm_ptr, tensorflow::kDlTensorCapsuleName, [](PyObject* capsule) {
          if (PyCapsule_IsValid(capsule, tensorflow::kDlTensorCapsuleName)) {
            void* dlm_rptr =
                PyCapsule_GetPointer(capsule, tensorflow::kDlTensorCapsuleName);
            if (dlm_rptr) {
              tensorflow::TFE_CallDLManagedTensorDeleter(dlm_rptr);
              PyCapsule_SetDestructor(capsule, nullptr);
            }
          }
        });
    return capsule;
  });

  m.def("TFE_FromDlpackCapsule", [](const py::capsule& pycapsule,
                                    const py::handle& context) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    if (absl::string_view(pycapsule.name()) !=
        tensorflow::kDlTensorCapsuleName) {
      status->status = tensorflow::errors::InvalidArgument(
          "DLPack tensor must be a capsule with name \"dltensor\", got \"%s\". "
          "Note that a DLPack tensor may be consumed at most once.",
          absl::string_view(pycapsule.name()));
      tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    }

    TFE_TensorHandle* thandle = tensorflow::TFE_HandleFromDLPack(
        pycapsule, status.get(), tensorflow::InputTFE_Context(context));

    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    PyCapsule_SetName(pycapsule.ptr(), "used_dltensor");
    PyCapsule_SetDestructor(pycapsule.ptr(), nullptr);

    PyObject* pyhandle = EagerTensorFromHandle(thandle);
    return tensorflow::PyoOrThrow(pyhandle);
  });

  m.def("TFE_Py_IsCustomDevice",
        [](const py::handle& context, const char* device_name) {
          return TFE_IsCustomDevice(tensorflow::InputTFE_Context(context),
                                    device_name);
        });

  m.def("TFE_Py_RegisterCustomDevice", [](const py::handle& context,
                                          const py::capsule& device,
                                          const char* device_name,
                                          const py::capsule& device_info) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    if (absl::string_view(device.name()) != "TFE_CustomDevice") {
      status->status = tensorflow::errors::InvalidArgument(
          "Expected a capsule named 'TFE_CustomDevice' for the `device` "
          "argument, got ",
          absl::string_view(device.name()));
      tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    }
    if (absl::string_view(device_info.name()) !=
        "TFE_CustomDevice_DeviceInfo") {
      status->status = tensorflow::errors::InvalidArgument(
          "Expected a capsule named 'TFE_CustomDevice_DeviceInfo' for "
          "the `device_info` argument, got ",
          absl::string_view(device_info.name()));
      tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    }
    // TFE_RegisterCustomDevice takes ownership
    PyCapsule_SetDestructor(device_info.ptr(), nullptr);
    TFE_RegisterCustomDevice(
        tensorflow::InputTFE_Context(context),
        *reinterpret_cast<TFE_CustomDevice*>(
            PyCapsule_GetPointer(device.ptr(), "TFE_CustomDevice")),
        device_name,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });

  py::class_<EagerContextThreadLocalDataWrapper>(m,
                                                 "EagerContextThreadLocalData")
      .def(py::init<py::handle, py::handle, py::handle>(),
           py::arg("py_eager_context"), py::arg("is_eager"),
           py::arg("device_spec"))
      .def_property("is_eager",
                    &EagerContextThreadLocalDataWrapper::get_is_eager,
                    &EagerContextThreadLocalDataWrapper::set_is_eager)
      .def_property(
          "invoking_op_callbacks",
          &EagerContextThreadLocalDataWrapper::get_invoking_op_callbacks,
          &EagerContextThreadLocalDataWrapper::set_invoking_op_callbacks)
      .def_property("device_name",
                    &EagerContextThreadLocalDataWrapper::get_device_name,
                    &EagerContextThreadLocalDataWrapper::set_device_name)
      .def_property("scope_name",
                    &EagerContextThreadLocalDataWrapper::get_scope_name,
                    &EagerContextThreadLocalDataWrapper::set_scope_name)
      .def_property("device_spec",
                    &EagerContextThreadLocalDataWrapper::get_device_spec,
                    &EagerContextThreadLocalDataWrapper::set_device_spec)
      .def_property(
          "function_call_options",
          &EagerContextThreadLocalDataWrapper::get_function_call_options,
          &EagerContextThreadLocalDataWrapper::set_function_call_options)
      .def_property("executor",
                    &EagerContextThreadLocalDataWrapper::get_executor,
                    &EagerContextThreadLocalDataWrapper::set_executor)
      .def_property("op_callbacks",
                    &EagerContextThreadLocalDataWrapper::get_op_callbacks,
                    &EagerContextThreadLocalDataWrapper::set_op_callbacks);

  // C API Enum

  py::enum_<TFE_ContextDevicePlacementPolicy>(
      m, "TFE_ContextDevicePlacementPolicy")
      .value("TFE_DEVICE_PLACEMENT_EXPLICIT", TFE_DEVICE_PLACEMENT_EXPLICIT)
      .value("TFE_DEVICE_PLACEMENT_WARN", TFE_DEVICE_PLACEMENT_WARN)
      .value("TFE_DEVICE_PLACEMENT_SILENT", TFE_DEVICE_PLACEMENT_SILENT)
      .value("TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32",
             TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32)
      .export_values();

  py::enum_<TF_AttrType>(m, "TF_AttrType")
      .value("TF_ATTR_STRING", TF_ATTR_STRING)
      .value("TF_ATTR_INT", TF_ATTR_INT)
      .value("TF_ATTR_FLOAT", TF_ATTR_FLOAT)
      .value("TF_ATTR_BOOL", TF_ATTR_BOOL)
      .value("TF_ATTR_TYPE", TF_ATTR_TYPE)
      .value("TF_ATTR_SHAPE", TF_ATTR_SHAPE)
      .value("TF_ATTR_TENSOR", TF_ATTR_TENSOR)
      .value("TF_ATTR_PLACEHOLDER", TF_ATTR_PLACEHOLDER)
      .value("TF_ATTR_FUNC", TF_ATTR_FUNC)
      .export_values();
};
