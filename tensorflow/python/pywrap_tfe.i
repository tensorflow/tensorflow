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

%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
%}


%include "tensorflow/c/tf_datatype.h"
%include "tensorflow/c/tf_status.h"

%ignoreall;

%rename("%s") TF_SetXlaEnableLazyCompilation;
%rename("%s") TF_SetTfXlaCpuGlobalJit;
%rename("%s") TF_SetXlaAutoJitMode;
%rename("%s") TF_SetXlaConstantFoldingDisabled;
%rename("%s") TF_GetXlaConstantFoldingDisabled;
%rename("%s") TF_SetXlaMinClusterSize;
%rename("%s") TFE_NewContext;
%rename("%s") TFE_DeleteContext;
%rename("%s") TFE_ContextListDevices;
%rename("%s") TFE_ContextAddFunction;
%rename("%s") TFE_ContextAddFunctionDef;
%rename("%s") TFE_ContextRemoveFunction;
%rename("%s") TFE_ContextHasFunction;
%rename("%s") TFE_ContextEnableRunMetadata;
%rename("%s") TFE_ContextDisableRunMetadata;
%rename("%s") TFE_ContextEnableGraphCollection;
%rename("%s") TFE_ContextDisableGraphCollection;
%rename("%s") TFE_ContextExportRunMetadata;
%rename("%s") TFE_ContextClearCaches;
%rename("%s") TFE_ContextGetDevicePlacementPolicy;
%rename("%s") TFE_ContextGetMirroringPolicy;
%rename("%s") TFE_ContextSetThreadLocalDevicePlacementPolicy;
%rename("%s") TFE_ContextSetThreadLocalMirroringPolicy;
%rename("%s") TFE_ContextSetServerDef;
%rename("%s") TFE_ContextUpdateServerDef;
%rename("%s") TFE_ContextCheckAlive;
%rename("%s") TFE_NewExecutor;
%rename("%s") TFE_DeleteExecutor;
%rename("%s") TFE_ExecutorIsAsync;
%rename("%s") TFE_ExecutorWaitForAllPendingNodes;
%rename("%s") TFE_ExecutorClearError;
%rename("%s") TFE_ContextSetExecutorForThread;
%rename("%s") TFE_ContextGetExecutorForThread;
%rename("%s") TFE_NewProfiler;
%rename("%s") TFE_ProfilerIsOk;
%rename("%s") TFE_DeleteProfiler;
%rename("%s") TFE_ProfilerSerializeToString;
%rename("%s") TFE_StartProfilerServer;
%rename("%s") TFE_ProfilerClientStartTracing;
%rename("%s") TFE_ProfilerClientMonitor;
%rename("%s") TFE_OpNameGetAttrType;
%rename("%s") TFE_Py_InitEagerTensor;
%rename("%s") TFE_Py_SetEagerTensorProfiler;
%rename("%s") TFE_Py_RegisterExceptionClass;
%rename("%s") TFE_Py_RegisterJVPFunction;
%rename("%s") TFE_Py_RegisterGradientFunction;
%rename("%s") TFE_Py_RegisterFallbackExceptionClass;
%rename("%s") TFE_Py_Execute;
%rename("%s") TFE_Py_ExecuteCancelable;
%rename("%s") TFE_Py_FastPathExecute;
%rename("%s") TFE_Py_RecordGradient;
%rename("%s") TFE_Py_UID;
%rename("%s") TFE_Py_TapeSetNew;
%rename("%s") TFE_Py_TapeSetAdd;
%rename("%s") TFE_Py_TapeSetRemove;
%rename("%s") TFE_Py_TapeSetStopOnThread;
%rename("%s") TFE_Py_TapeSetRestartOnThread;
%rename("%s") TFE_Py_TapeSetIsStopped;
%rename("%s") TFE_Py_TapeSetIsEmpty;
%rename("%s") TFE_Py_TapeSetShouldRecordBackprop;
%rename("%s") TFE_Py_TapeSetPossibleGradientTypes;
%rename("%s") TFE_Py_TapeSetDeleteTrace;
%rename("%s") TFE_Py_TapeSetRecordOperation;
%rename("%s") TFE_Py_TapeSetRecordOperationBackprop;
%rename("%s") TFE_Py_TapeSetRecordOperationForwardprop;
%rename("%s") TFE_Py_TapeGradient;
%rename("%s") TFE_Py_TapeVariableAccessed;
%rename("%s") TFE_Py_TapeWatch;
%rename("%s") TFE_Py_TapeWatchVariable;
%rename("%s") TFE_Py_TapeWatchedVariables;
%rename("%s") TFE_Py_ForwardAccumulatorNew;
%rename("%s") TFE_Py_ForwardAccumulatorSetAdd;
%rename("%s") TFE_Py_ForwardAccumulatorSetRemove;
%rename("%s") TFE_Py_ForwardAccumulatorWatch;
%rename("%s") TFE_Py_ForwardAccumulatorJVP;
%rename("%s") TFE_Py_ForwardAccumulatorPushState;
%rename("%s") TFE_Py_ForwardAccumulatorPopState;
%rename("%s") TFE_Py_PackJVPs;
%rename("%s") TFE_NewContextOptions;
%rename("%s") TFE_ContextOptionsSetConfig;
%rename("%s") TFE_ContextOptionsSetDevicePlacementPolicy;
%rename("%s") TFE_ContextOptionsSetMirroringPolicy;
%rename("%s") TFE_ContextOptionsSetAsync;
%rename("%s") TFE_ContextOptionsSetLazyRemoteInputsCopy;
%rename("%s") TFE_DeleteContextOptions;
%rename("%s") TFE_Py_TensorShapeSlice;
%rename("%s") TFE_Py_TensorShapeOnDevice;
%rename("%s") TFE_Py_EnableInteractivePythonLogging;
%rename("%s") TFE_Py_SetEagerContext;
%rename("%s") TFE_ContextStartStep;
%rename("%s") TFE_ContextEndStep;
%rename("%s") TFE_Py_RegisterVSpace;
%rename("%s") TFE_Py_EncodeArg;
%rename("%s") TFE_EnableCollectiveOps;
%rename("%s") TF_ListPhysicalDevices;
%rename("%s") TF_PickUnusedPortOrDie;
%rename("%s") TFE_MonitoringCounterCellIncrementBy;
%rename("%s") TFE_MonitoringCounterCellValue;
%rename("%s") TFE_MonitoringNewCounter0;
%rename("%s") TFE_MonitoringDeleteCounter0;
%rename("%s") TFE_MonitoringGetCellCounter0;
%rename("%s") TFE_MonitoringNewCounter1;
%rename("%s") TFE_MonitoringDeleteCounter1;
%rename("%s") TFE_MonitoringGetCellCounter1;
%rename("%s") TFE_MonitoringNewCounter2;
%rename("%s") TFE_MonitoringDeleteCounter2;
%rename("%s") TFE_MonitoringGetCellCounter2;
%rename("%s") TFE_MonitoringIntGaugeCellSet;
%rename("%s") TFE_MonitoringIntGaugeCellValue;
%rename("%s") TFE_MonitoringNewIntGauge0;
%rename("%s") TFE_MonitoringDeleteIntGauge0;
%rename("%s") TFE_MonitoringGetCellIntGauge0;
%rename("%s") TFE_MonitoringNewIntGauge1;
%rename("%s") TFE_MonitoringDeleteIntGauge1;
%rename("%s") TFE_MonitoringGetCellIntGauge1;
%rename("%s") TFE_MonitoringNewIntGauge2;
%rename("%s") TFE_MonitoringDeleteIntGauge2;
%rename("%s") TFE_MonitoringGetCellIntGauge2;
%rename("%s") TFE_MonitoringStringGaugeCellSet;
%rename("%s") TFE_MonitoringStringGaugeCellValue;
%rename("%s") TFE_MonitoringNewStringGauge0;
%rename("%s") TFE_MonitoringDeleteStringGauge0;
%rename("%s") TFE_MonitoringGetCellStringGauge0;
%rename("%s") TFE_MonitoringNewStringGauge1;
%rename("%s") TFE_MonitoringDeleteStringGauge1;
%rename("%s") TFE_MonitoringGetCellStringGauge1;
%rename("%s") TFE_MonitoringNewStringGauge2;
%rename("%s") TFE_MonitoringDeleteStringGauge2;
%rename("%s") TFE_MonitoringGetCellStringGauge2;
%rename("%s") TFE_MonitoringBoolGaugeCellSet;
%rename("%s") TFE_MonitoringBoolGaugeCellValue;
%rename("%s") TFE_MonitoringNewBoolGauge0;
%rename("%s") TFE_MonitoringDeleteBoolGauge0;
%rename("%s") TFE_MonitoringGetCellBoolGauge0;
%rename("%s") TFE_MonitoringNewBoolGauge1;
%rename("%s") TFE_MonitoringDeleteBoolGauge1;
%rename("%s") TFE_MonitoringGetCellBoolGauge1;
%rename("%s") TFE_MonitoringNewBoolGauge2;
%rename("%s") TFE_MonitoringDeleteBoolGauge2;
%rename("%s") TFE_MonitoringGetCellBoolGauge2;
%rename("%s") TFE_MonitoringSamplerCellAdd;
%rename("%s") TFE_MonitoringSamplerCellValue;
%rename("%s") TFE_MonitoringNewExponentialBuckets;
%rename("%s") TFE_MonitoringDeleteBuckets;
%rename("%s") TFE_MonitoringNewSampler0;
%rename("%s") TFE_MonitoringDeleteSampler0;
%rename("%s") TFE_MonitoringGetCellSampler0;
%rename("%s") TFE_MonitoringNewSampler1;
%rename("%s") TFE_MonitoringDeleteSampler1;
%rename("%s") TFE_MonitoringGetCellSampler1;
%rename("%s") TFE_MonitoringNewSampler2;
%rename("%s") TFE_MonitoringDeleteSampler2;
%rename("%s") TFE_MonitoringGetCellSampler2;
%rename("%s") TFE_NewCancellationManager;
%rename("%s") TFE_CancellationManagerIsCancelled;
%rename("%s") TFE_CancellationManagerStartCancel;
%rename("%s") TFE_DeleteCancellationManager;
%rename("%s") TF_ImportGraphDefOptionsSetValidateColocationConstraints;
%rename("%s") TFE_ClearScalarCache;

%{
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/util/util.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/common_runtime/device_factory.h"

static PyObject* TF_ListPhysicalDevices(TF_Status* status) {
  std::vector<string> devices;
  tensorflow::Status s = tensorflow::DeviceFactory::ListAllPhysicalDevices(&devices);
  tensorflow::Set_TF_Status_from_Status(status, s);
  if (!s.ok()) {
    Py_RETURN_NONE;
  };
  PyObject* result = PyList_New(devices.size());
  int i = 0;
  for (auto& dev : devices) {
    PyObject* dev_obj = PyBytes_FromStringAndSize(dev.data(), dev.size());
    PyList_SetItem(result, i, dev_obj);
    ++i;
  }
  return result;
}
%}
static PyObject* TF_ListPhysicalDevices(TF_Status* status);

%{
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"

static PyObject* TFE_ClearScalarCache() {
  tensorflow::TFE_TensorHandleCache::Get()->Clear();
  Py_RETURN_NONE;
}
%}
static PyObject* TFE_ClearScalarCache();

%typemap(in) (const void* proto) {
  char* c_string;
  Py_ssize_t py_size;
  // PyBytes_AsStringAndSize() does not copy but simply interprets the input
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }
  $1 = static_cast<void*>(c_string);
}

%typemap(in) int64_t {
  $1 = PyLong_AsLongLong($input);
}

%typemap(out) TF_DataType {
  $result = PyInt_FromLong($1);
}

%typemap(out) int64_t {
  $result = PyInt_FromLong($1);
}

%typemap(out) TF_AttrType {
  $result = PyInt_FromLong($1);
}

%typemap(in, numinputs=0) unsigned char* is_list (unsigned char tmp) {
  tmp = 0;
  $1 = &tmp;
}

%typemap(argout) unsigned char* is_list {
  if (*$1 == 1) {
    PyObject* list = PyList_New(1);
    PyList_SetItem(list, 0, $result);
    $result = list;
  }
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* serialized_function_def {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* device_name {
  if ($input == Py_None) {
    $1 = nullptr;
  } else {
    $1 = const_cast<char*>(TFE_GetPythonString($input));
  }
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* op_name {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* name {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}


// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* description {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* label {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* label1 {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* label2 {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* service_addr {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* logdir {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

// For const parameters in a function, SWIG pretty much ignores the const.
// See: http://www.swig.org/Doc2.0/SWIG.html#SWIG_nn13
// Hence the 'const_cast'.
%typemap(in) const char* worker_list {
  $1 = const_cast<char*>(TFE_GetPythonString($input));
}

%typemap(in) (TFE_Context*) {
  $1 = (TFE_Context*)PyCapsule_GetPointer($input, nullptr);

}
%typemap(out) (TFE_Context*) {
  // When the TFE_Context* returned is a nullptr, we expect the status is not
  // OK. This will raise an error (happens in another typemap).
  if ($1 != nullptr) {
    $result = PyCapsule_New($1, nullptr, TFE_DeleteContextCapsule);
  }
}

%rename("%s") TFE_ContextDevicePlacementPolicy;
%rename("%s") TFE_DEVICE_PLACEMENT_EXPLICIT;
%rename("%s") TFE_DEVICE_PLACEMENT_WARN;
%rename("%s") TFE_DEVICE_PLACEMENT_SILENT;
%rename("%s") TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32;

%rename("%s") TFE_ContextMirroringPolicy;
%rename("%s") TFE_MIRRORING_NONE;
%rename("%s") TFE_MIRRORING_ALL;

%include "tensorflow/c/eager/c_api.h"

%typemap(in) TFE_InputTensorHandles* inputs (TFE_InputTensorHandles temp) {
  $1 = &temp;
  if ($input != Py_None) {
    if (!PyList_Check($input)) {
      SWIG_exception_fail(SWIG_TypeError,
                          "must provide a list of Tensors as inputs");
    }
    Py_ssize_t len = PyList_Size($input);
    $1->resize(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PyList_GetItem($input, i);
      if (!elem) {
        SWIG_fail;
      }
      if (EagerTensor_CheckExact(elem)) {
        (*$1)[i] = EagerTensor_Handle(elem);
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
          (*$1)[i] = EagerTensor_Handle(value_attr.get());
        } else {
          // This is a subclass of EagerTensor that we don't support.
          PyErr_Clear();
          SWIG_exception_fail(
              SWIG_TypeError,
              tensorflow::strings::StrCat(
                  "Saw an object that is an instance of a strict subclass of "
                  "EagerTensor, which is not supported.  Item ",
                  i, " is type: ", elem->ob_type->tp_name)
                  .c_str());
        }
      } else if (tensorflow::swig::IsTensor(elem)) {
        // If it isnt an EagerTensor, but is still a Tensor, it must be a graph
        // tensor.
        tensorflow::Safe_PyObjectPtr name_attr(
            PyObject_GetAttrString(elem, "name"));
        SWIG_exception_fail(
            SWIG_TypeError,
            tensorflow::strings::StrCat(
                "An op outside of the function building code is being passed\n"
                "a \"Graph\" tensor. It is possible to have Graph tensors\n"
                "leak out of the function building context by including a\n"
                "tf.init_scope in your function building code.\n"
                "For example, the following function will fail:\n",
                "  @tf.function\n",
                "  def has_init_scope():\n",
                "    my_constant = tf.constant(1.)\n",
                "    with tf.init_scope():\n",
                "      added = my_constant * 2\n",
                "The graph tensor has name: ",
                name_attr ? TFE_GetPythonString(name_attr.get()) : "<unknown>"
            ).c_str());
      } else {
        SWIG_exception_fail(
            SWIG_TypeError,
            tensorflow::strings::StrCat(
                "provided list of inputs contains objects other "
                "than 'EagerTensor'. Item ",
                i, " is type: ", elem->ob_type->tp_name).c_str());
      }
    }
  }
}

// Temporary for the argout
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp) {
  if (!PyInt_Check($input)) {
    SWIG_exception_fail(SWIG_TypeError,
                        "expected an integer value (size of the number of "
                        "outputs of the operation)");
  }
  $1 = &temp;
  long sz = PyInt_AsLong($input);
  if (sz > 0) {
    $1->resize(PyInt_AsLong($input), nullptr);
  }
}

// Create new Status object.
%typemap(in, numinputs=0) TF_Status *out_status {
  $1 = GetStatus();
}

%typemap(freearg) (TF_Status* out_status) {
 ReturnStatus($1);
}

%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status) {
  if (MaybeRaiseExceptionFromTFStatus($2, nullptr)) {
    SWIG_fail;
  } else {
    int num_outputs = $1->size();
    Py_CLEAR($result);
    $result = PyList_New(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      PyObject *output;
      output = EagerTensorFromHandle($1->at(i));
      PyList_SetItem($result, i, output);
    }
  }
}

// SWIG usually unwraps the tuple that the native Python/C interface generates.
// Since we wanted to have a function with a variable length of arguments, we
// used the native Python/C interface directly (which by default supports
// passing all arguments as a tuple).
%native(TFE_Py_FastPathExecute) TFE_Py_FastPathExecute_C;

%include "tensorflow/python/eager/pywrap_tfe.h"
%include "tensorflow/c/c_api_experimental.h"
%include "tensorflow/c/eager/c_api_experimental.h"

// Clear all typemaps.
%typemap(out) TF_DataType;
%typemap(in) int64_t;
%typemap(out) int64_t;
%typemap(out) TF_AttrType;
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(argout) unsigned char* is_list;
%typemap(in) const char* description;
%typemap(in) const char* label1;
%typemap(in) const char* label2;
%typemap(in) (TFE_Context*);
%typemap(out) (TFE_Context*);
%typemap(in) TFE_OutputTensorHandles* outputs (TFE_OutputTensorHandles temp);
%typemap(in, numinputs=0) TF_Status *out_status;
%typemap(freearg) (TF_Status* out_status);
%typemap(argout) (TFE_OutputTensorHandles* outputs, TF_Status* out_status);
%typemap(in) (const void* proto);

%unignoreall
