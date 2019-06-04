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

#ifndef TENSORFLOW_PYTHON_EAGER_PYWRAP_TFE_H_
#define TENSORFLOW_PYTHON_EAGER_PYWRAP_TFE_H_

// Place `<locale>` before <Python.h> to avoid build failure in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

typedef tensorflow::gtl::InlinedVector<TFE_TensorHandle*, 4>
    TFE_InputTensorHandles;
typedef tensorflow::gtl::InlinedVector<TFE_TensorHandle*, 2>
    TFE_OutputTensorHandles;

// Execute a TensorFlow operation.
//
// 'device_name': Name of the device on which to execute the operation, or NULL
//                for automatic selection.
// 'op_name': Name of the TensorFlow op to execute.
// 'inputs': An array of TFE_TensorHandle*'s of size 'num_inputs'. These tensors
//           will be provided as input to the operation.
// 'attrs': A Python tuple alternating names and attr values.
// 'outputs': A pointer to a TFE_OutputTensorHandles in which outputs will
//            placed. On success, its elements will be filled in and the
//            caller takes ownership of each returned TFE_TensorHandle.
//            'outputs' MUST be sized to be at least as large as the number
//            of tensors produced by the operation and will be resized to
//            the actual number of tensors produced.
void TFE_Py_Execute(TFE_Context* ctx, const char* device_name,
                    const char* op_name, TFE_InputTensorHandles* inputs,
                    PyObject* attrs, TFE_OutputTensorHandles* outputs,
                    TF_Status* out_status);

// Registers e as the Exception class for handling not ok Status. Returns
// Py_None if registration succeeds, else throws a TypeError and returns NULL.
//
// This function is not thread-safe.
PyObject* TFE_Py_RegisterExceptionClass(PyObject* e);

// Registers e as the type of the ResourceVariable class.
// Returns Py_None if registration succeeds, else throws a TypeError and returns
// NULL.
//
// This function is not thread-safe.
PyObject* TFE_Py_RegisterResourceVariableType(PyObject* e);

// Registers e as the VSpace to use.
// `vspace` must be a imperative_grad.py:VSpace named tuple.
PyObject* TFE_Py_RegisterVSpace(PyObject* e);

// Registers e as the Exception to be raised when the conditions of
// TFE_Py_FastPathExecute_C have not been met. When this exception is set, it
// is a signal to the calling code that it should fall back to the safer (and
// more complete) code path.
//
// This function is not thread-safe.
PyObject* TFE_Py_RegisterFallbackExceptionClass(PyObject* e);

// Registers e as the gradient_function.
// The registered function takes
// (op_name, attrs, num_inputs, inputs, outputs, output_gradients) and returns
// the input gradients. This function will not correctly be able to generate
// gradients for functional ops - the gradients for those ops are calculated
// through a different codepath (see function.py for additional information).
//
// This function is not thread-safe.
PyObject* TFE_Py_RegisterGradientFunction(PyObject* e);

// Returns 0 if 'status' is TF_OK. Otherwise, raises an exception (using
// `exception` if not nullptr, else using the class registered via
// TFE_Py_RegisterExceptionClass), and returns -1.
int MaybeRaiseExceptionFromTFStatus(TF_Status* status, PyObject* exception);

// Returns 0 if 'status' is ok. Otherwise, raises an exception (using
// `exception` if not nullptr, else using the class registered via
// TFE_Py_RegisterExceptionClass), and returns -1.
int MaybeRaiseExceptionFromStatus(const tensorflow::Status& status,
                                  PyObject* exception);

// Returns the string associated with the passed-in python object.
const char* TFE_GetPythonString(PyObject* o);

// Returns a unique id on each call.
int64_t get_uid();

// Wraps the output of get_uid as a Python Long object. Ownership is passed to
// the caller.
PyObject* TFE_Py_UID();

// Deleter for Context objects, called from the Capsule that owns it.
void TFE_DeleteContextCapsule(PyObject* context);

// Returns true if o is an instance of EagerTensor, but not a subclass. Else
// returns false.
bool EagerTensor_CheckExact(const PyObject* o);

// Helper function to construct a new EagerTensor from a TFE_TensorHandle.
PyObject* EagerTensorFromHandle(TFE_TensorHandle* handle);

// Extracts the handle inside EagerTensor object `o`. Returns nullptr on error.
TFE_TensorHandle* EagerTensor_Handle(const PyObject* o);

// Creates the `EagerTensor` class by subclassing `base_class` and returns the
// newly created type, or nullptr on error.
PyObject* TFE_Py_InitEagerTensor(PyObject* base_class);

// Sets `profiler` as the current profiler to receive callbacks about events
// on eager tensors. Currently, the only reported event is creation.
// `profiler` is expected to have a `created(self, eager_tensor)` method that
// takes the created tensor as its single argument.
// Previous profiler, if any, is unset and will not receive any more
// callbacks.
// To unset the profiler, pass Py_None as the value of `profiler`.
PyObject* TFE_Py_SetEagerTensorProfiler(PyObject* profiler);

// Creates a new tape and adds it to the active set. `persistent` and
// `watch_accessed_variables` must be `PyBool_Type` (`Py_True` or `Py_False`).
PyObject* TFE_Py_TapeSetNew(PyObject* persistent,
                            PyObject* watch_accessed_variables);

// Removes the passed tape from the set of active tapes.
void TFE_Py_TapeSetRemove(PyObject* tape);

// Adds the passed tape to the set of active tapes.
void TFE_Py_TapeSetAdd(PyObject* tape);

// Returns true if the tape stack is empty.
PyObject* TFE_Py_TapeSetIsEmpty();

PyObject* TFE_Py_TapeSetShouldRecord(PyObject* tensors);
void TFE_Py_TapeWatch(PyObject* tape, PyObject* tensor);
void TFE_Py_TapeSetDeleteTrace(tensorflow::int64 tensor_id);

// Stops any gradient recording on the current thread.
void TFE_Py_TapeSetStopOnThread();

// Restarts gradient recording on the current thread.
void TFE_Py_TapeSetRestartOnThread();

// Records an operation in the gradient tape stack.type is a string for the
// operation type, used in the backprop code. output_tensors should be a list of
// python ops.Tensor objects. input_tensor_ids should be a list of python
// integers with the ids of the input tensors of the recorded
// operation. backward_function should be the function to be called during
// backprop to, given the gradients of the output tensors, produce the gradients
// of the input tensors.
void TFE_Py_TapeSetRecordOperation(PyObject* op_type, PyObject* output_tensors,
                                   PyObject* input_tensor_ids,
                                   PyObject* backward_function);

// Notifies all tapes that a variable has been accessed.
void TFE_Py_TapeVariableAccessed(PyObject* variable);

// Watches the given variable object on the given tape.
void TFE_Py_TapeWatchVariable(PyObject* tape, PyObject* variable);

// Computes a gradient based on information recorded on the tape.`tape` must
// have been produced by TFE_Py_NewTape. `target` and `sources` must be python
// lists of Tensor objects. `output_gradients` is either None or a python list
// of either Tensor or None, and if not None should have the same length as
// target.
PyObject* TFE_Py_TapeGradient(PyObject* tape, PyObject* target,
                              PyObject* sources, PyObject* output_gradients,
                              PyObject* sources_raw,
                              PyObject* unconnected_gradients,
                              TF_Status* status);

// Execute a tensorflow operation assuming that all provided inputs are
// correctly formatted (i.e. EagerTensors). If it doesn't find EagerTensors,
// it will simply fail with a NotImplementedError.
//
// The first PyObject* is unused.
// The "args" PyObject* is meant to be a tuple with the following structure:
//  Item 1: The TFE Context
//  Item 2: device_name: Name of the device on which to execute the operation,
//          or NULL for automatic selection.
//  Item 3: op_name: Name of the TensorFlow op to execute.
//  Item 4: name: An optional name for the operation.
//  Item 5: List representing all callbacks to execute after successful
//  op execute.
//  Item 6 onwards: inputs - This is a list of inputs followed by a list of
//        attrs. It is not necessary for type attrs to be present.
//
// This is named _C since there doesn't seem to be any way to make it visible
// in the SWIG interface without renaming due to the use of the %native
// directive.
PyObject* TFE_Py_FastPathExecute_C(PyObject*, PyObject* args);

// Record the gradient for a given op.
PyObject* TFE_Py_RecordGradient(PyObject* op_name, PyObject* inputs,
                                PyObject* attrs, PyObject* results,
                                PyObject* name);

// Returns all variables watched by the given tape in the order those variables
// were created.
PyObject* TFE_Py_TapeWatchedVariables(PyObject* tape);

// Returns an EagerTensor of dimension [len(`tensors`)] containing
// the `slice_dim`'th dimension of each tensor in `tensors`. In other words,
// TFE_Py_TensorShapeSlice takes a slice of dimensions of tensors in
// `tensors`. For example, if `tensors` contains tensors of with shapes
// [1, 2, 3], [4, 5], [6, 7, 8, 9], TFE_Py_TensorShapeSlice called with
// `slice_dim` equal to 1 will return [2, 5, 7].
// On error, returns nullptr and sets python exception.
// REQUIRES: `tensors` is a python list/tuple of EagerTensors
// REQUIRES: `slice_dim` is non-negative and smaller than the rank of all
//   tensors in `tensors`.
PyObject* TFE_Py_TensorShapeSlice(PyObject* tensors, int slice_dim);

// Returns the shape of this tensor's on-device representation.
// The shape is represented as a Python tuple of integers.
PyObject* TFE_Py_TensorShapeOnDevice(PyObject* tensor);

// Encodes the object as a tuple that is meant to be used as part of the key
// for the defun function cache.  If `include_tensor_ranks_only` is true,
// then the encoding only stores tensor ranks, and the key is
// agnostic to dimension sizes.  Otherwise, full tensor shape encodings are
// returned.
PyObject* TFE_Py_EncodeArg(PyObject*, bool include_tensor_ranks_only);

void TFE_Py_EnableInteractivePythonLogging();

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TFE_H_
