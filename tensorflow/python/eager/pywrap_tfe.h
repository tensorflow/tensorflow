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

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include <Python.h>

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
PyObject* TFE_Py_RegisterExceptionClass(PyObject* e);

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
char* TFE_GetPythonString(PyObject* o);

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

PyObject* TFE_Py_NewTape();
PyObject* TFE_Py_TapeShouldRecord(PyObject* py_tape, PyObject* tensors);
void TFE_Py_TapeWatch(PyObject* tape, tensorflow::int64 tensor_id);
void TFE_Py_TapeDeleteTrace(PyObject* tape, tensorflow::int64 tensor_id);

// Records an operation in the gradient tape. `tape` should point to an object
// returned by TFE_Py_NewTape. op_type is a string for the operation type, used
// in the backprop code. output_tensors should be a list of python ops.Tensor
// objects. input_tensor_ids should be a list of python integers with the ids of
// the input tensors of the recorded operation. backward_function should be the
// function to be called during backprop to, given the gradients of the output
// tensors, produce the gradients of the input tensors.
void TFE_Py_TapeRecordOperation(PyObject* tape, PyObject* op_type,
                                PyObject* output_tensors,
                                PyObject* input_tensor_ids,
                                PyObject* backward_function);

// Computes a gradient based on information recorded on the tape.`tape` must
// have been produced by TFE_Py_NewTape. `vspace` must be a
// imperative_grad.py:VSpace named tuple. `target` and `sources` must be python
// lists of Tensor objects. `output_gradients` is either None or a python list
// of either Tensor or None, and if not None should have the same length as
// target.
PyObject* TFE_Py_TapeGradient(PyObject* tape, PyObject* vspace,
                              PyObject* target, PyObject* sources,
                              PyObject* output_gradients, TF_Status* status);

// Returns an EagerTensor of dimension [len(`tensor_list`)] containing
// the `slice_dim`'th dimension of each tensor in `tensor_list`. In other words,
// TFE_Py_TensorShapeSlice takes a slice of dimensions of tensors in
// `tensor_list`. For example, if `tensor_list` contains tensors of with shapes
// [1, 2, 3], [4, 5], [6, 7, 8, 9], TFE_Py_TensorShapeSlice called with
// `slice_dim` equal to 1 will return [2, 5, 7].
// On error, returns nullptr and sets python exception.
// REQUIRES: `tensor_list` is a python list of EagerTensors
// REQUIRES: `slice_dim` is non-negative and smaller than the rank of all
//   tensors in `tensor_list`.
PyObject* TFE_Py_TensorShapeSlice(PyObject* tensor_list, int slice_dim);

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TFE_H_
