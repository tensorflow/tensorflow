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

// Execute a cancelable TensorFlow operation.
//
// Arguments as above (for TFE_Py_Execute), with the addition of:
// 'cancellation_manager': A pointer to a TFE_CancellationManager that can be
//                         used to cancel execution of the given operation.
typedef struct TFE_CancellationManager TFE_CancellationManager;
void TFE_Py_ExecuteCancelable(TFE_Context* ctx, const char* device_name,
                              const char* op_name,
                              TFE_InputTensorHandles* inputs, PyObject* attrs,
                              TFE_CancellationManager* cancellation_manager,
                              TFE_OutputTensorHandles* outputs,
                              TF_Status* out_status);

// Registers e as the Exception class for handling not ok Status. Returns
// Py_None if registration succeeds, else throws a TypeError and returns NULL.
//
// This function is not thread-safe.
PyObject* TFE_Py_RegisterExceptionClass(PyObject* e);

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

// Registers e as the forward_gradient_function.  The registered function takes
// (op_name, attrs, inputs, outputs, tangents) and returns the output
// tangents. This function is used only for operations, not for custom gradients
// or functional ops.
//
// This function is not thread-safe.
PyObject* TFE_Py_RegisterJVPFunction(PyObject* e);

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

// Check if any backward tape should record an operation given inputs.
//
// Does not take forward accumulators into account.
PyObject* TFE_Py_TapeSetShouldRecordBackprop(PyObject* tensors);

// Determine possible gradient types, taking forward accumulators into account.
//   - 0 if no tape will record (implies TFE_Py_TapeSetShouldRecordBackprop
//     is false and no forward accumulator is watching)
//   - 1 if first-order gradients may be requested
//   - 2 if higher-order gradients may be requested
PyObject* TFE_Py_TapeSetPossibleGradientTypes(PyObject* tensors);

void TFE_Py_TapeWatch(PyObject* tape, PyObject* tensor);
void TFE_Py_TapeSetDeleteTrace(tensorflow::int64 tensor_id);

// Stops any gradient recording on the current thread.
//
// Includes forward accumulators.
void TFE_Py_TapeSetStopOnThread();

// Restarts gradient recording on the current thread.
void TFE_Py_TapeSetRestartOnThread();

// Checks whether gradient recording is stopped on the current thread.
PyObject* TFE_Py_TapeSetIsStopped();

// Records an operation for the purpose of gradient computation.
//
// Arguments:
//  - op_type is a string for the operation type, used in the backprop code
//  - output_tensors are a list of Python Tensor objects output by the operation
//  - input_tensors are a list of input Tensors to the recorded operation
//  - backward_function is the function to be called during backprop or
//    forwardprop to, given the gradients of the output tensors, produce the
//    gradients of the input tensors. This function is automatically transposed
//    during forwardprop.
//  - forward_function is an optional special-case for fowardprop, taking input
//    jvps and returning output jvps.
//
// Records an operation both for backprop (gradient tape) and forwardprop
// (forward accumulator). Equivalent to calling both
// TFE_Py_TapeSetRecordOperationBackprop and
// TFE_Py_TapeSetRecordOperationForwardprop.
PyObject* TFE_Py_TapeSetRecordOperation(PyObject* op_type,
                                        PyObject* output_tensors,
                                        PyObject* input_tensors,
                                        PyObject* backward_function,
                                        PyObject* forward_function);

// Records an operation only for backprop (gradient tapes).
//
// Same arguments as TFE_Py_TapeSetRecordOperation.
PyObject* TFE_Py_TapeSetRecordOperationBackprop(PyObject* op_type,
                                                PyObject* output_tensors,
                                                PyObject* input_tensors,
                                                PyObject* backward_function);

// Records an operation only for forwardprop (forward accumulators).
//
// Arguments:
//  - op_type is a string for the operation type, used in the backprop code
//  - output_tensors are a list of Python Tensor objects output by the operation
//  - input_tensors are a list of input Tensors to the recorded operation
//  - backward_function is the function to be called to, given the gradients of
//    the output tensors, produce the gradients of the input tensors. This
//    function is automatically transposed to produce output gradients given
//    input gradients.
//  - forwardprop_output_indices indicates any output_tensors which contain
//    JVPs. Typically these will have come from TFE_Py_PackJVPs. May
//    be None or an empty sequence if there are no JVP outputs from the
//    operation.
PyObject* TFE_Py_TapeSetRecordOperationForwardprop(
    PyObject* op_type, PyObject* output_tensors, PyObject* input_tensors,
    PyObject* backward_function, PyObject* forwardprop_output_indices);

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
                                PyObject* attrs, PyObject* results);

// Returns all variables watched by the given tape in the order those variables
// were created.
PyObject* TFE_Py_TapeWatchedVariables(PyObject* tape);

// Creates a new forward accumulator. Does not add it to the active set.
PyObject* TFE_Py_ForwardAccumulatorNew();

// Adds a ForwardAccumulator to the active set, meaning it will watch executed
// operations. It must not already be in the active set.
PyObject* TFE_Py_ForwardAccumulatorSetAdd(PyObject* accumulator);
// Removes a forward accumulator from the active set, meaning it will no longer
// be watching operations.
void TFE_Py_ForwardAccumulatorSetRemove(PyObject* accumulator);

// Tell the forward accumulator `accumulator` to watch `tensor`, with a Tensor
// tangent vector `tangent` of matching shape and dtype.
void TFE_Py_ForwardAccumulatorWatch(PyObject* accumulator, PyObject* tensor,
                                    PyObject* tangent);

// Looks up the Jacobian-vector product of `tensor` in the forward accumulator
// `accumulator`. Returns None if no JVP is available.
PyObject* TFE_Py_ForwardAccumulatorJVP(PyObject* accumulator, PyObject* tensor);

// Temporarily push or pop transient state for accumulators in the active set.
//
// Allows an accumulator which is currently processing an operation to
// temporarily reset its state. This is useful when building forwardprop
// versions of functions, where an accumulator will trigger function building
// and then must process captured symbolic tensors while building it. Without
// pushing and poping, accumulators ignore operations executed as a direct
// result of their own jvp computations.
PyObject* TFE_Py_ForwardAccumulatorPushState();
PyObject* TFE_Py_ForwardAccumulatorPopState();

// Collects state from all current forward accumulators related to `tensors`.
//
// This is useful for packing JVPs as function inputs before executing a
// function which computes primals and JVPs at the same time.
//
// Does not include accumulators which are currently in the process of computing
// a jvp (and so appear somewhere on the current execution stack) or any
// accumulators more deeply nested.
//
// Includes JVPs for `tensors` and any higher-order JVPs for those
// (recursively). Returns a two-element tuple (indices, jvps):
//   indices: A sequence of sequences of two-element tuples. Each forward
//       accumulator is represented as a sequence of tuples with (primal_index,
//       jvp_index). Both integers index into the concatenated `tensors + jvps`
//       array.
//   jvps: A flat list of Tensors. Best interpreted as a sequence to be
//       appended to `tensors`.
PyObject* TFE_Py_PackJVPs(PyObject* tensors);

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

// Sets `python_context` as the current eager Context object (defined
// in eager/context.py). This function must be called at least once before
// eager tensors are created.
// If an error is encountered, sets python error and returns NULL. Else, returns
// Py_None.
//
// This function is not thread-safe.
PyObject* TFE_Py_SetEagerContext(PyObject* python_context);

// Returns the current eager Context object (defined in eager/context.py)
// that was last set using TFE_Py_SetEagerContext.
// If an error is encountered, sets python error and returns NULL.
// The returned PyObject is "new", i.e. the caller must call Py_DECREF on it at
// some point.
PyObject* GetPyEagerContext();

// These are exposed since there is SWIG code that calls these.
// Returns a pre-allocated status if it exists.
TF_Status* GetStatus();
// Returns the pre-allocated status to the code.
void ReturnStatus(TF_Status* status);
#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TFE_H_
