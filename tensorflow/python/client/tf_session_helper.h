/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_
#define TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_

// Must be included first
#include "tensorflow/python/lib/core/numpy.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

// Container types for the various arguments and temporary values used
// in the wrapper.

// A NameVector is a vector of tensor or operation names, as borrowed
// C strings.
typedef tensorflow::gtl::InlinedVector<const char*, 8> NameVector;

// A PyObjectVector is a vector of borrowed pointers to PyObjects.
typedef tensorflow::gtl::InlinedVector<PyObject*, 8> PyObjectVector;

// A TF_TensorVector is a vector of borrowed pointers to TF_Tensors.
typedef gtl::InlinedVector<TF_Tensor*, 8> TF_TensorVector;

TF_Session* TF_NewSessionRef(TF_Graph* graph, const TF_SessionOptions* opts,
                             TF_Status* status);

// Run the graph associated with the session starting with the
// supplied inputs[].  Regardless of success or failure, inputs[] are
// stolen by the implementation (i.e. the implementation will
// eventually call Py_DECREF on each array input).
//
// The PyObject* feed_dict must be a dictionary mapping strings to
// NumPy arrays. This function does not modify its reference count.
//
// On success, the tensors corresponding to output_names[0,noutputs-1]
// are placed in out_values[], and these outputs[] become the property
// of the caller (the caller must eventually call Py_DECREF on them).
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
void TF_Run_wrapper(TF_DeprecatedSession* session, const TF_Buffer* run_options,
                    PyObject* feed_dict, const NameVector& output_names,
                    const NameVector& target_nodes, TF_Status* out_status,
                    PyObjectVector* out_values, TF_Buffer* run_outputs);

// Python wrappers for the `Session::MakeCallable()` API.
void TF_DeprecatedSessionMakeCallable(TF_DeprecatedSession* session,
                                      const TF_Buffer* callable_options,
                                      int64_t* out_handle,
                                      TF_Status* out_status);
void TF_SessionMakeCallable(TF_Session* session,
                            const TF_Buffer* callable_options,
                            int64_t* out_handle, TF_Status* out_status);

// Python wrappers for the `Session::RunCallable()` API.
void TF_DeprecatedSessionRunCallable(TF_DeprecatedSession* session,
                                     int64_t handle, PyObject* feed_values,
                                     TF_Status* out_status,
                                     PyObjectVector* out_values,
                                     TF_Buffer* run_metadata);
void TF_SessionRunCallable(TF_Session* session, int64_t handle,
                           PyObject* feed_values, TF_Status* out_status,
                           PyObjectVector* out_values, TF_Buffer* run_metadata);

// Python wrappers for the `Session::ReleaseCallable()` API.
void TF_DeprecatedSessionReleaseCallable(TF_DeprecatedSession* session,
                                         int64_t handle, TF_Status* out_status);
void TF_SessionReleaseCallable(TF_Session* session, int64_t handle,
                               TF_Status* out_status);

// Set up the graph with the intended feeds and fetches for partial run.
// *out_handle is owned by the caller.
//
// On success, returns a handle that is used for subsequent PRun calls.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
void TF_PRunSetup_wrapper(TF_DeprecatedSession* session,
                          const NameVector& input_names,
                          const NameVector& output_names,
                          const NameVector& target_nodes, TF_Status* out_status,
                          const char** out_handle);

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
//
// The PyObject* feed_dict must be a dictionary mapping strings to
// NumPy arrays. This function does not modify its reference count.
//
// On success,  the tensors corresponding to output_names[0,noutputs-1]
// are placed in out_values[], and these outputs[] become the property
// of the caller (the caller must eventually call Py_DECREF on them).
//
// On failure,  out_status contains a tensorflow::Status with an error
// message.
void TF_PRun_wrapper(TF_DeprecatedSession* session, const char* handle,
                     PyObject* feed_dict, const NameVector& output_names,
                     TF_Status* out_status, PyObjectVector* out_values);

// Wrapper for TF_Reset that converts the string vectors to character arrays.
void TF_Reset_wrapper(const TF_SessionOptions* opt,
                      const NameVector& containers, TF_Status* out_status);

// Convenience wrapper around EqualGraphDef to make it easier to wrap.
// Returns an explanation if a difference is found, or the empty string
// for no difference.
string EqualGraphDefWrapper(const string& actual, const string& expected);

// Convenience wrapper around AreAttrValuesEqual to make it easier to wrap.
// The actual and expected strings must correspond to a serialized binary
// representation of two AttrValue proto instances.
// Returns an explanation if a difference is found, or the empty string
// for no difference.
string EqualAttrValueWrapper(const string& actual, const string& expected);

// Gets shape from C API Graph object.
//
// If shape is known, returns shape vector where -1 means "unknown
// dimension".  Sets unknown_shape to false.
//
// If shape is unknown, sets unknown_shape to true.
tensorflow::gtl::InlinedVector<int64_t, 6> TF_GraphGetTensorShapeHelper(
    TF_Graph* graph, TF_Output output, TF_Status* status, bool* unknown_shape);

// Runs the graph associated with the session starting with the supplied inputs.
// On success, `py_outputs` is populated with a numpy ndarray for each output
// (the caller must decref these ndarrays, although this will likely be handled
// by the Python gc). `session`, `out_status`, and `py_outputs` must be
// non-null. `py_outputs` should be empty.
void TF_SessionRun_wrapper(TF_Session* session, const TF_Buffer* run_options,
                           const std::vector<TF_Output>& inputs,
                           const std::vector<PyObject*>& input_ndarrays,
                           const std::vector<TF_Output>& outputs,
                           const std::vector<TF_Operation*>& targets,
                           TF_Buffer* run_metadata, TF_Status* status,
                           std::vector<PyObject*>* py_outputs);

// Set up the graph with the intended feeds (inputs) and fetches (output) for
// a sequence of partial run calls.
//
// On success, returns a handle that can be used for subsequent PRun calls. The
// handle is owned by the caller and should be deleted with TF_DeletePRunHandle
// when it is no longer needed.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
void TF_SessionPRunSetup_wrapper(TF_Session* session,
                                 const std::vector<TF_Output>& inputs,
                                 const std::vector<TF_Output>& outputs,
                                 const std::vector<TF_Operation*>& targets,
                                 const char** out_handle, TF_Status* status);

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
//
// On success, `py_outputs` is populated with a numpy ndarray for each output
// (the caller must decref these ndarrays, although this will likely be handled
// by the Python gc). `session`, `handle`, `out_status`, and `py_outputs` must
// be non-null. `py_outputs` should be empty.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
void TF_SessionPRun_wrapper(TF_Session* session, const char* handle,
                            const std::vector<TF_Output>& inputs,
                            const std::vector<PyObject*>& input_ndarrays,
                            const std::vector<TF_Output>& outputs,
                            TF_Status* status,
                            std::vector<PyObject*>* py_outputs);

// Retrieves the inputs of this operation.
std::vector<TF_Output> GetOperationInputs(TF_Operation* oper);

// Retrieves the control inputs of this operation.
std::vector<TF_Operation*> TF_OperationGetControlInputs_wrapper(
    TF_Operation* oper);

// Retrieves the control outputs of this operation.
std::vector<TF_Operation*> TF_OperationGetControlOutputs_wrapper(
    TF_Operation* oper);

// Retrieves the op names of the consumers of `oper_out`. The returned strings
// have the lifetime of the underlying TF_Graph.
std::vector<const char*> TF_OperationOutputConsumers_wrapper(
    TF_Output oper_out);

// `opers` equaling NULL are converted to `nopers = -1`.
// `output_names` must be empty or have the same length as `outputs`.
TF_Function* TF_GraphToFunction_wrapper(
    const TF_Graph* fn_body, const char* fn_name, bool append_hash_to_fn_name,
    const std::vector<TF_Operation*>* opers,
    const std::vector<TF_Output>& inputs, const std::vector<TF_Output>& outputs,
    const NameVector& output_names,
    const std::vector<TF_Operation*>* control_outputs,
    const NameVector& control_output_names, const TF_FunctionOptions* opts,
    const char* description, TF_Status* status);

// Set the shapes and types for the output's handle.
//
// The sizes of 'shapes', 'ranks', and 'types' must be equal; `shapes[i]`
// contains the shape of the handle's i-th value, `ranks[i]` contains the i-th
// shape's rank, and `types[i]` contains the i-th value's dtype. If the i-th
// shape is unknown, then `ranks[i]` must be equal to -1.
//
// The space between the double angle brackets below looks extraneous, but
// our version of SWIG cannot parse ">>".
void TF_GraphSetOutputHandleShapesAndTypes_wrapper(
    TF_Graph* graph, TF_Output output,
    const std::vector<std::vector<int64_t> >& shapes,
    const std::vector<int>& ranks, const std::vector<TF_DataType>& types,
    TF_Status* status);

// Set the shape of output. If unknown is true, `num_dims` must be set to
// -1 and `dims` is set to nullptr.
void TF_GraphSetTensorShape_wrapper(TF_Graph* graph, TF_Output output,
                                    const std::vector<int64_t>& dims,
                                    bool unknown_shape, TF_Status* status);

// Returns the string representations of the missing unused input mappings.
std::vector<string> TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
    TF_ImportGraphDefResults* results);

// If evaluation was possible, returns the numpy ndarray of the evaluated
// result. Otherwise returns None.
PyObject* TF_TryEvaluateConstant_wrapper(TF_Graph* graph, TF_Output output,
                                         TF_Status* status);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_
