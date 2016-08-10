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

// A FeedVector is a vector of tensor name and numpy array pairs. The
// name is a borrowed C string.
typedef tensorflow::gtl::InlinedVector<std::pair<const char*, PyArrayObject*>,
                                       8>
    FeedVector;

// A NameVector is a vector of tensor or operation names, as borrowed
// C strings.
typedef tensorflow::gtl::InlinedVector<const char*, 8> NameVector;

// A PyObjectVector is a vector of borrowed pointers to PyObjects.
typedef tensorflow::gtl::InlinedVector<PyObject*, 8> PyObjectVector;

// Safe containers for (an) owned PyObject(s). On destruction, the
// reference count of the contained object will be decremented.
inline void Py_DECREF_wrapper(PyObject* o) { Py_DECREF(o); }
typedef void (*Py_DECREF_wrapper_type)(PyObject*);
typedef std::unique_ptr<PyObject, Py_DECREF_wrapper_type> Safe_PyObjectPtr;
typedef std::vector<Safe_PyObjectPtr> Safe_PyObjectVector;
Safe_PyObjectPtr make_safe(PyObject* o);

// Run the graph associated with the session starting with the
// supplied inputs[].  Regardless of success or failure, inputs[] are
// stolen by the implementation (i.e. the implementation will
// eventually call Py_DECREF on each array input).
//
// On success, the tensors corresponding to output_names[0,noutputs-1]
// are placed in out_values[], and these outputs[] become the property
// of the caller (the caller must eventually call Py_DECREF on them).
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
void TF_Run_wrapper(TF_Session* session, const TF_Buffer* run_options,
                    const FeedVector& inputs, const NameVector& output_names,
                    const NameVector& target_nodes, TF_Status* out_status,
                    PyObjectVector* out_values, TF_Buffer* run_outputs);

// Set up the graph with the intended feeds and fetches for partial run.
// *out_handle is owned by the caller.
//
// On success, returns a handle that is used for subsequent PRun calls.
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
//
// NOTE: This is EXPERIMENTAL and subject to change.
void TF_PRunSetup_wrapper(TF_Session* session, const NameVector& input_names,
                          const NameVector& output_names,
                          const NameVector& target_nodes, TF_Status* out_status,
                          const char** out_handle);

// Continue to run the graph with additional feeds and fetches. The
// execution state is uniquely identified by the handle.
//
// On success,  the tensors corresponding to output_names[0,noutputs-1]
// are placed in out_values[], and these outputs[] become the property
// of the caller (the caller must eventually call Py_DECREF on them).
//
// On failure,  out_status contains a tensorflow::Status with an error
// message.
//
// NOTE: This is EXPERIMENTAL and subject to change.
void TF_PRun_wrapper(TF_Session* session, const char* handle,
                     const FeedVector& inputs, const NameVector& output_names,
                     TF_Status* out_status, PyObjectVector* out_values);

// Wrapper for TF_Reset that converts the string vectors to character arrays.
void TF_Reset_wrapper(const TF_SessionOptions* opt,
                      const NameVector& containers, TF_Status* out_status);

// Convenience wrapper around EqualGraphDef to make it easier to wrap.
// Returns an explanation if a difference is found, or the empty string
// for no difference.
string EqualGraphDefWrapper(const string& actual, const string& expected);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_
