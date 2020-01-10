#ifndef TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_
#define TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_

#include <Python.h>

#include "numpy/arrayobject.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor_c_api.h"

namespace tensorflow {

// Container types for the various arguments and temporary values used
// in the wrapper.

// A FeedVector is a vector of tensor name and numpy array pairs. The
// name is a borrowed C string.
typedef tensorflow::gtl::InlinedVector<std::pair<const char*, PyArrayObject*>,
                                       8> FeedVector;

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
// supplied inputs[].  Regardless of success of failure, inputs[] are
// stolen by the implementation (i.e. the implementation will
// eventually call Py_DECREF on each array input).
//
// On success, the tensors corresponding to output_names[0,noutputs-1]
// are placed in out_values[], and these outputs[] become the property
// of the caller (the caller must eventually call Py_DECREF on them).
//
// On failure, out_status contains a tensorflow::Status with an error
// message.
void TF_Run_wrapper(TF_Session* session, const FeedVector& inputs,
                    const NameVector& output_names,
                    const NameVector& target_nodes, Status* out_status,
                    PyObjectVector* out_values);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_CLIENT_TF_SESSION_HELPER_H_
