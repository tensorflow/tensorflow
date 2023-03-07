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

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "pybind11/chrono.h"  // from @pybind11
#include "pybind11/complex.h"  // from @pybind11
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace py = pybind11;

// TODO(amitpatankar): Move the custom type casters to separate common header
// only libraries.

namespace pybind11 {
namespace detail {

/* This is a custom type caster for the TensorShape object. For more
 * documentation please refer to this link:
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html#custom-type-casters
 * The PyCheckpointReader methods sometimes return the `TensorShape` object
 * and the `DataType` object as outputs. This custom type caster helps Python
 * handle it's conversion from C++ to Python. Since we do not accept these
 * classes as arguments from Python, it is not necessary to define the `load`
 * function to cast the object from Python to a C++ object.
 */

template <>
struct type_caster<tensorflow::TensorShape> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::TensorShape, _("tensorflow::TensorShape"));

  static handle cast(const tensorflow::TensorShape& src,
                     return_value_policy unused_policy, handle unused_handle) {
    // TODO(amitpatankar): Simplify handling TensorShape as output later.
    size_t dims = src.dims();
    tensorflow::Safe_PyObjectPtr value(PyList_New(dims));
    for (size_t i = 0; i < dims; ++i) {
#if PY_MAJOR_VERSION >= 3
      tensorflow::Safe_PyObjectPtr dim_value(
          tensorflow::make_safe(PyLong_FromLong(src.dim_size(i))));
#else
      tensorflow::Safe_PyObjectPtr dim_value(
          tensorflow::make_safe(PyInt_FromLong(src.dim_size(i))));
#endif
      PyList_SET_ITEM(value.get(), i, dim_value.release());
    }

    return value.release();
  }
};

template <>
struct type_caster<tensorflow::DataType> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::DataType, _("tensorflow::DataType"));

  static handle cast(const tensorflow::DataType& src,
                     return_value_policy unused_policy, handle unused_handle) {
#if PY_MAJOR_VERSION >= 3
    tensorflow::Safe_PyObjectPtr value(
        tensorflow::make_safe(PyLong_FromLong(src)));
#else
    tensorflow::Safe_PyObjectPtr value(
        tensorflow::make_safe(PyInt_FromLong(src)));
#endif
    return value.release();
  }
};

}  // namespace detail
}  // namespace pybind11

namespace tensorflow {

static py::object CheckpointReader_GetTensor(
    tensorflow::checkpoint::CheckpointReader* reader, const string& name) {
  Safe_TF_StatusPtr status = make_safe(TF_NewStatus());
  PyObject* py_obj = Py_None;
  std::unique_ptr<tensorflow::Tensor> tensor;
  reader->GetTensor(name, &tensor, status.get());

  // Error handling if unable to get Tensor.
  tensorflow::MaybeRaiseFromTFStatus(status.get());

  tensorflow::MaybeRaiseFromStatus(
      tensorflow::TensorToNdarray(*tensor, &py_obj));

  return tensorflow::PyoOrThrow(
      PyArray_Return(reinterpret_cast<PyArrayObject*>(py_obj)));
}

}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_checkpoint_reader, m) {
  // Initialization code to use numpy types in the type casters.
  import_array1();
  py::class_<tensorflow::checkpoint::CheckpointReader> checkpoint_reader_class(
      m, "CheckpointReader");
  checkpoint_reader_class
      .def(py::init([](const std::string& filename) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // pybind11 support smart pointers and will own freeing the memory when
        // complete.
        // https://pybind11.readthedocs.io/en/master/advanced/smart_ptrs.html#std-unique-ptr
        auto checkpoint =
            std::make_unique<tensorflow::checkpoint::CheckpointReader>(
                filename, status.get());
        tensorflow::MaybeRaiseFromTFStatus(status.get());
        return checkpoint;
      }))
      .def("debug_string",
           [](tensorflow::checkpoint::CheckpointReader& self) {
             return py::bytes(self.DebugString());
           })
      .def("get_variable_to_shape_map",
           &tensorflow::checkpoint::CheckpointReader::GetVariableToShapeMap)
      .def("_GetVariableToDataTypeMap",
           &tensorflow::checkpoint::CheckpointReader::GetVariableToDataTypeMap)
      .def("_HasTensor", &tensorflow::checkpoint::CheckpointReader::HasTensor)
      .def_static("CheckpointReader_GetTensor",
                  &tensorflow::CheckpointReader_GetTensor);
};
