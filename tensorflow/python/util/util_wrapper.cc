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

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/util/util.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_utils, m) {
  m.doc() = R"pbdoc(
    _pywrap_utils
    -----
  )pbdoc";
  m.def("RegisterType",
        [](const py::handle& type_name, const py::handle& type) {
          return tensorflow::PyoOrThrow(
              tensorflow::swig::RegisterType(type_name.ptr(), type.ptr()));
        });
  m.def("RegisterPyObject", [](const py::handle& name, const py::handle& type) {
    return tensorflow::PyoOrThrow(
        tensorflow::swig::RegisterPyObject(name.ptr(), type.ptr()));
  });
  m.def(
      "IsTensor",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsTensor(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Check if an object is a Tensor.
    )pbdoc");
  m.def(
      "IsNested",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsNested(o.ptr());
        return result;
      },
      R"pbdoc(
      Refer to `tf.nest.is_nested`.
    )pbdoc");
  m.def(
      "IsNestedOrComposite",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsNestedOrComposite(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns true if its input is a sequence or a `CompositeTensor`.

      Args:
        seq: an input sequence.

      Returns:
        True if the sequence is a not a string and is a collections.Sequence or a
        dict or a CompositeTensor or a TypeSpec (except string and TensorSpec).
    )pbdoc");
  m.def(
      "IsCompositeTensor",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsCompositeTensor(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns true if its input is a `CompositeTensor`.

      Args:
        seq: an input sequence.

      Returns:
        True if the sequence is a CompositeTensor.
    )pbdoc");
  m.def(
      "IsTypeSpec",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsTypeSpec(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns true if its input is a `TypeSpec`, but is not a `TensorSpec`.

      Args:
        seq: an input sequence.

      Returns:
        True if the sequence is a `TypeSpec`, but is not a `TensorSpec`.
    )pbdoc");
  m.def(
      "IsNamedtuple",
      [](const py::handle& o, bool strict) {
        return tensorflow::PyoOrThrow(
            tensorflow::swig::IsNamedtuple(o.ptr(), strict));
      },
      R"pbdoc(
      Check if an object is a NamedTuple.
    )pbdoc");
  m.def(
      "IsMapping",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsMapping(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns True if `instance` is a `collections.Mapping`.

      Args:
        instance: An instance of a Python object.

      Returns:
        True if `instance` is a `collections.Mapping`.
    )pbdoc");
  m.def(
      "IsMutableMapping",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsMutableMapping(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns True if `instance` is a `collections.MutableMapping`.

      Args:
        instance: An instance of a Python object.

      Returns:
        True if `instance` is a `collections.MutableMapping`.
    )pbdoc");
  m.def(
      "IsMappingView",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsMappingView(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns True if considered a mapping view for the purposes of Flatten()`.

      Args:
        instance: An instance of a Python object.

      Returns:
        True if considered a mapping view for the purposes of Flatten().
    )pbdoc");
  m.def(
      "IsAttrs",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsAttrs(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns True if `instance` is an instance of an `attr.s` decorated class.

      Args:
        instance: An instance of a Python object.

      Returns:
        True if `instance` is an instance of an `attr.s` decorated class.
    )pbdoc");
  m.def(
      "SameNamedtuples",
      [](const py::handle& o1, const py::handle& o2) {
        return tensorflow::PyoOrThrow(
            tensorflow::swig::SameNamedtuples(o1.ptr(), o2.ptr()));
      },
      R"pbdoc(
      Returns True if the two namedtuples have the same name and fields.
    )pbdoc");
  m.def(
      "AssertSameStructure",
      [](const py::handle& o1, const py::handle& o2, bool check_types,
         bool expand_composites) {
        bool result = tensorflow::swig::AssertSameStructure(
            o1.ptr(), o2.ptr(), check_types, expand_composites);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns True if the two structures are nested in the same way.
    )pbdoc");
  m.def(
      "Flatten",
      [](const py::handle& o, bool expand_composites) {
        return tensorflow::PyoOrThrow(
            tensorflow::swig::Flatten(o.ptr(), expand_composites));
      },
      R"pbdoc(
      Refer to `tf.nest.flatten`.
    )pbdoc");
  m.def(
      "IsNestedForData",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsNestedForData(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns a true if `seq` is a nested structure for tf.data.

      NOTE(mrry): This differs from `tensorflow.python.util.nest.is_nested()`,
      which *does* treat a Python list as a sequence. For ergonomic
      reasons, `tf.data` users would prefer to treat lists as
      implicit `tf.Tensor` objects, and dicts as (nested) sequences.

      Args:
        seq: an input sequence.

      Returns:
        True if the sequence is a not a string or list and is a
        collections.Sequence.
    )pbdoc");
  m.def(
      "FlattenForData",
      [](const py::handle& o) {
        return tensorflow::PyoOrThrow(
            tensorflow::swig::FlattenForData(o.ptr()));
      },
      R"pbdoc(
      Returns a flat sequence from a given nested structure.

      If `nest` is not a sequence, this returns a single-element list: `[nest]`.

      Args:
        nest: an arbitrarily nested structure or a scalar object.
          Note, numpy arrays are considered scalars.

      Returns:
        A Python list, the flattened version of the input.
    )pbdoc");
  m.def(
      "AssertSameStructureForData",
      [](const py::handle& o1, const py::handle& o2, bool check_types) {
        bool result = tensorflow::swig::AssertSameStructureForData(
            o1.ptr(), o2.ptr(), check_types);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns True if the two structures are nested in the same way in particular tf.data.
    )pbdoc");
  m.def(
      "IsResourceVariable",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsResourceVariable(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns 1 if `o` is a ResourceVariable.

      Args:
        instance: An instance of a Python object.

      Returns:
        True if `instance` is a `ResourceVariable`.
    )pbdoc");
  m.def(
      "IsVariable",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsVariable(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns 1 if `o` is a Variable.

      Args:
        instance: An instance of a Python object.

      Returns:
        True if `instance` is a `Variable`.
    )pbdoc");
  m.def(
      "IsBF16SupportedByOneDNNOnThisCPU",
      []() {
        bool result = tensorflow::port::TestCPUFeature(
            tensorflow::port::CPUFeature::AVX512F);
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns 1 if CPU has avx512f feature.

      Args:
       None

      Returns:
        True if CPU has avx512f feature.
    )pbdoc");
}
