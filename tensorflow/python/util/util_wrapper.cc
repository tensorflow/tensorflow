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

#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
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
          return tensorflow::pyo_or_throw(
              tensorflow::swig::RegisterType(type_name.ptr(), type.ptr()));
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
      "IsSequence",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsSequence(o.ptr());
        return result;
      },
      R"pbdoc(
      Returns true if its input is a collections.Sequence (except strings).

      Args:
        seq: an input sequence.

      Returns:
        True if the sequence is a not a string and is a collections.Sequence or a
        dict.
    )pbdoc");
  m.def(
      "IsSequenceOrComposite",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsSequenceOrComposite(o.ptr());
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
        return tensorflow::pyo_or_throw(
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
        return tensorflow::pyo_or_throw(
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
        return tensorflow::pyo_or_throw(
            tensorflow::swig::Flatten(o.ptr(), expand_composites));
      },
      R"pbdoc(
      Returns a flat list from a given nested structure.

      If `nest` is not a sequence, tuple, or dict, then returns a single-element
      list: `[nest]`.

      In the case of dict instances, the sequence consists of the values, sorted by
      key to ensure deterministic behavior. This is true also for `OrderedDict`
      instances: their sequence order is ignored, the sorting order of keys is
      used instead. The same convention is followed in `pack_sequence_as`. This
      correctly repacks dicts and `OrderedDict`s after they have been flattened,
      and also allows flattening an `OrderedDict` and then repacking it back using
      a corresponding plain dict, or vice-versa.
      Dictionaries with non-sortable keys cannot be flattened.

      Users must not modify any collections used in `nest` while this function is
      running.

      Args:
        nest: an arbitrarily nested structure or a scalar object. Note, numpy
            arrays are considered scalars.
        expand_composites: If true, then composite tensors such as `tf.SparseTensor`
            and `tf.RaggedTensor` are expanded into their component tensors.

      Returns:
        A Python list, the flattened version of the input.

      Raises:
        TypeError: The nest is or contains a dict with non-sortable keys.
    )pbdoc");
  m.def(
      "IsSequenceForData",
      [](const py::handle& o) {
        bool result = tensorflow::swig::IsSequenceForData(o.ptr());
        if (PyErr_Occurred()) {
          throw py::error_already_set();
        }
        return result;
      },
      R"pbdoc(
      Returns a true if `seq` is a Sequence or dict (except strings/lists).

      NOTE(mrry): This differs from `tensorflow.python.util.nest.is_sequence()`,
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
        return tensorflow::pyo_or_throw(
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
}
