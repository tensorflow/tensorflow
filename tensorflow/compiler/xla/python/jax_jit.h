/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Describes the abstract shape and dtype of an argument.
struct ArgSignature {
  // This is the XLA dtype of the object.
  xla::PrimitiveType dtype;
  // JAX arguments can be of weak type, if and only if they are Python scalars
  // or `DeviceArray` values such that `aval.weak_type` is true.
  bool weak_type;
  absl::InlinedVector<int64, 4> shape;
  bool operator==(const ArgSignature& other) const {
    return std::tie(dtype, weak_type, shape) ==
           std::tie(other.dtype, other.weak_type, other.shape);
  }
  bool operator!=(const ArgSignature& other) const { return !(*this == other); }
  std::string DebugString() const;
};

template <typename H>
H AbslHashValue(H h, const ArgSignature& s) {
  h = H::combine(std::move(h), s.dtype);
  h = H::combine_contiguous(std::move(h), s.shape.data(), s.shape.size());
  return h;
}

// The signature of Python jitted function call, partitioned into:
// - dynamic positional arguments (i.e. positional args which are not static)
// - static positional arguments (i.e. the args associated to static_argnums)
// - keyword arguments
// The CallSignature should unambiguously identify a function call, thus,
// equality is based on:
// (a) Same PyTree for all dynamic positional arguments and keyword arguments
// (a) equality of the arguments and keyword arguments ArgSignature
// (a) equality (delegated to Python) of the static arguments.
struct CallSignature {
  struct KwargEntry {
    // To avoid comparing strings, we intern the kwargs strings.
    // The compilation cache holds a reference to all the keys.
    pybind11::handle key;
    PyTreeDef value_treedef;
    bool operator==(const KwargEntry& other) const {
      return key.ptr() == other.key.ptr() &&
             value_treedef == other.value_treedef;
    }
    bool operator!=(const KwargEntry& other) const { return !(*this == other); }
  };

  // Only contains the arguments associated to `static_argnums`, sorted in the
  // order of their argnum index.
  std::vector<pybind11::object> static_args;
  // A PyTreeDef for each positional dynamic (i.e. not static) argument.
  std::vector<PyTreeDef> dynamic_positional_args_treedef;
  // Keyword arguments. Sorted by the keyword name.
  std::vector<KwargEntry> keyword_args;
  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by keyword name).
  std::vector<ArgSignature> dynamic_args_signatures;
  PjRtDevice* device;

  bool operator==(const CallSignature& other) const;
  bool operator!=(const CallSignature& other) const {
    return !(*this == other);
  }

  // To be used when we want to keep ownership of Python values referenced by
  // the `CallSignature` (i.e. when we insert an entry).
  void IncRef() const;
  // The destructor of the cache should call this on all entries.
  void DecRef() const;

  std::string DebugString() const;
};

template <typename H>
H AbslHashValue(H h, const CallSignature::KwargEntry& kw) {
  h = H::combine(std::move(h), kw.key.ptr(), kw.value_treedef);
  return h;
}

template <typename H>
H AbslHashValue(H h, const CallSignature& s);

// The function to call in `xla.cc` to add the bindings for this module.
void BuildJaxjitSubmodule(pybind11::module& m);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
