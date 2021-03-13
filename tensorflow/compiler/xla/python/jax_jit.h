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
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace jax {

// Returns the value for jax_enable_x64 (defined by a thread-local value if
// defined, defaulting to the value of the flag otherwise).
bool GetEnableX64();

// Describes the abstract shape and dtype of an argument.
struct ArgSignature {
  ArgSignature(xla::PrimitiveType dtype, absl::Span<const xla::int64> shape,
               bool weak_type)
      : dtype(dtype), shape(shape.begin(), shape.end()), weak_type(weak_type) {}
  // This is the XLA dtype of the object.
  const xla::PrimitiveType dtype;
  const absl::InlinedVector<xla::int64, 4> shape;
  // JAX arguments can be of weak type, if and only if they are Python scalars
  // or `DeviceArray` values such that `aval.weak_type` is true.
  const bool weak_type;
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
    xla::PyTreeDef value_treedef;
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
  std::vector<xla::PyTreeDef> dynamic_positional_args_treedef;
  // Keyword arguments. Sorted by the keyword name.
  std::vector<KwargEntry> keyword_args;
  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by keyword name).
  std::vector<ArgSignature> dynamic_args_signatures;
  xla::PjRtDevice* device;
  bool jax_enable_x64;

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

// The resulting information of the parsing and conversion of the arguments.
struct ParsedArgumentsAsBuffers {
  // The call signature will be filled during 2 steps:
  // - `ParseArguments` will fill the static arguments and the pytree
  //    structures
  // - the shapes and dtypes are filled later, by `ParseAndTransferArguments`.
  CallSignature signature;
  // The concatenation of the dynamic positional arguments and the sorted
  // keyword arguments.
  std::vector<pybind11::object> flat_dynamic_args;
  std::vector<pybind11::object> keep_alive_objects;

  // The following is only valid if the parsing succeeds.
  std::vector<xla::PjRtBuffer*> arg_buffers;
  // We may need to keep these objects around, because:
  // (a) we need to extend the lifetime of objects created within
  //    `ConvertArgsToBuffers`
  // (b) `arg_buffers` do not maintain ownership
  std::vector<std::unique_ptr<xla::PjRtBuffer>> keep_alive;
};

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(const pybind11::args& args,
                           const pybind11::kwargs& py_kwargs,
                           absl::Span<int const> static_argnums,
                           ParsedArgumentsAsBuffers& arguments);


// Returns the ArgSignature associated with an argument. Returns an error if
// the argument is not supported.
xla::StatusOr<ArgSignature> ArgSignatureOfValue(pybind11::handle arg,
                                                bool jax_enable_x64);


// The function to call in `xla.cc` to add the bindings for this module.
void BuildJaxjitSubmodule(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
