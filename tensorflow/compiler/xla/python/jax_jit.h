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

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace jax {

// Returns the value for jax_enable_x64 (defined by a thread-local value if
// defined, defaulting to the value of the flag otherwise).
bool GetEnableX64();

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
  // A PyTreeDef for each positional dynamic (i.e. not static) argument.
  absl::InlinedVector<xla::PyTreeDef, 2> dynamic_positional_args_treedef;
  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by keyword name).
  absl::InlinedVector<xla::PyArgSignature, 2> dynamic_args_signatures;
  xla::PjRtDevice* device;
  bool jax_enable_x64;
  // Opaque additional context that should be included as part of the cache key.
  pybind11::object extra_jit_context;

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
  // Keyword arguments. Sorted by the keyword name.
  std::vector<KwargEntry> keyword_args;


  bool operator==(const CallSignature& other) const;
  bool operator!=(const CallSignature& other) const {
    return !(*this == other);
  }

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
  absl::InlinedVector<pybind11::object, 2> flat_dynamic_args;
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
xla::Status ParseArguments(pybind11::handle args,
                           const absl::optional<pybind11::kwargs>& py_kwargs,
                           absl::Span<int const> static_argnums,
                           ParsedArgumentsAsBuffers& arguments);

// The function to call in `xla.cc` to add the bindings for this module.
void BuildJaxjitSubmodule(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
