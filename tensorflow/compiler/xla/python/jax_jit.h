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

#include <stdexcept>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace jax {

// Flags, such as JIT disable and the x64 mode, are controlled by:
// - a global flag value, e.g., associated to --jax_enable_x64
// - possibly a thread-local value, which initially is std::nullopt and
//   overrides the global value if set. The thread-local state is
//   used to implement context managers that locally override the global state.
struct JitState {
  ~JitState() {
    if (extra_jit_context) {
      // We likely do not hold the GIL if this JitState is thread-local, so we
      // hand the Python object to the global reference manager to destroy.
      pybind11::object o = std::move(*extra_jit_context);
      xla::GlobalPyRefManager()->AddGarbage(absl::MakeSpan(&o, 1));
      extra_jit_context = std::nullopt;
    }
  }

  std::optional<bool> disable_jit;
  std::optional<bool> enable_x64;
  std::optional<bool> jax_array;

  // Used to manually set the default device jax should use. May be unset even
  // in global state, indicating there is no manual override.
  // TODO(skyewm): make this a C++ type when all JAX backends support a single
  // C++ device interface
  std::optional<pybind11::object> default_device;

  // Extra context that should be included in the JIT cache key. Must be
  // hashable and have an equality defined.
  std::optional<pybind11::object> extra_jit_context;

  // A callback that, if present, is called when a JITted function is executed
  // from cache. May be unset even in global state.
  std::optional<pybind11::function> post_hook;
};

JitState& GetGlobalState();
JitState& GetLocalState();

// Getters for JitState fields that first look in thread-local state, then
// fallback to global state.
bool GetDisableJit();
bool GetEnableX64();
bool GetEnableJaxArray();
// TODO(skyewm): return a C++ type when all JAX backends support a single C++
// device interface
std::optional<pybind11::object> GetDefaultDevice();
std::optional<pybind11::function> GetPostHook();

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
  // Not part of the signature, but we need it for error messages.
  absl::string_view function_name;

  // A PyTreeDef for each dynamic argument, positional arguments first
  // followed by keyword arguments. Keyword arguments are in the order given
  // by dynamic_arg_names.
  absl::InlinedVector<xla::PyTreeDef, 2> dynamic_arg_treedefs;
  // Dynamic keyword argument names. Interned, and sorted by the keyword
  // name.
  std::vector<pybind11::object> dynamic_arg_names;
  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by keyword name).
  absl::InlinedVector<xla::PyArgSignature, 2> dynamic_arg_signatures;

  // Static arguments. Contains the positional arguments sorted in argument
  // order, followed by static keyword arguments in the order given by
  // `static_arg_names`.
  std::vector<pybind11::object> static_args;
  // Static keyword argument names. Interned, and sorted by keyword name.
  std::vector<pybind11::object> static_arg_names;

  // For JIT, we need this in the key because computation follows the data, so
  // we may have multiple executables depending on the devices the data is on.
  // This is not the case for PMAP, and is set to `nullptr`.
  xla::PjRtDevice* device = nullptr;
  bool jax_enable_x64;
  bool jax_array = false;

  // Opaque additional context that should be included as part of the cache key.
  std::optional<pybind11::object> global_extra_jit_context;
  std::optional<pybind11::object> thread_local_extra_jit_context;

  bool operator==(const CallSignature& other) const;
  bool operator!=(const CallSignature& other) const {
    return !(*this == other);
  }

  std::string DebugString() const;
};

template <typename H>
H AbslHashValue(H h, const CallSignature& s) {
  h = H::combine(std::move(h), s.dynamic_arg_treedefs,
                 s.dynamic_arg_signatures);
  for (const auto& name : s.dynamic_arg_names) {
    h = H::combine(std::move(h), name.ptr());
  }
  h = H::combine(std::move(h), s.dynamic_arg_names.size());
  for (const auto& static_arg : s.static_args) {
    ssize_t hash;
    try {
      hash = pybind11::hash(static_arg);
    } catch (const pybind11::error_already_set& e) {
      if (!e.matches(PyExc_TypeError)) throw;
      throw std::invalid_argument(absl::StrCat(
          "Non-hashable static arguments are not supported. An error occurred "
          "during a call to '",
          s.function_name, "' while trying to hash an object of type ",
          pybind11::cast<std::string>(
              pybind11::str(pybind11::type::of(static_arg))),
          ", ", pybind11::cast<std::string>(pybind11::str(static_arg)),
          ". The error was:\n", e.what(), "\n"));
    }
    h = H::combine(std::move(h), hash);
  }
  h = H::combine(std::move(h), s.static_args.size());
  for (const auto& name : s.static_arg_names) {
    h = H::combine(std::move(h), name.ptr());
  }
  h = H::combine(std::move(h), s.static_arg_names.size());
  h = H::combine(std::move(h), s.device, s.jax_enable_x64);

  // We do not hash the extra_jit_context fields since calling Python hash
  // functions is expensive (~300ns) and we don't expect a large number of
  // different contexts.
  return h;
}

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
  //    `CopyBuffersToDevice`
  // (b) `arg_buffers` do not maintain ownership
  std::vector<std::unique_ptr<xla::PjRtBuffer>> keep_alive;
};

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(pybind11::handle args,
                           const std::optional<pybind11::kwargs>& py_kwargs,
                           absl::Span<int const> static_argnums,
                           absl::Span<pybind11::str const> static_argnames,
                           ParsedArgumentsAsBuffers& arguments);

// The function to call in `xla.cc` to add the bindings for this module.
void BuildJaxjitSubmodule(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_JAX_JIT_H_
