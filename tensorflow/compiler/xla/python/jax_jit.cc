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

// This files implements the `jax.jit` dispatch and just-in-time feature.
//
// In a nutshell, `Jit(f)` returns a callable that will dispatch (i.e. forward
// based on passed arguments dtypes/shapes/identity) the execution to a
// just-in-time compiled XLA Executable. All of that is done in C++ for
// performance reasons.
//
// This file contains the utilities to:
// (a) inspect arguments and describe their structure, dtype/shapes, etc.
// (b) keep a mapping from function signatures to compiled XLA Executables.

#include "tensorflow/compiler/xla/python/jax_jit.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/compiler/xla/pjrt/lru_cache.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace jax {

namespace py = pybind11;

// TODO(phawkins): Add support for Tracers.
// TODO(jblespiau): Use absl Status.

namespace {

// `thread_local_state.extra_jit_context` is set from Python. It's done when
// loading the Python jax modules on the main-thread. For other threads, we
// need to initialize the field the first time we access `thread_local_state`.
py::object& initialize_local_state = *new py::object();

}  // namespace

JitState& GlobalJitState() {
  // Protected by the GIL.
  static JitState& global_state = *new JitState();
  return global_state;
}

JitState& ThreadLocalJitState() {
  // TODO(phawkins): Google style guide forbids thread-local values with
  // non-trivial destructors.
  ABSL_CONST_INIT thread_local JitState thread_local_state;  // NOLINT
  DCHECK(PyGILState_Check());
  if (thread_local_state.extra_jit_context == std::nullopt) {
    CHECK(initialize_local_state.ptr() != nullptr);
    // Avoids reentrant calls to the initialization function.
    thread_local_state.extra_jit_context = py::none();
    initialize_local_state();
  }
  return thread_local_state;
}

bool GetDisableJit() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  CHECK(global_state.disable_jit.has_value());
  return thread_local_state.disable_jit.value_or(*global_state.disable_jit);
}

bool GetEnableX64() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  CHECK(global_state.enable_x64.has_value());
  return thread_local_state.enable_x64.value_or(*global_state.enable_x64);
}

std::optional<py::object> GetDefaultDevice() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  return thread_local_state.default_device.has_value()
             ? thread_local_state.default_device
             : global_state.default_device;
}

std::optional<pybind11::function> GetPostHook() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  return thread_local_state.post_hook.has_value() ? thread_local_state.post_hook
                                                  : global_state.post_hook;
}

static std::string OptionalDebugString(
    const std::optional<py::object> optional) {
  if (optional.has_value()) {
    return py::cast<std::string>(py::str(optional.value()));
  } else {
    return "None";
  }
}

std::string CallSignature::DebugString() const {
  auto py_object_formatter = [](std::string* out, const py::object& o) {
    out->append(py::cast<std::string>(py::str(o)));
  };
  auto treedef_formatter = [](std::string* out, const xla::PyTreeDef& d) {
    out->append(d.ToString());
  };
  auto signature_formatter = [](std::string* out,
                                const xla::PyArgSignature& s) {
    out->append(s.DebugString());
  };
  auto bool_formatter = [](std::string* out, bool o) {
    out->append(o ? "true" : "false");
  };
  return absl::StrFormat(
      "static args (positional + keyword): %s\nstatic arg keyword names: %s\n"
      "dynamic arg signatures (positional + keyword): %s\n"
      "dynamic arg shardings: %s\n"
      "committed args: %s\n"
      "dynamic arg keyword names: %s\n"
      "dynamic arg treedefs: %s\n"
      "device: %s\n"
      "default_device: %s\n"
      "jax_enable_x64: %d\n"
      "global_extra_jit_context: %s\n"
      "thread_local_extra_jit_context: %s\n",
      absl::StrJoin(static_args, ",", py_object_formatter),
      absl::StrJoin(static_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_signatures, ", ", signature_formatter),
      absl::StrJoin(dynamic_arg_shardings, ", ", py_object_formatter),
      absl::StrJoin(committed_args, ",", bool_formatter),
      absl::StrJoin(dynamic_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_treedefs, "| ", treedef_formatter),  // new line
      device != nullptr ? device->DebugString() : "nullptr",
      OptionalDebugString(default_device), jax_enable_x64,
      OptionalDebugString(global_extra_jit_context),
      OptionalDebugString(thread_local_extra_jit_context));
}

bool CallSignature::operator==(const CallSignature& other) const {
  // TODO(chky): Consider implementing hashing and equality for sharding in cpp
  // instead of hashing and checking sharding's pointer values.
  return std::tie(dynamic_arg_treedefs, dynamic_arg_names,
                  dynamic_arg_signatures, device, jax_enable_x64,
                  static_arg_names, committed_args) ==
             std::tie(other.dynamic_arg_treedefs, other.dynamic_arg_names,
                      other.dynamic_arg_signatures, other.device,
                      other.jax_enable_x64, other.static_arg_names,
                      other.committed_args) &&
         // `==` on py:objects is the Python `is`. We need equal.
         std::equal(dynamic_arg_shardings.begin(), dynamic_arg_shardings.end(),
                    other.dynamic_arg_shardings.begin(),
                    other.dynamic_arg_shardings.end(),
                    [](const py::object& a, const py::object& b) {
                      return ShardingEqual(a, b);
                    }) &&
         std::equal(
             static_args.begin(), static_args.end(), other.static_args.begin(),
             other.static_args.end(),
             [this](const py::object& a, const py::object& b) {
               try {
                 return py::type::handle_of(a) == py::type::handle_of(b) &&
                        a.equal(b);
               } catch (const py::error_already_set& e) {
                 throw std::invalid_argument(absl::StrCat(
                     "static arguments should be comparable using __eq__."
                     "The following error was raised during a call to '",
                     function_name, "' when comparing two objects of types ",
                     py::cast<std::string>(py::str(py::type::of(a))), " and ",
                     py::cast<std::string>(py::str(py::type::of(b))),
                     ". The error was:\n", e.what()));
               }
             }) &&
         (global_extra_jit_context.has_value() ==
          other.global_extra_jit_context.has_value()) &&
         (!global_extra_jit_context.has_value() ||
          global_extra_jit_context->equal(*other.global_extra_jit_context)) &&
         (default_device.has_value() == other.default_device.has_value()) &&
         (!default_device.has_value() ||
          default_device->equal(*other.default_device)) &&
         (thread_local_extra_jit_context.has_value() ==
          other.thread_local_extra_jit_context.has_value()) &&
         (!thread_local_extra_jit_context.has_value() ||
          thread_local_extra_jit_context->equal(
              *other.thread_local_extra_jit_context));
}

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(absl::Span<PyObject* const> positional_args,
                           absl::Span<PyObject* const> keyword_args,
                           py::handle kwnames,
                           absl::Span<int const> static_argnums,
                           absl::Span<py::str const> static_argnames,
                           ParsedArgumentsAsBuffers& arguments) {
  tsl::profiler::TraceMe traceme("ParseArguments");

  arguments.flat_dynamic_args.reserve(positional_args.size() +
                                      keyword_args.size());
  if (static_argnums.empty()) {
    arguments.signature.dynamic_arg_treedefs.resize(positional_args.size());

    // Positional arguments.
    for (int i = 0; i < positional_args.size(); ++i) {
      xla::PyTreeDef& pytree_def = arguments.signature.dynamic_arg_treedefs[i];
      pytree_def.FlattenInto(positional_args[i], arguments.flat_dynamic_args);
    }
  } else {
    arguments.signature.dynamic_arg_treedefs.reserve(positional_args.size());

    // Positional arguments.
    for (int i = 0; i < positional_args.size(); ++i) {
      if (std::find(static_argnums.begin(), static_argnums.end(), i) ==
          static_argnums.end()) {
        arguments.signature.dynamic_arg_treedefs.emplace_back();
        xla::PyTreeDef& pytree_def =
            arguments.signature.dynamic_arg_treedefs.back();
        pytree_def.FlattenInto(positional_args[i], arguments.flat_dynamic_args);
      } else {
        arguments.signature.static_args.emplace_back(
            py::reinterpret_borrow<py::object>(positional_args[i]));
      }
    }
  }

  // Keyword arguments.
  if (!keyword_args.empty()) {
    std::vector<std::pair<py::handle, py::handle>> kwargs(keyword_args.size());
    // We first intern the keys, then sort them (by name, as in the Python path)
    // (see also xla::PyTreeDef::Flatten) and then create the signatures.
    // TODO(jblespiau): We should be able to sort the keys by interned-key
    // pointers, but this requires the Python compilation to do the same.
    for (int i = 0; i < keyword_args.size(); ++i) {
      // Intern the key if not already interned.
      kwargs[i].first = py::handle(PyTuple_GET_ITEM(kwnames.ptr(), i));
      kwargs[i].first.inc_ref();
      kwargs[i].second = py::handle(keyword_args[i]);
      if (!PyUnicode_CHECK_INTERNED(kwargs[i].first.ptr())) {
        PyUnicode_InternInPlace(&kwargs[i].first.ptr());
      }
    }

    std::sort(kwargs.begin(), kwargs.end(),
              [](const std::pair<py::handle, py::handle>& a,
                 const std::pair<py::handle, py::handle>& b) {
                return a.first < b.first;
              });
    auto kwarg_is_static = [&](py::handle name) {
      for (const auto& kw : static_argnames) {
        if (kw.ptr() == name.ptr()) return true;
      }
      return false;
    };

    arguments.signature.dynamic_arg_names.reserve(keyword_args.size());
    for (int i = 0; i < keyword_args.size(); ++i) {
      if (kwarg_is_static(kwargs[i].first)) {
        arguments.signature.static_arg_names.push_back(
            py::reinterpret_steal<py::object>(kwargs[i].first));
        arguments.signature.static_args.push_back(
            py::reinterpret_borrow<py::object>(kwargs[i].second));
      } else {
        arguments.signature.dynamic_arg_names.push_back(
            py::reinterpret_steal<py::object>(kwargs[i].first));
        arguments.signature.dynamic_arg_treedefs.emplace_back();
        xla::PyTreeDef& pytree_def =
            arguments.signature.dynamic_arg_treedefs.back();
        pytree_def.FlattenInto(kwargs[i].second, arguments.flat_dynamic_args);
      }
    }
  }
  return ::tsl::OkStatus();
}

void BuildJaxjitSubmodule(py::module& m) {
  py::module jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  py::class_<JitState> jit_state_(jitlib, "JitState");
  jit_state_.def_readwrite("disable_jit", &JitState::disable_jit);
  jit_state_.def_readwrite("enable_x64", &JitState::enable_x64);
  jit_state_.def_readwrite("default_device", &JitState::default_device);
  jit_state_.def_readwrite("extra_jit_context", &JitState::extra_jit_context);
  jit_state_.def_readwrite("post_hook", &JitState::post_hook);

  jitlib.def(
      "global_state", [&]() { return &GlobalJitState(); },
      py::return_value_policy::reference);
  jitlib.def(
      "thread_local_state", [&]() { return &ThreadLocalJitState(); },
      py::return_value_policy::reference);

  jitlib.def("jit_is_disabled", &GetDisableJit);
  jitlib.def("get_enable_x64", &GetEnableX64);
  jitlib.def("set_thread_local_state_initialization_callback",
             [](py::object f) { initialize_local_state = f; });

  // TODO(yashkatariya, phawkins): Remove after 3 months from March 20, 2023.
  struct CompiledFunction {};
  py::class_<CompiledFunction>(m, "CompiledFunction");

  py::class_<xla::PyArgSignature> arg_signature(jitlib, "PyArgSignature");
  arg_signature
      .def_property_readonly(
          "dtype",
          [](const xla::PyArgSignature& sig) {
            return xla::ValueOrThrow(PrimitiveTypeToDtype(sig.dtype));
          })
      .def_property_readonly(
          "shape",
          [](const xla::PyArgSignature& sig) {
            return xla::SpanToTuple(absl::MakeConstSpan(sig.shape));
          })
      .def_readonly("weak_type", &xla::PyArgSignature::weak_type);
  jitlib.def("_ArgSignatureOfValue",
             xla::ValueOrThrowWrapper(xla::PyArgSignatureOfValue));

  jitlib.def("_is_float0", &xla::IsFloat0);
}

}  // namespace jax
