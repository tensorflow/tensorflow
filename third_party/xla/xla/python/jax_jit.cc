/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/jax_jit.h"

#include <Python.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/py_values.h"
#include "xla/python/pytree.h"
#include "xla/python/sharding.h"
#include "xla/python/types.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax {

namespace nb = nanobind;

// TODO(phawkins): Add support for Tracers.
// TODO(jblespiau): Use absl Status.

namespace {

// `thread_local_state.extra_jit_context` is set from Python. It's done when
// loading the Python jax modules on the main-thread. For other threads, we
// need to initialize the field the first time we access `thread_local_state`.
nb::object& initialize_local_state = *new nb::object();

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
    thread_local_state.extra_jit_context = nb::none();
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

std::optional<nb::object> GetDefaultDevice() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  return thread_local_state.default_device.has_value()
             ? thread_local_state.default_device
             : global_state.default_device;
}

std::optional<nb::callable> GetPostHook() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  return thread_local_state.post_hook.has_value() ? thread_local_state.post_hook
                                                  : global_state.post_hook;
}

static std::string OptionalDebugString(
    const std::optional<nb::object> optional) {
  if (optional.has_value()) {
    return nb::cast<std::string>(nb::str(optional.value()));
  } else {
    return "None";
  }
}

bool FetchMemoriesFlag() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  CHECK(global_state.enable_memories.has_value());
  return thread_local_state.enable_memories.value_or(
      *global_state.enable_memories);
}

std::string ArgumentSignature::DebugString() const {
  auto py_object_formatter = [](std::string* out, const nb::object& o) {
    out->append(nb::cast<std::string_view>(nb::str(o)));
  };
  auto treedef_formatter = [](std::string* out, const xla::PyTreeDef& d) {
    out->append(d.ToString());
  };
  return absl::StrFormat(
      "static args (positional + keyword): [%s], "
      "static arg keyword names: [%s], "
      "dynamic arg signatures (positional + keyword): [%s]"
      "dynamic arg shardings: [%s]",
      absl::StrJoin(static_args, ",", py_object_formatter),
      absl::StrJoin(static_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_treedefs, "| ", treedef_formatter));
}

bool ArgumentSignature::operator==(const ArgumentSignature& other) const {
  if (dynamic_arg_treedefs != other.dynamic_arg_treedefs) {
    return false;
  }
  auto object_ptr_equality = [](nb::handle a, nb::handle b) {
    return a.ptr() == b.ptr();
  };
  if (!absl::c_equal(dynamic_arg_names, other.dynamic_arg_names,
                     object_ptr_equality)) {
    return false;
  }
  if (!absl::c_equal(static_arg_names, other.static_arg_names,
                     object_ptr_equality)) {
    return false;
  }
  return absl::c_equal(
      static_args, other.static_args,
      [](const nb::object& a, const nb::object& b) {
        try {
          return a.type().ptr() == b.type().ptr() && a.equal(b);
        } catch (const nb::python_error& e) {
          throw std::invalid_argument(absl::StrCat(
              "static arguments should be comparable using __eq__."
              "The following error was raised when comparing two objects of "
              "types ",
              nb::cast<std::string_view>(nb::str(a.type())), " and ",
              nb::cast<std::string_view>(nb::str(b.type())),
              ". The error was:\n", e.what()));
        }
      });
}

std::string CallSignature::DebugString() const {
  auto py_object_formatter = [](std::string* out, const nb::object& o) {
    out->append(nb::cast<std::string_view>(nb::str(o)));
  };
  auto signature_formatter = [](std::string* out,
                                const xla::PyArgSignature& s) {
    out->append(s.DebugString());
  };
  auto bool_formatter = [](std::string* out, bool o) {
    out->append(o ? "true" : "false");
  };
  return absl::StrFormat(
      "arg signature: %s\n"
      "dynamic arg signatures (positional + keyword): %s\n"
      "dynamic arg shardings: %s\n"
      "committed args: %s\n"
      "device: %s\n"
      "default_device: %s\n"
      "jax_enable_x64: %d\n"
      "jax_enable_memories: %d\n"
      "global_extra_jit_context: %s\n"
      "thread_local_extra_jit_context: %s\n",
      arg_signature.DebugString(),
      absl::StrJoin(dynamic_arg_signatures, ", ", signature_formatter),
      absl::StrJoin(dynamic_arg_shardings, ", ", py_object_formatter),
      absl::StrJoin(committed_args, ",", bool_formatter),
      device != nullptr ? device->DebugString() : "nullptr",
      OptionalDebugString(default_device), jax_enable_x64, jax_enable_memories,
      OptionalDebugString(global_extra_jit_context),
      OptionalDebugString(thread_local_extra_jit_context));
}

bool CallSignature::operator==(const CallSignature& other) const {
  if (arg_signature != other.arg_signature) {
    return false;
  }
  if (dynamic_arg_signatures != other.dynamic_arg_signatures) {
    return false;
  }
  if (device != other.device) {
    return false;
  }
  if (jax_enable_x64 != other.jax_enable_x64) {
    return false;
  }
  if (jax_enable_memories != other.jax_enable_memories) {
    return false;
  }
  if (committed_args != other.committed_args) {
    return false;
  }
  return
      // `==` on py:objects is the Python `is`. We need equal.
      absl::c_equal(dynamic_arg_shardings, other.dynamic_arg_shardings,
                    ShardingEqual) &&
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
absl::Status ParseArguments(
    absl::Span<PyObject* const> positional_args,
    absl::Span<PyObject* const> keyword_args, nb::handle kwnames,
    absl::Span<int const> static_argnums,
    absl::Span<nb::str const> static_argnames,
    xla::PyTreeRegistry* pytree_registry, ArgumentSignature& signature,
    absl::InlinedVector<nanobind::object, 2>& flat_dynamic_args) {
  tsl::profiler::TraceMe traceme("ParseArguments");

  flat_dynamic_args.reserve(positional_args.size() + keyword_args.size());
  if (static_argnums.empty()) {
    signature.dynamic_arg_treedefs.reserve(positional_args.size());

    // Positional arguments.
    for (int i = 0; i < positional_args.size(); ++i) {
      signature.dynamic_arg_treedefs.emplace_back(pytree_registry);
      xla::PyTreeDef& pytree_def = signature.dynamic_arg_treedefs.back();
      pytree_def.Flatten(nb::handle(positional_args[i]), flat_dynamic_args);
    }
  } else {
    signature.dynamic_arg_treedefs.reserve(positional_args.size());

    // Positional arguments.
    for (int i = 0; i < positional_args.size(); ++i) {
      if (std::find(static_argnums.begin(), static_argnums.end(), i) ==
          static_argnums.end()) {
        signature.dynamic_arg_treedefs.emplace_back(pytree_registry);
        xla::PyTreeDef& pytree_def = signature.dynamic_arg_treedefs.back();
        pytree_def.Flatten(positional_args[i], flat_dynamic_args);
      } else {
        signature.static_args.emplace_back(
            nb::borrow<nb::object>(positional_args[i]));
      }
    }
  }

  // Keyword arguments.
  if (!keyword_args.empty()) {
    std::vector<std::pair<nb::handle, nb::handle>> kwargs(keyword_args.size());
    // We first intern the keys, then sort them (by name, as in the Python path)
    // (see also xla::PyTreeDef::Flatten) and then create the signatures.
    // TODO(jblespiau): We should be able to sort the keys by interned-key
    // pointers, but this requires the Python compilation to do the same.
    for (int i = 0; i < keyword_args.size(); ++i) {
      // Intern the key if not already interned.
      PyObject* key = PyTuple_GET_ITEM(kwnames.ptr(), i);
      Py_INCREF(key);
      if (!PyUnicode_CHECK_INTERNED(key)) {
        PyUnicode_InternInPlace(&key);
      }
      kwargs[i].first = key;
      kwargs[i].second = keyword_args[i];
    }

    std::sort(kwargs.begin(), kwargs.end(),
              [](const std::pair<nb::handle, nb::handle>& a,
                 const std::pair<nb::handle, nb::handle>& b) {
                return a.first < b.first;
              });
    auto kwarg_is_static = [&](nb::handle name) {
      for (const auto& kw : static_argnames) {
        if (kw.ptr() == name.ptr()) return true;
      }
      return false;
    };

    signature.dynamic_arg_names.reserve(keyword_args.size());
    for (int i = 0; i < keyword_args.size(); ++i) {
      if (kwarg_is_static(kwargs[i].first)) {
        signature.static_arg_names.push_back(
            nb::steal<nb::object>(kwargs[i].first));
        signature.static_args.push_back(
            nb::borrow<nb::object>(kwargs[i].second));
      } else {
        signature.dynamic_arg_names.push_back(
            nb::steal<nb::object>(kwargs[i].first));
        signature.dynamic_arg_treedefs.emplace_back(pytree_registry);
        xla::PyTreeDef& pytree_def = signature.dynamic_arg_treedefs.back();
        pytree_def.Flatten(nb::handle(kwargs[i].second.ptr()),
                           flat_dynamic_args);
      }
    }
  }
  return absl::OkStatus();
}

void BuildJaxjitSubmodule(nb::module_& m) {
  nb::module_ jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  nb::class_<JitState> jit_state_(jitlib, "JitState");
  jit_state_.def_rw("disable_jit", &JitState::disable_jit, nb::arg().none());
  jit_state_.def_rw("enable_x64", &JitState::enable_x64, nb::arg().none());
  jit_state_.def_rw("enable_memories", &JitState::enable_memories,
                    nb::arg().none());
  jit_state_.def_rw("default_device", &JitState::default_device,
                    nb::arg().none());
  jit_state_.def_rw("extra_jit_context", &JitState::extra_jit_context,
                    nb::arg().none());
  jit_state_.def_rw("post_hook", &JitState::post_hook, nb::arg().none());

  GetEnableMemories = +[] { return FetchMemoriesFlag(); };

  jitlib.def(
      "global_state", [&]() { return &GlobalJitState(); },
      nb::rv_policy::reference);
  jitlib.def(
      "thread_local_state", [&]() { return &ThreadLocalJitState(); },
      nb::rv_policy::reference);

  jitlib.def(
      "swap_thread_local_state_disable_jit",
      [&](std::optional<bool> value) -> std::optional<bool> {
        auto tls = &ThreadLocalJitState();
        auto result = tls->disable_jit;
        tls->disable_jit = value;
        return result;
      },
      nb::arg("value").none(), nb::rv_policy::reference);

  jitlib.def("get_enable_x64", &GetEnableX64);
  jitlib.def("set_thread_local_state_initialization_callback",
             [](nb::object f) { initialize_local_state = f; });

  nb::class_<xla::PyArgSignature> arg_signature(jitlib, "PyArgSignature");
  arg_signature
      .def_prop_ro(
          "dtype",
          [](const xla::PyArgSignature& sig) {
            return xla::ValueOrThrow(xla::PrimitiveTypeToNbDtype(sig.dtype));
          })
      .def_prop_ro("shape",
                   [](const xla::PyArgSignature& sig) {
                     return xla::SpanToNbTuple(absl::MakeConstSpan(sig.shape));
                   })
      .def_ro("weak_type", &xla::PyArgSignature::weak_type);
  jitlib.def("_ArgSignatureOfValue",
             xla::ValueOrThrowWrapper(xla::PyArgSignatureOfValue));

  jitlib.def("_is_float0", &xla::IsFloat0);
}

}  // namespace jax
