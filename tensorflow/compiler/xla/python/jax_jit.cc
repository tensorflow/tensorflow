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
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/pjrt/lru_cache.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/pytree.h"
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

bool GetEnableJaxArray() {
  auto& global_state = GlobalJitState();
  auto& thread_local_state = ThreadLocalJitState();
  CHECK(global_state.jax_array.has_value());
  return thread_local_state.jax_array.value_or(*global_state.jax_array);
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
      "jax_array: %d\n"
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
      OptionalDebugString(default_device), jax_enable_x64, jax_array,
      OptionalDebugString(global_extra_jit_context),
      OptionalDebugString(thread_local_extra_jit_context));
}

bool CallSignature::operator==(const CallSignature& other) const {
  // TODO(chky): Consider implementing hashing and equality for sharding in cpp
  // instead of hashing and checking sharding's pointer values.
  return std::tie(dynamic_arg_treedefs, dynamic_arg_names,
                  dynamic_arg_signatures, device, jax_enable_x64, jax_array,
                  static_arg_names, committed_args) ==
             std::tie(other.dynamic_arg_treedefs, other.dynamic_arg_names,
                      other.dynamic_arg_signatures, other.device,
                      other.jax_enable_x64, other.jax_array,
                      other.static_arg_names, other.committed_args) &&
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

namespace {

// Elements of CacheEntry are protected by the GIL.
struct CacheEntry {
  // Ensures a single thread performs the compilation for a given executable.
  //
  // The first thread (holding the GIL) will create the CacheEntry associated to
  // a signature and fill it. Other threads will wait for the notification.
  // If an error occurred during the compilation, `fall_back_to_python` is set
  // to `true`, and other threads will fail with the same error.
  absl::Notification compilation_complete;
  std::thread::id thread_id = std::this_thread::get_id();

  std::shared_ptr<xla::PyLoadedExecutable> executable;
  xla::PyTreeDef out_pytree_def;
  // We use Python types within the vector because this is what we will be
  // returning to Python. No need to convert back and forth.
  // We need py::object to maintain the objects alive.
  std::vector<py::object> out_avals;
  std::vector<bool> out_weak_types;
  std::vector<py::dtype> out_dtypes;
  std::vector<std::vector<int64_t>> out_shapes;
  std::vector<py::object> out_shardings;
  std::vector<bool> committed;

  // Bitvector of kept arguments from Jaxpr DCE pass. Used to drop some `args`
  // in CompiledFunction::Call before calling into compiled computation.
  std::vector<bool> kept_var_bitvec;
  std::optional<xla::ClientAndPtr<xla::PjRtDevice>> sticky_device;

  // Fallback to Python happens:
  // - for trivial computations
  // - when running a jax(pmap)
  // - after a compilation error, for threads that did not compile it the first
  //   time
  bool fall_back_to_python = false;

  // Python objects (notably in the cache key) that must remain alive as long
  // as the cache entry does. Currently this is the `key` values in the kwarg
  // entries in the cache key.
  std::vector<py::object> keepalive;
};

// A CompiledFunctionCache represents a cache of compiled functions that can be
// shared between one or more CompiledFunction objects. It serves two goals:
// - reduce the number of lru caches (hash map) across multiple JITs.
// - make the cache global to increase cache hits (e.g. calling jit(f)(3) twice)
//   keeping entries alive as long as the underlying function f is alive.
// Assume the cache is protected by the GIL.
class CompiledFunctionCache {
 public:
  static constexpr int kDefaultCapacity = 4096;
  explicit CompiledFunctionCache(int capacity);

  // Cache entries are shared_ptr<>s because it's possible the cache entry
  // might be evicted before we finish tracing/compiling.
  typedef xla::LRUCache<CallSignature, std::shared_ptr<CacheEntry>> Cache;

  // We include as part of the cache key `donate_argnums` (and any other fields
  // that aren't subsumed by the CallSignature we compute for each call).
  std::shared_ptr<Cache> Lookup(py::handle function,
                                absl::Span<const int> donate_argnums);

  int Size() const { return lru_list_.Size(); }
  int Capacity() const { return lru_list_.Capacity(); }
  void Clear() { lru_list_.Clear(); }

 private:
  struct Key {
    py::handle function;  // Does not hold a reference.

    // Other fields that are part of the arguments to `jit`, but are not
    // otherwise part of CallSignature.
    std::vector<int> donate_argnums;

    bool operator==(const Key& other) const {
      return std::tie(function, donate_argnums) ==
             std::tie(other.function, other.donate_argnums);
    }
  };
  template <typename H>
  friend H AbslHashValue(H h, const Key& key) {
    h = H::combine(std::move(h), key.function.ptr());
    h = H::combine_contiguous(std::move(h), key.donate_argnums.data(),
                              key.donate_argnums.size());
    return h;
  }

  struct Value {
    explicit Value(std::shared_ptr<Cache> cache) : cache(std::move(cache)) {}
    std::shared_ptr<Cache> cache;

    // A weak reference to the key function. We use the weak reference to
    // register a callback that is triggered when the key function is destroyed.
    // We use a weak pointer because we want to allow caching across multiple
    // calls to `jax.jit(f)` if `f` remains alive, but we do not want the cache
    // to keep `f` alive if all other references are dropped.
    py::weakref weakref;
  };

  Cache::LRUList lru_list_;
  absl::flat_hash_map<Key, std::unique_ptr<Value>> functions_;
};

CompiledFunctionCache::CompiledFunctionCache(int capacity)
    : lru_list_(capacity) {}

std::shared_ptr<CompiledFunctionCache::Cache> CompiledFunctionCache::Lookup(
    py::handle function, absl::Span<const int> donate_argnums) {
  Key key;
  key.function = function;
  key.donate_argnums =
      std::vector<int>(donate_argnums.begin(), donate_argnums.end());
  auto insert = functions_.emplace(key, nullptr);
  if (!insert.second) {
    return insert.first->second->cache;
  }
  std::shared_ptr<Cache> cache = std::make_shared<Cache>(&lru_list_);
  py::cpp_function callback([this, key{std::move(key)}](py::handle weakref) {
    functions_.erase(key);
  });
  PyObject* weakref = PyWeakref_NewRef(function.ptr(), callback.ptr());
  if (weakref) {
    std::unique_ptr<Value>& entry = insert.first->second;
    entry = std::make_unique<Value>(cache);
    entry->weakref = py::reinterpret_steal<py::weakref>(weakref);
  } else {
    PyErr_Clear();
    // `function` is not weak-referenceable. Don't bother adding it to the
    // shared cache in that case; the `jit` object will hold the only shared
    // reference to the cache entry.
    functions_.erase(insert.first);
  }
  return cache;
}

// A `CompiledFunction` is associated to a `jax.jit(f)` and takes care of the
// bookkeeping of the different signatures used and the dispatch of calls to
// the correct underlying `PyLoadedExecutable`. This class is thread-safe.
class CompiledFunction {
 public:
  CompiledFunction(py::function fun, py::function cache_miss,
                   py::function get_device, bool has_explicit_device,
                   std::vector<int> static_argnums,
                   std::vector<py::str> static_argnames,
                   std::vector<int> donate_argnums,
                   std::shared_ptr<CompiledFunctionCache> cache);
  ~CompiledFunction();

  // pybind11::object typed subclass for CompiledFunction objects.
  class pyobject : public py::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    py::object, CompiledFunction::IsCompiledFunction);
    pyobject() = default;
    CompiledFunction* func() const {
      return CompiledFunction::AsCompiledFunctionUnchecked(*this);
    }
  };
  // Alias as ::object; outside the scope above we won't confuse pybind11's
  // macros.
  using object = pyobject;

  // Returns true if `h` is a CompiledFunction.
  static bool IsCompiledFunction(py::handle handle);
  // Converts `handle` to a CompiledFunction*. Does not do any checking.
  static CompiledFunction* AsCompiledFunctionUnchecked(py::handle handle);

  // This function will:
  // (a) flatten the inputs using pytree
  // (b) get buffer objects from the arguments
  // (c) call the executable
  // (d) construct `DeviceArray` objects from the outputs
  // (e) reconstruct the `PyTree`.
  xla::StatusOr<py::object> Call(py::handle callable, PyObject* const* args,
                                 size_t nargs, PyObject* kwnames);

  // This allows `inspect.signature(cpp_jitted_f)` from Python.
  py::object PythonSignature() {
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(fun_);
  }

  int cache_size() const { return executables_->Size(); }
  void ClearCache() {
// Setting `default_device_` to nullptr forces Call() to retrieve the
// device.
    default_client_ = nullptr;
    default_device_ = nullptr;
    executables_->Clear();
  }

  const py::function& fun() const { return fun_; }
  const py::function& cache_miss() const { return cache_miss_; }
  const py::function& get_device() const { return get_device_; }
  bool has_explicit_device() const { return has_explicit_device_; }
  const std::vector<int>& static_argnums() const { return static_argnums_; }
  const std::vector<py::str>& static_argnames() const {
    return static_argnames_;
  }
  const std::vector<int>& donate_argnums() const { return donate_argnums_; }
  const std::shared_ptr<CompiledFunctionCache>& cache() const { return cache_; }

  // Helper function used by the tp_clear GC method.
  void ClearPythonReferences() {
    py::function fun, cache_miss, get_device;
    // Swap values for nulls before they are destroyed. See the Python
    // Py_CLEAR() documentation for a discussion of this topic.
    std::swap(fun_, fun);
    std::swap(cache_miss_, cache_miss);
    std::swap(get_device_, get_device);
  }

  const std::string& function_name() const { return function_name_; }

 private:
  // Attempts to populate default_device_. May release the GIL; is
  // reentrant-safe.
  void TryToPopulateDefaultDevice();

  void PopulateCacheEntry(CacheEntry* entry, const CallSignature& signature,
                          const py::tuple& out_and_fastpath_data);
  bool always_fallback_to_python_ = false;

  py::function fun_;  // The Python function to jit.
  std::string function_name_;

  // See JAX _cpp_jit in api.py for documentation.
  py::function cache_miss_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyLoadedExecutable. In sorted order.
  std::vector<int> static_argnums_;
  // Keyword arguments, interned.
  std::vector<py::str> static_argnames_;
  std::vector<int> donate_argnums_;

  // Whether this function has an explicit device set by either the `device` or
  // `backend` arguments to jit.
  bool has_explicit_device_;

  // A function taking no arguments and returning the default device and whether
  // jax.jit has been committed to it.
  py::function get_device_;

  // Keeps the shared LRU cache alive as long as the CompiledFunction is alive.
  std::shared_ptr<CompiledFunctionCache> cache_;

  // The part of cache_ specific to this CompiledFunction.
  std::shared_ptr<CompiledFunctionCache::Cache> executables_;

  // The logic if the following:
  // - if `device` or `backend` are not specified to `jax.jit`, we will use
  //   the input sticky buffer device, or `default_device_` if there is no
  //   such sticky buffer.
  // - When one of `device` or `backend` is specified, this will determine
  //   the `default_device_` which will be used as the targeted device. In
  //   which case, we will always copy input buffers to this device.
  // These fields are protected by the GIL.

  xla::ifrt::Client* default_client_ = nullptr;
  xla::PjRtDevice* default_device_ = nullptr;
  bool is_committed_;
};

// This class keeps references to all CompiledFunctions. This class is
// thread-compatible.
class CompiledFunctionStore {
 public:
  void Insert(CompiledFunction* function) {
    compiled_functions_.insert(function);
  }

  void Erase(CompiledFunction* function) {
    compiled_functions_.erase(function);
  }

  void ClearFunctionCache() {
    for (auto* function : compiled_functions_) {
      function->ClearCache();
    }
  }

 private:
  absl::flat_hash_set<CompiledFunction*> compiled_functions_;
};

// Protected by GIL.
CompiledFunctionStore& GetGlobalCompiledFunctionStore() {
  static auto* const store = new CompiledFunctionStore();
  return *store;
}

CompiledFunction::CompiledFunction(py::function fun, py::function cache_miss,
                                   py::function get_device,
                                   bool has_explicit_device,
                                   std::vector<int> static_argnums,
                                   std::vector<py::str> static_argnames,
                                   std::vector<int> donate_argnums,
                                   std::shared_ptr<CompiledFunctionCache> cache)
    : fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      static_argnames_(std::move(static_argnames)),
      donate_argnums_(donate_argnums),
      has_explicit_device_(std::move(has_explicit_device)),
      get_device_(std::move(get_device)),
      cache_(std::move(cache)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
  for (py::str& s : static_argnames_) {
    PyUnicode_InternInPlace(&s.ptr());
  }
  executables_ = cache_->Lookup(fun_, donate_argnums);
  function_name_ = py::str(py::getattr(fun_, "__name__", fun));

  GetGlobalCompiledFunctionStore().Insert(this);
}

CompiledFunction::~CompiledFunction() {
  GetGlobalCompiledFunctionStore().Erase(this);
}

// Returns nullptr if arg has no sticky device
static xla::StatusOr<std::pair<xla::ifrt::Client*, xla::PjRtDevice*>>
GetJitArgumentStickyDevice(py::handle arg) {
  struct PythonTypes {
    py::object device_array;
  };
  static const auto& types = *[]() -> PythonTypes* {
    py::module xla_module(py::module::import("jax.interpreters.xla"));
    py::object device_array;
    if (py::hasattr(xla_module, "_DeviceArray")) {
      device_array = xla_module.attr("_DeviceArray");
    }
    return new PythonTypes{device_array};
  }();

  if (arg.get_type() == xla::PyArray::type()) {
    auto py_array = py::reinterpret_borrow<xla::PyArray>(arg);
    if (py_array.fastpath_enabled()) {
      if (py_array.num_shards() != 1) {
        return xla::InvalidArgument(
            "Only single-sharded Array is expected in C++ JIT.");
      }

      if (!py_array.committed()) {
        return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{nullptr,
                                                               nullptr};
      }
      return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{
          py_array.ifrt_array()->client(),
          py_array.ifrt_array()->sharding().devices().front()};
    }
  }

  // We specically only deal with DeviceArray (not ShardedDeviceArray).
  // (Can happen in jit(pmap), e.g. "test_jit_nested_donate_ignored").
  if (arg.get_type().ptr() == xla::PyBuffer::type()) {
    xla::PyBuffer* buffer = xla::PyBuffer::AsPyBufferUnchecked(arg);
    if (!buffer->sticky_device()) {
      return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{nullptr, nullptr};
    }
    return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{
        buffer->ifrt_array()->client(), buffer->sticky_device()};
  }

  if (arg.get_type().ptr() == types.device_array.ptr()) {
    if (arg.attr("_device").is_none()) {
      return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{nullptr, nullptr};
    }
    try {
      // This can fail, e.g. for cloud TPU 2VM buffers.
      TF_ASSIGN_OR_RETURN(xla::PyBuffer * buffer,
                          xla::PyBuffer::AsPyBuffer(arg.attr("device_buffer")));
      return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{
          buffer->ifrt_array()->client(),
          buffer->ifrt_array()->sharding().devices().front()};
    } catch (const py::cast_error& e) {
      return xla::InvalidArgument(
          "%s", absl::StrCat("[jaxjit] Unsupported subclass of `DeviceArray`: "
                             "`device_buffer` field is of type ",
                             py::cast<std::string>(
                                 arg.attr("device_buffer").get_type().str()),
                             " while a `PyBuffer` was expected."));
    }
  }

  return std::pair<xla::ifrt::Client*, xla::PjRtDevice*>{nullptr, nullptr};
}

// Compute signature for arguments.
//
// Returns `OkStatus()` on success. Returning an error should lead to
// calling the Python fallback.
xla::Status ComputeSignature(bool jax_enable_x64,
                             xla::ifrt::Client* default_client,
                             xla::PjRtDevice* default_device, bool is_committed,
                             ParsedArgumentsAsBuffers& arguments) {
  tsl::profiler::TraceMe traceme("ComputeSignature");

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  // When the jitted function is not committed, we first check whether any
  // sticky `DeviceArray` is present and on which device they live. See also:
  // https://github.com/google/jax/pull/1884
  // https://github.com/google/jax/pull/1916 for the rationale why the
  // computation follows the data locality.
  // It's also similar to PyTorch's behavior.
  xla::ifrt::Client* ifrt_client = nullptr;
  xla::PjRtDevice* data_device = nullptr;
  if (!is_committed) {
    for (int i = 0; i < num_flat_dynamic_args; ++i) {
      TF_ASSIGN_OR_RETURN(
          auto client_and_device,
          GetJitArgumentStickyDevice(arguments.flat_dynamic_args[i]));
      xla::ifrt::Client* client = client_and_device.first;
      xla::PjRtDevice* device = client_and_device.second;
      if (device) {
        if (data_device && (device != data_device)) {
          throw std::invalid_argument(absl::StrCat(
              "primitive arguments must be colocated on the same device ("
              "C++ jax.jit). Arguments are on devices: ",
              device->DebugString(), " and ", data_device->DebugString()));
        } else {
          ifrt_client = client;
          data_device = device;
        }
      }
    }
  }
  if (!data_device) {
    ifrt_client = default_client;
    data_device = default_device;
  }
  CHECK(data_device);
  arguments.ifrt_client = ifrt_client;
  arguments.signature.device = data_device;

  arguments.signature.dynamic_arg_signatures.reserve(num_flat_dynamic_args);
  for (int i = 0; i < num_flat_dynamic_args; ++i) {
    py::handle arg = arguments.flat_dynamic_args[i];
    TF_ASSIGN_OR_RETURN(auto sig,
                        xla::PyArgSignatureOfValue(arg, jax_enable_x64));
    arguments.signature.dynamic_arg_signatures.push_back(std::move(sig));
  }
  return ::tsl::OkStatus();
}

// Copy buffers to device, skipping pruned arguments.
// Returns `OkStatus()` on success. Returning an error should lead to
// calling the Python fallback.
xla::Status CopyBuffersToDevice(bool jax_enable_x64,
                                const std::vector<bool>& kept_args,
                                ParsedArgumentsAsBuffers& arguments) {
  std::vector<tsl::RCReference<xla::ifrt::Array>>& ifrt_arg_arrays =
      arguments.ifrt_arg_arrays;
  xla::PjRtDevice* data_device = arguments.signature.device;

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  xla::DevicePutOptions options;
  options.squash_64bit_types = !jax_enable_x64;
  options.allow_zero_copy = true;
  ifrt_arg_arrays.reserve(num_flat_dynamic_args);
  for (int i = 0; i < num_flat_dynamic_args; ++i) {
    if (!kept_args[i]) {
      continue;
    }

    py::handle arg = arguments.flat_dynamic_args[i];
    TF_ASSIGN_OR_RETURN(xla::DevicePutResult on_device,
                        DevicePut(arg,
                                  arguments.ifrt_client,
                                  data_device, options));

    ifrt_arg_arrays.push_back(std::move(on_device.ifrt_array));
    if (on_device.owning_pybuffer) {
      arguments.keep_alive_objects.push_back(
          std::move(on_device.owning_pybuffer));
    }
  }
  return ::tsl::OkStatus();
}

void CompiledFunction::PopulateCacheEntry(
    CacheEntry* cache_entry, const CallSignature& signature,
    const py::tuple& out_and_fastpath_data) {
  CHECK_EQ(out_and_fastpath_data.size(), 2);
  if (out_and_fastpath_data[1].is_none()) {
    cache_entry->fall_back_to_python = true;
    return;
  }

  py::tuple executable_handlers_out_tree =
      py::cast<py::tuple>(out_and_fastpath_data[1]);
  auto executable = py::cast<std::shared_ptr<xla::PyLoadedExecutable>>(
      executable_handlers_out_tree.attr("xla_executable"));
  cache_entry->executable = std::move(executable);
  int num_devices =
      cache_entry->executable->ifrt_executable()->addressable_devices().size();
  // The presence of jit(pmap) is detected from Python.
  CHECK_EQ(num_devices, 1);

  auto out_tree = py::cast<xla::PyTreeDef>(
      executable_handlers_out_tree.attr("out_pytree_def"));
  cache_entry->out_pytree_def = std::move(out_tree);

  cache_entry->sticky_device =
      py::cast<std::optional<xla::ClientAndPtr<xla::PjRtDevice>>>(
          executable_handlers_out_tree.attr("sticky_device"));
  auto avals = py::cast<py::list>(executable_handlers_out_tree.attr("avals"));

  cache_entry->out_avals.reserve(avals.size());
  cache_entry->out_weak_types.reserve(avals.size());
  cache_entry->out_dtypes.reserve(avals.size());
  cache_entry->out_shapes.reserve(avals.size());
  for (int i = 0; i < avals.size(); ++i) {
    py::object shaped_array = py::reinterpret_borrow<py::object>(avals[i]);

    cache_entry->out_avals.push_back(shaped_array);
    cache_entry->out_weak_types.push_back(
        py::cast<bool>(shaped_array.attr("weak_type")));
    cache_entry->out_dtypes.push_back(shaped_array.attr("dtype"));
    cache_entry->out_shapes.push_back(
        py::cast<std::vector<int64_t>>(shaped_array.attr("shape")));
  }

  auto shardings =
      py::cast<py::list>(executable_handlers_out_tree.attr("shardings"));
  cache_entry->out_shardings.reserve(shardings.size());
  for (const auto& sharding : shardings) {
    cache_entry->out_shardings.push_back(
        py::reinterpret_borrow<py::object>(sharding));
  }

  auto committed =
      py::cast<py::list>(executable_handlers_out_tree.attr("committed"));
  cache_entry->committed.reserve(shardings.size());
  for (const auto& c : committed) {
    cache_entry->committed.push_back(c.cast<bool>());
  }

  auto kept_var_bitvec =
      py::cast<py::list>(executable_handlers_out_tree.attr("kept_var_bitvec"));
  cache_entry->kept_var_bitvec.reserve(kept_var_bitvec.size());
  for (const auto& b : kept_var_bitvec) {
    cache_entry->kept_var_bitvec.push_back(b.cast<bool>());
  }
}

void CompiledFunction::TryToPopulateDefaultDevice() {
  // The following line calls Python and may release the GIL.
  py::object device_and_is_committed;
  try {
    device_and_is_committed = get_device_();
  } catch (py::error_already_set& e) {
    // Backend or device initialization failed. Handle this in Python.
    always_fallback_to_python_ = true;
    return;
  }
  // If the GIL was released by the call to get_device_, another thread may
  // have filled in default_device_.
  if (!default_device_) {
    try {
      auto default_pydevice = py::cast<xla::ClientAndPtr<xla::PjRtDevice>>(
          device_and_is_committed.attr("default_device"));
      is_committed_ =
          py::cast<bool>(device_and_is_committed.attr("committed_to_device"));
      default_client_ = default_pydevice.client->ifrt_client();
      default_device_ = default_pydevice.contents;
    } catch (const py::cast_error& e) {
      // Pathways, Cloud TPU 2VM, and UPTC runtime.
      always_fallback_to_python_ = true;
    }
  }
}

xla::StatusOr<py::object> CompiledFunction::Call(py::handle callable,
                                                 PyObject* const* args,
                                                 size_t nargs,
                                                 PyObject* kwnames) {
  VLOG(3) << "Calling CompiledFunction " << function_name_;

  // Make sure we trigger a garbage collection on JIT function calls. Otherwise
  // code like
  // f = jit(...)
  // while True:
  //   f(x)
  // may never free temporary buffers for copies of arguments.
  xla::GlobalPyRefManager()->MaybeCollectGarbage();

  auto& global_state = GlobalJitState();
  auto& tls = ThreadLocalJitState();
  if (GetDisableJit()) {
    return py::reinterpret_steal<py::object>(
        JAX_PyObject_Vectorcall(fun_.ptr(), args, nargs, kwnames));
  }

  // Calls the cache_miss_ function. This just calls the Python function; it may
  // return nullptr value if a Python exception is thrown.
  auto cache_miss = [&]() -> py::tuple {
    return py::reinterpret_steal<py::tuple>(
        JAX_PyObject_Vectorcall(cache_miss_.ptr(), args, nargs, kwnames));
  };

  // Call the cache_miss() function, extracting the output data and ignoring
  // the fastpath data. If the cache miss returns a Python error, returns
  // nullptr and leaves the Python error set.
  auto fallback_to_cache_miss = [&]() {
    py::tuple cache_miss_output = cache_miss();
    if (!cache_miss_output.ptr()) {
      return py::object();
    }
    return py::object(cache_miss_output[0]);
  };

  if (always_fallback_to_python_) {
    return fallback_to_cache_miss();
  }

  xla::ifrt::Client* client = nullptr;
  xla::PjRtDevice* device = nullptr;
  // Whether `device` should override an input with a sticky device.
  bool is_committed;
  if (!has_explicit_device_ && GetDefaultDevice().has_value()) {
    xla::ClientAndPtr<xla::PjRtDevice> pjrt_device_ptr;
    bool cast_success = true;
    try {
      pjrt_device_ptr =
          GetDefaultDevice()->cast<xla::ClientAndPtr<xla::PjRtDevice>>();
    } catch (py::cast_error& e) {
      // We assume GetDefaultDevice() returned a non-PJRT device object. Leave
      // `device` unset so we fallback to Python path and handle default device
      // there.
      cast_success = false;
    }
    if (cast_success) {
      client = pjrt_device_ptr.client->ifrt_client();
      device = pjrt_device_ptr.get();
      is_committed = false;
      VLOG(3) << "Using config.default_device (uncommitted): "
              << device->DebugString();
    }
  }
  if (device == nullptr) {
    // Call back into Python to find system default device, which will be stored
    // in default_device_.
    if (!default_device_) {
      // On the first call to `Call`, compute a default device. We need to wait
      // until after platform initialization is complete before doing so, but
      // @jit may be used as a decorator.
      TryToPopulateDefaultDevice();
      if (!default_device_) {
        return fallback_to_cache_miss();
      }
    }
    client = default_client_;
    device = default_device_;
    is_committed = is_committed_;
    VLOG(3) << "Using device from Python): " << device->DebugString()
            << ", committed: " << is_committed;
  }
  CHECK(device != nullptr);

  ParsedArgumentsAsBuffers arguments;
  arguments.signature.function_name = function_name_;
  size_t num_positional_args = PyVectorcall_NARGS(nargs);
  size_t num_keyword_args = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  absl::Span<PyObject* const> positional_args(args, num_positional_args);
  absl::Span<PyObject* const> keyword_args(args + num_positional_args,
                                           num_keyword_args);
  xla::Status status =
      ParseArguments(positional_args, keyword_args, kwnames, static_argnums_,
                     static_argnames_, arguments);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  bool jax_enable_x64 = GetEnableX64();
  arguments.signature.jax_enable_x64 = jax_enable_x64;
  arguments.signature.jax_array = GetEnableJaxArray();
  // The C++ jit do not support Tracers arguments inputs yet. The Python-based
  // jit function will be called if any of the dynamic arguments is unsupported.
  status =
      ComputeSignature(jax_enable_x64, client, device, is_committed, arguments);
  if (!status.ok()) {
    VLOG(2) << "ComputeSignature failed: " << status;
    return fallback_to_cache_miss();
  }
  arguments.signature.global_extra_jit_context = global_state.extra_jit_context;
  arguments.signature.thread_local_extra_jit_context = tls.extra_jit_context;

  VLOG(3) << "CallSignature:\n" << arguments.signature.DebugString();
  bool inserted = false;
  std::shared_ptr<CacheEntry> cache_entry = executables_->GetOrCreateIfAbsent(
      arguments.signature, [&inserted](const CallSignature& key) {
        inserted = true;
        return std::make_shared<CacheEntry>();
      });

  if (!cache_entry->compilation_complete.HasBeenNotified()) {
    // In case of several threads attempting to compile the executable, only
    // the one that inserted the item will perform the compilation.
    if (inserted) {
      py::object out_and_fastpath_data;
      py::tuple out_tuple;
      VLOG(2) << "Cache miss for\n" << arguments.signature.DebugString();
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = cache_miss();
        if (!out_and_fastpath_data.ptr()) {
          throw py::error_already_set();
        }
        out_tuple = py::cast<py::tuple>(out_and_fastpath_data);
        PopulateCacheEntry(cache_entry.get(), arguments.signature, out_tuple);
      } catch (const std::exception& e) {
        cache_entry->fall_back_to_python = true;
        cache_entry->compilation_complete.Notify();
        throw;
      }
      cache_entry->compilation_complete.Notify();

      // We have already computed the result in the miss path so we can return
      // it. We are even *required* to do so if there are donated arguments,
      // because any donated buffers will now be invalid.
      return py::object(out_tuple[0]);
    } else {
      if (cache_entry->thread_id == std::this_thread::get_id()) {
        auto error_string = absl::StrCat("Recursively calling jit: ",
                                         arguments.signature.DebugString());
        PyErr_SetString(PyExc_RecursionError, error_string.c_str());
        throw pybind11::error_already_set();
      }
      // Release the GIL while we wait, making sure the compile thread can
      // lock it.
      py::gil_scoped_release release;
      cache_entry->compilation_complete.WaitForNotification();
    }
  }
  // It's hard to reraise the exact same kind of errors when a compilation error
  // occurred. If the first compilation failed, other threads will also execute
  // the Python path.
  if (cache_entry->fall_back_to_python) {
    VLOG(2) << "fallback to python: " << function_name_;
    return fallback_to_cache_miss();
  }

  status = CopyBuffersToDevice(jax_enable_x64, cache_entry->kept_var_bitvec,
                               arguments);
  if (!status.ok()) {
    VLOG(2) << "CopyBuffersToDevice failed: " << status;
    return fallback_to_cache_miss();
  }

// Executes the computation.
  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(auto result,
                        cache_entry->executable->ifrt_executable()->Execute(
                            absl::MakeSpan(arguments.ifrt_arg_arrays),
                            cache_entry->executable->options(),
                            /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
  }
  auto traceback = xla::Traceback::Get();

  int num_outputs = output_arrays.size();
  absl::InlinedVector<py::object, 1> flat_device_arrays;
  flat_device_arrays.reserve(num_outputs);

  if (!cache_entry->out_shardings.empty()) {
    for (int i = 0; i < output_arrays.size(); ++i) {
      xla::PyArray array(
          cache_entry->out_avals[i], cache_entry->out_weak_types[i],
          cache_entry->out_dtypes[i], cache_entry->out_shapes[i],
          cache_entry->out_shardings.at(i), cache_entry->executable->client(),
          traceback, std::move(output_arrays[i]),
          /*committed=*/cache_entry->committed.at(i), /*skip_checks=*/true);
      flat_device_arrays.push_back(std::move(array));
    }
  } else {
    for (int i = 0; i < output_arrays.size(); ++i) {
      bool last = (i == (num_outputs - 1));
      xla::PyBuffer::object buffer = xla::PyBuffer::Make(
          cache_entry->executable->client(), std::move(output_arrays[i]),
          last ? std::move(traceback) : traceback);
      buffer.buf()->SetAval(cache_entry->out_avals[i]);
      buffer.buf()->set_weak_type(cache_entry->out_weak_types[i]);
      if (cache_entry->sticky_device.has_value()) {
        TF_RETURN_IF_ERROR(buffer.buf()->set_sticky_device(
            (*cache_entry->sticky_device).get()));
      }
      flat_device_arrays.push_back(std::move(buffer));
    }
  }
  py::object out = cache_entry->out_pytree_def.Unflatten(flat_device_arrays);

  // If there is a post-hook function, call it with the inputs and the outputs.
  std::optional<py::object> post_hook = GetPostHook();
  if (post_hook) {
    py::tuple args_tuple(num_positional_args);
    for (size_t i = 0; i < num_positional_args; ++i) {
      args_tuple[i] = args[i];
    }
    py::dict kwargs;
    if (kwnames) {
      for (size_t i = 0; i < num_keyword_args; ++i) {
        kwargs[py::handle(PyTuple_GET_ITEM(kwnames, i))] =
            args[num_positional_args + i];
      }
    }

    (*post_hook)(callable, args_tuple, kwargs, out);
  }
  return std::move(out);
}

struct JaxCompiledFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
  vectorcallfunc vectorcall;
  CompiledFunction fun;
};

PyObject* JaxCompiledFunction_Type = nullptr;

bool CompiledFunction::IsCompiledFunction(py::handle handle) {
  return handle.get_type() == JaxCompiledFunction_Type;
}

CompiledFunction* CompiledFunction::AsCompiledFunctionUnchecked(
    py::handle handle) {
  return &(reinterpret_cast<JaxCompiledFunctionObject*>(handle.ptr())->fun);
}

xla::StatusOr<CompiledFunction*> AsCompiledFunction(py::handle handle) {
  if (!CompiledFunction::IsCompiledFunction(handle)) {
    return xla::InvalidArgument("Expected a CompiledFunction");
  }
  return CompiledFunction::AsCompiledFunctionUnchecked(handle);
}

extern "C" {

PyObject* JaxCompiledFunction_tp_vectorcall(PyObject* callable,
                                            PyObject* const* args, size_t nargs,
                                            PyObject* kwnames) {
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(callable);
  tsl::profiler::TraceMe traceme([&] {
    return absl::StrCat("JaxCompiledFunction(", o->fun.function_name(), ")");
  });
  try {
    xla::StatusOr<py::object> out = o->fun.Call(callable, args, nargs, kwnames);
    if (!out.ok()) {
      PyErr_SetString(PyExc_ValueError, out.status().ToString().c_str());
      return nullptr;
    }
    return out.value().release().ptr();
  } catch (py::error_already_set& e) {
    e.restore();
    return nullptr;
  } catch (py::cast_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return nullptr;
  } catch (std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return nullptr;
  } catch (std::runtime_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return nullptr;
  }
}

PyObject* JaxCompiledFunction_tp_new(PyTypeObject* subtype, PyObject* args,
                                     PyObject* kwds) {
  JaxCompiledFunctionObject* self =
      reinterpret_cast<JaxCompiledFunctionObject*>(
          subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->dict = nullptr;
  self->weakrefs = nullptr;
  self->vectorcall = JaxCompiledFunction_tp_vectorcall;
  return reinterpret_cast<PyObject*>(self);
}

void JaxCompiledFunction_tp_dealloc(PyObject* self) {
  PyTypeObject* tp = Py_TYPE(self);
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(self);
  if (o->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  Py_CLEAR(o->dict);
  o->fun.~CompiledFunction();
  tp->tp_free(self);
  Py_DECREF(tp);
}

int JaxCompiledFunction_tp_traverse(PyObject* self, visitproc visit,
                                    void* arg) {
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(self);
#if PY_VERSION_HEX >= 0x03090000
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
#endif
  Py_VISIT(o->dict);
  Py_VISIT(o->fun.fun().ptr());
  Py_VISIT(o->fun.cache_miss().ptr());
  Py_VISIT(o->fun.get_device().ptr());
  return 0;
}

int JaxCompiledFunction_tp_clear(PyObject* self) {
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(self);
  Py_CLEAR(o->dict);
  o->fun.ClearPythonReferences();
  return 0;
}

// Implements the Python descriptor protocol so JIT-compiled functions can be
// used as bound methods. See:
// https://docs.python.org/3/howto/descriptor.html#functions-and-methods
PyObject* JaxCompiledFunction_tp_descr_get(PyObject* self, PyObject* obj,
                                           PyObject* type) {
  if (obj == nullptr || obj == Py_None) {
    Py_INCREF(self);
    return self;
  }
  return PyMethod_New(self, obj);
}

// Support d = instance.__dict__.
PyObject* JaxCompiledFunction_get_dict(PyObject* self, void*) {
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(self);
  if (!o->dict) {
    o->dict = PyDict_New();
  }
  Py_XINCREF(o->dict);
  return o->dict;
}

int JaxCompiledFunction_set_dict(PyObject* self, PyObject* new_dict, void*) {
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(self);
  if (!PyDict_Check(new_dict)) {
    PyErr_Format(PyExc_TypeError,
                 "__dict__ must be set to a dictionary, not a '%s'",
                 Py_TYPE(new_dict)->tp_name);
    return -1;
  }
  Py_INCREF(new_dict);
  Py_CLEAR(o->dict);
  o->dict = new_dict;
  return 0;
}

static PyGetSetDef JaxCompiledFunction_tp_getset[] = {
    // Having a __dict__ seems necessary to allow !functool.wraps to override
    // __doc__.
    {const_cast<char*>("__dict__"), JaxCompiledFunction_get_dict,
     JaxCompiledFunction_set_dict, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyObject* JaxCompiledFunction_tp_repr(PyObject* self) {
  try {
    const std::string& repr = absl::StrFormat(
        "<CompiledFunction of %s>",
        static_cast<std::string>(py::repr(py::getattr(self, "__wrapped__"))));
    return PyUnicode_FromString(repr.c_str());
  } catch (...) {
    // Ignore all errors when accessing a repr.
    return PyUnicode_FromString("<CompiledFunction>");
  }
}

void InitializeCompiledFunction(JaxCompiledFunctionObject* cfun,
                                py::function fun, py::function cache_miss,
                                py::function get_device,
                                bool has_explicit_device,
                                std::vector<int> static_argnums,
                                std::vector<py::str> static_argnames,
                                std::vector<int> donate_argnums,
                                std::shared_ptr<CompiledFunctionCache> cache) {
  new (&cfun->fun) CompiledFunction(
      std::move(fun), std::move(cache_miss), std::move(get_device),
      has_explicit_device, std::move(static_argnums),
      std::move(static_argnames), std::move(donate_argnums), std::move(cache));
}

}  // extern "C"

py::object MakeCompiledFunction(py::function fun, py::function cache_miss,
                                py::function get_device,
                                bool has_explicit_device,
                                std::vector<int> static_argnums,
                                std::vector<py::str> static_argnames,
                                std::vector<int> donate_argnums,
                                std::shared_ptr<CompiledFunctionCache> cache) {
  py::object obj = py::reinterpret_steal<py::object>(JaxCompiledFunction_tp_new(
      reinterpret_cast<PyTypeObject*>(JaxCompiledFunction_Type), nullptr,
      nullptr));
  JaxCompiledFunctionObject* buf =
      reinterpret_cast<JaxCompiledFunctionObject*>(obj.ptr());
  if (!cache) {
    cache = std::make_shared<CompiledFunctionCache>(
        CompiledFunctionCache::kDefaultCapacity);
  }
  InitializeCompiledFunction(
      buf, std::move(fun), std::move(cache_miss), std::move(get_device),
      has_explicit_device, std::move(static_argnums),
      std::move(static_argnames), std::move(donate_argnums), std::move(cache));
  return obj;
}

// Version numbers for the pickled representations of
// CompiledFunction/CompiledFunctionCache. Increment these if changing them.
const int kCompiledFunctionCachePickleVersion = 1;
const int kCompiledFunctionPickleVersion = 1;

}  // namespace

void BuildJaxjitSubmodule(py::module& m) {
  py::module jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  py::class_<CompiledFunctionCache, std::shared_ptr<CompiledFunctionCache>>
      cache(jitlib, "CompiledFunctionCache");
  cache.def(py::init<int>(),
            py::arg("capacity") = CompiledFunctionCache::kDefaultCapacity);
  cache.def("size", &CompiledFunctionCache::Size);
  cache.def("capacity", &CompiledFunctionCache::Capacity);
  cache.def("clear", &CompiledFunctionCache::Clear);
  cache.def_static("clear_all", []() {
    GetGlobalCompiledFunctionStore().ClearFunctionCache();
  });
  cache.def(py::pickle(
      // __getstate__
      // Pickles as an empty cache; the client can repopulate as needed.
      [](const CompiledFunctionCache& cache) {
        py::dict pickle;
        pickle["version"] = kCompiledFunctionCachePickleVersion;
        pickle["capacity"] = cache.Capacity();
        return pickle;
      },
      // __setstate__
      [](const py::dict& pickle) {
        int version = py::cast<int>(pickle["version"]);
        if (version != kCompiledFunctionCachePickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid CompiledFunction pickle version, got %d, expected %d",
              version, kCompiledFunctionCachePickleVersion));
        }
        int capacity = py::cast<int>(pickle["capacity"]);
        return std::make_shared<CompiledFunctionCache>(capacity);
      }));

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  py::object cfun;
  {
    py::str name = py::str("CompiledFunction");
    py::str qualname = py::str("CompiledFunction");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called. Otherwise the GC might see a half-constructed
    // type object.
    CHECK(heap_type) << "Unable to create heap type object";
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "CompiledFunction";
    type->tp_basicsize = sizeof(JaxCompiledFunctionObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE |
                     Py_TPFLAGS_HAVE_GC | JAX_TPFLAGS_HAVE_VECTORCALL;
    type->tp_new = JaxCompiledFunction_tp_new;
    type->tp_dealloc = JaxCompiledFunction_tp_dealloc;
    type->tp_dictoffset = offsetof(JaxCompiledFunctionObject, dict);
    type->tp_traverse = JaxCompiledFunction_tp_traverse;
    type->tp_clear = JaxCompiledFunction_tp_clear;
    type->tp_weaklistoffset = offsetof(JaxCompiledFunctionObject, weakrefs);
    type->tp_getset = JaxCompiledFunction_tp_getset;
    type->tp_descr_get = JaxCompiledFunction_tp_descr_get;
    type->tp_call = PyVectorcall_Call;
    type->tp_vectorcall_offset =
        offsetof(JaxCompiledFunctionObject, vectorcall);
    type->tp_repr = JaxCompiledFunction_tp_repr;
    CHECK_EQ(PyType_Ready(type), 0);
    JaxCompiledFunction_Type = reinterpret_cast<PyObject*>(type);
    cfun = py::reinterpret_borrow<py::object>(JaxCompiledFunction_Type);
  }
  py::object cfun_type =
      py::reinterpret_borrow<py::object>(JaxCompiledFunction_Type);

  // Add CompiledFunction to the xla_extension module so it can be pickled.
  m.attr("CompiledFunction") = cfun_type;
  cfun.attr("__module__") = m.attr("__name__");

  cfun.attr("__signature__") =
      property_readonly([](py::handle self) -> xla::StatusOr<py::object> {
        TF_ASSIGN_OR_RETURN(CompiledFunction * fun, AsCompiledFunction(self));
        return fun->PythonSignature();
      });
  cfun.attr("_cache_miss") =
      property_readonly([](py::handle self) -> xla::StatusOr<py::object> {
        TF_ASSIGN_OR_RETURN(CompiledFunction * fun, AsCompiledFunction(self));
        return fun->cache_miss();
      });
  cfun.attr("__getstate__") = py::cpp_function(
      [](const CompiledFunction::object& self) {
        CompiledFunction* fn = self.func();
        py::dict pickle;
        pickle["version"] = kCompiledFunctionPickleVersion;
        pickle["fun"] = fn->fun();
        pickle["cache_miss"] = fn->cache_miss();
        pickle["get_device"] = fn->get_device();
        pickle["has_explicit_device"] = fn->has_explicit_device();
        pickle["static_argnums"] = fn->static_argnums();
        pickle["static_argnames"] = fn->static_argnames();
        pickle["donate_argnums"] = fn->donate_argnums();
        pickle["cache"] = fn->cache();
        return pickle;
      },
      py::is_method(cfun_type));
  cfun.attr("__setstate__") = py::cpp_function(
      [](CompiledFunction::object& self, const py::dict& pickle) {
        int version = py::cast<int>(pickle["version"]);
        if (version != kCompiledFunctionPickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid CompiledFunction pickle version, got %d, expected %d. "
              "Pickling/Unpickling jitted functions using different JAX "
              "versions is not supported.",
              version, kCompiledFunctionPickleVersion));
        }
        py::function fun = py::cast<py::function>(pickle["fun"]);
        py::function cache_miss = py::cast<py::function>(pickle["cache_miss"]);
        py::function get_device = py::cast<py::function>(pickle["get_device"]);
        bool has_explicit_device =
            py::cast<bool>(pickle["has_explicit_device"]);
        std::vector<int> static_argnums =
            py::cast<std::vector<int>>(pickle["static_argnums"]);
        std::vector<py::str> static_argnames =
            py::cast<std::vector<py::str>>(pickle["static_argnames"]);
        std::vector<int> donate_argnums =
            py::cast<std::vector<int>>(pickle["donate_argnums"]);
        std::shared_ptr<CompiledFunctionCache> cache =
            py::cast<std::shared_ptr<CompiledFunctionCache>>(pickle["cache"]);
        InitializeCompiledFunction(
            reinterpret_cast<JaxCompiledFunctionObject*>(self.ptr()),
            std::move(fun), std::move(cache_miss), std::move(get_device),
            has_explicit_device, std::move(static_argnums),
            std::move(static_argnames), std::move(donate_argnums),
            std::move(cache));
      },
      py::is_method(cfun_type));

  py::class_<JitState> jit_state_(jitlib, "JitState");
  jit_state_.def_readwrite("disable_jit", &JitState::disable_jit);
  jit_state_.def_readwrite("enable_x64", &JitState::enable_x64);
  jit_state_.def_readwrite("jax_array", &JitState::jax_array);
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

  jitlib.def(
      "jit",
      [](py::function fun, py::function cache_miss, py::function get_device,
         std::vector<int> static_argnums, std::vector<py::str> static_argnames,
         std::vector<int> donate_argnums, bool has_explicit_device,
         std::shared_ptr<CompiledFunctionCache> cache) -> py::object {
        return MakeCompiledFunction(
            std::move(fun), std::move(cache_miss), std::move(get_device),
            has_explicit_device, std::move(static_argnums),
            std::move(static_argnames), std::move(donate_argnums),
            std::move(cache));
      },
      py::arg("fun"), py::arg("cache_miss"), py::arg("get_device"),
      py::arg("static_argnums"),
      py::arg("static_argnames") = std::vector<py::str>(),
      py::arg("donate_argnums") = std::vector<int>(),
      py::arg("has_explicit_device") = false, py::arg("cache") = nullptr);

  // This function is not yet a full replacement for the Python one, because:
  // (a) it does not support abstract types,
  // (b) it does not set the device stickiness yet.
  // TODO(jblespiau): Finish the replacement of the Python feature.
  jitlib.def(
      "device_put",
      [](py::handle obj, bool jax_enable_x64,
         xla::ClientAndPtr<xla::PjRtDevice> to_device)
          -> xla::StatusOr<py::object> {
        std::shared_ptr<xla::PyClient>& pyclient = to_device.client;
        xla::DevicePutOptions options;
        options.squash_64bit_types = !jax_enable_x64;
        options.allow_zero_copy = true;
        xla::StatusOr<xla::DevicePutResult> results = DevicePut(
            obj, pyclient->ifrt_client(), to_device.contents, options);
        if (!results.ok()) {
          throw xla::XlaRuntimeError(results.status().error_message());
        }
        if (results->ifrt_array) {
          auto buffer = xla::PyBuffer::Make(
              pyclient, std::move(results->ifrt_array), xla::Traceback::Get());

          static const auto* jax_core =
              new py::module(py::module::import("jax.core"));
          static const auto* shaped_array =
              new py::handle(jax_core->attr("ShapedArray"));
          buffer.buf()->SetAval((*shaped_array)(buffer.buf()->python_shape(),
                                                buffer.buf()->python_dtype(),
                                                results->weak_type));
          TF_RETURN_IF_ERROR(buffer.buf()->set_sticky_device(nullptr));

          return std::move(buffer);
        } else {
          return py::cast<py::object>(obj);
        }
      });

  py::class_<xla::PyArgSignature> arg_signature(jitlib, "PyArgSignature");
  arg_signature
      .def_property_readonly("dtype",
                             [](const xla::PyArgSignature& sig) {
                               return PrimitiveTypeToDtype(sig.dtype);
                             })
      .def_property_readonly(
          "shape",
          [](const xla::PyArgSignature& sig) {
            return xla::SpanToTuple(absl::MakeConstSpan(sig.shape));
          })
      .def_readonly("weak_type", &xla::PyArgSignature::weak_type);
  jitlib.def("_ArgSignatureOfValue", &xla::PyArgSignatureOfValue);

  // All private members are only for testing/debugging purposes
  cfun.attr("_cache_size") = py::cpp_function(
      [](py::handle self) -> xla::StatusOr<int> {
        TF_ASSIGN_OR_RETURN(CompiledFunction * fun, AsCompiledFunction(self));
        return fun->cache_size();
      },
      py::is_method(cfun));
  cfun.attr("_clear_cache") = py::cpp_function(
      [](py::handle self) -> xla::Status {
        TF_ASSIGN_OR_RETURN(CompiledFunction * fun, AsCompiledFunction(self));
        fun->ClearCache();
        return ::tsl::OkStatus();
      },
      py::is_method(cfun));
  jitlib.def("_is_float0", &xla::IsFloat0);
}

}  // namespace jax
