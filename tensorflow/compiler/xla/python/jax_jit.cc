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

#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/lru_cache.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace jax {

namespace py = pybind11;

// TODO(phawkins): Add support for Tracers.
// TODO(jblespiau): Use absl Status.

namespace {

// Flags, such as JIT disable and the x64 mode, are controlled by:
// - a global flag value, e.g., associated to --jax_enable_x64
// - possibly a thread-local value, which initially is absl::nullopt and
//   overrides the global value if set. The thread-local state is
//   used to implement context managers that locally override the global state.
// TODO(phawkins): consider changing the global state to optional types to
// catch cases where we fail to set it.
struct GlobalJitState {
  bool disable_jit = false;
  bool enable_x64 = false;

  // Extra context that should be included in the JIT cache key. Must be
  // hashable and have an equality defined.
  py::object extra_jit_context = py::none();

  // A callback that, if present, is called when a JITted function is executed
  // from cache.
  absl::optional<py::function> post_hook;
};

// Protected by the GIL.
GlobalJitState& global_state = *new GlobalJitState();

struct ThreadLocalJitState {
  ~ThreadLocalJitState() {
    if (extra_jit_context) {
      // We likely do not hold the GIL, so we hand the Python object to the
      // global reference manager to destroy.
      py::object o = std::move(*extra_jit_context);
      xla::GlobalPyRefManager()->AddGarbage(absl::MakeSpan(&o, 1));
      extra_jit_context = absl::nullopt;
    }
  }
  absl::optional<bool> disable_jit;
  absl::optional<bool> enable_x64;
  absl::optional<py::object> extra_jit_context;
  absl::optional<py::function> post_hook;
};

// TODO(phawkins): Google style guide forbids thread-local values with
// non-trivial destructors.
ABSL_CONST_INIT thread_local ThreadLocalJitState thread_local_state;  // NOLINT

bool JitIsDisabled() {
  return thread_local_state.disable_jit.value_or(global_state.disable_jit);
}

}  // namespace

bool GetEnableX64() {
  return thread_local_state.enable_x64.value_or(global_state.enable_x64);
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
  return absl::StrFormat(
      "static args (positional + keyword): %s\nstatic arg keyword names: %s\n"
      "dynamic arg signatures (positional + keyword): %s\n"
      "dynamic arg keyword names: %s\ndynamic arg treedefs: %s\n",
      absl::StrJoin(static_args, ",", py_object_formatter),
      absl::StrJoin(static_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_signatures, ", ", signature_formatter),
      absl::StrJoin(dynamic_arg_names, ",", py_object_formatter),
      absl::StrJoin(dynamic_arg_treedefs, "| ", treedef_formatter));
}

bool CallSignature::operator==(const CallSignature& other) const {
  return std::tie(dynamic_arg_treedefs, dynamic_arg_names,
                  dynamic_arg_signatures, device, jax_enable_x64,
                  static_arg_names) ==
             std::tie(other.dynamic_arg_treedefs, other.dynamic_arg_names,
                      other.dynamic_arg_signatures, other.device,
                      other.jax_enable_x64, static_arg_names) &&
         // `==` on py:objects is the Python `is`. We need equal.
         std::equal(
             static_args.begin(), static_args.end(), other.static_args.begin(),
             other.static_args.end(),
             [](const py::object& a, const py::object& b) {
               try {
                 return a.equal(b);
               } catch (const py::error_already_set& e) {
                 throw std::invalid_argument(absl::StrCat(
                     "static arguments should be comparable using __eq__."
                     "The following error was raised when comparing two "
                     "objects of types ",
                     py::cast<std::string>(py::str(py::type::of(a))), " and ",
                     py::cast<std::string>(py::str(py::type::of(b))),
                     ". The error was:\n", e.what()));
               }
             }) &&
         global_extra_jit_context.equal(other.global_extra_jit_context) &&
         (thread_local_extra_jit_context.has_value() ==
          other.thread_local_extra_jit_context.has_value()) &&
         (!thread_local_extra_jit_context.has_value() ||
          thread_local_extra_jit_context->equal(
              *other.thread_local_extra_jit_context));
}

template <typename H>
H AbslHashValue(H h, const CallSignature& s) {
  h = H::combine_contiguous(std::move(h), s.dynamic_arg_treedefs.data(),
                            s.dynamic_arg_treedefs.size());
  for (const auto& name : s.dynamic_arg_names) {
    h = H::combine(std::move(h), name.ptr());
  }
  h = H::combine_contiguous(std::move(h), s.dynamic_arg_signatures.data(),
                            s.dynamic_arg_signatures.size());
  for (const auto& static_arg : s.static_args) {
    ssize_t hash;
    try {
      hash = py::hash(static_arg);
    } catch (const py::error_already_set& e) {
      throw std::invalid_argument(absl::StrCat(
          "Non-hashable static arguments are not supported. An error occured "
          "while trying to hash an object of type ",
          py::cast<std::string>(py::str(py::type::of(static_arg))), ", ",
          py::cast<std::string>(py::str(static_arg)), ". The error was:\n",
          e.what(), "\n"));
    }
    h = H::combine(std::move(h), hash);
  }
  for (const auto& name : s.static_arg_names) {
    h = H::combine(std::move(h), name.ptr());
  }
  h = H::combine(std::move(h), s.device, s.jax_enable_x64);

  // We do not hash the extra_jit_context fields since calling Python hash
  // functions is expensive (~300ns) and we don't expect a large number of
  // different contexts.
  return h;
}

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(py::handle args,
                           const absl::optional<py::kwargs>& py_kwargs,
                           absl::Span<int const> static_argnums,
                           absl::Span<py::str const> static_argnames,
                           ParsedArgumentsAsBuffers& arguments) {
  tensorflow::profiler::TraceMe traceme("ParseArguments");
  int num_args = PyTuple_GET_SIZE(args.ptr());
  int num_kwargs = py_kwargs ? py_kwargs->size() : 0;
  if (static_argnums.size() > num_args) {
    return xla::InvalidArgument(
        "%s", "[jaxjit] Error with static argnums, executing the Python path.");
  }

  arguments.flat_dynamic_args.reserve(num_args + num_kwargs -
                                      static_argnums.size());
  if (static_argnums.empty()) {
    arguments.signature.dynamic_arg_treedefs.resize(num_args);

    // Positional arguments.
    for (int i = 0; i < num_args; ++i) {
      xla::PyTreeDef& pytree_def = arguments.signature.dynamic_arg_treedefs[i];
      pytree_def.FlattenInto(PyTuple_GET_ITEM(args.ptr(), i),
                             arguments.flat_dynamic_args);
    }
  } else {
    arguments.signature.dynamic_arg_treedefs.reserve(num_args -
                                                     static_argnums.size());

    // Positional arguments.
    for (int i = 0; i < num_args; ++i) {
      if (std::find(static_argnums.begin(), static_argnums.end(), i) ==
          static_argnums.end()) {
        arguments.signature.dynamic_arg_treedefs.emplace_back();
        xla::PyTreeDef& pytree_def =
            arguments.signature.dynamic_arg_treedefs.back();
        pytree_def.FlattenInto(PyTuple_GET_ITEM(args.ptr(), i),
                               arguments.flat_dynamic_args);
      } else {
        arguments.signature.static_args.emplace_back(
            py::reinterpret_borrow<py::object>(
                PyTuple_GET_ITEM(args.ptr(), i)));
      }
    }
  }

  // Keyword arguments.
  if (py_kwargs) {
    std::vector<std::pair<py::handle, py::handle>> kwargs(py_kwargs->begin(),
                                                          py_kwargs->end());
    // We first intern the keys, then sort them (by name, as in the Python path)
    // (see also xla::PyTreeDef::Flatten) and then create the signatures.
    // TODO(jblespiau): We should be able to sort the keys by interned-key
    // pointers, but this requires the Python compilation to do the same.
    for (int i = 0; i < num_kwargs; ++i) {
      // Intern the key if not already interned.
      kwargs[i].first.inc_ref();
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

    arguments.signature.dynamic_arg_names.reserve(num_kwargs);
    for (int i = 0; i < num_kwargs; ++i) {
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
  return xla::Status::OK();
}

namespace {

// Elements of CacheEntry are protected by the GIL.
struct CacheEntry {
  // Ensures a single thread performs the compilation for a given executable.
  //
  // The first thread (holding the GIL) will create the CacheEntry associated to
  // a signature and fill it. Other threads will wait for the notification.
  // If an error occured during the compilation, `fall_back_to_python` is set
  // to `true`, and other threads will fail with the same error.
  absl::Notification compilation_complete;

  std::shared_ptr<xla::PyExecutable> executable;
  xla::PyTreeDef out_pytree_def;
  // We use Python types within the vector because this is what we will be
  // returning to Python. No need to convert back and forth.
  // We need py::object to maintain the objects alive.
  std::vector<py::object> out_avals;
  std::vector<bool> out_weak_types;

  // The processing done in `AddCacheEntry` ensures that LazyExpr are stored as
  // `py::none()`.
  std::vector<py::object> out_lazy_exprs;

  // Bitvector of kept arguments from Jaxpr DCE pass. Used to drop some `args`
  // in CompiledFunction::Call before calling into compiled computation.
  absl::optional<std::vector<bool>> kept_var_bitvec;
  xla::PjRtDevice* sticky_device;

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

  // The lifetime of the returned cache lasts at least as long as the lifetime
  // of `function`, so if the caller holds a strong reference to `function`, the
  // `Cache` remains valid.
  // We include as part of the cache key `donate_argnums` (and any other fields
  // that aren't subsumed by the CallSignature we compute for each call).
  Cache* Lookup(py::handle function, absl::Span<const int> donate_argnums);

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
    explicit Value(Cache::LRUList* lru_list) : cache(lru_list) {}
    Cache cache;

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

CompiledFunctionCache::Cache* CompiledFunctionCache::Lookup(
    py::handle function, absl::Span<const int> donate_argnums) {
  Key key;
  key.function = function;
  key.donate_argnums =
      std::vector<int>(donate_argnums.begin(), donate_argnums.end());
  auto insert = functions_.emplace(key, nullptr);
  std::unique_ptr<Value>& entry = insert.first->second;
  if (insert.second) {
    entry = std::make_unique<Value>(&lru_list_);
    entry->weakref = py::weakref(
        function,
        py::cpp_function([this, key{std::move(key)}](py::handle weakref) {
          functions_.erase(key);
        }));
  }
  return &entry->cache;
}

// A `CompiledFunction` is associated to a `jax.jit(f)` and takes care of the
// bookkeeping of the different signatures used and the dispatch of calls to
// the correct underlying `PyExecutable`. This class is thread-safe.
class CompiledFunction {
 public:
  CompiledFunction(py::function fun, py::function cache_miss,
                   py::function get_device, std::vector<int> static_argnums,
                   std::vector<py::str> static_argnames,
                   std::vector<int> donate_argnums,
                   std::shared_ptr<CompiledFunctionCache> cache);
  ~CompiledFunction();

  // This function will:
  // (a) flatten the inputs using pytree
  // (b) get buffer objects from the arguments
  // (c) call the executable
  // (d) construct `DeviceArray` objects from the outputs
  // (e) reconstruct the `PyTree`.
  xla::StatusOr<py::object> Call(py::handle args,
                                 absl::optional<py::kwargs> kwargs);

  // This allows `inspect.signature(cpp_jitted_f)` from Python.
  py::object PythonSignature() {
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(fun_);
  }

  int cache_size() const { return executables_->Size(); }
  void ClearCache() { executables_->Clear(); }

  const py::function& fun() const { return fun_; }
  const py::function& cache_miss() const { return cache_miss_; }
  const py::function& get_device() const { return get_device_; }

  // Helper function used by the tp_clear GC method.
  void ClearPythonReferences() {
    py::function fun, cache_miss, get_device;
    // Swap values for nulls before they are destroyed. See the Python
    // Py_CLEAR() documentation for a discussion of this topic.
    std::swap(fun_, fun);
    std::swap(cache_miss_, cache_miss);
    std::swap(get_device_, get_device);
  }

  py::handle AsPyHandle();

 private:
  // Attempts to populate default_device_. May release the GIL; is
  // reentrant-safe.
  void TryToPopulateDefaultDevice();

  void PopulateCacheEntry(CacheEntry* entry, const CallSignature& signature,
                          const py::tuple& out_and_fastpath_data);
  bool always_fallback_to_python_ = false;

  py::function fun_;  // The Python function to jit.
  // See JAX _cpp_jit in api.py for documentation.
  py::function cache_miss_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyExecutable. In sorted order.
  std::vector<int> static_argnums_;
  // Keyword arguments, interned.
  std::vector<py::str> static_argnames_;

  // A function taking no arguments and returning the default device and whether
  // jax.jit has been committed to it.
  py::function get_device_;

  // Keeps the shared LRU cache alive as long as the CompiledFunction is alive.
  std::shared_ptr<CompiledFunctionCache> cache_;

  // The part of cache_ specific to this CompiledFunction.
  CompiledFunctionCache::Cache* executables_;

  // The logic if the following:
  // - if `device` or `backend` are not specified to `jax.jit`, we will use
  //   the input sticky buffer device, or `default_device_` if there is no
  //   such sticky buffer.
  // - When one of `device` or `backend` is specified, this will determine
  //   the `default_device_` which will be used as the targeted device. In
  //   which case, we will always copy input buffers to this device.
  // These fields are protected by the GIL.
  std::shared_ptr<xla::PyClient> default_pyclient_ = nullptr;
  xla::PjRtDevice* default_device_ = nullptr;
  bool is_committed_;
};

CompiledFunction::CompiledFunction(py::function fun, py::function cache_miss,
                                   py::function get_device,
                                   std::vector<int> static_argnums,
                                   std::vector<py::str> static_argnames,
                                   std::vector<int> donate_argnums,
                                   std::shared_ptr<CompiledFunctionCache> cache)
    : fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      static_argnames_(std::move(static_argnames)),
      get_device_(std::move(get_device)),
      cache_(std::move(cache)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
  for (py::str& s : static_argnames) {
    PyUnicode_InternInPlace(&s.ptr());
  }
  executables_ = cache_->Lookup(fun_, donate_argnums);
}

CompiledFunction::~CompiledFunction() = default;

// Compute signature for arguments.
//
// Returns `Status::OK()` on success. Returning an error should lead to
// calling the Python fallback.
xla::Status ComputeSignature(bool jax_enable_x64, xla::PyClient& pyclient,
                             xla::PjRtDevice* default_device, bool is_committed,
                             ParsedArgumentsAsBuffers& arguments) {
  tensorflow::profiler::TraceMe traceme("ComputeSignature");

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  struct PythonTypes {
    py::object device_array;
  };
  static const auto& types = *[]() -> PythonTypes* {
    py::module xla_module(py::module::import("jax.interpreters.xla"));
    py::object device_array(xla_module.attr("_DeviceArray"));
    return new PythonTypes{device_array};
  }();
  // When the jitted function is not committed, we first check whether any
  // sticky `DeviceArray` is present and on which device they live. See also:
  // https://github.com/google/jax/pull/1884
  // https://github.com/google/jax/pull/1916 for the rationale why the
  // computation follows the data locality.
  // It's also similar to PyTorch's behavior.
  xla::PjRtDevice* data_device = nullptr;
  if (is_committed) {
    data_device = default_device;
  } else {
    for (int i = 0; i < num_flat_dynamic_args; ++i) {
      py::handle arg = arguments.flat_dynamic_args[i];
      // We specically only deal with DeviceArray (not ShardedDeviceArray).
      // (Can happen in jit(pmap), e.g. "test_jit_nested_donate_ignored").
      xla::PjRtDevice* device = nullptr;
      if (arg.get_type().ptr() == xla::PyBuffer::type()) {
        xla::PyBuffer* buffer = xla::PyBuffer::AsPyBufferUnchecked(arg);
        if (!buffer->sticky_device()) {
          continue;
        }
        device = buffer->sticky_device();
      } else if (arg.get_type().ptr() == types.device_array.ptr()) {
        if (arg.attr("_device").is_none()) {  // Skip non-sticky devices.
          continue;
        }
        try {
          // This can fail, e.g. for cloud TPU 2VM buffers.
          TF_ASSIGN_OR_RETURN(
              xla::PyBuffer * buffer,
              xla::PyBuffer::AsPyBuffer(arg.attr("device_buffer")));
          device = buffer->buffer()->device();
        } catch (const py::cast_error& e) {
          return xla::InvalidArgument(
              "%s",
              absl::StrCat("[jaxjit] Unsupported subclass of `DeviceArray`: "
                           "`device_buffer` field is of type ",
                           py::cast<std::string>(
                               arg.attr("device_buffer").get_type().str()),
                           " while a `PyBuffer` was expected."

                           ));
        }
      }
      if (device) {
        if (data_device && (device != data_device)) {
          throw std::invalid_argument(absl::StrCat(
              "primitive arguments must be colocated on the same device ("
              "C++ jax.jit). Arguments are on devices: ",
              device->DebugString(), " and ", data_device->DebugString()));
        } else {
          data_device = device;
        }
      }
    }
  }
  if (!data_device) {
    // No `DeviceArray` were found default to `default_device`.
    data_device = default_device;
  }
  CHECK(data_device);
  arguments.signature.device = data_device;

  arguments.signature.dynamic_arg_signatures.reserve(num_flat_dynamic_args);
  for (int i = 0; i < num_flat_dynamic_args; ++i) {
    py::handle arg = arguments.flat_dynamic_args[i];
    TF_ASSIGN_OR_RETURN(auto sig,
                        xla::PyArgSignatureOfValue(arg, jax_enable_x64));
    arguments.signature.dynamic_arg_signatures.push_back(std::move(sig));
  }
  return xla::Status::OK();
}

// Copy buffers to device, skipping pruned arguments.
// Returns `Status::OK()` on success. Returning an error should lead to
// calling the Python fallback.
xla::Status CopyBuffersToDevice(
    bool jax_enable_x64, const absl::optional<std::vector<bool>>& kept_args,
    ParsedArgumentsAsBuffers& arguments) {
  std::vector<xla::PjRtBuffer*>& arg_buffers = arguments.arg_buffers;
  xla::PjRtDevice* data_device = arguments.signature.device;

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  xla::DevicePutOptions options;
  options.squash_64bit_types = !jax_enable_x64;
  // TODO(phawkins): consider allowing forces here.
  options.force_lazy_arrays = false;
  options.allow_zero_copy = true;
  arg_buffers.reserve(num_flat_dynamic_args);
  bool input_pruning_enabled = kept_args.has_value();
  for (int i = 0; i < num_flat_dynamic_args; ++i) {
    if (input_pruning_enabled && !kept_args.value()[i]) {
      continue;
    }

    py::handle arg = arguments.flat_dynamic_args[i];
    TF_ASSIGN_OR_RETURN(xla::DevicePutResult on_device,
                        DevicePut(arg, data_device, options));

    xla::PjRtBuffer* buffer = on_device.buffer;
    arg_buffers.push_back(buffer);
    if (on_device.owned_buffer) {
      arguments.keep_alive.push_back(std::move(on_device.owned_buffer));
    } else if (on_device.owning_pybuffer) {
      arguments.keep_alive_objects.push_back(
          std::move(on_device.owning_pybuffer));
    }
  }
  return xla::Status::OK();
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
  // TODO(zhangqiaorjc): Lookup NamedTuple by name after min jax version bump.
  size_t arity = executable_handlers_out_tree.size();
  if (arity != 5 && !py::hasattr(executable_handlers_out_tree, "_fields")) {
    throw std::runtime_error(absl::StrCat(
        "The versions of jaxlib and Jax are incompatible (jaxlib is too recent "
        "compared to Jax. Upgrade Jax is advised. The C++ code expects "
        "5 or 6 arguments but ",
        arity, " were provided: ",
        py::cast<std::string>(
            py::str(py::repr(executable_handlers_out_tree)))));
  }
  // (xla_executable, out_pytree_def, sticky_device, avals, lazy_exprs)
  auto executable = py::cast<std::shared_ptr<xla::PyExecutable>>(
      executable_handlers_out_tree[0]);
  cache_entry->executable = std::move(executable);
  int num_devices =
      cache_entry->executable->pjrt_executable().addressable_devices().size();
  // The presence of jit(pmap) is detected from Python.
  CHECK_EQ(num_devices, 1);

  auto out_tree = py::cast<xla::PyTreeDef>(executable_handlers_out_tree[1]);
  cache_entry->out_pytree_def = std::move(out_tree);

  cache_entry->sticky_device =
      py::cast<xla::PjRtDevice*>(executable_handlers_out_tree[2]);
  auto avals = py::cast<py::list>(executable_handlers_out_tree[3]);
  auto lazy_exprs = py::cast<py::list>(executable_handlers_out_tree[4]);
  CHECK_EQ(avals.size(), lazy_exprs.size());

  cache_entry->out_avals.reserve(avals.size());
  cache_entry->out_weak_types.reserve(avals.size());
  cache_entry->out_lazy_exprs.reserve(avals.size());
  for (int i = 0; i < avals.size(); ++i) {
    py::object shaped_array = py::reinterpret_borrow<py::object>(avals[i]);
    py::object lazy_expr = py::reinterpret_borrow<py::object>(lazy_exprs[i]);

    cache_entry->out_avals.push_back(shaped_array);
    cache_entry->out_weak_types.push_back(
        py::cast<bool>(shaped_array.attr("weak_type")));
    cache_entry->out_lazy_exprs.push_back(lazy_expr);
  }
  auto kept_var_bitvec_attr =
      py::getattr(executable_handlers_out_tree, "kept_var_bitvec", py::none());
  if (!kept_var_bitvec_attr.is_none()) {
    auto kept_var_bitvec = py::cast<py::list>(kept_var_bitvec_attr);
    cache_entry->kept_var_bitvec =
        absl::make_optional<std::vector<bool>>(kept_var_bitvec.size(), false);
    for (int i = 0; i < kept_var_bitvec.size(); ++i) {
      cache_entry->kept_var_bitvec.value()[i] =
          py::cast<bool>(kept_var_bitvec[i]);
    }
  }
}

void CompiledFunction::TryToPopulateDefaultDevice() {
  // The following line calls Python and may release the GIL.
  py::object device_and_is_committed = get_device_();
  // If the GIL was released by the call to get_device_, another thread may
  // have filled in default_device_.
  if (!default_device_) {
    try {
      auto default_pydevice = py::cast<xla::ClientAndPtr<xla::PjRtDevice>>(
          device_and_is_committed.attr("default_device"));
      is_committed_ =
          py::cast<bool>(device_and_is_committed.attr("committed_to_device"));
      default_pyclient_ = default_pydevice.client;
      default_device_ = default_pydevice.contents;
    } catch (const py::cast_error& e) {
      // Pathways, Cloud TPU 2VM, and UPTC runtime.
      always_fallback_to_python_ = true;
    }
  }
}

xla::StatusOr<py::object> CompiledFunction::Call(
    py::handle args, absl::optional<py::kwargs> kwargs) {
  // Make sure we trigger a garbage collection on JIT function calls. Otherwise
  // code like
  // f = jit(...)
  // while True:
  //   f(x)
  // may never free temporary buffers for copies of arguments.
  xla::GlobalPyRefManager()->MaybeCollectGarbage();

  auto& tls = thread_local_state;
  if (tls.disable_jit.value_or(global_state.disable_jit)) {
    return fun_(*py::reinterpret_borrow<py::args>(args),
                **kwargs.value_or(py::kwargs()));
  }
  if (always_fallback_to_python_) {
    return py::object(
        py::cast<py::tuple>(cache_miss_(*py::reinterpret_borrow<py::args>(args),
                                        **kwargs.value_or(py::kwargs())))[0]);
  }

  // On the first call to `Call`, compute a default device. We need to wait
  // until after platform initialization is complete before doing so, but @jit
  // may be used as a decorator.
  if (!default_device_) {
    TryToPopulateDefaultDevice();
    if (!default_device_) {
      return py::object(py::cast<py::tuple>(
          cache_miss_(*py::reinterpret_borrow<py::args>(args),
                      **kwargs.value_or(py::kwargs())))[0]);
    }
  }

  ParsedArgumentsAsBuffers arguments;
  if (!ParseArguments(args, kwargs, static_argnums_, static_argnames_,
                      arguments)
           .ok()) {
    return py::object(
        py::cast<py::tuple>(cache_miss_(*py::reinterpret_borrow<py::args>(args),
                                        **kwargs.value_or(py::kwargs())))[0]);
  }

  bool jax_enable_x64 = tls.enable_x64.value_or(global_state.enable_x64);
  arguments.signature.jax_enable_x64 = jax_enable_x64;
  // The C++ jit do not support Tracers arguments inputs yet. The Python-based
  // jit function will be called if any of the dynamic arguments is unsupported.
  if (!ComputeSignature(jax_enable_x64, *default_pyclient_, default_device_,
                        is_committed_, arguments)
           .ok()) {
    return py::object(
        py::cast<py::tuple>(cache_miss_(*py::reinterpret_borrow<py::args>(args),
                                        **kwargs.value_or(py::kwargs())))[0]);
  }
  arguments.signature.global_extra_jit_context = global_state.extra_jit_context;
  arguments.signature.thread_local_extra_jit_context = tls.extra_jit_context;

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
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = out_and_fastpath_data =
            cache_miss_(*py::reinterpret_borrow<py::args>(args),
                        **kwargs.value_or(py::kwargs()));
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
      // Release the GIL while we wait, making sure the compile thread can
      // lock it.
      py::gil_scoped_release release;
      cache_entry->compilation_complete.WaitForNotification();
    }
  }
  // It's hard to reraise the exact same kind of errors when a compilation error
  // occured. If the first compilation failed, other threads will also execute
  // the Python path.
  if (cache_entry->fall_back_to_python) {
    return py::object(
        py::cast<py::tuple>(cache_miss_(*py::reinterpret_borrow<py::args>(args),
                                        **kwargs.value_or(py::kwargs())))[0]);
  }

  if (!CopyBuffersToDevice(jax_enable_x64, cache_entry->kept_var_bitvec,
                           arguments)
           .ok()) {
    return py::object(
        py::cast<py::tuple>(cache_miss_(*py::reinterpret_borrow<py::args>(args),
                                        **kwargs.value_or(py::kwargs())))[0]);
  }

  // Executes the computation.
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> output_buffers;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        output_buffers,
        cache_entry->executable->mutable_pjrt_executable()->Execute(
            {arguments.arg_buffers}, cache_entry->executable->options()));
  }
  auto traceback = xla::Traceback::Get();

  int num_outputs = output_buffers[0].size();
  absl::InlinedVector<py::object, 1> flat_device_arrays;
  flat_device_arrays.reserve(num_outputs);
  for (int i = 0; i < output_buffers[0].size(); ++i) {
    bool last = (i == (num_outputs - 1));
    xla::PyBuffer::object buffer = xla::PyBuffer::Make(
        cache_entry->executable->client(), std::move(output_buffers[0][i]),
        last ? std::move(traceback) : traceback);
    if (cache_entry->out_lazy_exprs[i].is_none()) {  // No LazyExpr.
      buffer.buf()->SetAval(cache_entry->out_avals[i]);
      buffer.buf()->set_weak_type(cache_entry->out_weak_types[i]);
      TF_RETURN_IF_ERROR(
          buffer.buf()->set_sticky_device(cache_entry->sticky_device));
      flat_device_arrays.push_back(std::move(buffer));
    } else {
      static const auto* xla_module =
          new py::module(py::module::import("jax.interpreters.xla"));
      static const auto* device_array =
          new py::handle(xla_module->attr("_DeviceArray"));
      flat_device_arrays.push_back(
          (*device_array)(cache_entry->out_avals[i],
                          py::cast(WrapWithClient(default_pyclient_,
                                                  cache_entry->sticky_device)),
                          cache_entry->out_lazy_exprs[i], std::move(buffer)));
    }
  }
  py::object out = cache_entry->out_pytree_def.Unflatten(flat_device_arrays);

  // If there is a post-hook function, call it with the inputs and the outputs.
  absl::optional<py::object> post_hook =
      tls.post_hook.has_value() ? tls.post_hook : global_state.post_hook;
  if (post_hook) {
    (*post_hook)(AsPyHandle(), args,
                 py::cast<py::dict>(kwargs.value_or(py::kwargs())), out);
  }
  return std::move(out);
}

PyObject* JaxCompiledFunction_Type = nullptr;

struct JaxCompiledFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
  CompiledFunction fun;
};

bool JaxCompiledFunction_Check(py::handle handle) {
  return handle.get_type() == JaxCompiledFunction_Type;
}

CompiledFunction* AsCompiledFunctionUnchecked(py::handle handle) {
  return &(reinterpret_cast<JaxCompiledFunctionObject*>(handle.ptr())->fun);
}

xla::StatusOr<CompiledFunction*> AsCompiledFunction(py::handle handle) {
  if (!JaxCompiledFunction_Check(handle)) {
    return xla::InvalidArgument("Expected a CompiledFunction");
  }
  return AsCompiledFunctionUnchecked(handle);
}

py::handle CompiledFunction::AsPyHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(JaxCompiledFunctionObject, fun));
}

extern "C" {

PyObject* JaxCompiledFunction_tp_new(PyTypeObject* subtype, PyObject* args,
                                     PyObject* kwds) {
  JaxCompiledFunctionObject* self =
      reinterpret_cast<JaxCompiledFunctionObject*>(
          subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->dict = nullptr;
  self->weakrefs = nullptr;
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

PyObject* JaxCompiledFunction_tp_call(PyObject* self, PyObject* args,
                                      PyObject* kwargs) {
  tensorflow::profiler::TraceMe traceme("JaxCompiledFunction::tp_call");
  JaxCompiledFunctionObject* o =
      reinterpret_cast<JaxCompiledFunctionObject*>(self);
  absl::optional<py::kwargs> py_kwargs;
  if (kwargs) {
    py_kwargs = py::reinterpret_borrow<py::kwargs>(kwargs);
  }
  try {
    xla::StatusOr<py::object> out = o->fun.Call(args, std::move(py_kwargs));
    if (!out.ok()) {
      PyErr_SetString(PyExc_ValueError, out.status().ToString().c_str());
      return nullptr;
    }
    return out.ValueOrDie().release().ptr();
  } catch (py::error_already_set& e) {
    e.restore();
    return nullptr;
  } catch (py::cast_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return nullptr;
  } catch (std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return nullptr;
  }
}

}  // extern "C"

py::object MakeCompiledFunction(py::function fun, py::function cache_miss,
                                py::function get_device,
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
  new (&buf->fun) CompiledFunction(
      std::move(fun), std::move(cache_miss), std::move(get_device),
      std::move(static_argnums), std::move(static_argnames),
      std::move(donate_argnums), std::move(cache));
  return obj;
}

// Helpers for building Python properties
template <typename Func>
py::object property_readonly(Func&& get) {
  py::handle property(reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property(py::cpp_function(std::forward<Func>(get)), py::none(),
                  py::none(), "");
}

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
    type->tp_flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC;
    type->tp_new = JaxCompiledFunction_tp_new;
    type->tp_dealloc = JaxCompiledFunction_tp_dealloc;
    type->tp_dictoffset = offsetof(JaxCompiledFunctionObject, dict);
    type->tp_traverse = JaxCompiledFunction_tp_traverse;
    type->tp_clear = JaxCompiledFunction_tp_clear;
    type->tp_weaklistoffset = offsetof(JaxCompiledFunctionObject, weakrefs);
    type->tp_getset = JaxCompiledFunction_tp_getset;
    type->tp_descr_get = JaxCompiledFunction_tp_descr_get;
    type->tp_call = JaxCompiledFunction_tp_call;
    CHECK_EQ(PyType_Ready(type), 0);
    JaxCompiledFunction_Type = reinterpret_cast<PyObject*>(type);
    cfun = py::reinterpret_borrow<py::object>(JaxCompiledFunction_Type);
  }
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

  py::class_<GlobalJitState> global_state_(jitlib, "GlobalJitState");
  global_state_.def_readwrite("disable_jit", &GlobalJitState::disable_jit);
  global_state_.def_readwrite("enable_x64", &GlobalJitState::enable_x64);
  global_state_.def_readwrite("extra_jit_context",
                              &GlobalJitState::extra_jit_context);
  global_state_.def_readwrite("post_hook", &GlobalJitState::post_hook);

  py::class_<ThreadLocalJitState> thread_local_state_(jitlib,
                                                      "ThreadLocalJitState");
  thread_local_state_.def_readwrite("disable_jit",
                                    &ThreadLocalJitState::disable_jit);
  thread_local_state_.def_readwrite("enable_x64",
                                    &ThreadLocalJitState::enable_x64);
  thread_local_state_.def_readwrite("extra_jit_context",
                                    &ThreadLocalJitState::extra_jit_context);
  thread_local_state_.def_readwrite("post_hook",
                                    &ThreadLocalJitState::post_hook);

  jitlib.def(
      "global_state", [&]() { return &global_state; },
      py::return_value_policy::reference);
  jitlib.def(
      "thread_local_state", [&]() { return &thread_local_state; },
      py::return_value_policy::reference);

  jitlib.def("jit_is_disabled", &JitIsDisabled);
  jitlib.def("get_enable_x64", &GetEnableX64);

  // TODO(phawkins): delete the following methods after dropping compatibility
  // with JAX python versions older than 0.2.10.
  jitlib.def("set_disable_jit_cpp_flag",
             [&](bool disable_jit) { global_state.disable_jit = disable_jit; });
  jitlib.def("get_disable_jit_cpp_flag",
             [&]() { return global_state.disable_jit; });
  jitlib.def("set_disable_jit_thread_local",
             [&](absl::optional<bool> disable_jit) {
               thread_local_state.disable_jit = disable_jit;
             });
  jitlib.def("get_disable_jit_thread_local",
             [&]() { return thread_local_state.disable_jit; });
  // TODO(jblespiau): Remove from the Python code and remove this
  jitlib.def("set_disable_jit", [&](bool disable_jit) {
    thread_local_state.disable_jit = disable_jit;
  });
  jitlib.def("get_disable_jit",
             [&]() { return thread_local_state.disable_jit; });

  jitlib.def("set_enable_x64_cpp_flag",
             [&](bool enable_x64) { global_state.enable_x64 = enable_x64; });
  jitlib.def("get_enable_x64_cpp_flag",
             [&]() { return global_state.enable_x64; });
  jitlib.def("set_enable_x64_thread_local",
             [&](absl::optional<bool> enable_x64) {
               thread_local_state.enable_x64 = enable_x64;
             });
  jitlib.def("get_enable_x64_thread_local", [&](bool enable_x64) {
    thread_local_state.enable_x64 = enable_x64;
  });
  // TODO(phawkins): delete up to here.

  jitlib.def(
      "jit",
      [](py::function fun, py::function cache_miss, py::function get_device,
         std::vector<int> static_argnums, std::vector<py::str> static_argnames,
         std::vector<int> donate_argnums,
         std::shared_ptr<CompiledFunctionCache> cache) -> py::object {
        return MakeCompiledFunction(
            std::move(fun), std::move(cache_miss), std::move(get_device),
            std::move(static_argnums), std::move(static_argnames),
            std::move(donate_argnums), std::move(cache));
      },
      py::arg("fun"), py::arg("cache_miss"), py::arg("get_device"),
      py::arg("static_argnums"),
      py::arg("static_argnames") = std::vector<py::str>(),
      py::arg("donate_argnums") = std::vector<int>(),
      py::arg("cache") = nullptr);

  // This function is not yet a full replacement for the Python one, because:
  // (a) it does not support abstract types,
  // (b) it does not set the device stickiness yet.
  // TODO(jblespiau): Finish the replacement of the Python feature.
  jitlib.def("device_put",
             [](py::handle obj, bool jax_enable_x64,
                xla::ClientAndPtr<xla::PjRtDevice> to_device)
                 -> xla::StatusOr<py::object> {
               std::shared_ptr<xla::PyClient>& pyclient = to_device.client;
               xla::DevicePutOptions options;
               options.squash_64bit_types = !jax_enable_x64;
               options.force_lazy_arrays = true;
               options.allow_zero_copy = true;
               xla::StatusOr<xla::DevicePutResult> results =
                   DevicePut(obj, to_device.contents, options);
               if (!results.ok()) {
                 throw std::runtime_error(results.status().error_message());
               }
               if (results->owned_buffer) {
                 auto buffer = xla::PyBuffer::Make(
                     pyclient, std::move(results->owned_buffer),
                     xla::Traceback::Get());

                 static const auto* jax_core =
                     new py::module(py::module::import("jax.core"));
                 static const auto* shaped_array =
                     new py::handle(jax_core->attr("ShapedArray"));
                 buffer.buf()->SetAval((*shaped_array)(
                     buffer.buf()->python_shape(), buffer.buf()->python_dtype(),
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
      .def_property_readonly("shape",
                             [](const xla::PyArgSignature& sig) {
                               return xla::IntSpanToTuple(sig.shape);
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
        return xla::Status::OK();
      },
      py::is_method(cfun));
  jitlib.def("_is_float0", &xla::IsFloat0);
}

}  // namespace jax
