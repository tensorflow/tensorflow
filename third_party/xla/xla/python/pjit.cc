/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/pjit.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/lru_cache.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/jax_jit.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_array.h"
#include "xla/python/py_executable.h"
#include "xla/python/py_values.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/pytree.h"
#include "xla/python/sharding.h"
#include "xla/python/traceback.h"
#include "xla/python/transfer_guard_lib.h"
#include "xla/util.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax {
namespace {

namespace nb = nanobind;
namespace py = pybind11;

struct PjitCacheEntry {
  explicit PjitCacheEntry(xla::PyTreeRegistry* registry)
      : out_pytree_def(registry) {}
  std::shared_ptr<xla::PyLoadedExecutable> executable;
  std::vector<nb::object> in_shardings;
  std::vector<nb::object> out_avals;
  std::vector<xla::nb_dtype> out_dtypes;
  std::vector<std::vector<int64_t>> out_shapes;
  std::vector<bool> out_weak_types;
  std::vector<nb::object> out_shardings;
  std::vector<bool> out_committed;
  xla::PyTreeDef out_pytree_def;
  // Bitvector of kept arguments from Jaxpr DCE pass. Used to drop some `args`
  // in PjitFunction::Call before calling into compiled computation.
  std::vector<bool> kept_var_bitvec;

  // Ensures a single thread performs the compilation for a given executable.
  //
  // The first thread (holding the GIL) will create the CacheEntry associated to
  // a signature and if the object has been inserted already, other threads
  // will wait for the notification.
  absl::Notification compilation_complete;

  std::thread::id thread_id = std::this_thread::get_id();

  bool fall_back_to_python = false;
};

// A PjitFunctionCache represents a cache of compiled functions that can be
// shared between one or more PjitFunction objects. It serves two goals:
// - reduce the number of lru caches (hash map) across multiple JITs.
// - make the cache global to increase cache hits (e.g. calling jit(f)(3) twice)
//   keeping entries alive as long as the underlying function f is alive.
// Assume the cache is protected by the GIL.
class PjitFunctionCache {
 public:
  static constexpr int kDefaultCapacity = 4096;
  explicit PjitFunctionCache(int capacity);

  // Cache entries are shared_ptr<>s because it's possible the cache entry
  // might be evicted before we finish tracing/compiling.
  typedef xla::LRUCache<CallSignature, std::shared_ptr<PjitCacheEntry>> Cache;

  // We include as part of the cache key `donate_argnums` (and any other fields
  // that aren't subsumed by the CallSignature we compute for each call).
  std::shared_ptr<Cache> Lookup(nb::handle function,
                                absl::Span<const int> donate_argnums);
  std::shared_ptr<Cache> DefaultCache();

  int Size() const { return lru_list_.Size(); }
  int Capacity() const { return lru_list_.Capacity(); }
  void Clear() { lru_list_.Clear(); }

 private:
  struct Key {
    nb::handle function;  // Does not hold a reference.

    // Other fields that are part of the arguments to `jit`, but are not
    // otherwise part of CallSignature.
    std::vector<int> donate_argnums;

    bool operator==(const Key& other) const {
      return function.ptr() == other.function.ptr() &&
             donate_argnums == other.donate_argnums;
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
    // calls to `pjit(f)` if `f` remains alive, but we do not want the cache
    // to keep `f` alive if all other references are dropped.
    std::optional<nb::weakref> weakref;
  };

  Cache::LRUList lru_list_;
  absl::flat_hash_map<Key, std::unique_ptr<Value>> functions_;
};

PjitFunctionCache::PjitFunctionCache(int capacity) : lru_list_(capacity) {}

std::shared_ptr<PjitFunctionCache::Cache> PjitFunctionCache::DefaultCache() {
  return std::make_shared<Cache>(&lru_list_);
}

std::shared_ptr<PjitFunctionCache::Cache> PjitFunctionCache::Lookup(
    nb::handle function, absl::Span<const int> donate_argnums) {
  Key key;
  key.function = function;
  key.donate_argnums =
      std::vector<int>(donate_argnums.begin(), donate_argnums.end());
  auto insert = functions_.emplace(key, nullptr);
  if (!insert.second) {
    return insert.first->second->cache;
  }
  std::shared_ptr<Cache> cache = std::make_shared<Cache>(&lru_list_);
  auto callback =
      nb::cpp_function([this, key{std::move(key)}](nb::handle weakref) {
        functions_.erase(key);
      });
  PyObject* weakref = PyWeakref_NewRef(function.ptr(), callback.ptr());
  if (weakref) {
    std::unique_ptr<Value>& entry = insert.first->second;
    entry = std::make_unique<Value>(cache);
    entry->weakref = nb::steal<nb::weakref>(weakref);
  } else {
    PyErr_Clear();
    // `function` is not weak-referenceable. Don't bother adding it to the
    // shared cache in that case; the `jit` object will hold the only shared
    // reference to the cache entry.
    functions_.erase(insert.first);
  }
  return cache;
}

class PjitFunction {
 public:
  PjitFunction(std::string function_name, std::optional<nb::callable> fun,
               nb::callable cache_miss, std::vector<int> static_argnums,
               std::vector<nb::str> static_argnames,
               std::vector<int> donate_argnums,
               std::shared_ptr<xla::PyTreeRegistry> pytree_registry,
               nb::callable shard_arg_fallback,
               std::shared_ptr<PjitFunctionCache> cache);
  ~PjitFunction();

  PjitFunction(const PjitFunction&) = delete;
  PjitFunction& operator=(const PjitFunction&) = delete;
  PjitFunction(PjitFunction&&) = default;
  PjitFunction& operator=(PjitFunction&&) = default;

  // nb::object typed subclass for PjitFunction objects.
  class pyobject : public nb::object {
   public:
    NB_OBJECT(pyobject, nb::object, "PjitFunction",
              PjitFunction::IsPjitFunction);
    pyobject() = default;
    PjitFunction* func() const {
      return PjitFunction::AsPjitFunctionUnchecked(*this);
    }
  };
  // Alias as ::object; outside the scope above we won't confuse pybind11's
  // macros.
  using object = pyobject;

  // Returns true if `h` is a PjitFunction.
  static bool IsPjitFunction(nb::handle handle);
  // Converts `handle` to a PjitFunction*. Does not do any checking.
  static PjitFunction* AsPjitFunctionUnchecked(nb::handle handle);

  absl::StatusOr<nb::object> Call(nb::handle callable, PyObject* const* args,
                                  size_t nargs, PyObject* kwnames);

  void ClearPythonReferences();

  const std::string& function_name() const { return function_name_; }
  const std::optional<nb::callable>& fun() const { return fun_; }
  const nb::callable& cache_miss() const { return cache_miss_; }
  const std::shared_ptr<xla::PyTreeRegistry>& pytree_registry() const {
    return pytree_registry_;
  }
  const nb::callable& shard_arg_fallback() const { return shard_arg_fallback_; }

  const std::vector<int>& static_argnums() const { return static_argnums_; }
  const std::vector<nb::str>& static_argnames() const {
    return static_argnames_;
  }
  const std::vector<int>& donate_argnums() const { return donate_argnums_; }
  const std::shared_ptr<PjitFunctionCache>& cache() const { return cache_; }

  int cache_capacity() const { return executables_->Size(); }

  void ClearCache() { executables_->Clear(); }

  nb::object PythonSignature() {
    if (!fun_.has_value()) {
      throw nb::value_error(
          absl::StrFormat(
              "Calling __signature__ on PjitFunction(%s) not supported.",
              function_name_)
              .c_str());
    }
    static const auto* inspect =
        new nb::module_(nb::module_::import_("inspect"));
    return inspect->attr("signature")(*fun_);
  }

 private:
  absl::Status UpdateArgsSignature(ParsedArgumentsAsBuffers& arguments);

  void PopulateCacheEntry(PjitCacheEntry& cache_entry,
                          const CallSignature& signature,
                          const nb::tuple& out_and_fastpath_data);

  std::string function_name_;
  std::optional<nb::callable> fun_;
  nb::callable cache_miss_;
  std::vector<int> static_argnums_;
  std::vector<nb::str> static_argnames_;
  std::vector<int> donate_argnums_;

  std::shared_ptr<xla::PyTreeRegistry> pytree_registry_;
  nb::callable shard_arg_fallback_;
  std::shared_ptr<PjitFunctionCache> cache_;
  std::shared_ptr<PjitFunctionCache::Cache> executables_;
};

// thread-compatible.
class PjitFunctionStore {
 public:
  void Insert(PjitFunction* function) { compiled_functions_.insert(function); }

  void Erase(PjitFunction* function) { compiled_functions_.erase(function); }

  void ClearFunctionCache() {
    for (auto* function : compiled_functions_) {
      function->ClearCache();
    }
  }

 private:
  absl::flat_hash_set<PjitFunction*> compiled_functions_;
};

// Protected by GIL.
PjitFunctionStore& GetGlobalPjitFunctionStore() {
  static auto* const store = new PjitFunctionStore();
  return *store;
}

PjitFunction::PjitFunction(
    std::string function_name, std::optional<nb::callable> fun,
    nb::callable cache_miss, std::vector<int> static_argnums,
    std::vector<nb::str> static_argnames, std::vector<int> donate_argnums,
    std::shared_ptr<xla::PyTreeRegistry> pytree_registry,
    nb::callable shard_arg_fallback, std::shared_ptr<PjitFunctionCache> cache)
    : function_name_(std::move(function_name)),
      fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      donate_argnums_(donate_argnums),
      pytree_registry_(std::move(pytree_registry)),
      shard_arg_fallback_(std::move(shard_arg_fallback)),
      cache_(std::move(cache)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
  static_argnames.reserve(static_argnames.size());
  for (nb::str& name : static_argnames) {
    PyObject* s = name.inc_ref().ptr();
    PyUnicode_InternInPlace(&s);
    static_argnames_.push_back(nb::steal<nb::str>(s));
  }
  if (!fun_.has_value()) {
    executables_ = cache_->DefaultCache();
  } else {
    executables_ = cache_->Lookup(fun_.value(), donate_argnums);
  }

  GetGlobalPjitFunctionStore().Insert(this);
}

PjitFunction::~PjitFunction() { GetGlobalPjitFunctionStore().Erase(this); }

void CallShardArgFallback(
    nb::handle arg, nb::handle sharding, const nb::callable& fallback,
    std::vector<tsl::RCReference<xla::ifrt::Array>>& num_args_arrays,
    ParsedArgumentsAsBuffers& arguments) {
  tsl::profiler::TraceMe traceme("cpp_pjit_shard_arg_fallback");
  auto py_array_or_bufs = fallback(arg, sharding);
  // TODO(phawkins): simplify after nanobind transition is complete
  auto py_array = py::cast<xla::PyArray>(py::handle(py_array_or_bufs.ptr()));
  num_args_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
  arguments.keep_alive_objects.push_back(std::move(py_array_or_bufs));
}

// Prepares the input PjRtBuffers from the python arguments. This is equivalent
// to shard_args() in pxla.py but for only a few supported cases.
absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
PrepareIfrtInputs(const xla::PyLoadedExecutable& executable,
                  ParsedArgumentsAsBuffers& arguments,
                  const std::vector<bool>& kept_args,
                  const std::vector<nb::object>& in_shardings,
                  const nb::callable& shard_arg_fallback) {
  const auto& addressable_devices = executable.AddressableDevices();
  int num_args = arguments.flat_dynamic_args.size();

  std::vector<tsl::RCReference<xla::ifrt::Array>> num_args_arrays;
  num_args_arrays.reserve(num_args);

  xla::DevicePutOptions options;
  options.squash_64bit_types = !arguments.signature.jax_enable_x64;
  options.allow_zero_copy = true;
  xla::PjRtDevice* data_device = nullptr;
  if (executable.ifrt_loaded_executable()->num_devices() == 1) {
    data_device = executable.ifrt_loaded_executable()->addressable_devices()[0];
  }
  int dce_i = 0;
  for (int i = 0; i < num_args; ++i) {
    if (!kept_args[i]) {
      continue;
    }
    int dce_index = dce_i;
    ++dce_i;

    const nb::object& arg = arguments.flat_dynamic_args[i];

    auto transfer_guard_formatter = [] { return std::string(""); };

    if (arg.type().ptr() != xla::PyArray::type().ptr()) {
      if (data_device != nullptr) {
        TF_RETURN_IF_ERROR(
            jax::ApplyTransferGuardToHostToDevice(transfer_guard_formatter));
        TF_ASSIGN_OR_RETURN(
            xla::DevicePutResult on_device,
            DevicePut(arg, executable.ifrt_loaded_executable()->client(),
                      data_device, options, xla::ifrt::MemoryKind()));

        num_args_arrays.push_back(std::move(on_device.ifrt_array));
        if (on_device.owning_pybuffer) {
          // TODO(phawkins): use std::move after nanobind transition is complete
          arguments.keep_alive_objects.push_back(
              nb::steal(on_device.owning_pybuffer.release().ptr()));
        }
        continue;
      } else {
        CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                             shard_arg_fallback, num_args_arrays, arguments);
        continue;
      }
    }

    xla::PyArray py_array(py::reinterpret_borrow<py::object>(arg.ptr()));
    const auto& sharding = py_array.sharding();
    // TODO(phawkins): remove .ptr() after nanobind transition is complete.
    int sharding_num_devices = jax::Sharding::SafeNumDevices(sharding.ptr());

    // Currently only committed PyArray inputs or uncommitted PyArray on a
    // single device inputs are allowed. This is checked previously in the entry
    // point of PjitFunction::Call().
    DCHECK(py_array.committed() ||
           (!py_array.committed() && sharding_num_devices == 1));

    // TODO(phawkins): remove .ptr() after nanobind transition is complete.
    if (sharding.get_type().ptr() == jax::PmapSharding::type().ptr()) {
      CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                           shard_arg_fallback, num_args_arrays, arguments);
      continue;
    }

    if (py_array.num_shards() != addressable_devices.size()) {
      CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                           shard_arg_fallback, num_args_arrays, arguments);
      continue;
    }

    xla::ifrt::Array* ifrt_array = py_array.ifrt_array();
    // PyArray inputs should have already been checked in
    // `xla::PyArgSignatureOfValue()` called by
    // `PjitFunction::UpdateArgsSignature()`.
    DCHECK(ifrt_array != nullptr) << "PyArray has been unexpectedly deleted.";

    if (sharding_num_devices == 1 && ifrt_array->sharding().devices().front() !=
                                         addressable_devices[0].get()) {
      xla::ifrt::DeviceList::Devices ifrt_devices;
      ifrt_devices.push_back(addressable_devices[0].get());
      auto sharding = xla::ifrt::OpaqueSharding::Create(
          xla::ifrt::DeviceList(std::move(ifrt_devices)),
          ifrt_array->sharding().memory_kind());
      TF_ASSIGN_OR_RETURN(
          auto copied_ifrt_array,
          ifrt_array->Reshard(std::move(sharding),
                              xla::ifrt::ArrayCopySemantics::kReuseInput));
      num_args_arrays.push_back(std::move(copied_ifrt_array));
    } else {
      num_args_arrays.push_back(tsl::FormRef(ifrt_array));
    }

    arguments.keep_alive_objects.push_back(arg);
  }

  return num_args_arrays;
}

absl::StatusOr<nb::object> PjitFunction::Call(nb::handle callable,
                                              PyObject* const* args,
                                              size_t nargs, PyObject* kwnames) {
  tsl::profiler::TraceMe traceme(
      [&] { return absl::StrCat("PjitFunction(", function_name_, ")"); });
  ParsedArgumentsAsBuffers arguments;

  // Make sure we trigger a garbage collection on JIT function calls. Otherwise
  // code like
  // f = jit(...)
  // while True:
  //   f(x)
  // may never free temporary buffers for copies of arguments.
  xla::GlobalPyRefManager()->MaybeCollectGarbage();

  if (GetDisableJit()) {
    if (!fun_.has_value()) {
      throw nb::value_error(
          absl::StrFormat("Disable jit is not supported in the AOT path since "
                          "the function is not available for (%s)",
                          function_name_)
              .c_str());
    }
    return nb::steal<nb::object>(
        PyObject_Vectorcall(fun_.value().ptr(), args, nargs, kwnames));
  }

  // Calls the cache_miss_ function. This just calls the Python function; it may
  // return nullptr value if a Python exception is thrown.
  auto cache_miss = [&]() -> nb::tuple {
    return nb::steal<nb::tuple>(
        PyObject_Vectorcall(cache_miss_.ptr(), args, nargs, kwnames));
  };

  // Call the cache_miss() function, extracting the output data and ignoring
  // the fastpath data. If the cache miss returns a Python error, returns
  // nullptr and leaves the Python error set.
  auto fallback_to_cache_miss = [&]() {
    nb::tuple cache_miss_output = cache_miss();
    if (!cache_miss_output.ptr()) {
      return nb::object();
    }
    return nb::object(cache_miss_output[0]);
  };

  size_t num_positional_args = PyVectorcall_NARGS(nargs);
  size_t num_keyword_args = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  absl::Span<PyObject* const> positional_args(args, num_positional_args);
  absl::Span<PyObject* const> keyword_args(args + num_positional_args,
                                           num_keyword_args);
  auto status =
      ParseArguments(positional_args, keyword_args, kwnames, static_argnums_,
                     static_argnames_, pytree_registry_.get(), arguments);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  // Perform a few checks for the arguments. Currently we are only allowing
  // committed PyArray inputs. For other cases, e.g. Tracers or ShapedArray, it
  // will fallback to python. For jit, numpy arrays and scalars are also
  // allowed, which we will check later.
  for (const auto& arg : arguments.flat_dynamic_args) {
    if (arg.type().ptr() != xla::PyArray::type().ptr()) {
      continue;
    }

    xla::PyArray py_array(py::reinterpret_borrow<py::object>(arg.ptr()));
    if (!py_array.fastpath_enabled()) {
      return fallback_to_cache_miss();
    }

    // Only allow committed PyArray in cpp pjit for now as the logic on handling
    // sharding for uncommitted PyArray is complicated and still under
    // development.
    //
    // TODO(chky): Consider support uncommitted PyArray in cpp when the python
    // side stablizes.
    // TODO(phawkins): remove .ptr() after nanobind transition is complete.
    if (!py_array.committed() &&
        jax::Sharding::SafeNumDevices(py_array.sharding().ptr()) > 1) {
      VLOG(2) << "PyArray argument is not committed and number of global "
                 "devices is more than 1; fallback to python.";
      return fallback_to_cache_miss();
    }
  }

  status = UpdateArgsSignature(arguments);
  if (!status.ok()) {
    VLOG(2) << "UpdateArgsSignature failed: " << status;
    return fallback_to_cache_miss();
  }

  VLOG(2) << "CallSignature:\n" << arguments.signature.DebugString();
  bool inserted = false;
  std::shared_ptr<PjitCacheEntry> cache_entry =
      executables_->GetOrCreateIfAbsent(
          arguments.signature, [this, &inserted](const CallSignature& unused) {
            inserted = true;
            return std::make_shared<PjitCacheEntry>(pytree_registry_.get());
          });

  if (!cache_entry->compilation_complete.HasBeenNotified()) {
    // In case of several threads attempting to compile the executable, only
    // the one that inserted the item will perform the compilation.
    if (inserted) {
      nb::object out_and_fastpath_data;
      nb::tuple out_tuple;
      VLOG(2) << "Cache miss for " << arguments.signature.DebugString();
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = cache_miss();
        if (!out_and_fastpath_data.ptr()) {
          throw nb::python_error();
        }
        out_tuple = nb::cast<nb::tuple>(out_and_fastpath_data);

        PopulateCacheEntry(*cache_entry, arguments.signature, out_tuple);
      } catch (const std::exception& e) {
        VLOG(2) << "cache miss fail: " << e.what();
        cache_entry->fall_back_to_python = true;
        cache_entry->compilation_complete.Notify();
        throw;
      }
      cache_entry->compilation_complete.Notify();

      // We have already computed the result in the miss path so we can return
      // it. We are even *required* to do so if there are donated arguments,
      // because any donated buffers will now be invalid.
      return nb::object(out_tuple[0]);
    } else {
      if (cache_entry->thread_id == std::this_thread::get_id()) {
        auto error_string = absl::StrCat("Recursively calling jit: ",
                                         arguments.signature.DebugString());
        PyErr_SetString(PyExc_RecursionError, error_string.c_str());
        throw nb::python_error();
      }
      // Release the GIL while we wait, making sure the compile thread can
      // lock it.
      nb::gil_scoped_release release;
      cache_entry->compilation_complete.WaitForNotification();
    }
  }

  if (cache_entry->fall_back_to_python) {
    VLOG(2) << "cpp pjit fallback to python.";
    return fallback_to_cache_miss();
  }

  // A vector of [num_inputs].
  auto num_args_arrays = PrepareIfrtInputs(
      *cache_entry->executable, arguments, cache_entry->kept_var_bitvec,
      cache_entry->in_shardings, shard_arg_fallback_);

  if (!num_args_arrays.ok()) {
    VLOG(2) << "Failed to prepare IFRT inputs: " << num_args_arrays.status();
    return fallback_to_cache_miss();
  }

  // A vector of [num_outputs].
  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(auto result,
                        cache_entry->executable->ifrt_executable()->Execute(
                            absl::MakeSpan(*num_args_arrays),
                            cache_entry->executable->options(),
                            /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
  }

  auto traceback = xla::Traceback::Get();
  const auto& client = cache_entry->executable->client();

  // Convert the ifrt::Array objects to PyArray.
  int num_outputs = output_arrays.size();
  absl::InlinedVector<nb::object, 4> outputs;
  outputs.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    // Creating the PyArray result. In addition to the IFRT arrays, the metadata
    // like `aval` and `sharding` are retrieved from the cache for this
    // function, which are produced by the python path in `cache_miss`.
    xla::PyArray py_array(
        // TODO(phawkins): remove nanobind translation
        py::reinterpret_borrow<py::object>(cache_entry->out_avals[i].ptr()),
        cache_entry->out_weak_types[i],
        py::reinterpret_borrow<py::object>(cache_entry->out_dtypes[i].ptr()),
        cache_entry->out_shapes[i],
        py::reinterpret_borrow<py::object>(cache_entry->out_shardings[i].ptr()),
        cache_entry->executable->client(), traceback,
        std::move(output_arrays[i]),
        /*committed=*/cache_entry->out_committed.at(i), /*skip_checks=*/true);

    // TODO(phawkins): use std::move after nanobind transition is complete
    outputs.push_back(nb::steal(py_array.release().ptr()));
  }

  nb::object out = nb::steal<nb::object>(
      cache_entry->out_pytree_def.Unflatten(outputs).release().ptr());

  // If there is a post-hook function, call it with the inputs and the outputs.
  std::optional<nb::object> post_hook = GetPostHook();
  if (post_hook) {
    nb::tuple args_tuple =
        nb::steal<nb::tuple>(PyTuple_New(num_positional_args));
    for (size_t i = 0; i < num_positional_args; ++i) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(args_tuple.ptr(), i, args[i]);
    }
    nb::dict kwargs;
    if (kwnames) {
      for (size_t i = 0; i < num_keyword_args; ++i) {
        kwargs[nb::handle(PyTuple_GET_ITEM(kwnames, i))] =
            nb::borrow(args[num_positional_args + i]);
      }
    }
    (*post_hook)(nb::handle(callable.ptr()), args_tuple, kwargs,
                 nb::handle(out.ptr()));
  }

  return out;
}

absl::Status PjitFunction::UpdateArgsSignature(
    ParsedArgumentsAsBuffers& arguments) {
  arguments.signature.function_name = function_name_;

  // Get dynamic argument signatures.
  JitState& global_state = jax::GlobalJitState();
  JitState& tls = jax::ThreadLocalJitState();
  bool jax_enable_x64 = GetEnableX64();

  arguments.signature.default_device = GetDefaultDevice();
  arguments.signature.jax_enable_x64 = jax_enable_x64;
  arguments.signature.jax_enable_memories = GetEnableMemories();

  auto& dynamic_arg_signatures = arguments.signature.dynamic_arg_signatures;
  dynamic_arg_signatures.reserve(arguments.flat_dynamic_args.size());
  auto& dynamic_arg_shardings = arguments.signature.dynamic_arg_shardings;
  dynamic_arg_shardings.reserve(arguments.flat_dynamic_args.size());

  for (nb::handle arg : arguments.flat_dynamic_args) {
    TF_ASSIGN_OR_RETURN(auto signature,
                        xla::PyArgSignatureOfValue(arg, jax_enable_x64));
    arguments.signature.dynamic_arg_signatures.push_back(std::move(signature));

    // It should be already checked previously in the entry point of
    // PjitFunction::Call().
    if (arg.type().ptr() == xla::PyArray::type().ptr()) {
      auto py_array = py::reinterpret_borrow<xla::PyArray>(arg.ptr());

      // TODO(phawkins): remove nanobind translation
      arguments.signature.dynamic_arg_shardings.push_back(
          nb::borrow(py_array.sharding().ptr()));
      arguments.signature.committed_args.push_back(py_array.committed());
    } else {
      arguments.signature.dynamic_arg_shardings.push_back(nb::none());
      arguments.signature.committed_args.push_back(false);
    }
  }

  arguments.signature.thread_local_extra_jit_context = tls.extra_jit_context;
  arguments.signature.global_extra_jit_context = global_state.extra_jit_context;

  return absl::OkStatus();
}

void PjitFunction::PopulateCacheEntry(PjitCacheEntry& cache_entry,
                                      const CallSignature& signature,
                                      const nb::tuple& out_and_fastpath_data) {
  DCHECK_EQ(out_and_fastpath_data.size(), 2);

  if (out_and_fastpath_data[1].is_none()) {
    VLOG(2) << "fastpath_data is none";
    cache_entry.fall_back_to_python = true;
    return;
  }

  nb::tuple fastpath_data = nb::cast<nb::tuple>(out_and_fastpath_data[1]);

  // TODO(phawkins): remove nanobind translation
  cache_entry.executable = py::cast<std::shared_ptr<xla::PyLoadedExecutable>>(
      py::handle(fastpath_data.attr("xla_executable").ptr()));

  nb::list in_shardings = fastpath_data.attr("in_shardings");
  cache_entry.in_shardings.reserve(in_shardings.size());
  for (nb::handle sharding : in_shardings) {
    cache_entry.in_shardings.push_back(nb::borrow(sharding));
  }

  nb::list out_shardings = fastpath_data.attr("out_shardings");
  cache_entry.out_shardings.reserve(out_shardings.size());
  for (nb::handle sharding : out_shardings) {
    cache_entry.out_shardings.push_back(nb::borrow(sharding));
  }

  nb::list out_committed = fastpath_data.attr("out_committed");
  cache_entry.out_committed.reserve(out_committed.size());
  for (nb::handle c : out_committed) {
    cache_entry.out_committed.push_back(nb::cast<bool>(c));
  }

  nb::list out_avals = fastpath_data.attr("out_avals");
  cache_entry.out_avals.reserve(out_avals.size());
  cache_entry.out_dtypes.reserve(out_avals.size());
  cache_entry.out_shapes.reserve(out_avals.size());
  cache_entry.out_weak_types.reserve(out_avals.size());
  for (nb::handle aval : out_avals) {
    cache_entry.out_avals.push_back(nb::borrow(aval));
    cache_entry.out_dtypes.push_back(aval.attr("dtype"));
    cache_entry.out_shapes.push_back(
        nb::cast<std::vector<int64_t>>(aval.attr("shape")));
    cache_entry.out_weak_types.push_back(
        nb::cast<bool>(aval.attr("weak_type")));
  }

  cache_entry.out_pytree_def = nb::cast<xla::PyTreeDef>(
      nb::handle(fastpath_data.attr("out_pytree_def").ptr()));

  nb::list kept_var_bitvec = fastpath_data.attr("kept_var_bitvec");
  cache_entry.kept_var_bitvec.reserve(kept_var_bitvec.size());
  for (nb::handle k : kept_var_bitvec) {
    cache_entry.kept_var_bitvec.push_back(nb::cast<bool>(k));
  }
}

// Helper function used by the tp_clear GC method.
void PjitFunction::ClearPythonReferences() {
  // TODO(mattjj): phawkins@ observed that the xla::PyTreeRegistry
  // pytree_registry_ attribute of PjitFunction could in principle also have
  // python references to clear
  nb::callable cache_miss;
  std::optional<nb::callable> fun;
  nb::callable shard_arg_fallback;
  // Swap values for nulls before they are destroyed. See the Python
  // Py_CLEAR() documentation for a discussion of this topic.
  std::swap(cache_miss_, cache_miss);
  std::swap(fun_, fun);
  std::swap(shard_arg_fallback_, shard_arg_fallback);
}

struct PjitFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
  vectorcallfunc vectorcall;
  PjitFunction fun;
};

PyObject* PjitFunction_Type = nullptr;

bool PjitFunction::IsPjitFunction(nb::handle handle) {
  return handle.type().ptr() == PjitFunction_Type;
}

PjitFunction* PjitFunction::AsPjitFunctionUnchecked(nb::handle handle) {
  return &(reinterpret_cast<PjitFunctionObject*>(handle.ptr())->fun);
}

PjitFunction* AsPjitFunction(nb::handle handle) {
  if (!PjitFunction::IsPjitFunction(handle)) {
    throw xla::XlaRuntimeError(xla::InvalidArgument("Expected a PjitFunction"));
  }
  return PjitFunction::AsPjitFunctionUnchecked(handle);
}

extern "C" {

PyObject* PjitFunction_tp_vectorcall(PyObject* callable, PyObject* const* args,
                                     size_t nargs, PyObject* kwnames) {
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(callable);
  tsl::profiler::TraceMe traceme([&] {
    return absl::StrCat("PjitFunction(", o->fun.function_name(), ")");
  });
  try {
    absl::StatusOr<nb::object> out =
        o->fun.Call(callable, args, nargs, kwnames);
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
  } catch (nb::python_error& e) {
    e.restore();
    return nullptr;
  } catch (nb::cast_error& e) {
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

PyObject* PjitFunction_tp_new(PyTypeObject* subtype, PyObject* args,
                              PyObject* kwds) {
  PjitFunctionObject* self =
      reinterpret_cast<PjitFunctionObject*>(subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->dict = nullptr;
  self->weakrefs = nullptr;
  self->vectorcall = PjitFunction_tp_vectorcall;
  return reinterpret_cast<PyObject*>(self);
}

void PjitFunction_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
  if (o->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  Py_CLEAR(o->dict);
  o->fun.~PjitFunction();
  tp->tp_free(self);
  Py_DECREF(tp);
}

int PjitFunction_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  // TODO(mattjj): phawkins@ observed that the xla::PyTreeRegistry
  // pytree_registry_ attribute of PjitFunction could in principle also have
  // python references to visit
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
#if PY_VERSION_HEX >= 0x03090000
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
#endif
  Py_VISIT(o->dict);
  Py_VISIT(o->fun.cache_miss().ptr());
  Py_VISIT(o->fun.shard_arg_fallback().ptr());
  if (o->fun.fun()) {
    Py_VISIT(o->fun.fun()->ptr());
  }
  return 0;
}

int PjitFunction_tp_clear(PyObject* self) {
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
  Py_CLEAR(o->dict);
  o->fun.ClearPythonReferences();
  return 0;
}

// Implements the Python descriptor protocol so JIT-compiled functions can be
// used as bound methods. See:
// https://docs.python.org/3/howto/descriptor.html#functions-and-methods
PyObject* PjitFunction_tp_descr_get(PyObject* self, PyObject* obj,
                                    PyObject* type) {
  if (obj == nullptr || obj == Py_None) {
    Py_INCREF(self);
    return self;
  }
  return PyMethod_New(self, obj);
}

// Support d = instance.__dict__.
PyObject* PjitFunction_get_dict(PyObject* self, void*) {
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
  if (!o->dict) {
    o->dict = PyDict_New();
  }
  Py_XINCREF(o->dict);
  return o->dict;
}

int PjitFunction_set_dict(PyObject* self, PyObject* new_dict, void*) {
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
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

static PyGetSetDef PjitFunction_tp_getset[] = {
    // Having a __dict__ seems necessary to allow !functool.wraps to override
    // __doc__.
    {const_cast<char*>("__dict__"), PjitFunction_get_dict,
     PjitFunction_set_dict, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyObject* PjitFunction_tp_repr(PyObject* self) {
  try {
    const std::string& repr = absl::StrFormat(
        "<PjitFunction of %s>",
        nb::cast<std::string_view>(nb::repr(nb::getattr(self, "__wrapped__"))));
    return PyUnicode_FromString(repr.c_str());
  } catch (...) {
    // Ignore all errors when accessing a repr.
    return PyUnicode_FromString("<PjitFunction>");
  }
}

}  // extern "C"

void InitializePjitFunction(
    PjitFunctionObject* fn_obj, std::string function_name,
    std::optional<nb::callable> fun, nb::callable cache_miss,
    std::vector<int> static_argnums, std::vector<nb::str> static_argnames,
    std::vector<int> donate_argnums,
    std::shared_ptr<xla::PyTreeRegistry> pytree_registry,
    nb::callable shard_arg_fallback, std::shared_ptr<PjitFunctionCache> cache) {
  new (&fn_obj->fun) PjitFunction(
      std::move(function_name), std::move(fun), std::move(cache_miss),
      std::move(static_argnums), std::move(static_argnames),
      std::move(donate_argnums), std::move(pytree_registry),
      std::move(shard_arg_fallback), std::move(cache));
}

nb::object MakePjitFunction(
    std::string function_name, std::optional<nb::callable> fun,
    nb::callable cache_miss, std::vector<int> static_argnums,
    std::vector<nb::str> static_argnames, std::vector<int> donate_argnums,
    std::shared_ptr<xla::PyTreeRegistry> pytree_registry,
    nb::callable shard_arg_fallback,
    std::optional<std::shared_ptr<PjitFunctionCache>> cache) {
  nb::object obj = nb::steal<nb::object>(PjitFunction_tp_new(
      reinterpret_cast<PyTypeObject*>(PjitFunction_Type), nullptr, nullptr));
  PjitFunctionObject* fn_obj = reinterpret_cast<PjitFunctionObject*>(obj.ptr());
  if (!cache) {
    cache = std::make_shared<PjitFunctionCache>(
        PjitFunctionCache::kDefaultCapacity);
  }
  InitializePjitFunction(fn_obj, std::move(function_name), std::move(fun),
                         std::move(cache_miss), std::move(static_argnums),
                         std::move(static_argnames), std::move(donate_argnums),
                         std::move(pytree_registry),
                         std::move(shard_arg_fallback), std::move(*cache));
  return obj;
}

// Version numbers for the pickled representations of
// PjitFunction. Increment these if changing them.
const int kPjitFunctionPickleVersion = 1;

}  // namespace

void BuildPjitSubmodule(nb::module_& m) {
  nb::class_<PjitFunctionCache> cache(m, "PjitFunctionCache");
  cache.def(nb::init<int>(),
            nb::arg("capacity") = PjitFunctionCache::kDefaultCapacity);
  cache.def("size", &PjitFunctionCache::Size);
  cache.def("capacity", &PjitFunctionCache::Capacity);
  cache.def("clear", &PjitFunctionCache::Clear);
  cache.def_static("clear_all",
                   []() { GetGlobalPjitFunctionStore().ClearFunctionCache(); });
  cache.def("__getstate__",
            // Pickles as an empty cache; the client can repopulate as needed.
            [](const PjitFunctionCache& cache) {
              nb::dict pickle;
              pickle["version"] = kPjitFunctionPickleVersion;
              pickle["capacity"] = cache.Capacity();
              return pickle;
            });
  cache.def("__setstate__",
            [](PjitFunctionCache* cache, const nb::dict& pickle) {
              int version = nb::cast<int>(pickle["version"]);
              if (version != kPjitFunctionPickleVersion) {
                throw std::invalid_argument(absl::StrFormat(
                    "Invalid PjitFunction pickle version, got %d, expected %d",
                    version, kPjitFunctionPickleVersion));
              }
              int capacity = nb::cast<int>(pickle["capacity"]);
              new (cache) PjitFunctionCache(capacity);
            });

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  nb::object cfun;
  {
    nb::str name = nb::str("PjitFunction");
    nb::str qualname = nb::str("PjitFunction");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called. Otherwise the GC might see a half-constructed
    // type object.
    CHECK(heap_type) << "Unable to create heap type object";
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "PjitFunction";
    type->tp_basicsize = sizeof(PjitFunctionObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE |
                     Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL;
    type->tp_new = PjitFunction_tp_new;
    type->tp_dealloc = PjitFunction_tp_dealloc;
    type->tp_dictoffset = offsetof(PjitFunctionObject, dict);
    type->tp_traverse = PjitFunction_tp_traverse;
    type->tp_clear = PjitFunction_tp_clear;
    type->tp_weaklistoffset = offsetof(PjitFunctionObject, weakrefs);
    type->tp_getset = PjitFunction_tp_getset;
    type->tp_descr_get = PjitFunction_tp_descr_get;
    type->tp_call = PyVectorcall_Call;
    type->tp_vectorcall_offset = offsetof(PjitFunctionObject, vectorcall);
    type->tp_repr = PjitFunction_tp_repr;
    CHECK_EQ(PyType_Ready(type), 0);
    PjitFunction_Type = reinterpret_cast<PyObject*>(type);
    cfun = nb::borrow<nb::object>(PjitFunction_Type);
  }
  nb::object cfun_type = nb::borrow<nb::object>(PjitFunction_Type);

  // Add PjitFunction to the xla_extension module so it can be pickled.
  m.attr("PjitFunction") = cfun_type;
  cfun.attr("__module__") = m.attr("__name__");

  cfun.attr("__getstate__") = nb::cpp_function(
      [](const PjitFunction::object& self) {
        PjitFunction* fn = self.func();
        nb::dict pickle;
        pickle["version"] = kPjitFunctionPickleVersion;
        pickle["function_name"] = fn->function_name();
        if (fn->fun().has_value()) {
          pickle["fun"] = *fn->fun();
        }
        pickle["cache_miss"] = fn->cache_miss();
        pickle["static_argnums"] = fn->static_argnums();
        pickle["static_argnames"] = nb::cast(fn->static_argnames());
        pickle["donate_argnums"] = fn->donate_argnums();
        pickle["pytree_registry"] = nb::cast(fn->pytree_registry());
        pickle["shard_arg_fallback"] = fn->shard_arg_fallback();
        pickle["cache"] = fn->cache();
        return pickle;
      },
      nb::is_method());
  cfun.attr("__setstate__") = nb::cpp_function(
      [](nb::object& self, const nb::dict& pickle) {
        int version = nb::cast<int>(pickle["version"]);
        if (version != kPjitFunctionPickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid PjitFunction pickle version, got %d, expected %d. "
              "Pickling/Unpickling jitted functions using different JAX "
              "versions is not supported.",
              version, kPjitFunctionPickleVersion));
        }
        std::string function_name =
            nb::cast<std::string>(pickle["function_name"]);
        std::optional<nb::callable> fun;
        if (pickle.contains("fun")) {
          fun = nb::cast<nb::callable>(pickle["fun"]);
        }
        nb::callable cache_miss = nb::cast<nb::callable>(pickle["cache_miss"]);
        std::vector<int> static_argnums =
            nb::cast<std::vector<int>>(pickle["static_argnums"]);
        std::vector<nb::str> static_argnames =
            nb::cast<std::vector<nb::str>>(pickle["static_argnames"]);
        std::vector<int> donate_argnums =
            nb::cast<std::vector<int>>(pickle["donate_argnums"]);
        std::shared_ptr<xla::PyTreeRegistry> pytree_registry =
            nb::cast<std::shared_ptr<xla::PyTreeRegistry>>(
                nb::handle(pickle["pytree_registry"].ptr()));
        nb::callable shard_arg_fallback =
            nb::cast<nb::callable>(pickle["shard_arg_fallback"]);
        std::shared_ptr<PjitFunctionCache> cache =
            nb::cast<std::shared_ptr<PjitFunctionCache>>(pickle["cache"]);
        InitializePjitFunction(
            reinterpret_cast<PjitFunctionObject*>(self.ptr()),
            std::move(function_name), std::move(fun), std::move(cache_miss),
            std::move(static_argnums), std::move(static_argnames),
            std::move(donate_argnums), std::move(pytree_registry),
            std::move(shard_arg_fallback), std::move(cache));
      },
      nb::is_method());
  cfun.attr("__signature__") =
      xla::nb_property_readonly([](nb::handle self) -> nb::object {
        return AsPjitFunction(self)->PythonSignature();
      });
  cfun.attr("_cache_miss") =
      xla::nb_property_readonly([](nb::handle self) -> nb::object {
        return AsPjitFunction(self)->cache_miss();
      });
  // All private members are only for testing/debugging purposes
  cfun.attr("_cache_size") = nb::cpp_function(
      [](nb::handle self) -> int {
        return AsPjitFunction(self)->cache_capacity();
      },
      nb::is_method());
  cfun.attr("_clear_cache") = nb::cpp_function(
      [](nb::handle self) { AsPjitFunction(self)->ClearCache(); },
      nb::is_method());

  m.def(
      "pjit",
      [](std::string function_name, std::optional<nb::callable> fun,
         nb::callable cache_miss, std::vector<int> static_argnums,
         std::vector<nb::str> static_argnames, std::vector<int> donate_argnums,
         nb::object pytree_registry, nb::callable shard_arg_fallback,
         std::optional<std::shared_ptr<PjitFunctionCache>> cache) {
        std::shared_ptr<xla::PyTreeRegistry> registry =
            nb::cast<std::shared_ptr<xla::PyTreeRegistry>>(
                nb::handle(pytree_registry.ptr()));
        return MakePjitFunction(
            std::move(function_name), std::move(fun), std::move(cache_miss),
            std::move(static_argnums), std::move(static_argnames),
            std::move(donate_argnums), std::move(registry),
            std::move(shard_arg_fallback), std::move(cache));
      },
      nb::arg("function_name"), nb::arg("fun").none(), nb::arg("cache_miss"),
      nb::arg("static_argnums"), nb::arg("static_argnames"),
      nb::arg("donate_argnums"), nb::arg("pytree_registry"),
      nb::arg("shard_arg_fallback"), nb::arg("cache").none() = nb::none());
}

}  // namespace jax
