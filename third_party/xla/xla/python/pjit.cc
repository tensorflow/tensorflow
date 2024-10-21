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
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/lru_cache.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/guard_lib.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
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
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax {
namespace {

namespace nb = nanobind;

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
  std::vector<nb::object> in_device_local_layouts;

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

  // We include as part of the cache key `global_cache_key` (and any other
  // fields that aren't subsumed by the CallSignature we compute for each call).
  std::shared_ptr<Cache> Lookup(nb::handle function,
                                nb::object global_cache_key);
  std::shared_ptr<Cache> DefaultCache();

  int Size() const { return lru_list_.Size(); }
  int Capacity() const { return lru_list_.Capacity(); }
  void Clear() {
    lru_list_.Clear();
    functions_.clear();
  }

 private:
  struct Key {
    nb::handle function;  // Does not hold a reference.

    // Other fields that are part of the arguments to `jit`, but are not
    // otherwise part of CallSignature.
    nb::object global_cache_key;

    bool operator==(const Key& other) const {
      bool global_cache_eq;
      try {
        global_cache_eq = global_cache_key.equal(other.global_cache_key);
      } catch (const nanobind::python_error& e) {
        throw std::invalid_argument(
            absl::StrCat("Equality of  global cache key lead to an exception. "
                         "The error was:\n",
                         e.what(), "\n"));
      }
      return function.ptr() == other.function.ptr() && global_cache_eq;
    }
  };

  template <typename H>
  friend H AbslHashValue(H h, const Key& key) {
    h = H::combine(std::move(h), key.function.ptr());
    Py_hash_t hash;
    try {
      hash = nb::hash(key.global_cache_key);
    } catch (const nanobind::python_error& e) {
      if (!e.matches(PyExc_TypeError)) throw;
      throw std::invalid_argument(absl::StrCat(
          "Hashing global cache key lead to an exception. The error was:\n",
          e.what(), "\n"));
    }
    h = H::combine(std::move(h), hash);
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
  absl::Mutex mu_;  // Non-trivial hashes need to be mutex locked.
  // ABSL containers are not exception safe:
  std::unordered_map<Key, std::unique_ptr<Value>, absl::Hash<Key>> functions_;
};

PjitFunctionCache::PjitFunctionCache(int capacity) : lru_list_(capacity) {}

std::shared_ptr<PjitFunctionCache::Cache> PjitFunctionCache::DefaultCache() {
  return std::make_shared<Cache>(&lru_list_);
}

std::shared_ptr<PjitFunctionCache::Cache> PjitFunctionCache::Lookup(
    nb::handle function,
    nb::object global_cache_key) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  {
    // Because the gil can be released during cache insertion, this forces
    // the lock order to be mu_ then gil so we must release the gil first.
    nb::gil_scoped_release release;
    // Acquire a mutex to avoid problems where the gil is released during
    // cache insertion and then a second thread invalidates the cache order.
    mu_.Lock();
  }
  absl::Cleanup unlock = [this]() ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
  Key key;
  key.function = function;
  key.global_cache_key = global_cache_key;
  auto insert = functions_.emplace(key, nullptr);
  if (!insert.second) {
    return insert.first->second->cache;
  }
  std::shared_ptr<Cache> cache = std::make_shared<Cache>(&lru_list_);
  auto callback =
      nb::cpp_function([this, key{std::move(key)}](nb::handle weakref) {
        auto it = functions_.find(key);
        if (it != functions_.end()) {
          functions_.erase(it);
        }
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
               nb::object global_cache_key,
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
  // Alias as ::object; outside the scope above we won't confuse nanobind's
  // macros.
  using object = pyobject;

  // Returns true if `h` is a PjitFunction.
  static bool IsPjitFunction(nb::handle handle);
  // Converts `handle` to a PjitFunction*. Does not do any checking.
  static PjitFunction* AsPjitFunctionUnchecked(nb::handle handle);

  absl::StatusOr<nb::object> Call(nb::handle callable, PyObject* const* args,
                                  size_t nargs, PyObject* kwnames);

  void InitExecutables();

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
  const nb::object& global_cache_key() const { return global_cache_key_; }
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
  absl::Status ComputeCallSignature(
      absl::Span<nb::object const> flat_dynamic_args,
      CallSignature& call_signature);

  void PopulateCacheEntry(PjitCacheEntry& cache_entry,
                          const nb::tuple& out_and_fastpath_data);

  std::string function_name_;
  std::optional<nb::callable> fun_;
  nb::callable cache_miss_;
  std::vector<int> static_argnums_;
  std::vector<nb::str> static_argnames_;
  nb::object global_cache_key_;

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
    compiled_functions_.clear();
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
    std::vector<nb::str> static_argnames, nb::object global_cache_key,
    std::shared_ptr<xla::PyTreeRegistry> pytree_registry,
    nb::callable shard_arg_fallback, std::shared_ptr<PjitFunctionCache> cache)
    : function_name_(std::move(function_name)),
      fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      global_cache_key_(std::move(global_cache_key)),
      pytree_registry_(std::move(pytree_registry)),
      shard_arg_fallback_(std::move(shard_arg_fallback)),
      cache_(std::move(cache)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
  static_argnames_.reserve(static_argnames.size());
  for (nb::str& name : static_argnames) {
    PyObject* s = name.inc_ref().ptr();
    PyUnicode_InternInPlace(&s);
    static_argnames_.push_back(nb::steal<nb::str>(s));
  }

  GetGlobalPjitFunctionStore().Insert(this);
}

void PjitFunction::InitExecutables() {
  if (!fun_.has_value()) {
    executables_ = cache_->DefaultCache();
  } else {
    executables_ = cache_->Lookup(fun_.value(), global_cache_key_);
  }
}

PjitFunction::~PjitFunction() { GetGlobalPjitFunctionStore().Erase(this); }

void CallShardArgFallback(
    nb::handle arg, nb::handle sharding, nb::handle layout,
    const nb::callable& fallback,
    std::vector<tsl::RCReference<xla::ifrt::Array>>& num_args_arrays,
    std::vector<nb::object>& keep_alive_objects) {
  tsl::profiler::TraceMe traceme("cpp_pjit_shard_arg_fallback");
  auto py_array_or_bufs = fallback(arg, sharding, layout);
  auto py_array = nb::cast<xla::PyArray>(py_array_or_bufs);
  num_args_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
  keep_alive_objects.push_back(std::move(py_array_or_bufs));
}

// Prepares the input PjRtBuffers from the python arguments. This is equivalent
// to shard_args() in pxla.py but for only a few supported cases.
absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
PrepareIfrtInputs(const xla::PyLoadedExecutable& executable,
                  absl::Span<nb::object const> flat_dynamic_args,
                  bool enable_x64, const std::vector<bool>& kept_args,
                  const std::vector<nb::object>& in_shardings,
                  const std::vector<nb::object>& in_device_local_layouts,
                  const nb::callable& shard_arg_fallback,
                  std::vector<nb::object>& keep_alive_objects) {
  const auto& addressable_devices =
      executable.ifrt_loaded_executable()->addressable_devices();
  const auto& num_global_devices =
      executable.ifrt_loaded_executable()->num_devices();
  int num_args = flat_dynamic_args.size();

  std::vector<tsl::RCReference<xla::ifrt::Array>> num_args_arrays;
  num_args_arrays.reserve(num_args);

  struct CopyGroup {
    std::vector<int> indices;
    std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  };
  absl::flat_hash_map<std::pair<xla::ifrt::Device*, xla::ifrt::MemoryKind>,
                      CopyGroup>
      copy_groups;

  xla::DevicePutOptions options;
  options.squash_64bit_types = !enable_x64;
  options.allow_zero_copy = true;
  xla::ifrt::Device* data_device = nullptr;
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

    const nb::object& arg = flat_dynamic_args[i];
    const nb::object& in_device_local_layout =
        in_device_local_layouts[dce_index];

    auto transfer_guard_formatter = [] { return std::string(""); };

    if (arg.type().ptr() != xla::PyArray::type().ptr()) {
      if (data_device != nullptr && in_device_local_layout.is_none()) {
        TF_RETURN_IF_ERROR(
            jax::ApplyTransferGuardToHostToDevice(transfer_guard_formatter));
        TF_ASSIGN_OR_RETURN(
            auto on_device_fn,
            DevicePut(arg, executable.ifrt_loaded_executable()->client(),
                      data_device, options, xla::ifrt::MemoryKind()));
        TF_ASSIGN_OR_RETURN(xla::DevicePutResult on_device, [&]() {
          // Must release the GIL before calling IFRT because backends may
          // decide to block/sleep for device buffer allocation.
          nb::gil_scoped_release gil_release;
          return std::move(on_device_fn)();
        }());

        num_args_arrays.push_back(std::move(on_device.ifrt_array));
        if (on_device.owning_pybuffer) {
          keep_alive_objects.push_back(std::move(on_device.owning_pybuffer));
        }
        continue;
      } else {
        CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                             in_device_local_layout, shard_arg_fallback,
                             num_args_arrays, keep_alive_objects);
        continue;
      }
    }

    xla::PyArray py_array = nb::borrow<xla::PyArray>(arg);
    const auto& sharding = py_array.sharding();
    int sharding_num_devices = jax::Sharding::SafeNumDevices(sharding);

    // Currently only committed PyArray inputs or uncommitted PyArray on a
    // single device inputs are allowed. This is checked previously in the entry
    // point of PjitFunction::Call().
    DCHECK(py_array.committed() ||
           (!py_array.committed() && sharding_num_devices == 1));

    if (!in_device_local_layout.is_none()) {
      TF_ASSIGN_OR_RETURN(auto arr_layout, py_array.ifrt_array()->layout());
      xla::Layout in_xc_layout = nb::cast<xla::Layout>(
          in_device_local_layout.attr("_to_xla_layout")(py_array.dtype()));
      if (in_xc_layout != GetXlaLayoutUnsafe(arr_layout)) {
        CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                             in_device_local_layout, shard_arg_fallback,
                             num_args_arrays, keep_alive_objects);
        continue;
      }
    }

    if (sharding.type().ptr() == jax::PmapSharding::type().ptr()) {
      CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                           in_device_local_layout, shard_arg_fallback,
                           num_args_arrays, keep_alive_objects);
      continue;
    }

    if (sharding_num_devices != num_global_devices) {
      CallShardArgFallback(arg.ptr(), in_shardings[dce_index],
                           in_device_local_layout, shard_arg_fallback,
                           num_args_arrays, keep_alive_objects);
      continue;
    }

    xla::ifrt::Array* ifrt_array = py_array.ifrt_array();
    // PyArray inputs should have already been checked in
    // `xla::PyArgSignatureOfValue()` called by
    // `PjitFunction::ComputeCallSignature()`.
    DCHECK(ifrt_array != nullptr) << "PyArray has been unexpectedly deleted.";

    const auto& ifrt_sharding = ifrt_array->sharding();
    if (sharding_num_devices == 1 &&
        ifrt_sharding.devices()->devices().front() != addressable_devices[0]) {
      auto& copy_group =
          copy_groups[std::make_pair(ifrt_sharding.devices()->devices().front(),
                                     ifrt_sharding.memory_kind())];
      copy_group.indices.push_back(num_args_arrays.size());
      copy_group.arrays.push_back(tsl::FormRef(ifrt_array));
      num_args_arrays.push_back({});
    } else {
      num_args_arrays.push_back(tsl::FormRef(ifrt_array));
    }

    keep_alive_objects.push_back(arg);
  }

  if (!copy_groups.empty()) {
    xla::ifrt::Client* const ifrt_client =
        executable.ifrt_loaded_executable()->client();
    tsl::RCReference<xla::ifrt::DeviceList> ifrt_devices =
        xla::ifrt::BasicDeviceList::Create({addressable_devices[0]});
    for (auto& [key, group] : copy_groups) {
      TF_ASSIGN_OR_RETURN(
          auto copied_ifrt_arrays,
          ifrt_client->CopyArrays(absl::MakeSpan(group.arrays), ifrt_devices,
                                  /*memory_kind=*/std::nullopt,
                                  xla::ifrt::ArrayCopySemantics::kReuseInput));
      for (int i = 0; i < copied_ifrt_arrays.size(); ++i) {
        num_args_arrays[group.indices[i]] = std::move(copied_ifrt_arrays[i]);
      }
    }
  }

  return num_args_arrays;
}

absl::StatusOr<nb::object> PjitFunction::Call(nb::handle callable,
                                              PyObject* const* args,
                                              size_t nargs, PyObject* kwnames) {
  tsl::profiler::TraceMe traceme(
      [&] { return absl::StrCat("PjitFunction(", function_name_, ")"); });

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

  CallSignature call_signature;
  std::vector<nb::object> keep_alive_objects;
  absl::InlinedVector<nb::object, 2> flat_dynamic_args;
  auto status = ParseArguments(
      positional_args, keyword_args, kwnames, static_argnums_, static_argnames_,
      pytree_registry_.get(), call_signature.arg_signature, flat_dynamic_args);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  // Perform a few checks for the arguments. Currently we are only allowing
  // committed PyArray inputs. For other cases, e.g. Tracers or ShapedArray, it
  // will fallback to python. For jit, numpy arrays and scalars are also
  // allowed, which we will check later.
  for (const auto& arg : flat_dynamic_args) {
    if (arg.type().ptr() != xla::PyArray::type().ptr()) {
      continue;
    }

    xla::PyArray py_array = nb::borrow<xla::PyArray>(arg);

    // Only allow committed PyArray in cpp pjit for now as the logic on handling
    // sharding for uncommitted PyArray is complicated and still under
    // development.
    //
    // TODO(chky): Consider support uncommitted PyArray in cpp when the python
    // side stablizes.
    if (!py_array.committed() &&
        jax::Sharding::SafeNumDevices(py_array.sharding()) > 1) {
      VLOG(2) << "PyArray argument is not committed and number of global "
                 "devices is more than 1; fallback to python.";
      return fallback_to_cache_miss();
    }
  }

  status = ComputeCallSignature(flat_dynamic_args, call_signature);
  if (!status.ok()) {
    VLOG(2) << "ComputeCallSignature failed: " << status;
    return fallback_to_cache_miss();
  }

  VLOG(2) << "CallSignature:\n" << call_signature.DebugString();
  bool inserted = false;
  std::shared_ptr<PjitCacheEntry> cache_entry =
      executables_->GetOrCreateIfAbsent(
          call_signature, [this, &inserted](const CallSignature& unused) {
            inserted = true;
            return std::make_shared<PjitCacheEntry>(pytree_registry_.get());
          });

  if (!cache_entry->compilation_complete.HasBeenNotified()) {
    // In case of several threads attempting to compile the executable, only
    // the one that inserted the item will perform the compilation.
    if (inserted) {
      nb::object out_and_fastpath_data;
      nb::tuple out_tuple;
      VLOG(2) << "Cache miss for " << call_signature.DebugString();
      bool remove_cache = false;
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = cache_miss();
        if (!out_and_fastpath_data.ptr()) {
          throw nb::python_error();
        }
        out_tuple = nb::cast<nb::tuple>(out_and_fastpath_data);

        PopulateCacheEntry(*cache_entry, out_tuple);

        if (out_tuple.size() > 2 && out_tuple[2].is_valid()) {
          remove_cache = nb::cast<bool>(out_tuple[2]);
        }
      } catch (const std::exception& e) {
        VLOG(2) << "cache miss fail: " << e.what();
        cache_entry->fall_back_to_python = true;
        cache_entry->compilation_complete.Notify();
        throw;
      }
      cache_entry->compilation_complete.Notify();

      if (remove_cache) {
        executables_->Remove(call_signature);
      }

      // We have already computed the result in the miss path so we can return
      // it. We are even *required* to do so if there are donated arguments,
      // because any donated buffers will now be invalid.
      return nb::object(out_tuple[0]);
    } else {
      if (cache_entry->thread_id == std::this_thread::get_id()) {
        auto error_string = absl::StrCat("Recursively calling jit: ",
                                         call_signature.DebugString());
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
      *cache_entry->executable, flat_dynamic_args,
      call_signature.jax_enable_x64, cache_entry->kept_var_bitvec,
      cache_entry->in_shardings, cache_entry->in_device_local_layouts,
      shard_arg_fallback_, keep_alive_objects);

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
        cache_entry->out_avals[i], cache_entry->out_weak_types[i],
        cache_entry->out_dtypes[i], cache_entry->out_shapes[i],
        cache_entry->out_shardings[i], cache_entry->executable->client(),
        traceback, std::move(output_arrays[i]),
        /*committed=*/cache_entry->out_committed.at(i), /*skip_checks=*/true);

    outputs.push_back(std::move(py_array));
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

absl::Status PjitFunction::ComputeCallSignature(
    absl::Span<nb::object const> flat_dynamic_args, CallSignature& signature) {
  signature.function_name = function_name_;

  // Get dynamic argument signatures.
  JitState& global_state = jax::GlobalJitState();
  JitState& tls = jax::ThreadLocalJitState();
  bool jax_enable_x64 = GetEnableX64();

  signature.default_device = GetDefaultDevice();
  signature.jax_enable_x64 = jax_enable_x64;
  signature.jax_enable_memories = GetEnableMemories();

  auto& dynamic_arg_signatures = signature.dynamic_arg_signatures;
  dynamic_arg_signatures.reserve(flat_dynamic_args.size());
  auto& dynamic_arg_shardings = signature.dynamic_arg_shardings;
  dynamic_arg_shardings.reserve(flat_dynamic_args.size());

  for (nb::handle arg : flat_dynamic_args) {
    TF_ASSIGN_OR_RETURN(auto arg_signature,
                        xla::PyArgSignatureOfValue(arg, jax_enable_x64));
    signature.dynamic_arg_signatures.push_back(std::move(arg_signature));

    // It should be already checked previously in the entry point of
    // PjitFunction::Call().
    if (arg.type().ptr() == xla::PyArray::type().ptr()) {
      auto py_array = nb::borrow<xla::PyArray>(arg);
      signature.dynamic_arg_shardings.push_back(py_array.sharding());
      signature.committed_args.push_back(py_array.committed());
    } else {
      signature.dynamic_arg_shardings.push_back(nb::none());
      signature.committed_args.push_back(false);
    }
  }

  signature.thread_local_extra_jit_context = tls.extra_jit_context;
  signature.global_extra_jit_context = global_state.extra_jit_context;

  return absl::OkStatus();
}

void PjitFunction::PopulateCacheEntry(PjitCacheEntry& cache_entry,
                                      const nb::tuple& out_and_fastpath_data) {
  DCHECK_GE(out_and_fastpath_data.size(), 2);

  if (out_and_fastpath_data[1].is_none()) {
    VLOG(2) << "fastpath_data is none";
    cache_entry.fall_back_to_python = true;
    return;
  }

  nb::tuple fastpath_data = nb::cast<nb::tuple>(out_and_fastpath_data[1]);

  cache_entry.executable = nb::cast<std::shared_ptr<xla::PyLoadedExecutable>>(
      fastpath_data.attr("xla_executable"));

  nb::sequence in_shardings = fastpath_data.attr("in_shardings");
  cache_entry.in_shardings.reserve(nb::len(in_shardings));
  for (nb::handle sharding : in_shardings) {
    cache_entry.in_shardings.push_back(nb::borrow(sharding));
  }

  nb::sequence out_shardings = fastpath_data.attr("out_shardings");
  cache_entry.out_shardings.reserve(nb::len(out_shardings));
  for (nb::handle sharding : out_shardings) {
    cache_entry.out_shardings.push_back(nb::borrow(sharding));
  }

  nb::sequence out_committed = fastpath_data.attr("out_committed");
  cache_entry.out_committed.reserve(nb::len(out_committed));
  for (nb::handle c : out_committed) {
    cache_entry.out_committed.push_back(nb::cast<bool>(c));
  }

  nb::sequence out_avals = fastpath_data.attr("out_avals");
  cache_entry.out_avals.reserve(nb::len(out_avals));
  cache_entry.out_dtypes.reserve(nb::len(out_avals));
  cache_entry.out_shapes.reserve(nb::len(out_avals));
  cache_entry.out_weak_types.reserve(nb::len(out_avals));
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

  nb::sequence kept_var_bitvec = fastpath_data.attr("kept_var_bitvec");
  cache_entry.kept_var_bitvec.reserve(nb::len(kept_var_bitvec));
  for (nb::handle k : kept_var_bitvec) {
    cache_entry.kept_var_bitvec.push_back(nb::cast<bool>(k));
  }

  nb::sequence in_device_local_layouts =
      fastpath_data.attr("in_device_local_layouts");
  cache_entry.in_device_local_layouts.reserve(nb::len(in_device_local_layouts));
  for (nb::handle dll : in_device_local_layouts) {
    cache_entry.in_device_local_layouts.push_back(nb::borrow(dll));
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
#if PY_VERSION_HEX < 0x030C0000
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
#endif                 // PY_VERSION_HEX < 0x030C0000
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
#if PY_VERSION_HEX < 0x030C0000
  self->dict = nullptr;
  self->weakrefs = nullptr;
#endif  // PY_VERSION_HEX < 0x030C0000
  self->vectorcall = PjitFunction_tp_vectorcall;
  return reinterpret_cast<PyObject*>(self);
}

void PjitFunction_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
  PyObject_ClearWeakRefs(self);
#if PY_VERSION_HEX < 0x030C0000
  Py_CLEAR(o->dict);
#elif PY_VERSION_HEX < 0x030D0000
  _PyObject_ClearManagedDict(self);
#else
  PyObject_ClearManagedDict(self);
#endif  // PY_VERSION_HEX < 0x030C0000
  o->fun.~PjitFunction();
  tp->tp_free(self);
  Py_DECREF(tp);
}

int PjitFunction_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  // TODO(mattjj): phawkins@ observed that the xla::PyTreeRegistry
  // pytree_registry_ attribute of PjitFunction could in principle also have
  // python references to visit
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
#if PY_VERSION_HEX < 0x030C0000
  Py_VISIT(o->dict);
#elif PY_VERSION_HEX < 0x030D0000
  _PyObject_VisitManagedDict(self, visit, arg);
#else
  PyObject_VisitManagedDict(self, visit, arg);
#endif  // PY_VERSION_HEX < 0x030C0000
  Py_VISIT(o->fun.cache_miss().ptr());
  Py_VISIT(o->fun.shard_arg_fallback().ptr());
  if (o->fun.fun()) {
    Py_VISIT(o->fun.fun()->ptr());
  }
  return 0;
}

int PjitFunction_tp_clear(PyObject* self) {
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
#if PY_VERSION_HEX < 0x030C0000
  Py_CLEAR(o->dict);
#elif PY_VERSION_HEX < 0x030D0000
  _PyObject_ClearManagedDict(self);
#else
  PyObject_ClearManagedDict(self);
#endif  // PY_VERSION_HEX < 0x030C0000
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

static PyGetSetDef PjitFunction_tp_getset[] = {
    // Having a __dict__ seems necessary to allow !functool.wraps to override
    // __doc__.
    {const_cast<char*>("__dict__"), PyObject_GenericGetDict,
     PyObject_GenericSetDict, nullptr, nullptr},
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
    nb::object global_cache_key,
    std::shared_ptr<xla::PyTreeRegistry> pytree_registry,
    nb::callable shard_arg_fallback, std::shared_ptr<PjitFunctionCache> cache) {
  if (nb::isinstance<nb::list>(global_cache_key)) {
    global_cache_key = nb::tuple(global_cache_key);
  }
  new (&fn_obj->fun) PjitFunction(
      std::move(function_name), std::move(fun), std::move(cache_miss),
      std::move(static_argnums), std::move(static_argnames),
      std::move(global_cache_key), std::move(pytree_registry),
      std::move(shard_arg_fallback), std::move(cache));
  // Handled separately because it is not exception safe to call this
  // in the constructor because it leaves the object improperly constructed.
  fn_obj->fun.InitExecutables();
}

nb::object MakePjitFunction(
    std::string function_name, std::optional<nb::callable> fun,
    nb::callable cache_miss, std::vector<int> static_argnums,
    std::vector<nb::str> static_argnames, nb::object global_cache_key,
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
  InitializePjitFunction(
      fn_obj, std::move(function_name), std::move(fun), std::move(cache_miss),
      std::move(static_argnums), std::move(static_argnames),
      std::move(global_cache_key), std::move(pytree_registry),
      std::move(shard_arg_fallback), std::move(*cache));
  return obj;
}

// Version numbers for the pickled representations of
// PjitFunction. Increment these if changing them.
const int kPjitFunctionPickleVersion = 1;

PyMemberDef PjitFunction_members[] = {
    {"__vectorcalloffset__", T_PYSSIZET,
     static_cast<Py_ssize_t>(offsetof(PjitFunctionObject, vectorcall)),
     READONLY, nullptr},
#if PY_VERSION_HEX < 0x030C0000
    {"__dictoffset__", T_PYSSIZET,
     static_cast<Py_ssize_t>(offsetof(PjitFunctionObject, dict)), READONLY,
     nullptr},
    {"__weaklistoffset__", T_PYSSIZET,
     static_cast<Py_ssize_t>(offsetof(PjitFunctionObject, weakrefs)), READONLY,
     nullptr},
#endif  // PY_VERSION_HEX < 0x030C0000
    {nullptr, 0, 0, 0, nullptr},
};

PyType_Slot PjitFunction_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PjitFunction_tp_new)},
    {Py_tp_dealloc, reinterpret_cast<void*>(PjitFunction_tp_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void*>(PjitFunction_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(PjitFunction_tp_clear)},
    {Py_tp_getset, reinterpret_cast<void*>(PjitFunction_tp_getset)},
    {Py_tp_descr_get, reinterpret_cast<void*>(PjitFunction_tp_descr_get)},
    {Py_tp_call, reinterpret_cast<void*>(PyVectorcall_Call)},
    {Py_tp_repr, reinterpret_cast<void*>(PjitFunction_tp_repr)},
    {Py_tp_members, reinterpret_cast<void*>(PjitFunction_members)},
    {0, nullptr},
};

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
  std::string name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".PjitFunction");
  PyType_Spec PjitFunction_spec = {
#if PY_VERSION_HEX < 0x030B0000
      // Work around for https://github.com/python/cpython/issues/89478
      // CPython 3.10 and earlier assume that the .name value remains alive
      // forever.
      /*.name=*/strdup(name.c_str()),
#else
      /*.name=*/name.c_str(),
#endif  // PY_VERSION_HEX < 0x030B0000
      /*.basicsize=*/static_cast<int>(sizeof(PjitFunctionObject)),
      /*.itemsize=*/0,
#if PY_VERSION_HEX < 0x030C0000
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
          Py_TPFLAGS_HAVE_VECTORCALL,
#else   // PY_VERSION_HEX < 0x030C0000
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
          Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_MANAGED_DICT |
          Py_TPFLAGS_MANAGED_WEAKREF,
#endif  // PY_VERSION_HEX < 0x030C0000
      /*.slots=*/PjitFunction_slots,
  };
  PjitFunction_Type = PyType_FromSpec(&PjitFunction_spec);
  if (!PjitFunction_Type) {
    throw nb::python_error();
  }
  nb::object cfun = nb::borrow<nb::object>(PjitFunction_Type);

  // Add PjitFunction to the xla_extension module so it can be pickled.
  m.attr("PjitFunction") = cfun;
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
        pickle["global_cache_key"] = fn->global_cache_key();
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
        nb::object global_cache_key = pickle["global_cache_key"];
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
            std::move(global_cache_key), std::move(pytree_registry),
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
         std::vector<nb::str> static_argnames, nb::object global_cache_key,
         nb::object pytree_registry, nb::callable shard_arg_fallback,
         std::optional<std::shared_ptr<PjitFunctionCache>> cache) {
        std::shared_ptr<xla::PyTreeRegistry> registry =
            nb::cast<std::shared_ptr<xla::PyTreeRegistry>>(
                nb::handle(pytree_registry.ptr()));
        return MakePjitFunction(
            std::move(function_name), std::move(fun), std::move(cache_miss),
            std::move(static_argnums), std::move(static_argnames),
            std::move(global_cache_key), std::move(registry),
            std::move(shard_arg_fallback), std::move(cache));
      },
      nb::arg("function_name"), nb::arg("fun").none(), nb::arg("cache_miss"),
      nb::arg("static_argnums"), nb::arg("static_argnames"),
      nb::arg("global_cache_key"), nb::arg("pytree_registry"),
      nb::arg("shard_arg_fallback"), nb::arg("cache").none() = nb::none());
}

}  // namespace jax
