/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/pjit.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/pjrt/lru_cache.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/jax_jit.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/sharding.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace jax {
namespace {

namespace py = pybind11;

struct PjitCacheEntry {
  std::shared_ptr<xla::PyLoadedExecutable> executable;
  std::vector<py::object> in_shardings;
  std::vector<py::object> out_avals;
  std::vector<py::dtype> out_dtypes;
  std::vector<std::vector<int64_t>> out_shapes;
  std::vector<bool> out_weak_types;
  std::vector<py::object> out_shardings;
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
  std::shared_ptr<Cache> Lookup(pybind11::handle function,
                                absl::Span<const int> donate_argnums);
  std::shared_ptr<Cache> DefaultCache();

  int Size() const { return lru_list_.Size(); }
  int Capacity() const { return lru_list_.Capacity(); }
  void Clear() { lru_list_.Clear(); }

 private:
  struct Key {
    pybind11::handle function;  // Does not hold a reference.

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
    // calls to `pjit(f)` if `f` remains alive, but we do not want the cache
    // to keep `f` alive if all other references are dropped.
    pybind11::weakref weakref;
  };

  Cache::LRUList lru_list_;
  absl::flat_hash_map<Key, std::unique_ptr<Value>> functions_;
};

PjitFunctionCache::PjitFunctionCache(int capacity) : lru_list_(capacity) {}

std::shared_ptr<PjitFunctionCache::Cache> PjitFunctionCache::DefaultCache() {
  return std::make_shared<Cache>(&lru_list_);
}

std::shared_ptr<PjitFunctionCache::Cache> PjitFunctionCache::Lookup(
    pybind11::handle function, absl::Span<const int> donate_argnums) {
  Key key;
  key.function = function;
  key.donate_argnums =
      std::vector<int>(donate_argnums.begin(), donate_argnums.end());
  auto insert = functions_.emplace(key, nullptr);
  if (!insert.second) {
    return insert.first->second->cache;
  }
  std::shared_ptr<Cache> cache = std::make_shared<Cache>(&lru_list_);
  pybind11::cpp_function callback(
      [this, key{std::move(key)}](pybind11::handle weakref) {
        functions_.erase(key);
      });
  PyObject* weakref = PyWeakref_NewRef(function.ptr(), callback.ptr());
  if (weakref) {
    std::unique_ptr<Value>& entry = insert.first->second;
    entry = std::make_unique<Value>(cache);
    entry->weakref = pybind11::reinterpret_steal<pybind11::weakref>(weakref);
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
  PjitFunction(std::string function_name, std::optional<py::function> fun,
               py::function cache_miss, std::vector<int> static_argnums,
               std::vector<py::str> static_argnames,
               std::vector<int> donate_argnums,
               std::shared_ptr<PjitFunctionCache> cache);
  ~PjitFunction();

  PjitFunction(const PjitFunction&) = delete;
  PjitFunction& operator=(const PjitFunction&) = delete;
  PjitFunction(PjitFunction&&) = default;
  PjitFunction& operator=(PjitFunction&&) = default;

  // pybind11::object typed subclass for PjitFunction objects.
  class pyobject : public py::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    py::object, PjitFunction::IsPjitFunction);
    pyobject() = default;
    PjitFunction* func() const {
      return PjitFunction::AsPjitFunctionUnchecked(*this);
    }
  };
  // Alias as ::object; outside the scope above we won't confuse pybind11's
  // macros.
  using object = pyobject;

  // Returns true if `h` is a PjitFunction.
  static bool IsPjitFunction(py::handle handle);
  // Converts `handle` to a PjitFunction*. Does not do any checking.
  static PjitFunction* AsPjitFunctionUnchecked(py::handle handle);

  xla::StatusOr<py::object> Call(py::handle callable, PyObject* const* args,
                                 size_t nargs, PyObject* kwnames);

  void ClearPythonReferences();

  const std::string& function_name() const { return function_name_; }
  const std::optional<py::function>& fun() const { return fun_; }
  const py::function& cache_miss() const { return cache_miss_; }

  const std::vector<int>& static_argnums() const { return static_argnums_; }
  const std::vector<py::str>& static_argnames() const {
    return static_argnames_;
  }
  const std::vector<int>& donate_argnums() const { return donate_argnums_; }
  const std::shared_ptr<PjitFunctionCache>& cache() const { return cache_; }

  int cache_capacity() const { return executables_->Size(); }

  void ClearCache() { executables_->Clear(); }

  py::object PythonSignature() {
    if (!fun_.has_value()) {
      throw py::value_error(absl::StrFormat(
          "Calling __signature__ on PjitFunction(%s) not supported.",
          function_name_));
    }
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(*fun_);
  }

 private:
  xla::Status UpdateArgsSignature(ParsedArgumentsAsBuffers& arguments);

  void PopulateCacheEntry(PjitCacheEntry& cache_entry,
                          const CallSignature& signature,
                          const py::tuple& out_and_fastpath_data);

  std::string function_name_;
  std::optional<py::function> fun_;
  py::function cache_miss_;
  std::vector<int> static_argnums_;
  std::vector<py::str> static_argnames_;
  std::vector<int> donate_argnums_;
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

PjitFunction::PjitFunction(std::string function_name,
                           std::optional<py::function> fun,
                           py::function cache_miss,
                           std::vector<int> static_argnums,
                           std::vector<py::str> static_argnames,
                           std::vector<int> donate_argnums,
                           std::shared_ptr<PjitFunctionCache> cache)
    : function_name_(std::move(function_name)),
      fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      static_argnames_(std::move(static_argnames)),
      donate_argnums_(donate_argnums),
      cache_(std::move(cache)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
  for (py::str& s : static_argnames_) {
    PyUnicode_InternInPlace(&s.ptr());
  }
  if (!fun_.has_value()) {
    executables_ = cache_->DefaultCache();
  } else {
    executables_ = cache_->Lookup(fun_.value(), donate_argnums);
  }

  GetGlobalPjitFunctionStore().Insert(this);
}

PjitFunction::~PjitFunction() { GetGlobalPjitFunctionStore().Erase(this); }

// Prepares the input PjRtBuffers from the python arguments. This is equivalent
// to shard_args() in pxla.py but for only a few supported cases.
xla::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
PrepareIfrtInputs(const xla::PyLoadedExecutable& executable,
                  ParsedArgumentsAsBuffers& arguments,
                  const std::vector<bool>& kept_args) {
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

  for (int i = 0; i < num_args; ++i) {
    if (!kept_args[i]) {
      continue;
    }
    const py::object& arg = arguments.flat_dynamic_args[i];

    if (arg.get_type() != xla::PyArray::type()) {
      if (data_device != nullptr) {
        py::handle arg = arguments.flat_dynamic_args[i];
        TF_ASSIGN_OR_RETURN(
            xla::DevicePutResult on_device,
            DevicePut(arg, executable.ifrt_loaded_executable()->client(),
                      data_device, options));

        num_args_arrays.push_back(std::move(on_device.ifrt_array));
        if (on_device.owning_pybuffer) {
          arguments.keep_alive_objects.push_back(
              std::move(on_device.owning_pybuffer));
        }
        continue;
      }

      return xla::Unimplemented("Unhandled non PyArray argument.");
    }

    xla::PyArray py_array = arg;
    const auto& sharding = py_array.sharding();
    int sharding_num_devices = jax::Sharding::SafeNumDevices(sharding);

    // Currently only committed PyArray inputs or uncommitted PyArray on a
    // single device inputs are allowed. This is checked previously in the entry
    // point of PjitFunction::Call().
    DCHECK(py_array.committed() ||
           (!py_array.committed() && sharding_num_devices == 1));

    if (sharding.get_type() == jax::PmapSharding::type()) {
      return xla::Unimplemented(
          "Handling PyArray in PmapSharding is not implemented.");
    }

    if (py_array.num_shards() != addressable_devices.size()) {
      return xla::InvalidArgument(
          "Expected PyArray to have %d shards, but got %d",
          addressable_devices.size(), py_array.num_shards());
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
          xla::ifrt::DeviceList(std::move(ifrt_devices)));
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

xla::StatusOr<py::object> PjitFunction::Call(py::handle callable,
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
      throw py::value_error(
          absl::StrFormat("Disable jit is not supported in the AOT path since "
                          "the function is not available for (%s)",
                          function_name_));
    }
    return py::reinterpret_steal<py::object>(
        JAX_PyObject_Vectorcall(fun_.value().ptr(), args, nargs, kwnames));
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

  size_t num_positional_args = PyVectorcall_NARGS(nargs);
  size_t num_keyword_args = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  absl::Span<PyObject* const> positional_args(args, num_positional_args);
  absl::Span<PyObject* const> keyword_args(args + num_positional_args,
                                           num_keyword_args);
  auto status = ParseArguments(positional_args, keyword_args, kwnames,
                               static_argnums_, static_argnames_, arguments);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  // Perform a few checks for the arguments. Currently we are only allowing
  // committed PyArray inputs. For other cases, e.g. Tracers or ShapedArray, it
  // will fallback to python. For jit, numpy arrays and scalars are also
  // allowed, which we will check later.
  for (const auto& arg : arguments.flat_dynamic_args) {
    if (arg.get_type() != xla::PyArray::type()) {
      continue;
    }

    xla::PyArray py_array = arg;
    if (!py_array.fastpath_enabled()) {
      return fallback_to_cache_miss();
    }

    // Only allow committed PyArray in cpp pjit for now as the logic on handling
    // sharding for uncommited PyArray is complicated and still under
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

  status = UpdateArgsSignature(arguments);
  if (!status.ok()) {
    VLOG(2) << "UpdateArgsSignature failed: " << status;
    return fallback_to_cache_miss();
  }

  VLOG(2) << "CallSignature:\n" << arguments.signature.DebugString();
  bool inserted = false;
  std::shared_ptr<PjitCacheEntry> cache_entry =
      executables_->GetOrCreateIfAbsent(
          arguments.signature, [&inserted](const CallSignature& unused) {
            inserted = true;
            return std::make_shared<PjitCacheEntry>();
          });

  if (!cache_entry->compilation_complete.HasBeenNotified()) {
    // In case of several threads attempting to compile the executable, only
    // the one that inserted the item will perform the compilation.
    if (inserted) {
      py::object out_and_fastpath_data;
      py::tuple out_tuple;
      VLOG(2) << "Cache miss for " << arguments.signature.DebugString();
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = cache_miss();
        if (!out_and_fastpath_data.ptr()) {
          throw py::error_already_set();
        }
        out_tuple = py::cast<py::tuple>(out_and_fastpath_data);

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

  if (cache_entry->fall_back_to_python) {
    VLOG(2) << "cpp pjit fallback to python.";
    return fallback_to_cache_miss();
  }

  // A vector of [num_inputs].
  auto num_args_arrays = PrepareIfrtInputs(*cache_entry->executable, arguments,
                                           cache_entry->kept_var_bitvec);

  if (!num_args_arrays.ok()) {
    VLOG(2) << "Failed to prepare IFRT inputs: " << num_args_arrays.status();
    return fallback_to_cache_miss();
  }

  // A vector of [num_outputs].
  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  {
    py::gil_scoped_release gil_release;
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
  absl::InlinedVector<py::object, 4> outputs;
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

  py::object out = cache_entry->out_pytree_def.Unflatten(outputs);

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

  return out;
}

xla::Status PjitFunction::UpdateArgsSignature(
    ParsedArgumentsAsBuffers& arguments) {
  arguments.signature.function_name = function_name_;

  // Get dynamic argument signatures.
  JitState& global_state = jax::GlobalJitState();
  JitState& tls = jax::ThreadLocalJitState();
  bool jax_enable_x64 = GetEnableX64();

  arguments.signature.default_device = GetDefaultDevice();
  arguments.signature.jax_enable_x64 = jax_enable_x64;

  auto& dynamic_arg_signatures = arguments.signature.dynamic_arg_signatures;
  dynamic_arg_signatures.reserve(arguments.flat_dynamic_args.size());
  auto& dynamic_arg_shardings = arguments.signature.dynamic_arg_shardings;
  dynamic_arg_shardings.reserve(arguments.flat_dynamic_args.size());

  for (py::handle arg : arguments.flat_dynamic_args) {
    TF_ASSIGN_OR_RETURN(auto signature,
                        xla::PyArgSignatureOfValue(arg, jax_enable_x64));
    arguments.signature.dynamic_arg_signatures.push_back(std::move(signature));

    // It should be already checked previously in the entry point of
    // PjitFunction::Call().
    if (arg.get_type() == xla::PyArray::type()) {
      auto py_array = py::reinterpret_borrow<xla::PyArray>(arg);

      arguments.signature.dynamic_arg_shardings.push_back(py_array.sharding());
      arguments.signature.committed_args.push_back(py_array.committed());
    } else {
      arguments.signature.dynamic_arg_shardings.push_back(py::none());
      arguments.signature.committed_args.push_back(false);
    }
  }

  arguments.signature.thread_local_extra_jit_context = tls.extra_jit_context;
  arguments.signature.global_extra_jit_context = global_state.extra_jit_context;

  return xla::OkStatus();
}

void PjitFunction::PopulateCacheEntry(PjitCacheEntry& cache_entry,
                                      const CallSignature& signature,
                                      const py::tuple& out_and_fastpath_data) {
  DCHECK_EQ(out_and_fastpath_data.size(), 2);

  if (out_and_fastpath_data[1].is_none()) {
    VLOG(2) << "fastpath_data is none";
    cache_entry.fall_back_to_python = true;
    return;
  }

  py::tuple fastpath_data = py::cast<py::tuple>(out_and_fastpath_data[1]);

  cache_entry.executable = py::cast<std::shared_ptr<xla::PyLoadedExecutable>>(
      fastpath_data.attr("xla_executable"));

  py::list in_shardings = fastpath_data.attr("in_shardings");
  cache_entry.in_shardings.reserve(in_shardings.size());
  for (py::handle sharding : in_shardings) {
    cache_entry.in_shardings.push_back(
        py::reinterpret_borrow<py::object>(sharding));
  }

  py::list out_shardings = fastpath_data.attr("out_shardings");
  cache_entry.out_shardings.reserve(out_shardings.size());
  for (py::handle sharding : out_shardings) {
    cache_entry.out_shardings.push_back(
        py::reinterpret_borrow<py::object>(sharding));
  }

  py::list out_committed = fastpath_data.attr("out_committed");
  cache_entry.out_committed.reserve(out_committed.size());
  for (py::handle c : out_committed) {
    cache_entry.out_committed.push_back(py::cast<bool>(c));
  }

  py::list out_avals = fastpath_data.attr("out_avals");
  cache_entry.out_avals.reserve(out_avals.size());
  cache_entry.out_dtypes.reserve(out_avals.size());
  cache_entry.out_shapes.reserve(out_avals.size());
  cache_entry.out_weak_types.reserve(out_avals.size());
  for (py::handle aval : out_avals) {
    cache_entry.out_avals.push_back(py::reinterpret_borrow<py::object>(aval));
    cache_entry.out_dtypes.push_back(aval.attr("dtype"));
    cache_entry.out_shapes.push_back(
        py::cast<std::vector<int64_t>>(aval.attr("shape")));
    cache_entry.out_weak_types.push_back(
        py::cast<bool>(aval.attr("weak_type")));
  }

  cache_entry.out_pytree_def =
      py::cast<xla::PyTreeDef>(fastpath_data.attr("out_pytree_def"));

  py::list kept_var_bitvec = fastpath_data.attr("kept_var_bitvec");
  cache_entry.kept_var_bitvec.reserve(kept_var_bitvec.size());
  for (py::handle k : kept_var_bitvec) {
    cache_entry.kept_var_bitvec.push_back(py::cast<bool>(k));
  }
}

// Helper function used by the tp_clear GC method.
void PjitFunction::ClearPythonReferences() {
  py::function cache_miss;
  // Swap values for nulls before they are destroyed. See the Python
  // Py_CLEAR() documentation for a discussion of this topic.
  std::swap(cache_miss_, cache_miss);
}

struct PjitFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
  vectorcallfunc vectorcall;
  PjitFunction fun;
};

PyObject* PjitFunction_Type = nullptr;

bool PjitFunction::IsPjitFunction(py::handle handle) {
  return handle.get_type() == PjitFunction_Type;
}

PjitFunction* PjitFunction::AsPjitFunctionUnchecked(py::handle handle) {
  return &(reinterpret_cast<PjitFunctionObject*>(handle.ptr())->fun);
}

PjitFunction* AsPjitFunction(py::handle handle) {
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
  PjitFunctionObject* o = reinterpret_cast<PjitFunctionObject*>(self);
#if PY_VERSION_HEX >= 0x03090000
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
#endif
  Py_VISIT(o->dict);
  Py_VISIT(o->fun.cache_miss().ptr());
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
        static_cast<std::string>(py::repr(py::getattr(self, "__wrapped__"))));
    return PyUnicode_FromString(repr.c_str());
  } catch (...) {
    // Ignore all errors when accessing a repr.
    return PyUnicode_FromString("<PjitFunction>");
  }
}

}  // extern "C"

void InitializePjitFunction(
    PjitFunctionObject* fn_obj, std::string function_name,
    std::optional<py::function> fun, py::function cache_miss,
    std::vector<int> static_argnums, std::vector<py::str> static_argnames,
    std::vector<int> donate_argnums, std::shared_ptr<PjitFunctionCache> cache) {
  new (&fn_obj->fun) PjitFunction(
      std::move(function_name), std::move(fun), std::move(cache_miss),
      std::move(static_argnums), std::move(static_argnames),
      std::move(donate_argnums), std::move(cache));
}

py::object MakePjitFunction(std::string function_name,
                            std::optional<py::function> fun,
                            py::function cache_miss,
                            std::vector<int> static_argnums,
                            std::vector<py::str> static_argnames,
                            std::vector<int> donate_argnums,
                            std::shared_ptr<PjitFunctionCache> cache) {
  py::object obj = py::reinterpret_steal<py::object>(PjitFunction_tp_new(
      reinterpret_cast<PyTypeObject*>(PjitFunction_Type), nullptr, nullptr));
  PjitFunctionObject* fn_obj = reinterpret_cast<PjitFunctionObject*>(obj.ptr());
  if (!cache) {
    cache = std::make_shared<PjitFunctionCache>(
        PjitFunctionCache::kDefaultCapacity);
  }
  InitializePjitFunction(fn_obj, std::move(function_name), std::move(fun),
                         std::move(cache_miss), std::move(static_argnums),
                         std::move(static_argnames), std::move(donate_argnums),
                         std::move(cache));
  return obj;
}

// Version numbers for the pickled representations of
// PjitFunction. Increment these if changing them.
const int kPjitFunctionPickleVersion = 1;

}  // namespace

void BuildPjitSubmodule(py::module& m) {
  py::class_<PjitFunctionCache, std::shared_ptr<PjitFunctionCache>> cache(
      m, "PjitFunctionCache");
  cache.def(py::init<int>(),
            py::arg("capacity") = PjitFunctionCache::kDefaultCapacity);
  cache.def("size", &PjitFunctionCache::Size);
  cache.def("capacity", &PjitFunctionCache::Capacity);
  cache.def("clear", &PjitFunctionCache::Clear);
  cache.def_static("clear_all",
                   []() { GetGlobalPjitFunctionStore().ClearFunctionCache(); });
  cache.def(py::pickle(
      // __getstate__
      // Pickles as an empty cache; the client can repopulate as needed.
      [](const PjitFunctionCache& cache) {
        py::dict pickle;
        pickle["version"] = kPjitFunctionPickleVersion;
        pickle["capacity"] = cache.Capacity();
        return pickle;
      },
      // __setstate__
      [](const py::dict& pickle) {
        int version = py::cast<int>(pickle["version"]);
        if (version != kPjitFunctionPickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid PjitFunction pickle version, got %d, expected %d",
              version, kPjitFunctionPickleVersion));
        }
        int capacity = py::cast<int>(pickle["capacity"]);
        return std::make_shared<PjitFunctionCache>(capacity);
      }));

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  py::object cfun;
  {
    py::str name = py::str("PjitFunction");
    py::str qualname = py::str("PjitFunction");
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
                     Py_TPFLAGS_HAVE_GC | JAX_TPFLAGS_HAVE_VECTORCALL;
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
    cfun = py::reinterpret_borrow<py::object>(PjitFunction_Type);
  }
  py::object cfun_type = py::reinterpret_borrow<py::object>(PjitFunction_Type);

  // Add PjitFunction to the xla_extension module so it can be pickled.
  m.attr("PjitFunction") = cfun_type;
  cfun.attr("__module__") = m.attr("__name__");

  cfun.attr("__getstate__") = py::cpp_function(
      [](const PjitFunction::object& self) {
        PjitFunction* fn = self.func();
        py::dict pickle;
        pickle["version"] = kPjitFunctionPickleVersion;
        pickle["function_name"] = fn->function_name();
        if (fn->fun().has_value()) {
          pickle["fun"] = *fn->fun();
        }
        pickle["cache_miss"] = fn->cache_miss();
        pickle["static_argnums"] = fn->static_argnums();
        pickle["static_argnames"] = fn->static_argnames();
        pickle["donate_argnums"] = fn->donate_argnums();
        pickle["cache"] = fn->cache();
        return pickle;
      },
      py::is_method(cfun_type));
  cfun.attr("__setstate__") = py::cpp_function(
      [](py::object& self, const py::dict& pickle) {
        int version = py::cast<int>(pickle["version"]);
        if (version != kPjitFunctionPickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid PjitFunction pickle version, got %d, expected %d. "
              "Pickling/Unpickling jitted functions using different JAX "
              "versions is not supported.",
              version, kPjitFunctionPickleVersion));
        }
        std::string function_name =
            py::cast<std::string>(pickle["function_name"]);
        std::optional<py::function> fun;
        if (pickle.contains("fun")) {
          fun = py::cast<py::function>(pickle["fun"]);
        }
        py::function cache_miss = py::cast<py::function>(pickle["cache_miss"]);
        std::vector<int> static_argnums =
            py::cast<std::vector<int>>(pickle["static_argnums"]);
        std::vector<py::str> static_argnames =
            py::cast<std::vector<py::str>>(pickle["static_argnames"]);
        std::vector<int> donate_argnums =
            py::cast<std::vector<int>>(pickle["donate_argnums"]);
        std::shared_ptr<PjitFunctionCache> cache =
            py::cast<std::shared_ptr<PjitFunctionCache>>(pickle["cache"]);
        InitializePjitFunction(
            reinterpret_cast<PjitFunctionObject*>(self.ptr()),
            std::move(function_name), std::move(fun), std::move(cache_miss),
            std::move(static_argnums), std::move(static_argnames),
            std::move(donate_argnums), std::move(cache));
      },
      py::is_method(cfun_type));
  cfun.attr("__signature__") =
      property_readonly([](py::handle self) -> py::object {
        return AsPjitFunction(self)->PythonSignature();
      });
  cfun.attr("_cache_miss") =
      property_readonly([](py::handle self) -> py::object {
        return AsPjitFunction(self)->cache_miss();
      });
  // All private members are only for testing/debugging purposes
  cfun.attr("_cache_size") = py::cpp_function(
      [](py::handle self) -> int {
        return AsPjitFunction(self)->cache_capacity();
      },
      py::is_method(cfun));
  cfun.attr("_clear_cache") = py::cpp_function(
      [](py::handle self) { AsPjitFunction(self)->ClearCache(); },
      py::is_method(cfun));

  m.def(
      "pjit",
      [](std::string function_name, std::optional<py::function> fun,
         py::function cache_miss, std::vector<int> static_argnums,
         std::vector<py::str> static_argnames, std::vector<int> donate_argnums,
         std::shared_ptr<PjitFunctionCache> cache) {
        return MakePjitFunction(
            std::move(function_name), std::move(fun), std::move(cache_miss),
            std::move(static_argnums), std::move(static_argnames),
            std::move(donate_argnums), std::move(cache));
      },
      py::arg("function_name"), py::arg("fun"), py::arg("cache_miss"),
      py::arg("static_argnums"),
      py::arg("static_argnames") = std::vector<py::str>(),
      py::arg("donate_argnums") = std::vector<int>(),
      py::arg("cache") = nullptr);
}

}  // namespace jax
