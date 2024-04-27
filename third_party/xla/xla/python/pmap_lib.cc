/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/python/pmap_lib.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/variant.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/jax_jit.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_array.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device.h"
#include "xla/python/py_executable.h"
#include "xla/python/py_values.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/pytree.h"
#include "xla/python/sharded_device_array.h"
#include "xla/python/sharding.h"
#include "xla/python/traceback.h"
#include "xla/python/types.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax {

namespace nb = nanobind;

namespace {

// Specifies how to shard the inputs. Even though everything could be computed
// from `sharding_specs` and the argument shape, we cache derived computations
// for performance.
struct InputSpec {
  InputSpec(nb::object indices, nb::object array_sharding)
      : indices(std::move(indices)),
        array_sharding(std::move(array_sharding)) {}
  nb::object indices;
  nb::object array_sharding;
};

// An object containing the arguments to create Array from the
// output buffers.
struct ResultSpec {
 public:
  explicit ResultSpec(nb::object aval)
      : out_aval(std::move(aval)),
        weak_type(nb::cast<bool>(out_aval.attr("weak_type"))) {}
  nb::object out_aval;
  bool weak_type;
};

// The result of `ShardArg`.
struct ShardArgResult {
  // Points to the on-device array.
  // ifrt_array->sharding().num_shards() == `num_devices`.
  tsl::RCReference<xla::ifrt::Array> ifrt_array;
  // The Python argument will be always be copied to `owning_sda`.
  nb::object owning_sda;
};

// Shars a single argument over devices.
//
// We currently only support fully in C++, C++ Array. For all
// other usages, we call a Python function returning C++ Array
// that will be casted back to the C++ objects.
//
// This function is not usable for JAX extensions that do not comply with the
// PjRt interfaces.
//
// Arguments:
// `arg`: The object to shard across `devices`. If a `Array`,
//   a fast-path will be executed if it's already correctly sharded.
//
// Returns a failure Status when an unrecoverable error occurred, so we don't
// need to fallback to Python.
//
// Both `devices` and `sharding_spec` has the same length.
absl::StatusOr<ShardArgResult> ShardArg(
    nb::handle arg, absl::Span<xla::ifrt::Device* const> devices,
    const InputSpec& input_spec, nb::handle py_devices,
    const nb::callable& python_fallback) {
  if (arg.type().ptr() == xla::PyArray::type().ptr()) {
    auto py_array = nb::borrow<xla::PyArray>(arg);
    if (py_array.sharding().type().ptr() ==
        input_spec.array_sharding.type().ptr()) {
      auto* pmap_sharding =
          nb::cast<jax::PmapSharding*>(nb::handle(py_array.sharding().ptr()));
      auto* cached_pmap_sharding = nb::cast<jax::PmapSharding*>(
          nb::handle(input_spec.array_sharding.ptr()));

      if (pmap_sharding->sharding_spec() ==
          cached_pmap_sharding->sharding_spec()) {
        ShardArgResult result;
        result.owning_sda = nb::borrow<nb::object>(arg.ptr());
        result.ifrt_array = tsl::FormRef(py_array.ifrt_array());
        if (result.ifrt_array == nullptr) {
          return xla::InvalidArgument("Array has been deleted.");
        }
        if (result.ifrt_array->sharding().devices().devices() != devices) {
          xla::ifrt::DeviceList::Devices ifrt_devices;
          ifrt_devices.reserve(devices.size());
          ifrt_devices.insert(ifrt_devices.end(), devices.begin(),
                              devices.end());
          // pmap does not support memory_kind for now.
          auto sharding = xla::ifrt::OpaqueSharding::Create(
              xla::ifrt::DeviceList(std::move(ifrt_devices)),
              xla::ifrt::MemoryKind());
          TF_ASSIGN_OR_RETURN(auto copied_ifrt_array,
                              result.ifrt_array->Reshard(
                                  std::move(sharding),
                                  xla::ifrt::ArrayCopySemantics::kReuseInput));
          result.ifrt_array = std::move(copied_ifrt_array);
        }
        return result;
      }
    }
  }

  auto ndarray = xla::nb_numpy_ndarray::ensure(arg);
  if (ndarray && PyArray_CheckExact(arg.ptr()) &&
      xla::DtypeToPrimitiveType(ndarray.dtype()).status().ok()) {
    tsl::profiler::TraceMe traceme("ndarray pmap ShardArg");
    nb::list indices = nb::list(input_spec.indices);
    nb::list py_devices_list = nb::cast<nb::list>(py_devices);
    auto n_devices = py_devices_list.size();
    if (indices.size() != n_devices) {
      return xla::InvalidArgument("indices vs devices mismatch: %d vs %d",
                                  indices.size(), n_devices);
    }

    std::vector<tsl::RCReference<xla::ifrt::Array>> per_device_arrays;
    per_device_arrays.reserve(n_devices);
    xla::ifrt::DeviceList::Devices devices;
    devices.reserve(n_devices);
    // TODO(hyeontaek): The created array will never be disassembled. We should
    // omit collecting shapes and make the OpaqueSharding non-disassemblable?
    std::vector<xla::ifrt::Shape> shapes;
    shapes.reserve(n_devices);

    nb::list owning_pylist;
    ShardArgResult result;
    result.owning_sda = owning_pylist;
    const bool jax_enable_x64 = GetEnableX64();

    std::vector<xla::DevicePutResultFn> device_put_fns;
    device_put_fns.reserve(n_devices);
    xla::DevicePutOptions options;
    options.squash_64bit_types = !jax_enable_x64;
    options.allow_zero_copy = true;
    for (size_t i = 0; i < n_devices; ++i) {
      auto to_device = nb::cast<xla::PyDevice*>(py_devices_list[i]);
      if (to_device->client().get() == nullptr) {
        return xla::InvalidArgument("Cannot copy to unattached devices.");
      }

      TF_ASSIGN_OR_RETURN(
          device_put_fns.emplace_back(),
          DevicePut(arg[indices[i]], to_device->client()->ifrt_client(),
                    to_device->device(), options, xla::ifrt::MemoryKind()));
    }
    std::vector<xla::DevicePutResult> device_puts;
    device_puts.reserve(n_devices);
    {
      nb::gil_scoped_release gil_release;
      for (auto& device_put_fn : device_put_fns) {
        TF_ASSIGN_OR_RETURN(auto device_put, std::move(device_put_fn)());
        device_puts.push_back(std::move(device_put));
      }
    }
    for (auto& device_put : device_puts) {
      per_device_arrays.push_back(std::move(device_put.ifrt_array));
      devices.push_back(per_device_arrays.back()->sharding().devices().front());
      shapes.push_back(per_device_arrays.back()->shape());
      if (device_put.owning_pybuffer) {
        owning_pylist.append(device_put.owning_pybuffer);
      }
    }

    // TODO(hyeontaek): The logical shape here is inaccurate. We
    // may want to avoid creating a new Array or specialize Array
    // to disallow access to the logical shape.
    xla::ifrt::Shape shape = per_device_arrays.front()->shape();
    // pmap does not support memory_kind for now.
    auto ifrt_sharding = xla::ifrt::ConcreteSharding::Create(
        xla::ifrt::DeviceList(std::move(devices)), xla::ifrt::MemoryKind(),
        /*shape=*/shape,
        /*shard_shapes=*/std::move(shapes));
    TF_ASSIGN_OR_RETURN(result.ifrt_array,
                        per_device_arrays.front()
                            ->client()
                            ->AssembleArrayFromSingleDeviceArrays(
                                std::move(shape), std::move(ifrt_sharding),
                                absl::MakeSpan(per_device_arrays),
                                xla::ifrt::ArrayCopySemantics::kReuseInput));
    return result;
  }
  tsl::profiler::TraceMe traceme("pmap_lib_shard_arg_python_fallback");
  auto py_array_or_bufs = python_fallback(arg, input_spec.array_sharding);

  auto py_array = nb::cast<xla::PyArray>(py_array_or_bufs);
  ShardArgResult result;
  result.owning_sda = nb::borrow(py_array_or_bufs.ptr());
  result.ifrt_array = tsl::FormRef(py_array.ifrt_array());
  return result;
}

struct PmapCacheEntry {
  explicit PmapCacheEntry(xla::PyTreeRegistry* registry)
      : out_pytree_def(registry) {}
  std::shared_ptr<xla::PyLoadedExecutable> executable;
  // The value `backend.local_devices()`.
  nb::object py_devices;  // To pass back to Python.
  std::vector<xla::ifrt::Device*> devices;
  std::vector<InputSpec> input_specs;
  xla::PyTreeDef out_pytree_def;
  // Objects necessary to build the out Array objects.
  std::vector<ResultSpec> out_result_specs;

  std::vector<nb::object> out_array_shardings;
  std::vector<xla::nb_dtype> out_dtypes;
  std::vector<std::vector<int64_t>> out_shapes;
  std::vector<bool> out_committed;

  // Ensures a single thread performs the compilation for a given executable.
  //
  // The first thread (holding the GIL) will create the CacheEntry associated to
  // a signature and if the object has been inserted already, other threads
  // will wait for the notification.
  absl::Notification compilation_complete;

  bool fall_back_to_python = false;
};

}  // namespace

// A `PmapFunction` is associated to a `jax.pmap(f)` and takes care of the
// bookkeeping of the different signatures used and the dispatch of calls to
// the correct underlying `PyLoadedExecutable`. This class is thread-safe.
class PmapFunction {
 public:
  PmapFunction(nb::callable fun, nb::callable cache_miss,
               std::vector<int> static_argnums,
               nb::callable python_shard_arg_fallback,
               std::shared_ptr<xla::PyTreeRegistry> pytree_registry)
      : fun_(std::move(fun)),
        cache_miss_(std::move(cache_miss)),
        static_argnums_(std::move(static_argnums)),
        pytree_registry_(std::move(pytree_registry)),
        python_shard_arg_fallback_(std::move(python_shard_arg_fallback)) {
    std::sort(static_argnums_.begin(), static_argnums_.end());

    function_name_ =
        nb::cast<std::string>(nb::str(nb::getattr(fun_, "__name__", fun_)));
  }
  PmapFunction(const PmapFunction&) = delete;
  PmapFunction& operator=(const PmapFunction& other) = delete;
  PmapFunction(PmapFunction&&) = default;
  PmapFunction& operator=(PmapFunction&&) = default;

  // This function will:
  // (a) flatten the inputs using pytree
  // (b) get buffer objects from the arguments
  // (c) call the executable
  // (d) construct `Array` objects from the outputs
  // (e) reconstruct the `PyTree`.
  absl::StatusOr<nb::object> Call(nb::handle callable, PyObject* const* args,
                                  size_t nargs, PyObject* kwnames);

  nb::object PythonSignature() {
    static const auto* inspect =
        new nb::module_(nb::module_::import_("inspect"));
    return inspect->attr("signature")(fun_);
  }

  int cache_size() const { return executables_.size(); }
  void cache_clear() { return executables_.clear(); }
  const nb::callable& fun() const { return fun_; }
  const nb::callable& cache_miss() const { return cache_miss_; }
  const std::string& function_name() const { return function_name_; }
  const std::shared_ptr<xla::PyTreeRegistry>& pytree_registry() const {
    return pytree_registry_;
  }
  const nb::callable& python_shard_arg_fallback() const {
    return python_shard_arg_fallback_;
  }
  const std::vector<int>& static_argnums() const { return static_argnums_; }

  // nb::object typed subclass for PmapFunction objects.
  class pyobject : public nb::object {
   public:
    NB_OBJECT(pyobject, nb::object, "PmapFunction",
              PmapFunction::IsPmapFunction);
    pyobject() = default;
    PmapFunction* func() const {
      return PmapFunction::AsPmapFunctionUnchecked(*this);
    }
  };
  // Alias as ::object; outside the scope above we won't confuse nanobind's
  // macros.
  using object = pyobject;

  // Returns true if `h` is a PmapFunction.
  static bool IsPmapFunction(nb::handle handle);
  // Converts `handle` to a PmapFunction*. Does not do any checking.
  static PmapFunction* AsPmapFunctionUnchecked(nb::handle handle);

  // Helper function used by the tp_clear GC method.
  void ClearPythonReferences() {
    nb::callable fun, cache_miss, python_shard_arg_fallback;
    // Swap values for nulls before they are destroyed. See the Python
    // Py_CLEAR() documentation for a discussion of this topic.
    std::swap(fun_, fun);
    std::swap(cache_miss_, cache_miss);
    std::swap(python_shard_arg_fallback_, python_shard_arg_fallback);
  }

  // Updates the signature of arguments for a pmapped function.
  //
  // It deals with the arguments signatures and also of the global and
  // thread-local jit context.
  absl::Status ComputeCallSignature(
      absl::Span<nb::object const> flat_dynamic_args,
      CallSignature& signature) {
    signature.function_name = function_name_;

    // Get dynamic argument signatures.
    JitState& global_state = jax::GlobalJitState();
    JitState& tls = jax::ThreadLocalJitState();
    const bool jax_enable_x64 = GetEnableX64();
    signature.jax_enable_x64 = jax_enable_x64;
    for (nb::handle arg : flat_dynamic_args) {
      auto signature_or_error = xla::PyArgSignatureOfValue(arg, jax_enable_x64);
      if (!signature_or_error.ok()) {
        VLOG(2) << "PyArgSignatureOfValue failed: "
                << signature_or_error.status();
        return signature_or_error.status();
      }
      signature.dynamic_arg_signatures.push_back(
          std::move(signature_or_error).value());
    }
    signature.thread_local_extra_jit_context = tls.extra_jit_context;
    signature.global_extra_jit_context = global_state.extra_jit_context;
    return absl::Status();
  }

  // Returns, for debugging purposes (e.g. finding why some call misses the
  // cache and recompiles), the list of the string representations of the keys.
  //
  // The format can change at any time.
  std::string DebugCacheKeys() const {
    std::vector<std::string> key_strings = {
        absl::StrCat("The cache contains ", executables_.size(), " elements:")};
    // We will be able to use auto& [key, _] when TF uses C++ 17.
    for (auto& pair : executables_) {
      key_strings.push_back(pair.first.DebugString());
    }
    return absl::StrJoin(key_strings, "\n\n");
  }

 private:
  // Mutates `cache_entry` in place.
  void PopulateCacheEntry(PmapCacheEntry& cache_entry,
                          const nb::tuple& out_and_fastpath_data);

  bool always_fallback_to_python_ = false;

  nb::callable fun_;  // The Python function to pmap.
  std::string function_name_;
  // See JAX _cpp_pmap in api.py for documentation.
  nb::callable cache_miss_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyLoadedExecutable. In sorted order.
  std::vector<int> static_argnums_;
  std::shared_ptr<xla::PyTreeRegistry> pytree_registry_;
  // We need a `unique_ptr` here to ensure value pointer stability.
  absl::flat_hash_map<CallSignature, std::unique_ptr<PmapCacheEntry>>
      executables_;

  // The fallback function to use with `ShardArgs`.
  // TODO(jblespiau): Add support for more types from C++.
  nb::callable python_shard_arg_fallback_;
};

void PmapFunction::PopulateCacheEntry(PmapCacheEntry& cache_entry,
                                      const nb::tuple& out_and_fastpath_data) {
  CHECK_EQ(out_and_fastpath_data.size(), 2);
  if (out_and_fastpath_data[1].is_none()) {
    cache_entry.fall_back_to_python = true;
    return;
  }

  nb::tuple pmap_data = nb::cast<nb::tuple>(out_and_fastpath_data[1]);
  if (nb::cast<int>(pmap_data.attr("version")) != 1) {
    throw xla::XlaRuntimeError(absl::StrCat(
        "The versions of jaxlib and Jax are incompatible (pmap cpp version 1 "
        "expected, but got ",
        nb::cast<int>(pmap_data.attr("version")),
        "Upgrade jaxlib and jax. Provided data was:",
        nb::cast<std::string>(nb::str(nb::repr(pmap_data)))));
  }
  // See api.nb::_PmapFastpathData in the JAX code base for the expected
  // namedtuple.
  std::shared_ptr<xla::PyLoadedExecutable> executable;
  try {
    executable = nb::cast<std::shared_ptr<xla::PyLoadedExecutable>>(
        pmap_data.attr("xla_executable"));
  } catch (const nb::cast_error& e) {
    // Backends that don't implement the C++ PjRt APIs
    cache_entry.fall_back_to_python = true;
    always_fallback_to_python_ = true;
    return;
  }
  cache_entry.executable = std::move(executable);
  const std::vector<xla::nb_class_ptr<xla::PyDevice>>& devices =
      cache_entry.executable->AddressableDevices();
  cache_entry.devices.reserve(devices.size());
  for (auto& device : devices) {
    cache_entry.devices.push_back(device->device());
  }

  // Inputs shard args details.
  nb::list input_indices = pmap_data.attr("input_indices");

  cache_entry.py_devices = pmap_data.attr("input_devices");
  auto input_devices = nb::cast<std::vector<xla::nb_class_ptr<xla::PyDevice>>>(
      pmap_data.attr("input_devices"));

  nb::list input_array_shardings = pmap_data.attr("input_array_shardings");

  cache_entry.input_specs.reserve(input_array_shardings.size());

  for (int i = 0; i < input_array_shardings.size(); ++i) {
    cache_entry.input_specs.emplace_back(input_indices[i],
                                         input_array_shardings[i]);
  }

  // Outputs specs.
  auto out_tree = nb::cast<xla::PyTreeDef>(
      nb::handle(pmap_data.attr("out_pytree_def").ptr()));
  cache_entry.out_pytree_def = std::move(out_tree);
  nb::list out_avals = pmap_data.attr("out_avals");

  cache_entry.out_result_specs.reserve(out_avals.size());
  cache_entry.out_dtypes.reserve(out_avals.size());
  cache_entry.out_shapes.reserve(out_avals.size());

  for (int i = 0; i < out_avals.size(); ++i) {
    cache_entry.out_dtypes.push_back(out_avals[i].attr("dtype"));
    cache_entry.out_shapes.push_back(
        nb::cast<std::vector<int64_t>>(out_avals[i].attr("shape")));
    cache_entry.out_result_specs.emplace_back(out_avals[i]);
  }

  nb::list out_array_shardings = pmap_data.attr("out_array_shardings");

  DCHECK(out_array_shardings.size() == 0 ||
         out_avals.size() == out_array_shardings.size());

  cache_entry.out_array_shardings.reserve(out_array_shardings.size());
  for (nb::handle out_array_sharding : out_array_shardings) {
    cache_entry.out_array_shardings.push_back(
        nb::borrow<nb::object>(out_array_sharding));
  }

  nb::list out_committed = pmap_data.attr("out_committed");

  DCHECK(out_committed.size() == 0 || out_avals.size() == out_committed.size());

  cache_entry.out_committed.reserve(out_committed.size());
  for (nb::handle c : out_committed) {
    cache_entry.out_committed.push_back(nb::cast<bool>(c));
  }
}

absl::StatusOr<nb::object> PmapFunction::Call(nb::handle callable,
                                              PyObject* const* args,
                                              size_t nargs, PyObject* kwnames) {
  xla::GlobalPyRefManager()->MaybeCollectGarbage();

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

  if (always_fallback_to_python_) {
    return fallback_to_cache_miss();
  }

  size_t num_positional_args = PyVectorcall_NARGS(nargs);
  size_t num_keyword_args = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  absl::Span<PyObject* const> positional_args(args, num_positional_args);
  absl::Span<PyObject* const> keyword_args(args + num_positional_args,
                                           num_keyword_args);
  CallSignature call_signature;
  absl::InlinedVector<nb::object, 2> flat_dynamic_args;
  std::vector<nb::object> keep_alive_objects;
  absl::Status status =
      ParseArguments(positional_args, keyword_args, kwnames, static_argnums_,
                     /*static_argnames=*/{}, pytree_registry_.get(),
                     call_signature.arg_signature, flat_dynamic_args);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  status = ComputeCallSignature(flat_dynamic_args, call_signature);
  if (!status.ok()) {
    return fallback_to_cache_miss();
  }

  // Retrieve/Maybe add the executable to the cache.
  absl::flat_hash_map<CallSignature, std::unique_ptr<PmapCacheEntry>>::iterator
      it;
  bool inserted;
  std::tie(it, inserted) = executables_.try_emplace(
      call_signature, std::unique_ptr<PmapCacheEntry>());
  if (inserted) {
    it->second = std::make_unique<PmapCacheEntry>(pytree_registry_.get());
  }
  PmapCacheEntry& cache_entry = *(it->second);

  if (!cache_entry.compilation_complete.HasBeenNotified()) {
    // In case of several threads attempting to compile the executable, only
    // the one that inserted the item will perform the compilation.
    if (inserted) {
      nb::object out_and_fastpath_data;
      nb::tuple out_tuple;
      VLOG(2) << "Cache miss for " << call_signature.DebugString();
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = cache_miss();
        if (!out_and_fastpath_data.ptr()) {
          throw nb::python_error();
        }
        out_tuple = nb::cast<nb::tuple>(out_and_fastpath_data);

        PopulateCacheEntry(cache_entry, out_tuple);
      } catch (const std::exception& e) {
        cache_entry.fall_back_to_python = true;
        cache_entry.compilation_complete.Notify();
        throw;
      }
      cache_entry.compilation_complete.Notify();

      // We have already computed the result in the miss path so we can return
      // it. We are even *required* to do so if there are donated arguments,
      // because any donated buffers will now be invalid.
      return nb::object(out_tuple[0]);
    } else {
      // Release the GIL while we wait, making sure the compile thread can
      // lock it.
      nb::gil_scoped_release release;
      cache_entry.compilation_complete.WaitForNotification();
    }
  }
  if (cache_entry.fall_back_to_python) {
    return fallback_to_cache_miss();
  }

  // 1. Parse arguments.
  std::vector<xla::ifrt::Device*>& input_devices = cache_entry.devices;
  std::vector<InputSpec>& input_specs = cache_entry.input_specs;
  const int num_args = flat_dynamic_args.size();

  // We need [num_args] for the `Execute` call below.
  std::vector<tsl::RCReference<xla::ifrt::Array>> num_args_arrays(num_args);
  for (int i = 0; i < num_args; ++i) {
    TF_ASSIGN_OR_RETURN(
        ShardArgResult sharded_arg,
        ShardArg(flat_dynamic_args[i].ptr(), input_devices, input_specs[i],
                 cache_entry.py_devices, python_shard_arg_fallback_));

    num_args_arrays[i] = std::move(sharded_arg.ifrt_array);
    if (sharded_arg.owning_sda) {
      keep_alive_objects.push_back(std::move(sharded_arg.owning_sda));
    }
  }

  // A vector of [num_outputs].
  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  {
    nb::gil_scoped_release gil_release;
    auto ifrt_executable = cache_entry.executable->ifrt_executable();
    TF_ASSIGN_OR_RETURN(
        auto result, ifrt_executable->Execute(absl::MakeSpan(num_args_arrays),
                                              cache_entry.executable->options(),
                                              /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
  }

  // TODO(jblespiau): We don't need to create the PyBuffer objects.
  // Having a C++ `Array`, keeping internally the PjRtBuffer
  // objects is sufficient, and we can lazily create the `PyBuffer` only if
  // we access them from Python.
  auto traceback = xla::Traceback::Get();
  // TODO(jblespiau): Change the `client` function to return a reference.
  xla::nb_class_ptr<xla::PyClient> client = cache_entry.executable->client();

  // Convert the PjRtBuffer objects to PyBuffer, and invert the order from
  // [num_devices, num_args] to [num_args, num_devices].
  const int num_outputs = output_arrays.size();
  std::vector<nb::object> flat_sharded_device_arrays;
  flat_sharded_device_arrays.reserve(num_outputs);

  const auto& output_specs = cache_entry.out_result_specs;

  TF_RET_CHECK(cache_entry.out_array_shardings.size() == num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const ResultSpec& result_spec = output_specs[i];
    xla::PyArray py_array(
        result_spec.out_aval, result_spec.weak_type, cache_entry.out_dtypes[i],
        cache_entry.out_shapes[i], cache_entry.out_array_shardings[i], client,
        traceback, std::move(output_arrays[i]), cache_entry.out_committed[i],
        /*skip_checks=*/true);

    flat_sharded_device_arrays.push_back(std::move(py_array));
  }

  nb::object out =
      cache_entry.out_pytree_def.Unflatten(flat_sharded_device_arrays);

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

struct JaxPmapFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
  vectorcallfunc vectorcall;
  PmapFunction fun;
};

PyObject* JaxPmapFunction_Type = nullptr;

bool PmapFunction::IsPmapFunction(nb::handle handle) {
  return handle.type().ptr() == JaxPmapFunction_Type;
}

PmapFunction* PmapFunction::AsPmapFunctionUnchecked(nb::handle handle) {
  return &(reinterpret_cast<JaxPmapFunctionObject*>(handle.ptr())->fun);
}

absl::StatusOr<PmapFunction*> AsPmapFunction(nb::handle handle) {
  if (!PmapFunction::IsPmapFunction(handle)) {
    return xla::InvalidArgument("Expected a PmapFunction");
  }
  return PmapFunction::AsPmapFunctionUnchecked(handle);
}

namespace {

extern "C" {

PyObject* JaxPmapFunction_tp_vectorcall(PyObject* callable,
                                        PyObject* const* args, size_t nargs,
                                        PyObject* kwnames) {
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(callable);
  tsl::profiler::TraceMe traceme([&] {
    return absl::StrCat("JaxPmapFunction(", o->fun.function_name(), ")");
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
  }
}

PyObject* JaxPmapFunction_tp_new(PyTypeObject* subtype, PyObject* args,
                                 PyObject* kwds) {
  JaxPmapFunctionObject* self =
      reinterpret_cast<JaxPmapFunctionObject*>(subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->dict = nullptr;
  self->weakrefs = nullptr;
  self->vectorcall = JaxPmapFunction_tp_vectorcall;
  return reinterpret_cast<PyObject*>(self);
}

void JaxPmapFunction_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(self);
  if (o->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  Py_CLEAR(o->dict);
  o->fun.~PmapFunction();
  tp->tp_free(self);
  Py_DECREF(tp);
}

int JaxPmapFunction_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(self);
#if PY_VERSION_HEX >= 0x03090000
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
#endif
  Py_VISIT(o->dict);
  Py_VISIT(o->fun.fun().ptr());
  Py_VISIT(o->fun.cache_miss().ptr());
  return 0;
}

int JaxPmapFunction_tp_clear(PyObject* self) {
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(self);
  Py_CLEAR(o->dict);
  o->fun.ClearPythonReferences();
  return 0;
}

// Implements the Python descriptor protocol so PMAP-compiled functions can be
// used as bound methods. See:
// https://docs.python.org/3/howto/descriptor.html#functions-and-methods
PyObject* JaxPmapFunction_tp_descr_get(PyObject* self, PyObject* obj,
                                       PyObject* type) {
  if (obj == nullptr || obj == Py_None) {
    Py_INCREF(self);
    return self;
  }
  return PyMethod_New(self, obj);
}

// Support d = instance.__dict__.
PyObject* JaxPmapFunction_get_dict(PyObject* self, void*) {
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(self);
  if (!o->dict) {
    o->dict = PyDict_New();
  }
  Py_XINCREF(o->dict);
  return o->dict;
}

int JaxPmapFunction_set_dict(PyObject* self, PyObject* new_dict, void*) {
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(self);
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

static PyGetSetDef JaxPmapFunction_tp_getset[] = {
    // Having a __dict__ seems necessary to allow !functool.wraps to override
    // __doc__.
    {const_cast<char*>("__dict__"), JaxPmapFunction_get_dict,
     JaxPmapFunction_set_dict, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

}  // extern "C"

nb::object MakePmapFunction(
    nb::callable fun, nb::callable cache_miss, std::vector<int> static_argnums,
    nb::callable python_shard_arg_fallback,
    std::shared_ptr<xla::PyTreeRegistry> pytree_registry) {
  nb::object obj = nb::steal<nb::object>(JaxPmapFunction_tp_new(
      reinterpret_cast<PyTypeObject*>(JaxPmapFunction_Type), nullptr, nullptr));
  JaxPmapFunctionObject* buf =
      reinterpret_cast<JaxPmapFunctionObject*>(obj.ptr());
  new (&buf->fun) PmapFunction(
      std::move(fun), std::move(cache_miss), std::move(static_argnums),
      std::move(python_shard_arg_fallback), std::move(pytree_registry));
  return obj;
}

// Version numbers for the pickled representations.
// Increment these if changing them.
const int kPmapFunctionPickleVersion = 1;

}  // namespace

void BuildPmapSubmodule(nb::module_& m) {
  nb::module_ pmap_lib = m.def_submodule("pmap_lib", "Jax C++ pmap library");
  nb::module_ pmap_lib_nb = nb::cast<nb::module_>(nb::borrow(pmap_lib.ptr()));

  nb::class_<NoSharding> no_sharding(pmap_lib_nb, "NoSharding");
  no_sharding.def(nb::init<>())
      .def("__getstate__",
           [](const NoSharding& self) { return nb::make_tuple(); })
      .def("__setstate__",
           [](NoSharding& self, nb::tuple t) { new (&self) NoSharding(); })
      .def("__repr__",
           [](const NoSharding& chuncked) { return "NoSharding()"; })
      .def("__eq__",
           [](const NoSharding& self, nb::object obj) {
             return nb::isinstance<NoSharding>(obj);
           })
      .def("__hash__", [](const NoSharding& self) {
        const size_t hash = absl::HashOf(self);
        return nb::int_(hash);
      });

  nb::class_<Chunked> chunked(pmap_lib_nb, "Chunked");
  chunked.def(nb::init<std::vector<int>>())
      .def("__getstate__",
           [](const Chunked& self) { return nb::make_tuple(self.chunks); })
      .def("__setstate__",
           [](Chunked& self, nb::tuple t) {
             new (&self) Chunked{nb::cast<std::vector<int>>(t[0])};
           })
      .def_ro("chunks", &Chunked::chunks)
      .def("__repr__",
           [](const Chunked& chuncked) {
             return absl::StrCat("Chunked(",
                                 absl::StrJoin(chuncked.chunks, ","), ")");
           })
      .def("__eq__", [](const Chunked& self, nb::object other) {
        if (!nb::isinstance<Chunked>(other)) {
          return false;
        }
        return self == nb::cast<const Chunked&>(other);
      });

  nb::class_<Unstacked> unstacked(pmap_lib_nb, "Unstacked");
  unstacked.def(nb::init<int>())
      .def("__getstate__",
           [](const Unstacked& self) { return nb::make_tuple(self.size); })
      .def("__setstate__",
           [](Unstacked& self, nb::tuple t) {
             new (&self) Unstacked{nb::cast<int>(t[0])};
           })
      .def_ro("size", &Unstacked::size)
      .def("__repr__",
           [](const Unstacked& x) {
             return absl::StrCat("Unstacked(", x.size, ")");
           })
      .def("__eq__", [](const Unstacked& self, nb::object other) {
        if (!nb::isinstance<Unstacked>(other)) {
          return false;
        }
        return self == nb::cast<const Unstacked&>(other);
      });

  nb::class_<ShardedAxis> sharded_axis(pmap_lib_nb, "ShardedAxis");
  sharded_axis.def(nb::init<int>())
      .def("__getstate__",
           [](const ShardedAxis& self) { return nb::make_tuple(self.axis); })
      .def("__setstate__",
           [](ShardedAxis& self, nb::tuple t) {
             new (&self) ShardedAxis{nb::cast<int>(t[0])};
           })
      .def_ro("axis", &ShardedAxis::axis)
      .def("__repr__",
           [](const ShardedAxis& x) {
             return absl::StrCat("ShardedAxis(axis=", x.axis, ")");
           })
      .def("__eq__", [](const ShardedAxis& self, const ShardedAxis& other) {
        return self == other;
      });

  nb::class_<Replicated> replicated(pmap_lib_nb, "Replicated");
  replicated.def(nb::init<int>())
      .def("__getstate__",
           [](const Replicated& self) { return nb::make_tuple(self.replicas); })
      .def("__setstate__",
           [](Replicated& self, nb::tuple t) {
             new (&self) Replicated{nb::cast<int>(t[0])};
           })
      .def_ro("replicas", &Replicated::replicas)
      .def("__repr__",
           [](const Replicated& x) {
             return absl::StrCat("Replicated(replicas=", x.replicas, ")");
           })
      .def("__eq__", [](const Replicated& self, const Replicated& other) {
        return self == other;
      });

  nb::class_<ShardingSpec> sharding_spec(pmap_lib_nb, "ShardingSpec");
  sharding_spec
      .def(nb::init<nb::iterable, nb::iterable>(), nb::arg("sharding"),
           nb::arg("mesh_mapping"))
      .def("__getstate__",
           [](const ShardingSpec& self) {
             auto sharding =
                 xla::SpanToNbTuple(absl::MakeConstSpan(self.GetSharding()));
             auto mesh_mapping =
                 xla::SpanToNbTuple(absl::MakeConstSpan(self.GetMeshMapping()));
             return nb::make_tuple(sharding, mesh_mapping);
           })
      .def("__setstate__",
           [](ShardingSpec& self, nb::tuple t) {
             new (&self)
                 ShardingSpec{nb::cast<std::vector<AvalDimSharding>>(t[0]),
                              nb::cast<std::vector<MeshDimAssignment>>(t[1])};
           })
      .def_prop_ro(
          "sharding",
          [](const ShardingSpec& self) {
            return xla::SpanToNbTuple(absl::MakeConstSpan(self.GetSharding()));
          })
      .def_prop_ro("mesh_mapping",
                   [](const ShardingSpec& self) {
                     return xla::SpanToNbTuple(
                         absl::MakeConstSpan(self.GetMeshMapping()));
                   })
      .def("__eq__", [](const ShardingSpec& self,
                        const ShardingSpec& other) { return self == other; })
      .def("__hash__", [](const ShardingSpec& self) {
        const size_t hash = absl::HashOf(self);
        return nb::int_(hash);
      });

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  nb::object cfun;
  {
    nb::str name = nb::str("PmapFunction");
    nb::str qualname = nb::str("PmapFunction");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called. Otherwise the GC might see a half-constructed
    // type object.
    CHECK(heap_type) << "Unable to create heap type object";
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "PmapFunction";
    type->tp_basicsize = sizeof(JaxPmapFunctionObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE |
                     Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL;
    type->tp_new = JaxPmapFunction_tp_new;
    type->tp_dealloc = JaxPmapFunction_tp_dealloc;
    type->tp_dictoffset = offsetof(JaxPmapFunctionObject, dict);
    type->tp_traverse = JaxPmapFunction_tp_traverse;
    type->tp_clear = JaxPmapFunction_tp_clear;
    type->tp_weaklistoffset = offsetof(JaxPmapFunctionObject, weakrefs);
    type->tp_getset = JaxPmapFunction_tp_getset;
    type->tp_descr_get = JaxPmapFunction_tp_descr_get;
    type->tp_call = PyVectorcall_Call;
    type->tp_vectorcall_offset = offsetof(JaxPmapFunctionObject, vectorcall);
    CHECK_EQ(PyType_Ready(type), 0);
    JaxPmapFunction_Type = reinterpret_cast<PyObject*>(type);
    cfun = nb::borrow<nb::object>(JaxPmapFunction_Type);
  }
  nb::object cfun_type = nb::borrow<nb::object>(JaxPmapFunction_Type);

  // Add PmapFunction to the xla_extension module so it can be pickled.
  m.attr("PmapFunction") = cfun_type;

  cfun.attr("__signature__") =
      xla::nb_property_readonly([](nb::handle self) -> nb::object {
        PmapFunction* fun = xla::ValueOrThrow(AsPmapFunction(self));
        return fun->PythonSignature();
      });
  // Required by `post_hook`.
  cfun.attr("_cache_miss") =
      xla::nb_property_readonly([](nb::handle self) -> nb::object {
        PmapFunction* fun = xla::ValueOrThrow(AsPmapFunction(self));
        return fun->cache_miss();
      });
  cfun.attr("__getstate__") = nb::cpp_function(
      [](const PmapFunction::object& self) {
        PmapFunction* fn = self.func();
        nb::dict pickle;
        pickle["version"] = kPmapFunctionPickleVersion;
        pickle["fun"] = fn->fun();
        pickle["cache_miss"] = fn->cache_miss();
        pickle["static_argnums"] = fn->static_argnums();
        pickle["python_shard_arg_fallback"] = fn->python_shard_arg_fallback();
        pickle["pytree_registry"] = nb::cast(fn->pytree_registry());
        return pickle;
      },
      nb::is_method());
  cfun.attr("__setstate__") = nb::cpp_function(
      [](PmapFunction::object& self, const nb::dict& pickle) {
        int version = nb::cast<int>(pickle["version"]);
        if (version != kPmapFunctionPickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid PmapFunction pickle version, got %d, expected %d. "
              "Pickling/Unpickling jitted functions using different JAX "
              "versions is not supported.",
              version, kPmapFunctionPickleVersion));
        }
        nb::callable fun = nb::cast<nb::callable>(pickle["fun"]);
        nb::callable cache_miss = nb::cast<nb::callable>(pickle["cache_miss"]);
        std::vector<int> static_argnums =
            nb::cast<std::vector<int>>(pickle["static_argnums"]);
        nb::callable python_shard_arg_fallback =
            nb::cast<nb::callable>(pickle["python_shard_arg_fallback"]);
        std::shared_ptr<xla::PyTreeRegistry> pytree_registry =
            nb::cast<std::shared_ptr<xla::PyTreeRegistry>>(
                nb::handle(pickle["pytree_registry"].ptr()));
        new (&(reinterpret_cast<JaxPmapFunctionObject*>(self.ptr())->fun))
            PmapFunction(std::move(fun), std::move(cache_miss),
                         std::move(static_argnums),
                         std::move(python_shard_arg_fallback),
                         std::move(pytree_registry));
      },
      nb::is_method());

  // This is only for testing/debugging purposes.
  cfun.attr("_cache_size") =
      xla::nb_property_readonly([](nb::handle self) -> nb::object {
        PmapFunction* fun = xla::ValueOrThrow(AsPmapFunction(self));
        return nb::cast<int>(fun->cache_size());
      });

  cfun.attr("_cache_clear") = nb::cpp_function(
      [](nb::handle self) {
        PmapFunction* fun = xla::ValueOrThrow(AsPmapFunction(self));
        fun->cache_clear();
      },
      nb::is_method());

  cfun.attr("_debug_cache_keys") = nb::cpp_function(
      [](nb::handle self) -> std::string {
        PmapFunction* fun = xla::ValueOrThrow(AsPmapFunction(self));
        return fun->DebugCacheKeys();
      },
      nb::is_method());

  pmap_lib.def(
      "pmap",
      [](nb::callable fun, nb::callable cache_miss,
         std::vector<int> static_argnums, nb::callable shard_arg_fallback,
         nb::object pytree_registry) -> nb::object {
        std::shared_ptr<xla::PyTreeRegistry> registry =
            nb::cast<std::shared_ptr<xla::PyTreeRegistry>>(
                nb::handle(pytree_registry.ptr()));
        return MakePmapFunction(
            std::move(fun), std::move(cache_miss), std::move(static_argnums),
            std::move(shard_arg_fallback), std::move(registry));
      },
      nb::arg("fun"), nb::arg("cache_miss"), nb::arg("static_argnums"),
      nb::arg("shard_arg_fallback"), nb::arg("pytree_registry"));
}

}  // namespace jax
