/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/pmap_lib.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/python/jax_jit.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/sharded_device_array.h"
#include "tensorflow/compiler/xla/python/sharding.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace jax {

namespace py = pybind11;

namespace {

// Specifies how to shard the inputs. Even though everything could be computed
// from `sharding_specs` and the argument shape, we cache derived computations
// for performance.
struct InputSpec {
  InputSpec(ShardingSpec sharding_spec, py::object indices,
            py::object array_sharding)
      : sharding_spec(std::move(sharding_spec)),
        indices(std::move(indices)),
        array_sharding(std::move(array_sharding)) {}
  ShardingSpec sharding_spec;
  py::object indices;
  py::object array_sharding;
};

// An object containing the arguments to create Array from the
// output buffers.
struct ResultSpec {
 public:
  ResultSpec(py::object aval, ShardingSpec out_spec, py::object out_indices)
      : out_aval(std::move(aval)),
        weak_type(py::cast<bool>(out_aval.attr("weak_type"))),
        out_spec(std::move(out_spec)),
        out_indices(std::move(out_indices)) {}
  py::object out_aval;
  bool weak_type;
  ShardingSpec out_spec;
  py::object out_indices;
};

// The result of `ShardArg`.
struct ShardArgResult {
  // Points to the on-device array.
  // ifrt_array->sharding().num_shards() == `num_devices`.
  tsl::RCReference<xla::ifrt::Array> ifrt_array;
  // The Python argument will be always be copied to `owning_sda`.
  py::object owning_sda;
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
xla::StatusOr<ShardArgResult> ShardArg(
    py::handle arg, absl::Span<xla::PjRtDevice* const> devices,
    const InputSpec& input_spec, py::handle py_devices,
    const py::function& python_fallback) {
  if (arg.get_type() == xla::PyArray::type()) {
    auto py_array = py::reinterpret_borrow<xla::PyArray>(arg);
    if (py_array.fastpath_enabled()) {
      if (py_array.sharding().get_type() ==
          input_spec.array_sharding.get_type()) {
        auto* pmap_sharding = py_array.sharding().cast<jax::PmapSharding*>();
        auto* cached_pmap_sharding =
            input_spec.array_sharding.cast<jax::PmapSharding*>();

        if (pmap_sharding->sharding_spec() ==
            cached_pmap_sharding->sharding_spec()) {
          ShardArgResult result;
          result.owning_sda = py::reinterpret_borrow<py::object>(arg);
          result.ifrt_array = tsl::FormRef(py_array.ifrt_array());
          if (result.ifrt_array == nullptr) {
            return xla::InvalidArgument("Array has been deleted.");
          }
          if (result.ifrt_array->sharding().devices().devices() != devices) {
            xla::ifrt::DeviceList::Devices ifrt_devices;
            ifrt_devices.reserve(devices.size());
            ifrt_devices.insert(ifrt_devices.end(), devices.begin(),
                                devices.end());
            auto sharding = xla::ifrt::OpaqueSharding::Create(
                xla::ifrt::DeviceList(std::move(ifrt_devices)));
            TF_ASSIGN_OR_RETURN(
                auto copied_ifrt_array,
                result.ifrt_array->Reshard(
                    std::move(sharding),
                    xla::ifrt::ArrayCopySemantics::kReuseInput));
            result.ifrt_array = std::move(copied_ifrt_array);
          }
          return result;
        }
      }
    }
  }

  static auto ndarray_type = py::module::import("numpy").attr("ndarray").ptr();
  auto ndarray = py::array::ensure(arg);
  if (ndarray && py::type::of(arg) == ndarray_type &&
      xla::DtypeToPrimitiveType(ndarray.dtype()).status().ok()) {
    tsl::profiler::TraceMe traceme("ndarray pmap ShardArg");
    py::list indices = input_spec.indices;
    py::list py_devices_list = py::cast<py::list>(py_devices);
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

    py::list owning_pylist(n_devices);
    ShardArgResult result;
    result.owning_sda = owning_pylist;
    const bool jax_enable_x64 = GetEnableX64();

    xla::DevicePutOptions options;
    options.squash_64bit_types = !jax_enable_x64;
    options.allow_zero_copy = true;
    for (size_t i = 0; i < n_devices; ++i) {
      auto to_device =
          py::cast<xla::ClientAndPtr<xla::PjRtDevice>>(py_devices_list[i]);

      TF_ASSIGN_OR_RETURN(
          xla::DevicePutResult on_device,
          DevicePut(arg[indices[i]], to_device.client->ifrt_client(),
                    to_device.contents, options));

      per_device_arrays.push_back(std::move(on_device.ifrt_array));
      devices.push_back(per_device_arrays.back()->sharding().devices().front());
      shapes.push_back(per_device_arrays.back()->shape());
      if (on_device.owning_pybuffer) {
        owning_pylist.append(on_device.owning_pybuffer);
      }
    }

    TF_ASSIGN_OR_RETURN(
        result.ifrt_array,
        per_device_arrays.front()
            ->client()
            ->AssembleArrayFromSingleDeviceArrays(
                // TODO(hyeontaek): The logical shape here is inaccurate. We
                // may want to avoid creating a new Array or specialize Array
                // to disallow access to the logical shape.
                per_device_arrays.front()->shape(),
                xla::ifrt::OpaqueSharding::Create(
                    xla::ifrt::DeviceList(std::move(devices)),
                    xla::ifrt::OpaqueSharding::MakeDisassembleFuncFromShapes(
                        std::move(shapes))),
                absl::MakeSpan(per_device_arrays),
                xla::ifrt::ArrayCopySemantics::kReuseInput));
    return result;
  }
  tsl::profiler::TraceMe traceme("pmap_lib_shard_arg_python_fallback");
  auto py_array_or_bufs = python_fallback(arg, py_devices, input_spec.indices,
                                          input_spec.array_sharding);

  if (py_array_or_bufs.get_type() == xla::PyArray::type()) {
    auto py_array = py::cast<xla::PyArray>(py_array_or_bufs);
    ShardArgResult result;
    result.owning_sda = py_array_or_bufs;
    result.ifrt_array = tsl::FormRef(py_array.ifrt_array());
    return result;
  }

  // This fallback is better than nothing, but ideally we should be able to
  // convert the argument in C++. At least, we can call the C++ DevicePut from
  // Python.
  auto per_device_pybuffers = py::cast<py::list>(py_array_or_bufs);
  ShardArgResult result;
  result.owning_sda = py::reinterpret_borrow<py::object>(per_device_pybuffers);
  if (!per_device_pybuffers.empty()) {
    std::vector<tsl::RCReference<xla::ifrt::Array>> per_device_arrays;
    per_device_arrays.reserve(per_device_pybuffers.size());
    xla::ifrt::DeviceList::Devices devices;
    devices.reserve(per_device_pybuffers.size());
    // TODO(hyeontaek): The created array will never be disassembled. We should
    // omit collecting shapes and make the OpaqueSharding non-disassemblable?
    std::vector<xla::ifrt::Shape> shapes;
    shapes.reserve(per_device_pybuffers.size());

    // The JAX Python shard_arg function is expected to return JAX PyBuffer
    // objects. If executing a JAX extension, it should have fallbacked to
    // Python well before this point.
    TF_RET_CHECK(xla::PyBuffer::IsPyBuffer(per_device_pybuffers[0]));
    for (py::handle per_device_pybuffer : per_device_pybuffers) {
      auto b = xla::PyBuffer::AsPyBuffer(per_device_pybuffer).value();
      per_device_arrays.push_back(tsl::FormRef(b->ifrt_array()));
      devices.push_back(per_device_arrays.back()->sharding().devices().front());
      shapes.push_back(per_device_arrays.back()->shape());
    }
    TF_ASSIGN_OR_RETURN(
        result.ifrt_array,
        per_device_arrays.front()
            ->client()
            ->AssembleArrayFromSingleDeviceArrays(
                // TODO(hyeontaek): The logical shape here is inaccurate. We
                // may want to avoid creating a new Array or specialize Array
                // to disallow access to the logical shape.
                per_device_arrays.front()->shape(),
                xla::ifrt::OpaqueSharding::Create(
                    xla::ifrt::DeviceList(std::move(devices)),
                    xla::ifrt::OpaqueSharding::MakeDisassembleFuncFromShapes(
                        std::move(shapes))),
                absl::MakeSpan(per_device_arrays),
                xla::ifrt::ArrayCopySemantics::kReuseInput));
  }
  return result;
}

struct PmapCacheEntry {
  std::shared_ptr<xla::PyLoadedExecutable> executable;
  // The value `backend.local_devices()`.
  py::object py_devices;  // To pass back to Python.
  std::vector<xla::PjRtDevice*> devices;
  std::vector<InputSpec> input_specs;
  xla::PyTreeDef out_pytree_def;
  // Objects necessary to build the out Array objects.
  std::vector<ResultSpec> out_result_specs;

  std::vector<py::object> out_array_shardings;
  std::vector<py::dtype> out_dtypes;
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
  PmapFunction(py::function fun, py::function cache_miss,
               std::vector<int> static_argnums,
               py::function python_shard_arg_fallback)
      : fun_(std::move(fun)),
        cache_miss_(std::move(cache_miss)),
        static_argnums_(std::move(static_argnums)),
        python_shard_arg_fallback_(std::move(python_shard_arg_fallback)) {
    std::sort(static_argnums_.begin(), static_argnums_.end());

    function_name_ = py::str(py::getattr(fun_, "__name__", fun));
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
  xla::StatusOr<py::object> Call(py::handle callable, PyObject* const* args,
                                 size_t nargs, PyObject* kwnames);

  py::object PythonSignature() {
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(fun_);
  }

  int cache_size() const { return executables_.size(); }
  const py::function& fun() const { return fun_; }
  const py::function& cache_miss() const { return cache_miss_; }
  const std::string& function_name() const { return function_name_; }
  const py::function& python_shard_arg_fallback() const {
    return python_shard_arg_fallback_;
  }
  const std::vector<int>& static_argnums() const { return static_argnums_; }

  // pybind11::object typed subclass for PmapFunction objects.
  class pyobject : public py::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    py::object, PmapFunction::IsPmapFunction);
    pyobject() = default;
    PmapFunction* func() const {
      return PmapFunction::AsPmapFunctionUnchecked(*this);
    }
  };
  // Alias as ::object; outside the scope above we won't confuse pybind11's
  // macros.
  using object = pyobject;

  // Returns true if `h` is a PmapFunction.
  static bool IsPmapFunction(py::handle handle);
  // Converts `handle` to a PmapFunction*. Does not do any checking.
  static PmapFunction* AsPmapFunctionUnchecked(py::handle handle);

  // Helper function used by the tp_clear GC method.
  void ClearPythonReferences() {
    py::function fun, cache_miss, python_shard_arg_fallback;
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
  xla::Status UpdateArgsSignature(ParsedArgumentsAsBuffers& arguments) {
    arguments.signature.function_name = function_name_;

    // Get dynamic argument signatures.
    JitState& global_state = jax::GlobalJitState();
    JitState& tls = jax::ThreadLocalJitState();
    const bool jax_enable_x64 = GetEnableX64();
    arguments.signature.jax_enable_x64 = jax_enable_x64;
    for (py::handle arg : arguments.flat_dynamic_args) {
      auto signature_or_error = xla::PyArgSignatureOfValue(arg, jax_enable_x64);
      if (!signature_or_error.ok()) {
        VLOG(2) << "PyArgSignatureOfValue failed: "
                << signature_or_error.status();
        return signature_or_error.status();
      }
      arguments.signature.dynamic_arg_signatures.push_back(
          std::move(signature_or_error).value());
    }
    try {
      py::object pxla_module = py::module::import("jax").attr("config");
      py::object sda = py::getattr(pxla_module, "_trace_context", py::none());
      if (!sda.is_none()) {
        arguments.signature.thread_local_extra_jit_context = sda();
      }
    } catch (const py::error_already_set& e) {
      // Ignore; jax may not be present.
    }
    if (!arguments.signature.thread_local_extra_jit_context.has_value()) {
      arguments.signature.thread_local_extra_jit_context =
          tls.extra_jit_context;
      arguments.signature.global_extra_jit_context =
          global_state.extra_jit_context;
    }
    return xla::Status();
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
                          const CallSignature& signature,
                          const py::tuple& out_and_fastpath_data);

  bool always_fallback_to_python_ = false;

  py::function fun_;  // The Python function to pmap.
  std::string function_name_;
  // See JAX _cpp_pmap in api.py for documentation.
  py::function cache_miss_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyLoadedExecutable. In sorted order.
  std::vector<int> static_argnums_;
  // We need a `unique_ptr` here to ensure value pointer stability.
  absl::flat_hash_map<CallSignature, std::unique_ptr<PmapCacheEntry>>
      executables_;

  // The fallback function to use with `ShardArgs`.
  // TODO(jblespiau): Add support for more types from C++.
  py::function python_shard_arg_fallback_;
};

void PmapFunction::PopulateCacheEntry(PmapCacheEntry& cache_entry,
                                      const CallSignature& signature,
                                      const py::tuple& out_and_fastpath_data) {
  CHECK_EQ(out_and_fastpath_data.size(), 2);
  if (out_and_fastpath_data[1].is_none()) {
    cache_entry.fall_back_to_python = true;
    return;
  }

  py::tuple pmap_data = py::cast<py::tuple>(out_and_fastpath_data[1]);
  if (py::cast<int>(pmap_data.attr("version")) != 1) {
    throw xla::XlaRuntimeError(absl::StrCat(
        "The versions of jaxlib and Jax are incompatible (pmap cpp version 1 "
        "expected, but got ",
        py::cast<int>(pmap_data.attr("version")),
        "Upgrade jaxlib and jax. Provided data was:",
        py::cast<std::string>(py::str(py::repr(pmap_data)))));
  }
  // See api.py::_PmapFastpathData in the JAX code base for the expected
  // namedtuple.
  std::shared_ptr<xla::PyLoadedExecutable> executable;
  try {
    executable = py::cast<std::shared_ptr<xla::PyLoadedExecutable>>(
        pmap_data.attr("xla_executable"));
  } catch (const py::cast_error& e) {
    // Backends that don't implement the C++ PjRt APIs
    always_fallback_to_python_ = true;
    return;
  }
  cache_entry.executable = std::move(executable);
  const std::vector<xla::ClientAndPtr<xla::PjRtDevice>>& client_and_devices =
      cache_entry.executable->AddressableDevices();
  cache_entry.devices.reserve(client_and_devices.size());
  for (auto& client_and_device : client_and_devices) {
    cache_entry.devices.push_back(client_and_device.get());
  }

  // Inputs shard args details.
  auto input_sharding_specs = py::cast<std::vector<ShardingSpec>>(
      pmap_data.attr("input_sharding_specs"));
  py::list input_indices = pmap_data.attr("input_indices");

  cache_entry.py_devices = pmap_data.attr("input_devices");
  auto input_devices =
      py::cast<std::vector<xla::PjRtDevice*>>(pmap_data.attr("input_devices"));

  py::list input_array_shardings = pmap_data.attr("input_array_shardings");

  CHECK_EQ(input_sharding_specs.size(), input_indices.size());
  cache_entry.input_specs.reserve(input_sharding_specs.size());

  if (input_array_shardings.empty()) {
    for (int i = 0; i < input_sharding_specs.size(); ++i) {
      cache_entry.input_specs.emplace_back(input_sharding_specs[i],
                                           input_indices[i],
                                           /*array_sharding=*/py::object());
    }
  } else {
    DCHECK_EQ(input_array_shardings.size(), input_sharding_specs.size());
    for (int i = 0; i < input_sharding_specs.size(); ++i) {
      cache_entry.input_specs.emplace_back(
          input_sharding_specs[i], input_indices[i], input_array_shardings[i]);
    }
  }

  // Outputs specs.
  auto out_tree = py::cast<xla::PyTreeDef>(pmap_data.attr("out_pytree_def"));
  cache_entry.out_pytree_def = std::move(out_tree);
  py::list out_avals = pmap_data.attr("out_avals");
  py::list out_indices = pmap_data.attr("out_indices");
  auto out_sharding_specs =
      py::cast<std::vector<ShardingSpec>>(pmap_data.attr("out_sharding_specs"));
  CHECK_EQ(out_avals.size(), out_indices.size());
  CHECK_EQ(out_indices.size(), out_sharding_specs.size());

  cache_entry.out_result_specs.reserve(out_avals.size());
  cache_entry.out_dtypes.reserve(out_avals.size());
  cache_entry.out_shapes.reserve(out_avals.size());

  for (int i = 0; i < out_avals.size(); ++i) {
    cache_entry.out_dtypes.push_back(out_avals[i].attr("dtype"));
    cache_entry.out_shapes.push_back(
        py::cast<std::vector<int64_t>>(out_avals[i].attr("shape")));
    cache_entry.out_result_specs.emplace_back(
        out_avals[i], std::move(out_sharding_specs[i]), out_indices[i]);
  }

  py::list out_array_shardings = pmap_data.attr("out_array_shardings");

  DCHECK(out_array_shardings.empty() ||
         out_avals.size() == out_array_shardings.size());

  cache_entry.out_array_shardings.reserve(out_array_shardings.size());
  for (py::handle out_array_sharding : out_array_shardings) {
    cache_entry.out_array_shardings.push_back(
        py::reinterpret_borrow<py::object>(out_array_sharding));
  }

  py::list out_committed = pmap_data.attr("out_committed");

  DCHECK(out_committed.empty() || out_avals.size() == out_committed.size());

  cache_entry.out_committed.reserve(out_committed.size());
  for (py::handle c : out_committed) {
    cache_entry.out_committed.push_back(py::cast<bool>(c));
  }
}

xla::StatusOr<py::object> PmapFunction::Call(py::handle callable,
                                             PyObject* const* args,
                                             size_t nargs, PyObject* kwnames) {
  xla::GlobalPyRefManager()->MaybeCollectGarbage();

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

  size_t num_positional_args = PyVectorcall_NARGS(nargs);
  size_t num_keyword_args = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  absl::Span<PyObject* const> positional_args(args, num_positional_args);
  absl::Span<PyObject* const> keyword_args(args + num_positional_args,
                                           num_keyword_args);
  ParsedArgumentsAsBuffers arguments;
  xla::Status status =
      ParseArguments(positional_args, keyword_args, kwnames, static_argnums_,
                     /*static_argnames=*/{}, arguments);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  status = UpdateArgsSignature(arguments);
  if (!status.ok()) {
    return fallback_to_cache_miss();
  }

  // Retrieve/Maybe add the executable to the cache.
  absl::flat_hash_map<CallSignature, std::unique_ptr<PmapCacheEntry>>::iterator
      it;
  bool inserted;
  std::tie(it, inserted) = executables_.try_emplace(
      arguments.signature, std::unique_ptr<PmapCacheEntry>());
  if (inserted) {
    it->second = std::make_unique<PmapCacheEntry>();
  }
  PmapCacheEntry& cache_entry = *(it->second);

  if (!cache_entry.compilation_complete.HasBeenNotified()) {
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
        PopulateCacheEntry(cache_entry, arguments.signature, out_tuple);
      } catch (const std::exception& e) {
        cache_entry.fall_back_to_python = true;
        cache_entry.compilation_complete.Notify();
        throw;
      }
      cache_entry.compilation_complete.Notify();

      // We have already computed the result in the miss path so we can return
      // it. We are even *required* to do so if there are donated arguments,
      // because any donated buffers will now be invalid.
      return py::object(out_tuple[0]);
    } else {
      // Release the GIL while we wait, making sure the compile thread can
      // lock it.
      py::gil_scoped_release release;
      cache_entry.compilation_complete.WaitForNotification();
    }
  }
  if (cache_entry.fall_back_to_python) {
    return fallback_to_cache_miss();
  }

  // 1. Parse arguments.
  std::vector<xla::PjRtDevice*>& input_devices = cache_entry.devices;
  std::vector<InputSpec>& input_specs = cache_entry.input_specs;
  const int num_args = arguments.flat_dynamic_args.size();

  // We need [num_args] for the `Execute` call below.
  std::vector<tsl::RCReference<xla::ifrt::Array>> num_args_arrays(num_args);
  for (int i = 0; i < num_args; ++i) {
    TF_ASSIGN_OR_RETURN(
        ShardArgResult sharded_arg,
        ShardArg(arguments.flat_dynamic_args[i], input_devices, input_specs[i],
                 cache_entry.py_devices, python_shard_arg_fallback_));

    num_args_arrays[i] = std::move(sharded_arg.ifrt_array);
    if (sharded_arg.owning_sda) {
      arguments.keep_alive_objects.push_back(std::move(sharded_arg.owning_sda));
    }
  }

  // A vector of [num_outputs].
  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  {
    py::gil_scoped_release gil_release;
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
  std::shared_ptr<xla::PyClient> client = cache_entry.executable->client();

  // Convert the PjRtBuffer objects to PyBuffer, and invert the order from
  // [num_devices, num_args] to [num_args, num_devices].
  const int num_outputs = output_arrays.size();
  std::vector<py::object> flat_sharded_device_arrays;
  flat_sharded_device_arrays.reserve(num_outputs);

  const auto& output_specs = cache_entry.out_result_specs;

  TF_RET_CHECK(!cache_entry.out_array_shardings.empty());
  for (int i = 0; i < num_outputs; ++i) {
    const ResultSpec& result_spec = output_specs[i];
    xla::PyArray py_array(
        result_spec.out_aval, result_spec.weak_type, cache_entry.out_dtypes[i],
        cache_entry.out_shapes[i], cache_entry.out_array_shardings[i], client,
        traceback, std::move(output_arrays[i]), cache_entry.out_committed[i]);

    flat_sharded_device_arrays.push_back(std::move(py_array));
  }

  py::object out =
      cache_entry.out_pytree_def.Unflatten(flat_sharded_device_arrays);

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

struct JaxPmapFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
  vectorcallfunc vectorcall;
  PmapFunction fun;
};

PyObject* JaxPmapFunction_Type = nullptr;

bool PmapFunction::IsPmapFunction(py::handle handle) {
  return handle.get_type() == JaxPmapFunction_Type;
}

PmapFunction* PmapFunction::AsPmapFunctionUnchecked(py::handle handle) {
  return &(reinterpret_cast<JaxPmapFunctionObject*>(handle.ptr())->fun);
}

xla::StatusOr<PmapFunction*> AsPmapFunction(py::handle handle) {
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

void InitializePmapFunction(JaxPmapFunctionObject* cfun, py::function fun,
                            py::function cache_miss,
                            std::vector<int> static_argnums,
                            py::function python_shard_arg_fallback) {
  new (&cfun->fun) PmapFunction(std::move(fun), std::move(cache_miss),
                                std::move(static_argnums),
                                std::move(python_shard_arg_fallback));
}

}  // extern "C"

py::object MakePmapFunction(py::function fun, py::function cache_miss,
                            std::vector<int> static_argnums,
                            py::function python_shard_arg_fallback) {
  py::object obj = py::reinterpret_steal<py::object>(JaxPmapFunction_tp_new(
      reinterpret_cast<PyTypeObject*>(JaxPmapFunction_Type), nullptr, nullptr));
  JaxPmapFunctionObject* buf =
      reinterpret_cast<JaxPmapFunctionObject*>(obj.ptr());
  InitializePmapFunction(buf, std::move(fun), std::move(cache_miss),
                         std::move(static_argnums),
                         std::move(python_shard_arg_fallback));
  return obj;
}

// Version numbers for the pickled representations.
// Increment these if changing them.
const int kPmapFunctionPickleVersion = 1;

}  // namespace

void BuildPmapSubmodule(py::module& m) {
  py::module pmap_lib = m.def_submodule("pmap_lib", "Jax C++ pmap library");

  py::class_<NoSharding> no_sharding(pmap_lib, "NoSharding");
  no_sharding.def(py::init<>())
      .def(py::pickle([](const NoSharding& self) { return py::make_tuple(); },
                      [](py::tuple t) { return NoSharding{}; }))
      .def("__repr__",
           [](const NoSharding& chuncked) { return "NoSharding()"; })
      .def("__eq__",
           [](const NoSharding& self, py::object obj) {
             return py::isinstance<NoSharding>(obj);
           })
      .def("__hash__", [](const NoSharding& self) {
        const size_t hash = absl::HashOf(self);
        return py::int_(hash);
      });

  py::class_<Chunked> chunked(pmap_lib, "Chunked");
  chunked.def(py::init<std::vector<int>>())
      .def(py::pickle(
          [](const Chunked& self) { return py::make_tuple(self.chunks); },
          [](py::tuple t) { return Chunked{t[0].cast<std::vector<int>>()}; }))
      .def_readonly("chunks", &Chunked::chunks)
      .def("__repr__",
           [](const Chunked& chuncked) {
             return absl::StrCat("Chunked(",
                                 absl::StrJoin(chuncked.chunks, ","), ")");
           })
      .def("__eq__", [](const Chunked& self, py::object other) {
        if (!py::isinstance<Chunked>(other)) {
          return false;
        }
        return self == py::cast<const Chunked&>(other);
      });

  py::class_<Unstacked> unstacked(pmap_lib, "Unstacked");
  unstacked.def(py::init<int>())
      .def(py::pickle(
          [](const Unstacked& self) { return py::make_tuple(self.size); },
          [](py::tuple t) { return Unstacked{t[0].cast<int>()}; }))
      .def_readonly("size", &Unstacked::size)
      .def("__repr__",
           [](const Unstacked& x) {
             return absl::StrCat("Unstacked(", x.size, ")");
           })
      .def("__eq__", [](const Unstacked& self, py::object other) {
        if (!py::isinstance<Unstacked>(other)) {
          return false;
        }
        return self == py::cast<const Unstacked&>(other);
      });

  py::class_<ShardedAxis> sharded_axis(pmap_lib, "ShardedAxis");
  sharded_axis.def(py::init<int>())
      .def(py::pickle(
          [](const ShardedAxis& self) { return py::make_tuple(self.axis); },
          [](py::tuple t) { return ShardedAxis{t[0].cast<int>()}; }))
      .def_readonly("axis", &ShardedAxis::axis)
      .def("__repr__",
           [](const ShardedAxis& x) {
             return absl::StrCat("ShardedAxis(axis=", x.axis, ")");
           })
      .def("__eq__", [](const ShardedAxis& self, const ShardedAxis& other) {
        return self == other;
      });

  py::class_<Replicated> replicated(pmap_lib, "Replicated");
  replicated.def(py::init<int>())
      .def(py::pickle(
          [](const Replicated& self) { return py::make_tuple(self.replicas); },
          [](py::tuple t) { return Replicated{t[0].cast<int>()}; }))
      .def_readonly("replicas", &Replicated::replicas)
      .def("__repr__",
           [](const Replicated& x) {
             return absl::StrCat("Replicated(replicas=", x.replicas, ")");
           })
      .def("__eq__", [](const Replicated& self, const Replicated& other) {
        return self == other;
      });

  py::class_<ShardingSpec> sharding_spec(pmap_lib, "ShardingSpec");
  sharding_spec
      .def(py::init<py::iterable, py::iterable>(), py::arg("sharding"),
           py::arg("mesh_mapping"))
      .def(py::pickle(
          [](const ShardingSpec& self) {
            auto sharding =
                xla::SpanToTuple(absl::MakeConstSpan(self.GetSharding()));
            auto mesh_mapping =
                xla::SpanToTuple(absl::MakeConstSpan(self.GetMeshMapping()));
            return py::make_tuple(sharding, mesh_mapping);
          },
          [](py::tuple t) {
            return ShardingSpec{t[0].cast<std::vector<AvalDimSharding>>(),
                                t[1].cast<std::vector<MeshDimAssignment>>()};
          }))
      .def_property_readonly(
          "sharding",
          [](const ShardingSpec& self) {
            return xla::SpanToTuple(absl::MakeConstSpan(self.GetSharding()));
          })
      .def_property_readonly(
          "mesh_mapping",
          [](const ShardingSpec& self) {
            return xla::SpanToTuple(absl::MakeConstSpan(self.GetMeshMapping()));
          })
      .def("__eq__", [](const ShardingSpec& self,
                        const ShardingSpec& other) { return self == other; })
      .def("__hash__", [](const ShardingSpec& self) {
        const size_t hash = absl::HashOf(self);
        return py::int_(hash);
      });

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  py::object cfun;
  {
    py::str name = py::str("PmapFunction");
    py::str qualname = py::str("PmapFunction");
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
                     Py_TPFLAGS_HAVE_GC | JAX_TPFLAGS_HAVE_VECTORCALL;
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
    cfun = py::reinterpret_borrow<py::object>(JaxPmapFunction_Type);
  }
  py::object cfun_type =
      py::reinterpret_borrow<py::object>(JaxPmapFunction_Type);

  // Add PmapFunction to the xla_extension module so it can be pickled.
  m.attr("PmapFunction") = cfun_type;

  cfun.attr("__signature__") =
      property_readonly([](py::handle self) -> xla::StatusOr<py::object> {
        TF_ASSIGN_OR_RETURN(PmapFunction * fun, AsPmapFunction(self));
        return fun->PythonSignature();
      });
  // Required by `post_hook`.
  cfun.attr("_cache_miss") =
      property_readonly([](py::handle self) -> xla::StatusOr<py::object> {
        TF_ASSIGN_OR_RETURN(PmapFunction * fun, AsPmapFunction(self));
        return fun->cache_miss();
      });
  cfun.attr("__getstate__") = py::cpp_function(
      [](const PmapFunction::object& self) {
        PmapFunction* fn = self.func();
        py::dict pickle;
        pickle["version"] = kPmapFunctionPickleVersion;
        pickle["fun"] = fn->fun();
        pickle["cache_miss"] = fn->cache_miss();
        pickle["static_argnums"] = fn->static_argnums();
        pickle["python_shard_arg_fallback"] = fn->python_shard_arg_fallback();
        return pickle;
      },
      py::is_method(cfun_type));
  cfun.attr("__setstate__") = py::cpp_function(
      [](PmapFunction::object& self, const py::dict& pickle) {
        int version = py::cast<int>(pickle["version"]);
        if (version != kPmapFunctionPickleVersion) {
          throw std::invalid_argument(absl::StrFormat(
              "Invalid PmapFunction pickle version, got %d, expected %d. "
              "Pickling/Unpickling jitted functions using different JAX "
              "versions is not supported.",
              version, kPmapFunctionPickleVersion));
        }
        py::function fun = py::cast<py::function>(pickle["fun"]);
        py::function cache_miss = py::cast<py::function>(pickle["cache_miss"]);
        std::vector<int> static_argnums =
            py::cast<std::vector<int>>(pickle["static_argnums"]);
        py::function python_shard_arg_fallback =
            py::cast<py::function>(pickle["python_shard_arg_fallback"]);

        InitializePmapFunction(
            reinterpret_cast<JaxPmapFunctionObject*>(self.ptr()),
            std::move(fun), std::move(cache_miss), std::move(static_argnums),
            std::move(python_shard_arg_fallback));
      },
      py::is_method(cfun_type));

  // This is only for testing/debugging purposes.
  cfun.attr("_cache_size") =
      property_readonly([](py::handle self) -> xla::StatusOr<py::object> {
        TF_ASSIGN_OR_RETURN(PmapFunction * fun, AsPmapFunction(self));
        return py::cast<int>(fun->cache_size());
      });

  cfun.attr("_debug_cache_keys") = py::cpp_function(
      [](py::handle self) -> xla::StatusOr<std::string> {
        TF_ASSIGN_OR_RETURN(PmapFunction * fun, AsPmapFunction(self));
        return fun->DebugCacheKeys();
      },
      py::is_method(cfun_type));

  pmap_lib.def("pmap",
               [](py::function fun, py::function cache_miss,
                  std::vector<int> static_argnums,
                  py::function python_shard_arg_fallback) -> py::object {
                 return MakePmapFunction(std::move(fun), std::move(cache_miss),
                                         std::move(static_argnums),
                                         std::move(python_shard_arg_fallback));
               });
}

}  // namespace jax
