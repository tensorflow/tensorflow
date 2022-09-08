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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/jax_jit.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/sharded_device_array.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/tsl/platform/logging.h"

namespace jax {

namespace py = pybind11;

namespace {

// Specifies how to shard the inputs. Even though everything could be computed
// from `sharding_specs` and the argument shape, we cache derived computations
// for performance.
struct InputSpec {
  InputSpec(ShardingSpec sharding_spec, py::object indices)
      : sharding_spec(std::move(sharding_spec)), indices(std::move(indices)) {}
  ShardingSpec sharding_spec;
  py::object indices;
};

// An object containing the arguments to create ShardedDeviceArray from the
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
  // Points to the on-device buffers. Not owned.
  // Size `num_devices`.
  std::vector<xla::PjRtBuffer*> per_device_buffers;

  // The Python argument will be always be copied to `owning_sda`.
  // If we need to copy data to a device, the newly created buffers will be
  // added to `owned_buffers`.
  std::vector<std::unique_ptr<xla::PjRtBuffer>> owned_buffers;
  py::object owning_sda;
};

// Shars a single argument over devices.
//
// We currently only support fully in C++, C++ ShardedDeviceArray. For all
// other usages, we call a Python function returning C++ ShardedDeviceArray
// that will be casted back to the C++ objects.
//
// This function is not usable for JAX extensions that do not comply with the
// PjRt interfaces.
//
// Arguments:
// `arg`: The object to shard across `devices`. If a `ShardedDeviceArray`,
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
  if (ShardedDeviceArray::IsShardedDeviceArray(arg)) {
    ShardedDeviceArray* sda =
        ShardedDeviceArray::AsShardedDeviceArrayUnchecked(arg);
    const ShardingSpec& sharding_spec = input_spec.sharding_spec;
    if (sharding_spec == sda->GetShardingSpec()) {
      const int num_devices = devices.size();
      TF_ASSIGN_OR_RETURN(absl::Span<xla::PjRtBuffer* const> sda_buffers,
                          sda->GetPjRtBuffers());
      CHECK_EQ(sda_buffers.size(), num_devices);

      ShardArgResult result;
      result.owning_sda = py::reinterpret_borrow<py::object>(arg);
      std::vector<xla::PjRtBuffer*>& per_device_buffers =
          result.per_device_buffers;
      per_device_buffers.reserve(num_devices);

      for (int i = 0; i < num_devices; ++i) {
        xla::PjRtBuffer* current_buffer = sda_buffers[i];
        if (devices[i] == current_buffer->device()) {  // Pointer equality.
          per_device_buffers.push_back(current_buffer);
        } else {
          TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> out,
                              current_buffer->CopyToDevice(devices[i]));
          per_device_buffers.push_back(out.get());
          result.owned_buffers.push_back(std::move(out));
        }
      }
      return result;
    }
  }

  // This fallback is better than nothing, but ideally we should be able to
  // convert the argument in C++. At least, we can call the C++ DevicePut from
  // Python.
  auto per_device_pybuffers =
      py::cast<py::list>(python_fallback(arg, py_devices, input_spec.indices));
  ShardArgResult result;
  result.owning_sda = py::reinterpret_borrow<py::object>(per_device_pybuffers);
  std::vector<xla::PjRtBuffer*>& per_device_buffers = result.per_device_buffers;
  if (!per_device_pybuffers.empty()) {
    per_device_buffers.reserve(per_device_pybuffers.size());

    // The JAX Python shard_arg function is expected to return JAX PyBuffer
    // objects. If executing a JAX extension, it should have fallbacked to
    // Python well before this point.
    TF_RET_CHECK(xla::PyBuffer::IsPyBuffer(per_device_pybuffers[0]));
    for (py::handle per_device_pybuffer : per_device_pybuffers) {
      xla::PjRtBuffer* buf =
          xla::PyBuffer::AsPyBuffer(per_device_pybuffer).value()->buffer();
      per_device_buffers.push_back(buf);
    }
  }
  return result;
}

struct PmapCacheEntry {
  std::shared_ptr<xla::PyExecutable> executable;
  // The value `backend.local_devices()`.
  py::object py_devices;  // To pass back to Python.
  std::vector<xla::PjRtDevice*> devices;
  std::vector<InputSpec> input_specs;
  xla::PyTreeDef out_pytree_def;
  // Objects necessary to build the out ShardedDeviceArray objects.
  std::vector<ResultSpec> out_result_specs;

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
// the correct underlying `PyExecutable`. This class is thread-safe.
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
  // (d) construct `ShardedDeviceArray` objects from the outputs
  // (e) reconstruct the `PyTree`.
  xla::StatusOr<py::object> Call(py::args args, py::kwargs kwargs);

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

  py::handle AsPyHandle();
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
  xla::Status UpdateArgsSignature(const py::args& args,
                                  const py::kwargs& kwargs,
                                  ParsedArgumentsAsBuffers& arguments) {
    arguments.signature.function_name = function_name_;

    // Get dynamic argument signatures.
    JitState& global_state = jax::GetGlobalState();
    JitState& tls = jax::GetLocalState();
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
  // passed to the underlying PyExecutable. In sorted order.
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
  std::shared_ptr<xla::PyExecutable> executable;
  try {
    executable = py::cast<std::shared_ptr<xla::PyExecutable>>(
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
  CHECK_EQ(input_sharding_specs.size(), input_indices.size());
  cache_entry.input_specs.reserve(input_sharding_specs.size());
  for (int i = 0; i < input_sharding_specs.size(); ++i) {
    cache_entry.input_specs.emplace_back(input_sharding_specs[i],
                                         input_indices[i]);
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
  for (int i = 0; i < out_avals.size(); ++i) {
    cache_entry.out_result_specs.emplace_back(
        out_avals[i], std::move(out_sharding_specs[i]), out_indices[i]);
  }
}

xla::StatusOr<py::object> PmapFunction::Call(py::args args, py::kwargs kwargs) {
  if (always_fallback_to_python_) {
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }

  ParsedArgumentsAsBuffers arguments;
  xla::Status status = ParseArguments(args, kwargs, static_argnums_,
                                      /*static_argnames=*/{}, arguments);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }

  status = UpdateArgsSignature(args, kwargs, arguments);
  if (!status.ok()) {
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }

  // Retrieve/Maybe add the executable to the cache.
  absl::flat_hash_map<CallSignature, std::unique_ptr<PmapCacheEntry>>::iterator
      it;
  bool inserted;
  std::tie(it, inserted) = executables_.try_emplace(
      arguments.signature, std::make_unique<PmapCacheEntry>());
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
        out_and_fastpath_data = cache_miss_(*args, **kwargs);
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
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }

  // 1. Parse arguments.
  std::vector<xla::PjRtDevice*>& input_devices = cache_entry.devices;
  const int num_computations =
      cache_entry.executable->AddressableDevices().size();
  std::vector<InputSpec>& input_specs = cache_entry.input_specs;
  const int num_args = arguments.flat_dynamic_args.size();

  // We need [num_computation, num_args] for the `Execute` call bellow,
  std::vector<std::vector<xla::PjRtBuffer*>> num_computation_num_args_buffers(
      num_computations);
  for (int computation = 0; computation < num_computations; ++computation) {
    num_computation_num_args_buffers[computation].resize(num_args);
  }
  for (int i = 0; i < num_args; ++i) {
    TF_ASSIGN_OR_RETURN(
        ShardArgResult sharded_arg,
        ShardArg(arguments.flat_dynamic_args[i], input_devices, input_specs[i],
                 cache_entry.py_devices, python_shard_arg_fallback_));

    std::vector<xla::PjRtBuffer*>& per_device_buffers =
        sharded_arg.per_device_buffers;
    for (int computation = 0; computation < num_computations; ++computation) {
      num_computation_num_args_buffers[computation][i] =
          per_device_buffers[computation];
    }
    for (auto& owned_buffer : sharded_arg.owned_buffers) {
      arguments.keep_alive.push_back(std::move(owned_buffer));
    }
    if (sharded_arg.owning_sda) {
      arguments.keep_alive_objects.push_back(std::move(sharded_arg.owning_sda));
    }
  }

  // A vector of [num_devices, num_outputs].
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> output_buffers;
  {
    py::gil_scoped_release gil_release;
    auto pjrt_executable = cache_entry.executable->mutable_pjrt_executable();
    TF_ASSIGN_OR_RETURN(output_buffers, pjrt_executable->Execute(
                                            num_computation_num_args_buffers,
                                            cache_entry.executable->options()));
  }

  // TODO(jblespiau): We don't need to create the PyBuffer objects.
  // Having a C++ `ShardedDeviceArray`, keeping internally the PjRtBuffer
  // objects is sufficient, and we can lazily create the `PyBuffer` only if
  // we access them from Python.
  auto traceback = xla::Traceback::Get();
  // TODO(jblespiau): Change the `client` function to return a reference.
  std::shared_ptr<xla::PyClient> client = cache_entry.executable->client();

  // Convert the PjRtBuffer objects to PyBuffer, and invert the order from
  // [num_devices, num_args] to [num_args, num_devices].
  const int num_outputs = output_buffers[0].size();
  std::vector<std::vector<xla::PyBuffer::object>> outputs;
  outputs.resize(num_outputs);
  for (int output_id = 0; output_id < num_outputs; ++output_id) {
    outputs[output_id].reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      outputs[output_id].push_back(xla::PyBuffer::Make(
          client, std::move(output_buffers[computation][output_id]),
          traceback));
    }
  }

  py::list outputs_as_python_objects;
  const auto& output_specs = cache_entry.out_result_specs;

  std::vector<py::object> flat_sharded_device_arrays;
  flat_sharded_device_arrays.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const ResultSpec& result_spec = output_specs[i];
    flat_sharded_device_arrays.push_back(ShardedDeviceArray::Make(
        /*aval=*/result_spec.out_aval,
        /*sharding_spec=*/result_spec.out_spec,
        /*device_buffers=*/py::cast(std::move(outputs[i])),
        /*indices=*/result_spec.out_indices,
        /*weak_type=*/result_spec.weak_type));
  }
  py::object out =
      cache_entry.out_pytree_def.Unflatten(flat_sharded_device_arrays);

  // If there is a post-hook function, call it with the inputs and the outputs.
  std::optional<py::object> post_hook = GetPostHook();
  if (post_hook) {
    (*post_hook)(this->AsPyHandle(), args, kwargs, out);
  }

  return out;
}

struct JaxPmapFunctionObject {
  PyObject_HEAD;
  PyObject* dict;      // Dictionary for __dict__
  PyObject* weakrefs;  // Weak references; for use by the Python interpreter.
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

py::handle PmapFunction::AsPyHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(JaxPmapFunctionObject, fun));
}

namespace {

extern "C" {

PyObject* JaxPmapFunction_tp_new(PyTypeObject* subtype, PyObject* args,
                                 PyObject* kwds) {
  JaxPmapFunctionObject* self =
      reinterpret_cast<JaxPmapFunctionObject*>(subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->dict = nullptr;
  self->weakrefs = nullptr;
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

PyObject* JaxPmapFunction_tp_call(PyObject* self, PyObject* args,
                                  PyObject* kwargs) {
  JaxPmapFunctionObject* o = reinterpret_cast<JaxPmapFunctionObject*>(self);
  tensorflow::profiler::TraceMe traceme([&] {
    return absl::StrCat("JaxPmapFunction(", o->fun.function_name(), ")");
  });
  py::kwargs py_kwargs;
  if (kwargs) {
    py_kwargs = py::reinterpret_borrow<py::kwargs>(kwargs);
  }
  try {
    xla::StatusOr<py::object> out = o->fun.Call(
        py::reinterpret_borrow<py::args>(args), std::move(py_kwargs));
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
  sharded_axis.def(py::init<int>()).def_readonly("axis", &ShardedAxis::axis);
  sharded_axis
      .def("__repr__",
           [](const ShardedAxis& x) {
             return absl::StrCat("ShardedAxis(axis=", x.axis, ")");
           })
      .def("__eq__", [](const ShardedAxis& self, const ShardedAxis& other) {
        return self == other;
      });

  py::class_<Replicated> replicated(pmap_lib, "Replicated");
  replicated.def(py::init<int>())
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

  TF_CHECK_OK(ShardedDeviceArray::RegisterTypes(pmap_lib));

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
    type->tp_flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC;
    type->tp_new = JaxPmapFunction_tp_new;
    type->tp_dealloc = JaxPmapFunction_tp_dealloc;
    type->tp_dictoffset = offsetof(JaxPmapFunctionObject, dict);
    type->tp_traverse = JaxPmapFunction_tp_traverse;
    type->tp_clear = JaxPmapFunction_tp_clear;
    type->tp_weaklistoffset = offsetof(JaxPmapFunctionObject, weakrefs);
    type->tp_getset = JaxPmapFunction_tp_getset;
    type->tp_descr_get = JaxPmapFunction_tp_descr_get;
    type->tp_call = JaxPmapFunction_tp_call;
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

  // Accepts _arbitrary_ arguments for a pmapped function and returns the
  // corresponding signatures that are used as cache keys. No-op.
  //
  // This function allows to pass partial args, which is especially useful when
  // the full list of arguments is too long and results in enormous signatures.
  // For example, this function can be multiple times as
  // > fn._debug_compute_cache_key(arg[0])
  // > fn._debug_compute_cache_key(arg[1])
  // > fn._debug_compute_cache_key(arg[-3:-1])
  // ...
  cfun.attr("_debug_compute_cache_key") = py::cpp_function(
      [](const PmapFunction::object& self, const py::args& args,
         const py::kwargs& kwargs) -> xla::StatusOr<std::string> {
        ParsedArgumentsAsBuffers arguments;
        TF_ASSIGN_OR_RETURN(PmapFunction * fun, AsPmapFunction(self));
        TF_RETURN_IF_ERROR(ParseArguments(args, kwargs, fun->static_argnums(),
                                          /*static_argnames=*/{}, arguments));
        TF_RETURN_IF_ERROR(fun->UpdateArgsSignature(args, kwargs, arguments));
        return arguments.signature.DebugString();
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
