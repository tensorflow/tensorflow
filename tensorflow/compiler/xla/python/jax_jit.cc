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
// TODO(jblespiau): Remove the "xla::" prefixes when not needed.

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

py::object ExtraJitContext() {
  return thread_local_state.extra_jit_context.value_or(
      global_state.extra_jit_context);
}

absl::optional<py::object> PostHook() {
  return thread_local_state.post_hook.has_value() ? thread_local_state.post_hook
                                                  : global_state.post_hook;
}
}  // namespace

bool GetEnableX64() {
  return thread_local_state.enable_x64.value_or(global_state.enable_x64);
}

std::string CallSignature::DebugString() const {
  std::vector<std::string> static_args_str;
  static_args_str.reserve(static_args.size());
  for (auto& static_arg : static_args) {
    static_args_str.emplace_back(py::cast<std::string>(py::str(static_arg)));
  }

  std::vector<std::string> signature_str;
  signature_str.reserve(dynamic_args_signatures.size());

  for (auto& arg_signature : dynamic_args_signatures) {
    signature_str.emplace_back(arg_signature.DebugString());
  }
  std::vector<std::string> tree_def_str;
  signature_str.reserve(dynamic_positional_args_treedef.size());
  for (auto& tree_def : dynamic_positional_args_treedef) {
    tree_def_str.emplace_back(tree_def.ToString());
  }
  std::vector<std::string> keyword_names;
  keyword_names.reserve(keyword_args.size());
  for (auto& kwarg_entry : keyword_args) {
    keyword_names.emplace_back(py::cast<std::string>(kwarg_entry.key));
    tree_def_str.emplace_back(kwarg_entry.value_treedef.ToString());
  }
  return absl::StrCat(
      static_args.size(), " static_args: ", absl::StrJoin(static_args_str, ","),
      "\n",  // new line
      keyword_args.size(), " keyword args:", absl::StrJoin(keyword_names, ","),
      "\n",  // new-line
      dynamic_positional_args_treedef.size(), " positional args.\n",
      dynamic_args_signatures.size(),
      " dynamic args (positional+keyword):\n   - ",
      absl::StrJoin(signature_str, ", "), "\n   - ",
      absl::StrJoin(tree_def_str, " | "));
}

bool CallSignature::operator==(const CallSignature& other) const {
  return std::tie(dynamic_positional_args_treedef, keyword_args,
                  dynamic_args_signatures, device, jax_enable_x64) ==
             std::tie(other.dynamic_positional_args_treedef, other.keyword_args,
                      other.dynamic_args_signatures, other.device,
                      other.jax_enable_x64) &&
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
         extra_jit_context.equal(other.extra_jit_context);
}

template <typename H>
H AbslHashValue(H h, const CallSignature& s) {
  h = H::combine_contiguous(std::move(h),
                            s.dynamic_positional_args_treedef.data(),
                            s.dynamic_positional_args_treedef.size());
  h = H::combine_contiguous(std::move(h), s.keyword_args.data(),
                            s.keyword_args.size());
  h = H::combine_contiguous(std::move(h), s.dynamic_args_signatures.data(),
                            s.dynamic_args_signatures.size());
  h = H::combine(std::move(h), s.device);
  h = H::combine(std::move(h), s.jax_enable_x64);
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
  // We do not hash extra_jit_context since its current hash function costs
  // ~300ns and we don't expect a large number of different contexts.
  return h;
}

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(const py::args& args, const py::kwargs& py_kwargs,
                           absl::Span<int const> static_argnums,
                           ParsedArgumentsAsBuffers& arguments) {
  tensorflow::profiler::TraceMe traceme("ParseArguments");
  if (static_argnums.size() > args.size()) {
    return xla::InvalidArgument(
        "%s", "[jaxjit] Error with static argnums, executing the Python path.");
  }
  arguments.flat_dynamic_args.reserve(args.size() + py_kwargs.size() -
                                      static_argnums.size());
  arguments.signature.dynamic_positional_args_treedef.reserve(
      args.size() - static_argnums.size());

  // Positional arguments.
  for (size_t i = 0; i < args.size(); ++i) {
    if (std::find(static_argnums.begin(), static_argnums.end(), i) ==
        static_argnums.end()) {
      xla::PyTreeDef pytree_def;
      pytree_def.FlattenInto(args[i], arguments.flat_dynamic_args);
      arguments.signature.dynamic_positional_args_treedef.push_back(pytree_def);
    } else {
      arguments.signature.static_args.emplace_back(
          // borrow is mandatory here.
          py::reinterpret_borrow<py::object>(args[i]));
    }
  }

  // Keyword arguments.
  std::vector<std::pair<py::handle, py::handle>> kwargs(py_kwargs.begin(),
                                                        py_kwargs.end());
  // We first intern the keys, then sort them (by name, as in the Python path)
  // (see also xla::PyTreeDef::Flatten) and then create the signatures.
  // TODO(jblespiau): We should be able to sort the keys by interned-key
  // pointers, but this requires the Python compilation to do the same.
  arguments.signature.keyword_args.resize(kwargs.size());
  for (size_t i = 0; i < kwargs.size(); ++i) {
    // Intern the key if not already interned.
    if (!PyUnicode_CHECK_INTERNED(kwargs[i].first.ptr())) {
      PyObject* key = kwargs[i].first.ptr();
      kwargs[i].first.inc_ref();
      PyUnicode_InternInPlace(&key);
      arguments.keep_alive_objects.push_back(
          py::reinterpret_steal<py::object>(key));
      kwargs[i].first = py::handle(key);
    }
  }

  std::sort(kwargs.begin(), kwargs.end(),
            [](const std::pair<py::handle, py::handle>& a,
               const std::pair<py::handle, py::handle>& b) {
              return a.first < b.first;
            });
  for (size_t i = 0; i < kwargs.size(); ++i) {
    arguments.signature.keyword_args[i].key = kwargs[i].first;
    arguments.signature.keyword_args[i].value_treedef.FlattenInto(
        kwargs[i].second, arguments.flat_dynamic_args);
  }
  return xla::Status::OK();
}

namespace {}  // namespace

namespace {

// Elements of CacheEntry are protected by the GIL.
struct CacheEntry {
  // Has this cache entry been fully populated?
  // The first thread to determine a compilation result sets `ready` to true
  // after populating all necessary fields of the cache entry.
  bool ready = false;

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
  xla::PjRtDevice* sticky_device;

  // Trivial computation will fallback to Python.
  // Running a jax(pmap) will also fallback to Python.
  bool fall_back_to_python = false;

  // Python objects (notably in the cache key) that must remain alive as long
  // as the cache entry does. Currently this is the `key` values in the kwarg
  // entries in the cache key.
  std::vector<py::object> keepalive;
};

// A `CompiledFunction` is associated to a `jax.jit(f)` and takes care of the
// bookkeeping of the different signatures used and the dispatch of calls to
// the correct underlying `PyExecutable`. This class is thread-safe.
class CompiledFunction {
 public:
  CompiledFunction(py::function fun, py::function cache_miss,
                   py::function get_device, std::vector<int> static_argnums,
                   int cache_size);
  ~CompiledFunction();

  // This function will:
  // (a) flatten the inputs using pytree
  // (b) get buffer objects from the arguments
  // (c) call the executable
  // (d) construct `DeviceArray` objects from the outputs
  // (e) reconstruct the `PyTree`.
  xla::StatusOr<py::object> Call(py::args args, py::kwargs kwargs);

  // This allows `inspect.signature(cpp_jitted_f)` from Python.
  py::object PythonSignature() {
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(fun_);
  }

  int cache_size() const { return executables_.Size(); }
  void ClearCache() { executables_.Clear(); }

  const py::function& cache_miss() const { return cache_miss_; }

 private:
  void PopulateCacheEntry(CacheEntry* entry, const py::args& args,
                          const py::kwargs& kwargs,
                          const CallSignature& signature,
                          const py::tuple& out_and_fastpath_data);
  bool always_fallback_to_python_ = false;

  const py::function fun_;  // The Python function to jit.
  // See JAX _cpp_jit in api.py for documentation.
  const py::function cache_miss_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyExecutable. In sorted order.
  std::vector<int> static_argnums_;

  // A function taking no arguments and returning the default device and whether
  // jax.jit has been committed to it.
  const py::function get_device_;

  // Cache entries are shared_ptr<>s because it's possible the cache entry
  // might be evicted before we finish tracing/compiling. Protected by the GIL.
  xla::LRUCache<CallSignature, std::shared_ptr<CacheEntry>> executables_;

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
                                   int cache_size)
    : fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      get_device_(std::move(get_device)),
      executables_(cache_size) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
}

CompiledFunction::~CompiledFunction() = default;

// Converts flattened arguments contained in ParsedArgumentsAsBuffers in
// place. If arguments are `DeviceArray`, they must all be on the same `Device`.
//
// Returns `Status::OK()` on success. Returning an error should lead to
// calling the Python fallback.
xla::Status ConvertArgsToBuffers(bool jax_enable_x64, xla::PyClient& pyclient,
                                 xla::PjRtDevice* default_device,
                                 bool is_committed,
                                 ParsedArgumentsAsBuffers& arguments) {
  tensorflow::profiler::TraceMe traceme("ConvertArgsToBuffers");
  std::vector<xla::PjRtBuffer*>& arg_buffers = arguments.arg_buffers;
  auto& keep_alive = arguments.keep_alive;

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  arg_buffers.reserve(num_flat_dynamic_args);
  arguments.signature.dynamic_args_signatures.reserve(num_flat_dynamic_args);

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

  xla::DevicePutOptions options;
  options.squash_64bit_types = !jax_enable_x64;
  // TODO(phawkins): consider allowing forces here.
  options.force_lazy_arrays = false;
  options.allow_zero_copy = true;
  for (int i = 0; i < num_flat_dynamic_args; ++i) {
    py::handle arg = arguments.flat_dynamic_args[i];
    TF_ASSIGN_OR_RETURN(xla::DevicePutResult on_device,
                        DevicePut(arg, data_device, options));

    xla::PjRtBuffer* buffer = on_device.buffer;
    arg_buffers.push_back(buffer);
    if (on_device.owned_buffer) {
      keep_alive.push_back(std::move(on_device.owned_buffer));
    } else if (on_device.owning_pybuffer) {
      arguments.keep_alive_objects.push_back(
          std::move(on_device.owning_pybuffer));
    }

    xla::PyArgSignature sig(buffer->on_device_shape().element_type(),
                            buffer->on_device_shape().dimensions(),
                            on_device.weak_type);
    arguments.signature.dynamic_args_signatures.push_back(std::move(sig));
  }
  return xla::Status::OK();
}

void CompiledFunction::PopulateCacheEntry(
    CacheEntry* cache_entry, const py::args& args, const py::kwargs& kwargs,
    const CallSignature& signature, const py::tuple& out_and_fastpath_data) {
  CHECK_EQ(out_and_fastpath_data.size(), 2);
  if (out_and_fastpath_data[1].is_none()) {
    cache_entry->fall_back_to_python = true;
    cache_entry->ready = true;
    return;
  }

  py::tuple executable_handlers_out_tree =
      py::cast<py::tuple>(out_and_fastpath_data[1]);
  if (executable_handlers_out_tree.size() != 5) {
    throw std::runtime_error(absl::StrCat(
        "The versions of jaxlib and Jax are incompatible (jaxlib is too recent "
        "compared to Jax. Upgrade Jax is advised. The C++ code expects "
        "5 arguments but ",
        executable_handlers_out_tree.size(), " where provided: ",
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

  cache_entry->ready = true;
}

xla::StatusOr<py::object> CompiledFunction::Call(py::args args,
                                                 py::kwargs kwargs) {
  if (JitIsDisabled()) {
    return fun_(*args, **kwargs);
  }
  if (always_fallback_to_python_) {
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }
  // Delayed values are retrieved on the first call to `Call`.
  if (!default_device_) {
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
      if (!default_device_) {
        return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
      }
    }
  }

  ParsedArgumentsAsBuffers arguments;
  if (!ParseArguments(args, kwargs, static_argnums_, arguments).ok()) {
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }

  bool jax_enable_x64 = GetEnableX64();
  arguments.signature.jax_enable_x64 = jax_enable_x64;
  // The C++ jit do not support Tracers arguments inputs yet. The Python-based
  // jit function will be called if any of the dynamic arguments is unsupported.
  if (!ConvertArgsToBuffers(jax_enable_x64, *default_pyclient_, default_device_,
                            is_committed_, arguments)
           .ok()) {
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }
  arguments.signature.extra_jit_context = ExtraJitContext();

  std::shared_ptr<CacheEntry> cache_entry = executables_.GetOrCreateIfAbsent(
      arguments.signature, [](const CallSignature& key) {
        auto entry = std::make_shared<CacheEntry>();
        for (const auto& kw : key.keyword_args) {
          entry->keepalive.push_back(
              py::reinterpret_borrow<py::object>(kw.key));
        }
        return entry;
      });

  if (!cache_entry->ready) {
    // Calls Python and may release the GIL. May also throw if
    // compilation/tracing fails.
    // Multiple threads may reach this point and compile the same computation
    // concurrently. Only the first thread to call PopulateCacheEntry ends
    // up committing its compilation result to the cache.
    // TODO(phawkins): it may be preferable to force other threads to wait if
    // a cache miss is already happening.
    py::object out_and_fastpath_data = cache_miss_(*args, **kwargs);
    py::tuple out_tuple = py::cast<py::tuple>(out_and_fastpath_data);

    // Another thread might have populated the cache entry while we were calling
    // cache_miss_. We therefore check again that the cache entry hasn't been
    // populated now that we have reacquired the GIL.
    if (!cache_entry->ready) {
      PopulateCacheEntry(cache_entry.get(), args, kwargs, arguments.signature,
                         out_tuple);
    }

    // We have already computed the result in the miss path so we can return it.
    // We are even *required* to do so if there are donated arguments, because
    // any donated buffers will now be invalid.
    return py::object(out_tuple[0]);
  }

  if (cache_entry->fall_back_to_python) {
    return py::object(py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0]);
  }
  std::vector<xla::PyBuffer::object> outputs =
      ValueOrThrow(cache_entry->executable->PjRtExecute(arguments.arg_buffers));

  const std::vector<py::object>& out_lazy_exprs = cache_entry->out_lazy_exprs;
  xla::PjRtDevice* sticky_device = cache_entry->sticky_device;

  std::vector<py::object> flat_device_arrays;
  flat_device_arrays.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    auto& buffer = outputs[i];
    if (out_lazy_exprs[i].is_none()) {  // No LazyExpr.
      buffer.buf()->SetAval(cache_entry->out_avals[i]);
      buffer.buf()->set_weak_type(cache_entry->out_weak_types[i]);
      TF_RETURN_IF_ERROR(buffer.buf()->set_sticky_device(sticky_device));
      flat_device_arrays.push_back(std::move(outputs[i]));
    } else {
      static const auto* xla_module =
          new py::module(py::module::import("jax.interpreters.xla"));
      static const auto* device_array =
          new py::handle(xla_module->attr("_DeviceArray"));
      flat_device_arrays.push_back((*device_array)(
          cache_entry->out_avals[i],
          py::cast(WrapWithClient(default_pyclient_, sticky_device)),
          out_lazy_exprs[i], std::move(outputs[i])));
    }
  }
  py::object out = cache_entry->out_pytree_def.Unflatten(flat_device_arrays);
  absl::optional<py::object> post_hook = PostHook();
  if (post_hook) {
    (*post_hook)(this, args, kwargs, out);
  }
  return out;
}

}  // namespace

void BuildJaxjitSubmodule(pybind11::module& m) {
  py::module jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  // We allow dynamic attributes on compiled functions because they are often
  // passed to @wraps(...).
  py::class_<CompiledFunction, std::unique_ptr<CompiledFunction>> cfun(
      jitlib, "CompiledFunction", py::dynamic_attr());
  cfun.def("__call__", &CompiledFunction::Call);
  cfun.def_property_readonly("__signature__",
                             &CompiledFunction::PythonSignature);
  cfun.def_property_readonly("_cache_miss", &CompiledFunction::cache_miss);

  // Implements the Python descriptor protocol so JIT-compiled functions can be
  // used as bound methods. See:
  // https://docs.python.org/3/howto/descriptor.html#functions-and-methods
  py::object method_type = py::module::import("types").attr("MethodType");
  cfun.def(
      "__get__",
      [method_type](py::object self, py::object obj, py::object objtype) {
        if (obj.is_none()) {
          return self;
        }
        return method_type(self, obj);
      },
      py::arg("obj"), py::arg("objtype") = py::none());

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
         std::vector<int> static_argnums,
         int cache_size) -> std::unique_ptr<CompiledFunction> {
        return std::make_unique<CompiledFunction>(
            std::move(fun), std::move(cache_miss), std::move(get_device),
            std::move(static_argnums), cache_size);
      },
      py::arg("fun"), py::arg("cache_miss"), py::arg("get_device"),
      py::arg("static_argnums"), py::arg("cache_size") = 4096);

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
  cfun.def("_cache_size", &CompiledFunction::cache_size);
  cfun.def("_clear_cache", &CompiledFunction::ClearCache);
  jitlib.def("_is_float0", &xla::IsFloat0);
}

}  // namespace jax
