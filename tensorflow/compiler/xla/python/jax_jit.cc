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
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/status.h"

namespace jax {

namespace py = pybind11;

// TODO(phawkins): Add support for Tracers.
// TODO(jblespiau): Use absl Status.
// TODO(jblespiau): Remove the "xla::" prefixes when not needed.

std::string ArgSignature::DebugString() const {
  std::string result = "";
  if (weak_type) {
    absl::StrAppend(&result, "weak_");
  }
  absl::StrAppend(&result, xla::PrimitiveType_Name(dtype));
  absl::StrAppend(&result, "[", absl::StrJoin(shape, ","), "]");
  return result;
}

bool CallSignature::operator==(const CallSignature& other) const {
  return std::tie(dynamic_positional_args_treedef, keyword_args,
                  dynamic_args_signatures, device) ==
             std::tie(other.dynamic_positional_args_treedef, other.keyword_args,
                      other.dynamic_args_signatures, other.device) &&
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
             });
}

void CallSignature::IncRef() const {
  for (const auto& kw : keyword_args) {
    kw.key.inc_ref();
  }
}

void CallSignature::DecRef() const {
  for (const auto& kw : keyword_args) {
    kw.key.dec_ref();
  }
}

namespace {

// These 2 constants are protected by the GIL.
ABSL_CONST_INIT bool disable_jit_flag = false;
ABSL_CONST_INIT bool enable_x64_flag = false;

ABSL_CONST_INIT thread_local absl::optional<bool> disable_jit_thread_local =
    absl::nullopt;
ABSL_CONST_INIT thread_local absl::optional<bool> jax_enable_x64_thread_local =
    absl::nullopt;

// The x64 mode is controlled by:
// - a global flag value, associated to --jax_enable_x64
// - possibly a thread-local value, which initially is absl::nullopt and which
//   will default to the flag value as long as it's not set.
void SetEnableX64Flag(bool jax_enable_x64) { enable_x64_flag = jax_enable_x64; }
bool GetEnableX64Flag() { return enable_x64_flag; }
void SetEnableX64ThreadLocal(absl::optional<bool> jax_enable_x64) {
  jax_enable_x64_thread_local = jax_enable_x64;
}
absl::optional<bool> GetEnableX64ThreadLocal() {
  return jax_enable_x64_thread_local;
}

void SetDisableJitFlag(bool disable_jit) { disable_jit_flag = disable_jit; }
bool GetDisableJitFlag() { return disable_jit_flag; }
void SetDisableJitThreadLocal(absl::optional<bool> disable_jit) {
  disable_jit_thread_local = disable_jit;
}
absl::optional<bool> GetDisableJitThreadLocal() {
  return disable_jit_thread_local;
}

bool JitIsDisabled() {
  if (disable_jit_thread_local != absl::nullopt) {
    return disable_jit_thread_local.value();
  } else {
    return disable_jit_flag;
  }
}

}  // namespace

bool GetEnableX64() {
  if (jax_enable_x64_thread_local != absl::nullopt) {
    return jax_enable_x64_thread_local.value();
  } else {
    return enable_x64_flag;
  }
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
  return h;
}

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
xla::Status ParseArguments(const py::args& args, const py::kwargs& py_kwargs,
                           absl::Span<int const> static_argnums,
                           ParsedArgumentsAsBuffers& arguments) {
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

namespace {

bool IsFloat0(py::array arg) {
  static const auto* dtypes_module =
      new py::module(py::module::import("jax.dtypes"));
  static const auto* float0_dtype =
      new py::handle(dtypes_module->attr("float0"));
  return float0_dtype->is(arg.attr("dtype"));
}

using ToArgSignatureHandler =
    std::function<xla::StatusOr<ArgSignature>(py::handle, bool)>;

}  // namespace

xla::StatusOr<ArgSignature> ArgSignatureOfValue(pybind11::handle arg,
                                                bool jax_enable_x64) {
  static const absl::flat_hash_map<PyObject*,
                                   ToArgSignatureHandler>* const handlers = [] {
    auto p = new absl::flat_hash_map<PyObject*, ToArgSignatureHandler>();

    const auto xla_module = py::module::import("jax.interpreters.xla");
    const auto& device_array = xla_module.attr("_DeviceArray");

    const xla::NumpyScalarTypes& dtypes = xla::GetNumpyScalarTypes();

    // The 4 Python native types.
    ToArgSignatureHandler bool_handler =
        [](py::handle, bool) -> xla::StatusOr<ArgSignature> {
      return ArgSignature(xla::PrimitiveType::PRED, {}, true);
    };
    ToArgSignatureHandler int_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      // TODO(phawkins): we should consider checking for integer overflow.
      if (jax_enable_x64) {
        return ArgSignature(xla::PrimitiveType::S64, {}, true);
      } else {
        return ArgSignature(xla::PrimitiveType::S32, {}, true);
      }
    };
    ToArgSignatureHandler float_handler =
        [&dtypes](py::handle h,
                  bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      // Only Python native types has a True weak_type.
      bool weak_type = !py::isinstance(h, dtypes.np_float64);
      if (jax_enable_x64) {
        return ArgSignature(xla::PrimitiveType::F64, {}, weak_type);
      } else {
        return ArgSignature(xla::PrimitiveType::F32, {}, weak_type);
      }
    };
    ToArgSignatureHandler complex_handler =
        [&dtypes](py::handle h,
                  bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      // Note that this branch is also taken  for np.complex128:
      // isinstance(np.complex128(3), complex) returns True
      // isinstance(np.complex64(3), complex) returns False
      bool weak_type = !py::isinstance(h, dtypes.np_complex128);
      if (jax_enable_x64) {
        return ArgSignature(xla::PrimitiveType::C128, {}, weak_type);
      } else {
        return ArgSignature(xla::PrimitiveType::C64, {}, weak_type);
      }
    };

    (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] = bool_handler;
    (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = int_handler;
    (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] = float_handler;
    (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] = complex_handler;

    // The Buffer types
    // PyBuffer necessarily has a trivial LazyExpr, no need to check it.
    ToArgSignatureHandler buffer_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      xla::PyBuffer* buffer = py::cast<xla::PyBuffer*>(h);
      bool weak_type = py::cast<py::bool_>(h.attr("aval").attr("weak_type"));
      return ArgSignature(buffer->buffer()->on_device_shape().element_type(),
                          buffer->buffer()->on_device_shape().dimensions(),
                          weak_type);
    };
    (*p)[py::type::handle_of<xla::DeviceArrayBase>().ptr()] = buffer_handler;
    ToArgSignatureHandler device_array_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      py::handle aval = h.attr("aval");
      TF_ASSIGN_OR_RETURN(auto dtype,
                          xla::DtypeToPrimitiveType(aval.attr("dtype")));
      return ArgSignature(dtype,
                          py::cast<std::vector<xla::int64>>(aval.attr("shape")),
                          py::cast<py::bool_>(aval.attr("weak_type")));
    };
    // ShardedDeviceArray is covered by the MRO fallback on _DeviceArray.
    (*p)[device_array.ptr()] = device_array_handler;

    ToArgSignatureHandler numpy_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      py::array numpy_array = py::cast<py::array>(h);
      if (IsFloat0(numpy_array)) {
        return xla::InvalidArgument(
            "float0 numpy arrays not supported in C++. "
            "Falling back to Python.");
      }
      TF_ASSIGN_OR_RETURN(xla::PrimitiveType dtype,
                          xla::DtypeToPrimitiveType(numpy_array.dtype()));
      if (!jax_enable_x64) {
        dtype = xla::Squash64BitTypes(dtype);
      }
      // We use reinterpret_cast<> to defend against environments where ssize_t
      // may not be precisely the same type as int64_t, even if it is the same
      // size (long vs long long).
      static_assert(sizeof(int64_t) == sizeof(ssize_t),
                    "Code assumes ssize_t is the same as int64_t");
      return ArgSignature(dtype,
                          absl::MakeConstSpan(reinterpret_cast<const int64_t*>(
                                                  numpy_array.shape()),
                                              numpy_array.ndim()),
                          /*weak_type=*/false);
    };
    const auto numpy = py::module::import("numpy");
    const auto& ndarray = numpy.attr("ndarray");
    (*p)[ndarray.ptr()] = numpy_handler;

    ToArgSignatureHandler np_uint64_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      if (jax_enable_x64) {
        return ArgSignature(xla::PrimitiveType::U64, {}, /*weak_type=*/false);
      } else {
        return ArgSignature(xla::PrimitiveType::U32, {}, /*weak_type=*/false);
      }
    };
    ToArgSignatureHandler np_int_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      if (jax_enable_x64) {
        return ArgSignature(xla::PrimitiveType::S64, {}, /*weak_type=*/false);
      } else {
        return ArgSignature(xla::PrimitiveType::S32, {}, /*weak_type=*/false);
      }
    };
    ToArgSignatureHandler numpy_array_handler =
        [](py::handle h, bool jax_enable_x64) -> xla::StatusOr<ArgSignature> {
      // This block deals with all numpy scalar types, except for int64_dt,
      // float64_dt and complex128_dt which are taken care of in previous if
      // blocks.
      TF_ASSIGN_OR_RETURN(auto dtype,
                          xla::DtypeToPrimitiveType(h.attr("dtype")));
      return ArgSignature(dtype, {}, /*weak_type=*/false);
    };

    // This block deals with all numpy scalar types, except for int64_dt,
    // float64_dt and complex128_dt which are taken care of in previous if
    // blocks.
    (*p)[dtypes.np_bool.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_int8.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_int16.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_int32.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_int64.ptr()] = np_int_handler;
    (*p)[dtypes.np_uint8.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_uint16.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_uint32.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_uint64.ptr()] = np_uint64_handler;
    (*p)[dtypes.np_float16.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_bfloat16.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_float32.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_float64.ptr()] = float_handler;
    (*p)[dtypes.np_complex64.ptr()] = numpy_array_handler;
    (*p)[dtypes.np_complex128.ptr()] = complex_handler;
    (*p)[dtypes.np_longlong.ptr()] = np_int_handler;
    (*p)[dtypes.np_intc.ptr()] = numpy_array_handler;

    return p;
  }();

  auto res = handlers->find(arg.get_type().ptr());
  if (res == handlers->end()) {
    // We attempt to look at the MRO classes
    for (auto base_class : arg.get_type().attr("mro")()) {
      res = handlers->find(base_class.ptr());
      if (res != handlers->end()) {
        return res->second(arg, jax_enable_x64);
      }
    }
    return xla::InvalidArgument(
        "%s", absl::StrCat("Not supported: The C++ ToArgSignature only accepts "
                           "Buffer/DeviceArray/ShardedDeviceArray, Numpy "
                           "arrays scalars of supported types "
                           "(see implementation), or Python scalars. Got type ",
                           py::cast<std::string>(py::str(arg.get_type()))));
  } else {
    return res->second(arg, jax_enable_x64);
  }
}

namespace {

struct CacheEntry {
  std::shared_ptr<xla::PyExecutable> executable;
  xla::PyTreeDef out_pytree_def;
  // We use Python types within the vector because this is what we will be
  // returning to Python. No need to convert back and forth.
  // We need py::object to maintain the objects alive.
  std::vector<py::object> out_avals;
  // The processing done in `AddCacheEntry` ensures that LazyExpr are stored as
  // `py::none()`.
  std::vector<py::object> out_lazy_exprs;
  py::object sticky_device;

  // Ensures a single thread performs the compilation for a given executable.
  //
  // The first thread (holding the GIL) will create the CacheEntry associated to
  // a signature and if the object has been insterted already, other threads
  // will wait for the notification.
  absl::Notification compilation_complete;
  absl::optional<xla::Status> compilation_error = absl::nullopt;
  // Trivial computation will fallback to Python.
  // Running a jax(pmap) will also fallback to Python.
  bool fall_back_to_python = false;
};

// A `CompiledFunction` is associated to a `jax.jit(f)` and takes care of the
// bookkeeping of the different signatures used and the dispatch of calls to
// the correct underlying `PyExecutable`. This class is thread-safe.
class CompiledFunction {
 public:
  CompiledFunction(py::function fun, py::function cache_miss,
                   py::function get_device, std::vector<int> static_argnums);
  ~CompiledFunction();

  // This function will:
  // (a) flatten the inputs using pytree
  // (b) get buffer objects from the arguments
  // (c) call the executable
  // (d) construct `DeviceArray` objects from the outputs
  // (e) reconstruct the `PyTree`.
  py::object Call(py::args args, py::kwargs kwargs);

  // This allows `inspect.signature(cpp_jitted_f)` from Python.
  py::object PythonSignature() {
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(fun_);
  }

  int cache_size() const { return executables_.size(); }

 private:
  // Returns nullptr if not present in the cache.
  CacheEntry* GetCacheEntryIfPresent(const CallSignature& signature);
  // Should never return nullptr.
  CacheEntry* AddCacheEntry(const py::args& args, const py::kwargs& kwargs,
                            const CallSignature& signature,
                            py::object out_and_fastpath_data);
  bool always_fallback_to_python_ = false;

  const py::function fun_;  // The Python function to jit.
  // See JAX _cpp_jit in api.py for documentation.
  const py::function cache_miss_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyExecutable. In sorted order.
  std::vector<int> static_argnums_;
  // We need a `unique_ptr` here to ensure value pointer stability.
  absl::flat_hash_map<CallSignature, std::unique_ptr<CacheEntry>> executables_;

  // A function taking no arguments and returning the default device and whether
  // jax.jit has been committed to it.
  const py::function get_device_;

  // The writing of the following is protected by the mutex.
  absl::Mutex mu_;

  // The logic if the following:
  // - if `device` or `backend` are not specified to `jax.jit`, we will use
  //   the input sticky buffer device, or `default_device_` if there is no
  //   such sticky buffer.
  // - When one of `device` or `backend` is specified, this will determine
  //   the `default_device_` which will be used as the targeted device. In
  //   which case, we will always copy input buffers to this device.
  std::shared_ptr<xla::PyClient> default_pyclient_ = nullptr;
  xla::ClientAndPtr<xla::PjRtDevice> default_pydevice_;
  xla::PjRtDevice* default_device_ = nullptr;
  bool is_committed_;
};

CompiledFunction::CompiledFunction(py::function fun, py::function cache_miss,
                                   py::function get_device,
                                   std::vector<int> static_argnums)
    : fun_(std::move(fun)),
      cache_miss_(std::move(cache_miss)),
      static_argnums_(std::move(static_argnums)),
      get_device_(std::move(get_device)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
}

CompiledFunction::~CompiledFunction() {
  for (const auto& entry : executables_) {
    entry.first.DecRef();
  }
}

// Converts flattened arguments contained in ParsedArgumentsAsBuffers in
// place. If arguments are `DeviceArray`, they must all be on the same `Device`.
//
// Returns `Okxla::Status()` on success. Returning an error should lead to
// calling the Python fallback.
xla::Status ConvertArgsToBuffers(bool jax_enable_x64, xla::PyClient& pyclient,
                                 xla::PjRtDevice* default_device,
                                 bool is_committed,
                                 ParsedArgumentsAsBuffers& arguments) {
  std::vector<xla::PjRtBuffer*>& arg_buffers = arguments.arg_buffers;
  auto& keep_alive = arguments.keep_alive;

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  arg_buffers.reserve(num_flat_dynamic_args);
  arguments.signature.dynamic_args_signatures.reserve(num_flat_dynamic_args);

  static const auto* xla_module =
      new py::module(py::module::import("jax.interpreters.xla"));
  const auto& device_array = xla_module->attr("_DeviceArray");

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
    for (py::handle arg : arguments.flat_dynamic_args) {
      // We specically only deal with DeviceArray (not ShardedDeviceArray).
      // (Can happen in jit(pmap), e.g. "test_jit_nested_donate_ignored").
      if (py::isinstance<xla::PyBuffer>(arg) ||
          arg.get_type().is(device_array)) {
        xla::PyBuffer* buffer;
        if (arg.attr("_device").is_none()) {  // Skip non-sticky devices.
          continue;
        }
        try {
          // This can fail, e.g. when device_buffer is a `DeviceConstant`.
          buffer = py::cast<xla::PyBuffer*>(arg.attr("device_buffer"));
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
        xla::PjRtDevice* device = buffer->buffer()->device();
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
  for (py::handle arg : arguments.flat_dynamic_args) {
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

    ArgSignature sig(buffer->on_device_shape().element_type(),
                     buffer->on_device_shape().dimensions(),
                     on_device.weak_type);
    arguments.signature.dynamic_args_signatures.push_back(std::move(sig));
  }
  return xla::Status::OK();
}

CacheEntry* CompiledFunction::GetCacheEntryIfPresent(
    const CallSignature& signature) {
  auto found_iterator = executables_.find(signature);
  if (found_iterator != executables_.end()) {  // Cache hit!
    if (!found_iterator->second->compilation_complete.HasBeenNotified()) {
      py::gil_scoped_release gil_release;
      found_iterator->second->compilation_complete.WaitForNotification();
    }
    if (found_iterator->second->compilation_error) {
      throw std::invalid_argument(
          found_iterator->second->compilation_error.value().error_message());
    }
    return found_iterator->second.get();
  }
  return nullptr;
}

CacheEntry* CompiledFunction::AddCacheEntry(const py::args& args,
                                            const py::kwargs& kwargs,
                                            const CallSignature& signature,
                                            py::object out_and_fastpath_data) {
  // We need to insert the element.
  auto result = executables_.emplace(signature, std::make_unique<CacheEntry>());
  auto it = result.first;
  CacheEntry* cache_entry = it->second.get();
  // CallSignatures in the cache own their keyword argument reference.
  result.first->first.IncRef();

  py::tuple tuple = py::cast<py::tuple>(out_and_fastpath_data);
  CHECK_EQ(tuple.size(), 2);
  if (tuple[1].is_none()) {
    cache_entry->fall_back_to_python = true;
    cache_entry->compilation_complete.Notify();
    return cache_entry;
  }

  py::tuple executable_handlers_out_tree = py::cast<py::tuple>(tuple[1]);
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
      py::cast<py::object>(executable_handlers_out_tree[2]);
  auto avals = py::cast<py::list>(executable_handlers_out_tree[3]);
  auto lazy_exprs = py::cast<py::list>(executable_handlers_out_tree[4]);
  CHECK_EQ(avals.size(), lazy_exprs.size());

  cache_entry->out_avals.reserve(avals.size());
  cache_entry->out_lazy_exprs.reserve(avals.size());
  for (int i = 0; i < avals.size(); ++i) {
    py::object shaped_array = py::reinterpret_borrow<py::object>(avals[i]);
    py::object lazy_expr = py::reinterpret_borrow<py::object>(lazy_exprs[i]);

    cache_entry->out_avals.push_back(shaped_array);
    cache_entry->out_lazy_exprs.push_back(lazy_expr);
  }

  cache_entry->compilation_complete.Notify();
  return cache_entry;
}

py::object CompiledFunction::Call(py::args args, py::kwargs kwargs) {
  if (JitIsDisabled()) {
    return fun_(*args, **kwargs);
  }
  if (always_fallback_to_python_) {
    return py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0];
  }
  // Delayed values are retrieved on the first call to `Call`.
  if (!default_device_) {
    // As we are calling Python code, that may release the GIL, we first hold
    // mu_ before holding the GIL.
    py::gil_scoped_release gil_release;
    {
      absl::MutexLock lock1(&mu_);
      py::gil_scoped_acquire gil_aquire;

      if (!default_device_) {
        py::object device_and_is_committed = get_device_();
        try {
          default_pydevice_ = py::cast<xla::ClientAndPtr<xla::PjRtDevice>>(
              device_and_is_committed.attr("default_device"));
        } catch (const py::cast_error& e) {
          // Pathways and Cloud TPU 2VM runtime.
          always_fallback_to_python_ = true;
          return py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0];
        }
        default_pyclient_ = default_pydevice_.client;
        default_device_ = default_pydevice_.contents;
        if (!default_device_) {  // UPTC
          always_fallback_to_python_ = true;
          return py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0];
        }
        is_committed_ =
            py::cast<bool>(device_and_is_committed.attr("committed_to_device"));
      }
    }
  }
  CHECK(default_device_);

  ParsedArgumentsAsBuffers arguments;
  if (!ParseArguments(args, kwargs, static_argnums_, arguments).ok()) {
    return py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0];
  }

  bool jax_enable_x64 = GetEnableX64();
  arguments.signature.jax_enable_x64 = jax_enable_x64;
  // The C++ jit do not support Tracers arguments inputs yet. The Python-based
  // jit function will be called if any of the dynamic arguments is unsupported.
  if (!ConvertArgsToBuffers(jax_enable_x64, *default_pyclient_, default_device_,
                            is_committed_, arguments)
           .ok()) {
    return py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0];
  }

  CacheEntry* cache_entry = GetCacheEntryIfPresent(arguments.signature);

  if (!cache_entry) {
    py::object out_and_fastpath_data = cache_miss_(*args, **kwargs);
    cache_entry = GetCacheEntryIfPresent(arguments.signature);
    if (!cache_entry) {
      cache_entry = AddCacheEntry(args, kwargs, arguments.signature,
                                  out_and_fastpath_data);
    }
    CHECK(cache_entry);
    if (cache_entry->fall_back_to_python) {
      return py::cast<py::tuple>(out_and_fastpath_data)[0];
    }
    // As we have already computed the results, we can return it.
    // It's even *required* e.g. if there are donated arguments, because
    // otherwise the buffer which has been donated already will be invalid.
    return py::cast<py::tuple>(out_and_fastpath_data)[0];
  }
  CHECK(cache_entry);
  if (cache_entry->fall_back_to_python) {
    return py::cast<py::tuple>(cache_miss_(*args, **kwargs))[0];
  }
  std::vector<std::unique_ptr<xla::PyBuffer>> outputs =
      ValueOrThrow(cache_entry->executable->PjRtExecute(arguments.arg_buffers));

  const std::vector<py::object>& out_avals = cache_entry->out_avals;
  const std::vector<py::object>& out_lazy_exprs = cache_entry->out_lazy_exprs;
  const py::object& sticky_device = cache_entry->sticky_device;

  py::list flat_device_arrays;
  for (int i = 0; i < outputs.size(); ++i) {
    auto& buffer = outputs[i];
    if (out_lazy_exprs[i].is_none()) {  // No LazyExpr.
      buffer->SetAval(out_avals[i]);
      buffer->SetStickyDevice(sticky_device);
      flat_device_arrays.append(py::cast(std::move(outputs[i])));
    } else {
      static const auto* xla_module =
          new py::module(py::module::import("jax.interpreters.xla"));
      static const auto* device_array =
          new py::handle(xla_module->attr("_DeviceArray"));
      flat_device_arrays.append(
          (*device_array)(out_avals[i], sticky_device, out_lazy_exprs[i],
                          py::cast(std::move(outputs[i]))));
    }
  }
  return cache_entry->out_pytree_def.Unflatten(flat_device_arrays);
}

}  // namespace

void BuildJaxjitSubmodule(pybind11::module& m) {
  py::module jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  py::class_<CompiledFunction, std::unique_ptr<CompiledFunction>> cfun(
      jitlib, "CompiledFunction");
  cfun.def("__call__", &CompiledFunction::Call);
  cfun.def_property_readonly("__signature__",
                             &CompiledFunction::PythonSignature);

  jitlib.def("set_disable_jit_cpp_flag", &SetDisableJitFlag);
  jitlib.def("get_disable_jit_cpp_flag", &GetDisableJitFlag);
  jitlib.def("set_disable_jit_thread_local", &SetDisableJitThreadLocal);
  jitlib.def("get_disable_jit_thread_local", &GetDisableJitThreadLocal);
  jitlib.def("jit_is_disabled", &JitIsDisabled);
  // TODO(jblespiau): Remove from the Python code and remove this
  jitlib.def("set_disable_jit", &SetDisableJitThreadLocal);
  jitlib.def("get_disable_jit", &GetDisableJitThreadLocal);

  jitlib.def("set_enable_x64_cpp_flag", &SetEnableX64Flag);
  jitlib.def("get_enable_x64_cpp_flag", &GetEnableX64Flag);
  jitlib.def("set_enable_x64_thread_local", &SetEnableX64ThreadLocal);
  jitlib.def("get_enable_x64_thread_local", &GetEnableX64ThreadLocal);
  jitlib.def("get_enable_x64", &GetEnableX64);

  jitlib.def(
      "jit",
      [](py::function fun, py::function cache_miss, py::function get_device,
         std::vector<int> static_argnums) -> std::unique_ptr<CompiledFunction> {
        return std::make_unique<CompiledFunction>(
            std::move(fun), std::move(cache_miss), std::move(get_device),
            std::move(static_argnums));
      });

  // This function is yet a full replacement for the Python one, because:
  // (a) it does not support abstract types,
  // (b) it does not set the device stickiness yet.
  // TODO(jblespiau): Finish the replacement of the Python feature.
  jitlib.def("device_put", [](py::handle obj, bool jax_enable_x64,
                              xla::ClientAndPtr<xla::PjRtDevice> to_device) {
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
      auto buffer = std::make_unique<xla::PyBuffer>(
          pyclient, std::move(results->owned_buffer), xla::Traceback::Get());

      static const auto* jax_core =
          new py::module(py::module::import("jax.core"));
      static const auto* shaped_array =
          new py::handle(jax_core->attr("ShapedArray"));
      buffer->SetAval((*shaped_array)(
          buffer->python_shape(), buffer->python_dtype(), results->weak_type));
      buffer->SetStickyDevice(py::none());

      return py::cast(std::move(buffer));
    } else {
      return py::cast<py::object>(obj);
    }
  });

  py::class_<ArgSignature> arg_signature(jitlib, "ArgSignature");
  arg_signature
      .def_property_readonly("dtype",
                             [](const ArgSignature& sig) {
                               return PrimitiveTypeToDtype(sig.dtype);
                             })
      .def_property_readonly("shape",
                             [](const ArgSignature& sig) {
                               return xla::IntSpanToTuple(sig.shape);
                             })
      .def_readonly("weak_type", &ArgSignature::weak_type);
  jitlib.def("_ArgSignatureOfValue", &ArgSignatureOfValue);

  // All private members are only for testing purposes
  cfun.def("_cache_size", &CompiledFunction::cache_size);
  jitlib.def("_is_float0", &IsFloat0);
}

}  // namespace jax
