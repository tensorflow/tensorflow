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

#include <exception>
#include <memory>
#include <stdexcept>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/status.h"

namespace xla {

namespace py = pybind11;

// TODO(phawkins): Add support for Tracers.
// TODO(jblespiau): Use absl Status.

namespace {

thread_local bool disable_jit;
void SetDisableJit(bool disable_jit_) { disable_jit = disable_jit_; }
bool GetDisableJit() { return disable_jit; }

// Describes the abstract shape and dtype of an argument.
struct ArgSignature {
  // This is the XLA dtype of the object.
  xla::PrimitiveType dtype;
  // JAX arguments can be of weak type, if and only if they are Python scalars
  // or `DeviceArray` values such that `aval.weak_type` is true.
  bool weak_type;
  absl::InlinedVector<int64, 4> shape;
  bool operator==(const ArgSignature& other) const {
    return std::tie(dtype, weak_type, shape) ==
           std::tie(other.dtype, other.weak_type, other.shape);
  }
  bool operator!=(const ArgSignature& other) const { return !(*this == other); }

  std::string DebugString() const {
    std::string result = "";
    if (weak_type) {
      absl::StrAppend(&result, "weak_");
    }
    absl::StrAppend(&result, xla::PrimitiveType_Name(dtype));
    absl::StrAppend(&result, "[", absl::StrJoin(shape, ","), "]");
    return result;
  }
};

template <typename H>
H AbslHashValue(H h, const ArgSignature& s) {
  h = H::combine(std::move(h), s.dtype);
  if (!s.shape.empty()) {
    h = H::combine_contiguous(std::move(h), &s.shape.front(), s.shape.size());
  }
  return h;
}

// The signature of Python jitted function call, partitioned into:
// - dynamic positional arguments (i.e. positional args which are not static)
// - static positional arguments (i.e. the args associated to static_argnums)
// - keyword arguments
// The CallSignature should unambiguously identify a function call, thus,
// equality is based on:
// (a) Same PyTree for all dynamic positional arguments and keyword arguments
// (a) equality of the arguments and keyword arguments ArgSignature
// (a) equality (delegated to Python) of the static arguments.
struct CallSignature {
  struct KwargEntry {
    // To avoid comparing strings, we intern the kwargs strings.
    // The compilation cache holds a reference to all the keys.
    py::handle key;
    PyTreeDef value_treedef;
    bool operator==(const KwargEntry& other) const {
      return key.ptr() == other.key.ptr() &&
             value_treedef == other.value_treedef;
    }
    bool operator!=(const KwargEntry& other) const { return !(*this == other); }
  };

  // Only contains the arguments associated to `static_argnums`, sorted in the
  // order of their argnum index.
  std::vector<py::object> static_args;
  // A PyTreeDef for each positional dynamic (i.e. not static) argument.
  std::vector<PyTreeDef> dynamic_positional_args_treedef;
  // Keyword arguments. Sorted by the interned keyword pointers.
  std::vector<KwargEntry> keyword_args;
  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by interned keyword pointers).
  std::vector<ArgSignature> dynamic_args_signatures;

  bool operator==(const CallSignature& other) const {
    return std::tie(dynamic_positional_args_treedef, static_args, keyword_args,
                    dynamic_args_signatures) ==
           std::tie(other.dynamic_positional_args_treedef, other.static_args,
                    other.keyword_args, other.dynamic_args_signatures);
  }
  bool operator!=(const CallSignature& other) const {
    return !(*this == other);
  }

  // To be used when we want to keep ownership of Python values referenced by
  // the `CallSignature` (i.e. when we insert an entry).
  void IncRef() const;
  // The destructor of the cache should call this on all entries.
  void DecRef() const;

  std::string DebugString() const;
};

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

template <typename H>
H AbslHashValue(H h, const CallSignature::KwargEntry& kw) {
  h = H::combine(std::move(h), kw.key.ptr(), kw.value_treedef);
  return h;
}

template <typename H>
H AbslHashValue(H h, const CallSignature& s) {
  // /!\ important: We cannot include static arguments to the hash, because
  // the py::object must be hashable for absl. We can try delegating to the
  // Python __hash__, but there are many non-hashable Python types such as
  // np.ndarray.
  // TODO(jblespiau): We should either ban non-hashable objects from jit or we
  // should hash them by object identity.
  h = H::combine_contiguous(std::move(h),
                            &s.dynamic_positional_args_treedef.front(),
                            s.dynamic_positional_args_treedef.size());
  h = H::combine_contiguous(std::move(h), &s.keyword_args.front(),
                            s.keyword_args.size());
  h = H::combine_contiguous(std::move(h), &s.dynamic_args_signatures.front(),
                            s.dynamic_args_signatures.size());
  return h;
}

std::string CallSignature::DebugString() const {
  std::vector<std::string> static_args_str;
  static_args_str.reserve(static_args.size());
  for (auto& static_arg : static_args) {
    static_args_str.emplace_back(py::cast<std::string>(static_arg.str()));
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

struct CacheEntry {
  std::shared_ptr<xla::PyExecutable> executable;
  xla::PjRtDevice* device;
  PyTreeDef out_pytree_def;
  // These are the objects required to create a `DeviceArray` object.
  // We use Python types within the vector because this is what we will be
  // returning to Python. No need to convert back and forth.
  // We need py::object to maintain the objects alive.
  std::vector<py::object> out_avals;
  std::vector<py::object> out_lazy_exprs;
  // Ensures a single thread performs the compilation for a given executable.
  //
  // The first thread (holding the GIL) will create the CacheEntry associated to
  // a signature and if the object has been insterted already, other threads
  // will wait for the notification.
  absl::Notification compilation_complete;
  absl::optional<std::exception> compilation_error = absl::nullopt;
};

// A `CompiledFunction` is associated to a `jax.jit(f)` and takes care of the
// bookkeeping of the different signatures used and the dispatch of calls to
// the correct underlying `PyExecutable`. This class is thread-safe.
class CompiledFunction {
 public:
  CompiledFunction(py::function fun, py::function cache_miss_fun,
                   py::function python_f_jitted, bool jax_enable_x64,
                   bool jax_disable_jit, std::vector<int> static_argnums);
  ~CompiledFunction();

  // This function will:
  // (a) flatten the inputs using pytree
  // (b) get buffer objects from the arguments
  // (c) call the executable
  // (d) construct `DeviceArray` objects from the outputs
  // (e) reconstruct the `PyTree`.
  py::object Call(py::args args, py::kwargs kwargs);

  // This allows `inspect.signature(cpp_jitted_f)` from Python.
  py::object __signature__() {
    static const auto* inspect = new py::module(py::module::import("inspect"));
    return inspect->attr("signature")(fun_);
  }

 private:
  CacheEntry& GetCacheEntry(const py::args& args, const py::kwargs& kwargs,
                            const CallSignature& signature,
                            absl::optional<py::tuple> cache_miss_return);
  CacheEntry& SetAndReturnCacheEntry(
      const py::args& args, const py::kwargs& kwargs,
      const CallSignature& signature,
      absl::optional<py::tuple> cache_miss_return = absl::nullopt);
  bool JitIsDisabled() { return GetDisableJit() || jax_disable_jit_; }

  const py::function fun_;  // The Python function to jit.
  // The Python function in charge of returning a `xla::PyExecutable` from
  // the arguments passed to `jitted_f`.
  const py::function cache_miss_fun_;
  // A function to call as fallback. This is the result of calling the Python
  // `jax.jit`.
  // TODO(jblespiau): Delete this when the C++ codepath supports all features.
  const py::function python_f_jitted_;

  // The value of the Python flag when the object was created.
  const bool jax_enable_x64_;
  const bool jax_disable_jit_;

  // We need to know the static arguments to remove them from the arguments
  // passed to the underlying PyExecutable. In sorted order.
  std::vector<int> static_argnums_;
  // We need a `unique_ptr` here to ensure value pointer stability.
  absl::flat_hash_map<CallSignature, std::unique_ptr<CacheEntry>> executables_;

  // As top-level functions are decorated with `jax.jit`, when
  // `CompiledFunction` is being instantiated from Python, the clients are not
  // yet available (done after GoogleInit). They will be during the first call
  // to `Call`.
  std::shared_ptr<xla::PyClient> pyclient_ = nullptr;
  xla::PjRtDevice* default_device_ = nullptr;

  // IMPORTANT: The GIL is not always held, because we call back to Python and
  // Python will release the GIL.
  // Thus, we protect the critical section modifying the `executables_` map
  // and more generally the compilation with some `absl::Notification`.
  // The first thread reaching such point will be responsible to create the
  // notification for the executable and others will wait until notified.
  // It's safe because the first thread will be holding the GIL while
  // initializing the `Notification`.
  //
  // absl::optional<absl::Notification> is not supported
  bool first_compilation_started_ = false;
  absl::Notification first_compilation_complete_;
  absl::optional<std::exception> first_compilation_error_ = absl::nullopt;
};

CompiledFunction::CompiledFunction(py::function fun,
                                   py::function cache_miss_fun,
                                   py::function python_f_jitted,
                                   bool jax_enable_x64, bool jax_disable_jit,
                                   std::vector<int> static_argnums)
    : fun_(std::move(fun)),
      cache_miss_fun_(std::move(cache_miss_fun)),
      python_f_jitted_(std::move(python_f_jitted)),
      jax_enable_x64_(jax_enable_x64),
      jax_disable_jit_(jax_disable_jit),
      static_argnums_(std::move(static_argnums)) {
  std::sort(static_argnums_.begin(), static_argnums_.end());
}

CompiledFunction::~CompiledFunction() {
  for (const auto& entry : executables_) {
    entry.first.DecRef();
  }
}

namespace {

// The resulting information of the parsing and conversion of the arguments.
struct ParsedArgumentsAsBuffers {
  // The call signature will be filled during 2 steps:
  // - `FlattenArguments` will fill the static arguments and the pytree
  //    structures
  // - the shapes and dtypes are filled later, by `ParseAndTransferArguments`.
  CallSignature signature;
  // The concatenation of the dynamic positional arguments and the sorted
  // keyword arguments. We do not need ownership, thus the py::handle.
  // TODO(jblespiau): We do not need py::object here and py::handle suffice and
  // will prevent any counter increment.
  std::vector<py::object> flat_dynamic_args;
  std::vector<py::object> keep_alive_objects;

  // The following is only valid if the parsing succeeds.
  std::vector<xla::PjRtBuffer*> arg_buffers;
  // We may need to keep some objects around, because:
  // (a) we need to extend the lifetime of objects created within
  //    `ConvertArgsToBuffers`
  // (b) `arg_buffers` do not maintain ownership
  std::vector<absl::variant<std::unique_ptr<xla::PyBuffer>,
                            std::unique_ptr<xla::PjRtBuffer>>>
      keep_alive;
};

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
void FlattenArguments(const py::args& args, const py::kwargs& py_kwargs,
                      absl::Span<int const> static_argnums,
                      ParsedArgumentsAsBuffers& arguments) {
  arguments.flat_dynamic_args.reserve(args.size() + py_kwargs.size() -
                                      static_argnums.size());
  arguments.signature.dynamic_positional_args_treedef.reserve(
      args.size() - static_argnums.size());

  // Positional arguments.
  for (size_t i = 0; i < args.size(); ++i) {
    if (std::find(static_argnums.begin(), static_argnums.end(), i) ==
        static_argnums.end()) {
      PyTreeDef pytree_def;
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
  // We first intern the keys, then sort them (by pointer) and then create
  // the signatures.
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
              return a.first.ptr() < b.first.ptr();
            });
  for (size_t i = 0; i < kwargs.size(); ++i) {
    arguments.signature.keyword_args[i].key = kwargs[i].first;
    arguments.signature.keyword_args[i].value_treedef.FlattenInto(
        kwargs[i].second, arguments.flat_dynamic_args);
  }
}

template <typename CppType, typename Pybind11Type>
std::unique_ptr<xla::PjRtBuffer> ConvertToScalarBuffer(
    const py::handle& scalar, xla::PjRtClient* client,
    xla::PjRtDevice* device) {
  CppType data = py::cast<Pybind11Type>(scalar);
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithType<CppType>({});
  return ValueOrThrow(xla::PjRtBuffer::FromHostBuffer(
      &data, shape,
      xla::PjRtBuffer::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
      client, device));
}

// Convert a scalar to the associated PjRtBuffer or raises an error if it is
// not convertible (thus, this must be called after other checks).
StatusOr<std::unique_ptr<xla::PjRtBuffer>> ScalarToBuffer(
    py::handle scalar, bool jax_enable_x64, xla::PjRtClient* client,
    xla::PjRtDevice* device) {
  // Important: In Python, isinstance(True, int) returns True. Thus, we have
  // to check for bool before int.
  if (py::isinstance<py::bool_>(scalar)) {
    return ConvertToScalarBuffer<bool, py::bool_>(scalar, client, device);
  } else if (py::isinstance<py::int_>(scalar)) {
    if (jax_enable_x64) {
      return ConvertToScalarBuffer<int64, py::int_>(scalar, client, device);
    } else {
      return ConvertToScalarBuffer<int, py::int_>(scalar, client, device);
    }
  } else if (py::isinstance<py::float_>(scalar)) {
    if (jax_enable_x64) {
      return ConvertToScalarBuffer<double, py::float_>(scalar, client, device);

    } else {
      return ConvertToScalarBuffer<float, py::float_>(scalar, client, device);
    }
  } else if (PyComplex_Check(scalar.ptr())) {
    Py_complex result = PyComplex_AsCComplex(scalar.ptr());
    if (result.real == -1.0 && PyErr_Occurred()) {
      PyErr_Clear();
      throw std::runtime_error("Could not convert the complex number");
    }
    if (jax_enable_x64) {
      xla::complex128 data(result.real, result.imag);
      xla::Shape shape = xla::ShapeUtil::MakeShapeWithType<xla::complex128>({});
      return ValueOrThrow(xla::PjRtBuffer::FromHostBuffer(
          &data, shape,
          xla::PjRtBuffer::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, client, device));
    } else {
      xla::complex64 data(result.real, result.imag);
      xla::Shape shape = xla::ShapeUtil::MakeShapeWithType<xla::complex64>({});
      return ValueOrThrow(xla::PjRtBuffer::FromHostBuffer(
          &data, shape,
          xla::PjRtBuffer::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, client, device));
    }
  }
  return InvalidArgument(
      "%s", absl::StrCat(
                "Not supported: The C++ jax jit execution path, only accepts "
                "DeviceArray, Numpy arrays, or Python scalars. Got type ",
                py::cast<std::string>(scalar.get_type().str())));
}

const py::dtype* DtypeTo32BitDtype(const py::dtype& dtype) {
  static const auto* int64_dt = new py::dtype("int64");
  static const auto* int32_dt = new py::dtype("int32");
  static const auto* uint64_dt = new py::dtype("uint64");
  static const auto* uint32_dt = new py::dtype("uint32");
  static const auto* float64_dt = new py::dtype("float64");
  static const auto* float32_dt = new py::dtype("float32");
  static const auto* complex64_dt = new py::dtype("complex64");
  static const auto* complex128_dt = new py::dtype("complex128");

  if (dtype == *int64_dt) {
    return int32_dt;
  }
  if (dtype == *float64_dt) {
    return float32_dt;
  }
  if (dtype == *uint64_dt) {
    return uint32_dt;
  }
  if (dtype == *complex128_dt) {
    return complex64_dt;
  }

  return nullptr;
}

// Converts flattened arguments contained in ParsedArgumentsAsBuffers in
// place. If arguments are `DeviceArray`, they must all be on the same `Device`.
//
// Returns `OkStatus()` on success.
Status ConvertArgsToBuffers(bool jax_enable_x64, xla::PyClient& pyclient,
                            xla::PjRtDevice* default_device,
                            ParsedArgumentsAsBuffers& arguments) {
  std::vector<xla::PjRtBuffer*>& arg_buffers = arguments.arg_buffers;
  auto& keep_alive = arguments.keep_alive;

  int num_flat_dynamic_args = arguments.flat_dynamic_args.size();
  arg_buffers.reserve(num_flat_dynamic_args);
  arguments.signature.dynamic_args_signatures.reserve(num_flat_dynamic_args);

  static const auto* xla_module =
      new py::module(py::module::import("jax.interpreters.xla"));
  const auto& device_array = xla_module->attr("DeviceArray");

  static const auto* numpy_module = new py::module(py::module::import("numpy"));
  const auto& array = numpy_module->attr("array");

  // TODO(phawkins): consider device stickiness.
  // We first check whether any `DeviceArray` is present and whether they are
  // attached to any specific device. See also
  // https://github.com/google/jax/pull/1884
  // https://github.com/google/jax/pull/1916 for the rationale why the
  // computation follows the data locality.
  // It's also similar to PyTorch's behavior.
  xla::PjRtDevice* data_device = nullptr;
  for (py::handle arg : arguments.flat_dynamic_args) {
    if (py::isinstance(arg, device_array)) {
      xla::PyBuffer* buffer;
      try {
        // This can fail, e.g. when device_buffer is a `DeviceConstant`.
        buffer = py::cast<xla::PyBuffer*>(arg.attr("device_buffer"));
      } catch (const py::cast_error& e) {
        return InvalidArgument(
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
        return InvalidArgument(
            "%s",
            absl::StrCat(
                "Arguments to a jit-compiled function must be colocated on the "
                "same device. Arguments were found to be on the two following "
                "different devices: ",
                device->DebugString(), " and ", data_device->DebugString()));
      } else {
        data_device = device;
      }
    }
  }
  if (!data_device) {
    // No `DeviceArray` were found default to `default_device`.
    data_device = default_device;
  }
  xla::PjRtClient* pjrt_client = data_device->client();

  for (py::handle arg : arguments.flat_dynamic_args) {
    // We do not support here d2d transparent transfers.
    // We assumes all the `DeviceArray` are already on the correct and shared
    // device.
    if (py::isinstance(arg, device_array)) {
      xla::PyBuffer* buffer =
          py::cast<xla::PyBuffer*>(arg.attr("device_buffer"));
      arg_buffers.push_back(buffer->buffer());
      ArgSignature sig;
      sig.dtype = buffer->shape().element_type();
      sig.shape.assign(buffer->shape().dimensions().begin(),
                       buffer->shape().dimensions().end());
      sig.weak_type = py::cast<py::bool_>(arg.attr("aval").attr("weak_type"));
      arguments.signature.dynamic_args_signatures.push_back(std::move(sig));
    } else if (py::isinstance<py::array>(arg)) {
      // TODO(jblespiau): Can we improve this call? Do we need the underlying
      // GlobalPyRefManager() and co?
      py::array numpy_array = py::cast<py::array>(arg);
      // If jax_enable_x64 is not set, we need to coerce 32 bits types.
      // Note that this is calling back to Python!
      if (!jax_enable_x64) {
        const py::dtype* to_dtype = DtypeTo32BitDtype(numpy_array.dtype());
        if (to_dtype) {
          numpy_array = array(numpy_array, to_dtype);
        }
      }
      std::unique_ptr<xla::PyBuffer> buffer =
          ValueOrThrow(pyclient.BufferFromPyval(
              numpy_array, data_device,
              /*force_copy=*/false, /*host_buffer_semantics=*/
              xla::PjRtBuffer::HostBufferSemantics::kZeroCopy));
      arg_buffers.push_back(buffer->buffer());

      ArgSignature sig;
      sig.dtype = buffer->shape().element_type();
      sig.weak_type = false;
      sig.shape.assign(buffer->shape().dimensions().begin(),
                       buffer->shape().dimensions().end());
      arguments.signature.dynamic_args_signatures.push_back(sig);

      keep_alive.emplace_back(std::move(buffer));
    } else {
      StatusOr<std::unique_ptr<xla::PjRtBuffer>> buffer =
          ScalarToBuffer(arg, jax_enable_x64, pjrt_client, data_device);
      if (!buffer.ok()) {
        return buffer.status();
      }
      arg_buffers.push_back(buffer.ValueOrDie().get());
      ArgSignature sig;
      sig.dtype = buffer.ValueOrDie()->on_host_shape().element_type();
      sig.weak_type = true;
      arguments.signature.dynamic_args_signatures.push_back(sig);

      keep_alive.emplace_back(std::move(buffer).ValueOrDie());
    }
  }
  return Status::OK();
}

}  // namespace

CacheEntry& CompiledFunction::GetCacheEntry(
    const py::args& args, const py::kwargs& kwargs,
    const CallSignature& signature,
    absl::optional<py::tuple> cache_miss_return) {
  auto found_iterator = executables_.find(signature);
  if (found_iterator != executables_.end()) {  // Cache hit!
    if (!found_iterator->second->compilation_complete.HasBeenNotified()) {
      py::gil_scoped_release gil_release;
      found_iterator->second->compilation_complete.WaitForNotification();
      if (found_iterator->second->compilation_error) {
        throw found_iterator->second->compilation_error.value();
      }
    }
    return *(found_iterator->second);
  }
  return SetAndReturnCacheEntry(args, kwargs, signature, cache_miss_return);
}
CacheEntry& CompiledFunction::SetAndReturnCacheEntry(
    const py::args& args, const py::kwargs& kwargs,
    const CallSignature& signature,
    absl::optional<py::tuple> cache_miss_return) {
  // We need to insert the element.
  auto result = executables_.emplace(signature, std::make_unique<CacheEntry>());
  auto it = result.first;
  CacheEntry& cache_entry = *(it->second.get());
  // CallSignatures in the cache own their keyword argument reference.
  result.first->first.IncRef();

  // Cache miss? Call the Python cache miss function.
  py::tuple executable_and_pytree;
  if (cache_miss_return) {
    executable_and_pytree = cache_miss_return.value();
  } else {
    try {
      executable_and_pytree = cache_miss_fun_(*args, **kwargs);
    } catch (const std::exception& e) {
      cache_entry.compilation_error = e;
      cache_entry.compilation_complete.Notify();
      throw;
    }
  }
  if (executable_and_pytree.size() != 4) {
    throw std::runtime_error(
        "AssertionError: The cache miss function should return 4 "
        "arguments.");
  }
  cache_entry.executable = py::cast<std::shared_ptr<xla::PyExecutable>>(
      std::move(executable_and_pytree[0]));
  int num_devices =
      cache_entry.executable->pjrt_executable().local_devices().size();
  if (num_devices != 1) {
    throw std::runtime_error(absl::StrCat(
        "Running on more than a single device is not currently supported."
        "The underlying PjRtExecutable has ",
        num_devices));
  }
  cache_entry.device =
      cache_entry.executable->pjrt_executable().local_devices()[0];
  cache_entry.out_pytree_def = py::cast<PyTreeDef>(executable_and_pytree[1]);

  py::list shaped_arrays =
      py::reinterpret_borrow<py::object>(executable_and_pytree[2]);
  py::list lazy_expressions =
      py::reinterpret_borrow<py::object>(executable_and_pytree[3]);

  cache_entry.out_avals.reserve(shaped_arrays.size());
  cache_entry.out_lazy_exprs.reserve(lazy_expressions.size());

  int num_outputs = shaped_arrays.size();
  for (int i = 0; i < num_outputs; ++i) {
    py::object shaped_array =
        py::reinterpret_borrow<py::object>(shaped_arrays[i]);
    py::object lazy_expr =
        py::reinterpret_borrow<py::object>(lazy_expressions[i]);

    cache_entry.out_avals.push_back(shaped_array);
    cache_entry.out_lazy_exprs.push_back(lazy_expr);
  }

  cache_entry.compilation_complete.Notify();
  return cache_entry;
}

py::object CompiledFunction::Call(py::args args, py::kwargs kwargs) {
  if (JitIsDisabled()) {
    return fun_(*args, **kwargs);
  }
  ParsedArgumentsAsBuffers arguments;
  FlattenArguments(args, kwargs, static_argnums_, arguments);

  // TODO(jblespiau): It would be preferable to have a single location for
  // locking code.
  absl::optional<py::tuple> cache_miss_result = absl::nullopt;
  if (!default_device_) {
    // TODO(jblespiau): This code will deadlock if a jitted function
    // recursively calls itself.
    if (first_compilation_started_) {
      if (!first_compilation_complete_.HasBeenNotified()) {
        py::gil_scoped_release gil_release;
        first_compilation_complete_.WaitForNotification();
        if (first_compilation_error_) {
          throw first_compilation_error_.value();
        }
      }
    } else {
      first_compilation_started_ = true;
      try {
        cache_miss_result = cache_miss_fun_(*args, **kwargs);
      } catch (const std::exception& e) {
        first_compilation_error_ = e;
        first_compilation_complete_.Notify();
        throw;
      }
      auto executable = py::cast<std::shared_ptr<xla::PyExecutable>>(
          cache_miss_result.value()[0]);

      pyclient_ = executable->client();
      default_device_ = executable->LocalDevices()[0].contents;
      first_compilation_complete_.Notify();
    }
  }

  // The C++ jit do not support Tracers arguments yet. The Python-based jit
  // function will be called if any of the dynamic arguments is unsupported.
  if (!ConvertArgsToBuffers(jax_enable_x64_, *pyclient_, default_device_,
                            arguments)
           .ok()) {
    return python_f_jitted_(*args, **kwargs);
  }

  CacheEntry& cache_entry =
      GetCacheEntry(args, kwargs, arguments.signature, cache_miss_result);

  std::vector<std::unique_ptr<xla::PyBuffer>> outputs =
      ValueOrThrow(cache_entry.executable->PjRtExecute(arguments.arg_buffers));

  static const auto* xla_module =
      new py::module(py::module::import("jax.interpreters.xla"));
  const auto& device_array = xla_module->attr("DeviceArray");

  const std::vector<py::object>& out_avals = cache_entry.out_avals;
  const std::vector<py::object>& out_lazy_exprs = cache_entry.out_lazy_exprs;

  py::list flat_device_arrays;
  for (int i = 0; i < outputs.size(); ++i) {
    flat_device_arrays.append(device_array(
        /*aval=*/out_avals[i], /*device=*/outputs[i]->device(),
        /*lazy_expr=*/out_lazy_exprs[i],
        /*device_buffer=*/std::move(outputs[i])));
  }
  return cache_entry.out_pytree_def.Unflatten(flat_device_arrays);
}

}  // namespace

void BuildJaxjitSubmodule(pybind11::module& m) {
  py::module jitlib = m.def_submodule("jax_jit", "Jax C++ jit library");

  py::class_<CompiledFunction, std::unique_ptr<CompiledFunction>> cfun(
      jitlib, "CompiledFunction");
  cfun.def("__call__", &CompiledFunction::Call);
  cfun.def_property_readonly("__signature__", &CompiledFunction::__signature__);

  jitlib.def("set_disable_jit", &SetDisableJit);
  jitlib.def("get_disable_jit", &GetDisableJit);
  jitlib.def(
      "jit",
      [](py::function fun, py::function cache_miss_fun,
         py::function fallback_on_unsupported_argument, bool jax_enable_x64,
         bool jax_disable_jit,
         std::vector<int> static_argnums) -> std::unique_ptr<CompiledFunction> {
        return std::make_unique<CompiledFunction>(
            std::move(fun), std::move(cache_miss_fun),
            std::move(fallback_on_unsupported_argument), jax_enable_x64,
            jax_disable_jit, std::move(static_argnums));
      });

  // Only for testing purposes
  jitlib.def("_ScalarToBuffer", [](py::handle scalar, bool jax_enable_x64,
                                   std::shared_ptr<xla::PyClient> client) {
    xla::PjRtClient* pjrt_client = client->pjrt_client();

    return std::make_unique<xla::PyBuffer>(
        client,
        ScalarToBuffer(scalar, jax_enable_x64, pjrt_client,
                       pjrt_client->local_devices()[0])
            .ValueOrDie(),
        nullptr);
  });
}

}  // namespace xla
