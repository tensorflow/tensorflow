/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/py_values.h"

#include <Python.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/complex.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/py_array.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/sharding.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace nb = nanobind;

namespace xla {

namespace {

using DevicePutFunc = std::function<absl::StatusOr<DevicePutResultFn>(
    nb::handle, ifrt::Client*, ifrt::Device*, const DevicePutOptions& options,
    ifrt::MemoryKind to_memory_kind)>;

template <typename T, typename SquashedT>
absl::StatusOr<DevicePutResultFn> HandlePythonScalar(
    nb::handle obj, ifrt::Client* client, ifrt::Device* to_device,
    const DevicePutOptions& options, ifrt::MemoryKind to_memory_kind) {
  T value;
  try {
    value = nb::cast<T>(obj);
  } catch (const std::exception& e) {
    return InvalidArgument(
        "Unable to convert Python scalar to %s. This most likely means the "
        "value (%s) overflows the range of the type.",
        PrimitiveType_Name(primitive_util::NativeToPrimitiveType<T>()),
        nb::cast<absl::string_view>(nb::repr(obj)));
  }

  std::variant<T, SquashedT> data;
  Shape shape;
  PrimitiveType type;
  if (std::is_same<T, SquashedT>() || !options.squash_64bit_types) {
    data.template emplace<0>(value);
    type = primitive_util::NativeToPrimitiveType<T>();
  } else {
    // TODO(phawkins): we should check for overflow here, e.g., because of bugs
    // like https://github.com/google/jax/issues/2006
    data.template emplace<1>(static_cast<SquashedT>(value));
    type = primitive_util::NativeToPrimitiveType<SquashedT>();
  }

  return [client, data, type, to_device,
          to_memory_kind]() -> absl::StatusOr<DevicePutResult> {
    const void* ptr = std::visit(
        [](const auto& v) { return static_cast<const void*>(&v); }, data);
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
    // TODO(yashkatariya): Plumb sharding or memory_kind here.
    TF_ASSIGN_OR_RETURN(
        auto ifrt_array,
        client->MakeArrayFromHostBuffer(
            ptr, ifrt_dtype, /*shape=*/ifrt::Shape({}), /*byte_strides=*/{},
            ifrt::SingleDeviceSharding::Create(to_device, to_memory_kind),
            ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/{}));
    return DevicePutResult(std::move(ifrt_array), /*weak_type=*/true);
  };
}

absl::StatusOr<DevicePutResultFn> HandlePythonInt(
    nb::handle obj, ifrt::Client* client, ifrt::Device* to_device,
    const DevicePutOptions& options, ifrt::MemoryKind to_memory_kind) {
  PrimitiveType type;
  std::variant<int64_t, int32_t> data;

  if (options.squash_64bit_types) {
    try {
      data.emplace<1>(nb::cast<int32_t>(obj));
    } catch (const std::exception& e) {
      return InvalidArgument(
          "Unable to convert Python scalar to %s. This most likely means the "
          "value (%s) overflows the range of the type.",
          PrimitiveType_Name(primitive_util::NativeToPrimitiveType<int32_t>()),
          nb::cast<absl::string_view>(nb::repr(obj)));
    }
    type = S32;
  } else {
    try {
      data.emplace<0>(nb::cast<int64_t>(obj));
    } catch (const std::exception& e) {
      return InvalidArgument(
          "Unable to convert Python scalar to %s. This most likely means the "
          "value (%s) overflows the range of the type.",
          PrimitiveType_Name(primitive_util::NativeToPrimitiveType<int64_t>()),
          nb::cast<absl::string_view>(nb::repr(obj)));
    }
    type = S64;
  }
  return [client, data, type, to_device,
          to_memory_kind]() -> absl::StatusOr<DevicePutResult> {
    const void* ptr = std::visit(
        [](const auto& v) { return static_cast<const void*>(&v); }, data);
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
    // TODO(yashkatariya): Plumb sharding or memory_kind here.
    TF_ASSIGN_OR_RETURN(
        auto ifrt_array,
        client->MakeArrayFromHostBuffer(
            ptr, ifrt_dtype, /*shape=*/xla::ifrt::Shape({}),
            /*byte_strides=*/{},
            ifrt::SingleDeviceSharding::Create(to_device, to_memory_kind),
            ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));
    return DevicePutResult(std::move(ifrt_array), /*weak_type=*/true);
  };
}

template <typename T, typename SquashedT = T>
absl::StatusOr<DevicePutResultFn> HandleNumpyScalar(
    nb::handle h, ifrt::Client* client, ifrt::Device* to_device,
    const DevicePutOptions& options, ifrt::MemoryKind to_memory_kind) {
  std::variant<T, SquashedT, void*> data;
  PrimitiveType type;
  // For extension types, ScalarAsCtype returns a pointer to the data.
  if (std::is_same<T, xla::s2>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = S2;
  } else if (std::is_same<T, xla::s4>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = S4;
  } else if (std::is_same<T, xla::u2>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = U2;
  } else if (std::is_same<T, xla::u4>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = U4;
  } else if (std::is_same<T, bfloat16>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = BF16;
  } else if (std::is_same<T, tsl::float8_e3m4>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E3M4;
  } else if (std::is_same<T, tsl::float8_e4m3>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3;
  } else if (std::is_same<T, tsl::float8_e4m3fn>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3FN;
  } else if (std::is_same<T, tsl::float8_e4m3b11fnuz>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3B11FNUZ;
  } else if (std::is_same<T, tsl::float8_e5m2>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E5M2;
  } else if (std::is_same<T, tsl::float8_e4m3fnuz>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E4M3FNUZ;
  } else if (std::is_same<T, tsl::float8_e5m2fnuz>()) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<2>());
    type = F8E5M2FNUZ;
  } else if (std::is_same<T, SquashedT>() || !options.squash_64bit_types) {
    PyArray_ScalarAsCtype(h.ptr(), &data.template emplace<0>());
    type = primitive_util::NativeToPrimitiveType<T>();
  } else {
    T value;
    PyArray_ScalarAsCtype(h.ptr(), &value);
    data.template emplace<1>(static_cast<SquashedT>(value));
    type = primitive_util::NativeToPrimitiveType<SquashedT>();
  }
  std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref;
  if (data.index() == 2) {
    py_buffer_ref =
        GlobalPyRefManager()->ManageReference(nb::cast<nb::object>(h));
  }
  return [client, data, py_buffer_ref, type, to_device,
          to_memory_kind]() mutable -> absl::StatusOr<DevicePutResult> {
    const void* ptr = std::visit(
        [](const auto& v) -> const void* {
          if constexpr (std::is_same_v<std::decay_t<decltype(v)>, void*>) {
            return v;
          } else {
            return static_cast<const void*>(&v);
          }
        },
        data);
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
    // TODO(yashkatariya): Plumb sharding or memory_kind here.
    TF_ASSIGN_OR_RETURN(
        auto ifrt_array,
        client->MakeArrayFromHostBuffer(
            ptr, ifrt_dtype, /*shape=*/xla::ifrt::Shape({}),
            /*byte_strides=*/{},
            ifrt::SingleDeviceSharding::Create(to_device, to_memory_kind),
            ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/
            [py_buffer_ref = std::move(
                 py_buffer_ref)]() { /* keeps py_buffer_ref alive */ }));
    return DevicePutResult(std::move(ifrt_array), /*weak_type=*/false);
  };
}

absl::StatusOr<DevicePutResultFn> HandleNumpyArray(
    nb::handle h, ifrt::Client* client, ifrt::Device* to_device,
    const DevicePutOptions& options, ifrt::MemoryKind to_memory_kind) {
  xla::nb_numpy_ndarray array = nb::cast<xla::nb_numpy_ndarray>(h);
  TF_ASSIGN_OR_RETURN(PrimitiveType type, DtypeToPrimitiveType(array.dtype()));

  PrimitiveType squashed_type;
  if (options.squash_64bit_types) {
    squashed_type = Squash64BitTypes(type);
    if (squashed_type != type) {
      TF_ASSIGN_OR_RETURN(xla::nb_dtype squashed_dtype,
                          PrimitiveTypeToNbDtype(squashed_type));
      array = nb::steal<xla::nb_numpy_ndarray>(PyArray_CastToType(
          reinterpret_cast<PyArrayObject*>(array.ptr()),
          reinterpret_cast<PyArray_Descr*>(squashed_dtype.release().ptr()),
          /*fortran=*/0));
    }
  } else {
    squashed_type = type;
  }

  absl::InlinedVector<int64_t, 4> dims(array.ndim());
  absl::InlinedVector<int64_t, 4> byte_strides(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims[i] = array.shape(i);
    byte_strides[i] = array.strides(i);
  }
  const void* data = array.data();
  std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref =
      GlobalPyRefManager()->ManageReference(std::move(array));
  return [client, data, squashed_type, dims = std::move(dims),
          byte_strides = std::move(byte_strides),
          py_buffer_ref = std::move(py_buffer_ref),
          allow_zero_copy = options.allow_zero_copy, to_device,
          to_memory_kind]() mutable -> absl::StatusOr<DevicePutResult> {
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(squashed_type));

    ifrt::Client::HostBufferSemantics host_buffer_semantics =
        ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall;
    std::function<void()> on_done_with_host_buffer;
    if (allow_zero_copy) {
      on_done_with_host_buffer =
          [py_buffer_ref{
              std::move(py_buffer_ref)}]() { /* keeps py_buffer_ref alive */ };
      host_buffer_semantics =
          ifrt::Client::HostBufferSemantics::kImmutableZeroCopy;
    }

    TF_ASSIGN_OR_RETURN(
        auto ifrt_array,
        client->MakeArrayFromHostBuffer(
            data, ifrt_dtype, ifrt::Shape(dims), byte_strides,
            xla::ifrt::SingleDeviceSharding::Create(to_device, to_memory_kind),
            host_buffer_semantics, std::move(on_done_with_host_buffer)));
    return DevicePutResult(std::move(ifrt_array), /*weak_type=*/false);
  };
}

absl::StatusOr<DevicePutResultFn> HandlePyArray(
    nb::handle obj, ifrt::Client* client, ifrt::Device* to_device,
    const DevicePutOptions& options, ifrt::MemoryKind to_memory_kind) {
  auto py_array = nb::borrow<PyArray>(obj);

  // We only allow single device case for PyArray in device put.
  if (py_array.num_shards() != 1) {
    return InvalidArgument(
        "device_put expects an array with exactly one shard, got an array with "
        "with %d shards.",
        py_array.num_shards());
  }

  ifrt::Array* ifrt_array = py_array.ifrt_array();
  if (ifrt_array == nullptr) {
    return InvalidArgument("Array has been deleted.");
  }

  // Fallback to python for non-matching clients or pmap sharding.
  if (py_array.sharding().type().ptr() == jax::PmapSharding::type().ptr() ||
      ifrt_array->sharding().devices()->devices().front()->client() !=
          to_device->client()) {
    return HandleNumpyArray(obj.attr("_value"), client, to_device, options,
                            to_memory_kind);
  }

  if (ifrt_array->sharding().devices()->devices().front() == to_device &&
      (!to_memory_kind.memory_kind().has_value() ||
       !ifrt_array->sharding().memory_kind().memory_kind().has_value() ||
       ifrt_array->sharding().memory_kind() == to_memory_kind)) {
    DevicePutResult result(tsl::FormRef(ifrt_array), py_array.weak_type(),
                           /*owning_pybuffer=*/nb::borrow<nb::object>(obj));
    return [result = std::move(result)]() mutable { return std::move(result); };
  } else {
    return [ifrt_array = tsl::FormRef(ifrt_array), to_device, to_memory_kind,
            owning_pybuffer = py_array.weak_type()]() mutable
           -> absl::StatusOr<DevicePutResult> {
      auto* ifrt_client = ifrt_array->client();
      TF_ASSIGN_OR_RETURN(
          auto copied_ifrt_arrays,
          ifrt_client->CopyArrays(absl::MakeSpan(&ifrt_array, 1),
                                  ifrt::BasicDeviceList::Create({to_device}),
                                  to_memory_kind,
                                  ifrt::ArrayCopySemantics::kReuseInput));
      return DevicePutResult(std::move(copied_ifrt_arrays[0]),
                             std::move(owning_pybuffer));
    };
  }
}

}  // namespace

absl::StatusOr<DevicePutResultFn> DevicePut(nb::handle arg,
                                            ifrt::Client* client,
                                            ifrt::Device* to_device,
                                            const DevicePutOptions& options,
                                            ifrt::MemoryKind to_memory_kind) {
  tsl::profiler::TraceMe traceme("DevicePut");
  static const absl::flat_hash_map<PyObject*, DevicePutFunc>* const handlers =
      [] {
        auto p = new absl::flat_hash_map<PyObject*, DevicePutFunc>();
        const NumpyScalarTypes& dtypes = GetNumpyScalarTypes();
        // Python scalar types.
        static_assert(sizeof(bool) == 1,
                      "Conversion code assumes bool is 1 byte");
        (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] =
            HandlePythonScalar<bool, bool>;
        (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = HandlePythonInt;
        (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] =
            HandlePythonScalar<double, float>;
        (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] =
            HandlePythonScalar<complex128, complex64>;

        (*p)[reinterpret_cast<PyObject*>(&PyArray_Type)] = HandleNumpyArray;

        // Numpy scalar types. For some of them, we share the handler with
        // Python types (np_int64, np_float64, np_complex128).
        (*p)[dtypes.np_bool.ptr()] = HandleNumpyScalar<bool>;
        (*p)[dtypes.np_int4.ptr()] = HandleNumpyScalar<xla::s4>;
        if (dtypes.np_int2.has_value()) {
          (*p)[dtypes.np_int2->ptr()] = HandleNumpyScalar<xla::s2>;
        }
        (*p)[dtypes.np_int8.ptr()] = HandleNumpyScalar<int8_t>;
        (*p)[dtypes.np_int16.ptr()] = HandleNumpyScalar<int16_t>;
        (*p)[dtypes.np_int32.ptr()] = HandleNumpyScalar<int32_t>;
        (*p)[dtypes.np_int64.ptr()] = HandleNumpyScalar<int64_t, int32_t>;
        if (dtypes.np_uint2.has_value()) {
          (*p)[dtypes.np_uint2->ptr()] = HandleNumpyScalar<xla::u2>;
        }
        (*p)[dtypes.np_uint4.ptr()] = HandleNumpyScalar<xla::u4>;
        (*p)[dtypes.np_uint8.ptr()] = HandleNumpyScalar<uint8_t>;
        (*p)[dtypes.np_uint16.ptr()] = HandleNumpyScalar<uint16_t>;
        (*p)[dtypes.np_uint32.ptr()] = HandleNumpyScalar<uint32_t>;
        (*p)[dtypes.np_uint64.ptr()] = HandleNumpyScalar<uint64_t, uint32_t>;
        if (dtypes.np_float8_e3m4.has_value()) {
          (*p)[dtypes.np_float8_e3m4->ptr()] =
              HandleNumpyScalar<tsl::float8_e3m4>;
        }
        if (dtypes.np_float8_e4m3.has_value()) {
          (*p)[dtypes.np_float8_e4m3->ptr()] =
              HandleNumpyScalar<tsl::float8_e4m3>;
        }
        (*p)[dtypes.np_float8_e4m3fn.ptr()] =
            HandleNumpyScalar<tsl::float8_e4m3fn>;
        (*p)[dtypes.np_float8_e4m3b11fnuz.ptr()] =
            HandleNumpyScalar<tsl::float8_e4m3b11fnuz>;
        (*p)[dtypes.np_float8_e5m2.ptr()] = HandleNumpyScalar<tsl::float8_e5m2>;
        (*p)[dtypes.np_float8_e4m3fnuz.ptr()] =
            HandleNumpyScalar<tsl::float8_e4m3fnuz>;
        (*p)[dtypes.np_float8_e5m2fnuz.ptr()] =
            HandleNumpyScalar<tsl::float8_e5m2fnuz>;
        (*p)[dtypes.np_bfloat16.ptr()] = HandleNumpyScalar<bfloat16>;
        (*p)[dtypes.np_float16.ptr()] = HandleNumpyScalar<half>;
        (*p)[dtypes.np_float32.ptr()] = HandleNumpyScalar<float>;
        (*p)[dtypes.np_float64.ptr()] = HandleNumpyScalar<double, float>;
        (*p)[dtypes.np_complex64.ptr()] = HandleNumpyScalar<complex64>;
        (*p)[dtypes.np_complex128.ptr()] =
            HandleNumpyScalar<complex128, complex64>;
        static_assert(sizeof(long long) == sizeof(int64_t),  // NOLINT
                      "long long must be the same size as int64_t");
        (*p)[dtypes.np_longlong.ptr()] = HandleNumpyScalar<int64_t, int32_t>;
        static_assert(sizeof(int) == sizeof(int32_t),
                      "int must be the same size as int32_t");
        (*p)[dtypes.np_intc.ptr()] = HandleNumpyScalar<int32_t>;

        return p;
      }();

  if (arg.type().ptr() == PyArray::type().ptr()) {
    auto array = nb::borrow<PyArray>(arg);
    return HandlePyArray(arg, client, to_device, options, to_memory_kind);
  }

  auto res = handlers->find(arg.type().ptr());
  if (res == handlers->end()) {
    for (auto base_class : arg.type().attr("__mro__")) {
      res = handlers->find(base_class.ptr());
      if (res != handlers->end()) {
        return res->second(arg, client, to_device, options, to_memory_kind);
      }
    }
    return InvalidArgument(
        "%s", absl::StrCat(
                  "Not supported: The C++ jax jit execution path, only accepts "
                  "DeviceArray, Numpy arrays scalars of supported types "
                  "(see implementation), or Python scalars. Got type ",
                  nb::cast<absl::string_view>(nb::str(arg.type()))));
  }
  return res->second(arg, client, to_device, options, to_memory_kind);
}

bool IsFloat0(xla::nb_numpy_ndarray arg) {
  static const auto* dtypes_module =
      new nb::module_(nb::module_::import_("jax.dtypes"));
  static const auto* float0_dtype =
      new nb::handle(dtypes_module->attr("float0"));
  return float0_dtype->is(arg.attr("dtype"));
}

std::string PyArgSignature::DebugString() const {
  std::string result = "";
  if (weak_type) {
    absl::StrAppend(&result, "weak_");
  }
  absl::StrAppend(&result, xla::PrimitiveType_Name(dtype));
  absl::StrAppend(&result, "[", absl::StrJoin(shape, ","), "]");
  return result;
}

using ToPyArgSignatureHandler =
    std::function<absl::StatusOr<PyArgSignature>(nb::handle, bool)>;

absl::StatusOr<PyArgSignature> PyArgSignatureOfValue(nb::handle arg,
                                                     bool jax_enable_x64) {
  static const absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>* const
      handlers = [] {
        auto p = new absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>();

        const NumpyScalarTypes& dtypes = GetNumpyScalarTypes();

        // The 4 Python native types.
        ToPyArgSignatureHandler bool_handler =
            [](nb::handle, bool) -> absl::StatusOr<PyArgSignature> {
          return PyArgSignature(PrimitiveType::PRED, {}, true);
        };
        ToPyArgSignatureHandler int_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // TODO(phawkins): we should consider checking for integer overflow.
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::S64, {}, true);
          } else {
            return PyArgSignature(PrimitiveType::S32, {}, true);
          }
        };
        ToPyArgSignatureHandler float_handler =
            [&dtypes](nb::handle h,
                      bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // Only Python native types has a True weak_type.
          bool weak_type = !nb::isinstance(h, dtypes.np_float64);
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::F64, {}, weak_type);
          } else {
            return PyArgSignature(PrimitiveType::F32, {}, weak_type);
          }
        };
        ToPyArgSignatureHandler complex_handler =
            [&dtypes](nb::handle h,
                      bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // Note that this branch is also taken  for np.complex128:
          // isinstance(np.complex128(3), complex) returns True
          // isinstance(np.complex64(3), complex) returns False
          bool weak_type = !nb::isinstance(h, dtypes.np_complex128);
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::C128, {}, weak_type);
          } else {
            return PyArgSignature(PrimitiveType::C64, {}, weak_type);
          }
        };

        (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] = bool_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = int_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] = float_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] = complex_handler;

        ToPyArgSignatureHandler numpy_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          xla::nb_numpy_ndarray numpy_array =
              nb::cast<xla::nb_numpy_ndarray>(h);
          TF_ASSIGN_OR_RETURN(PrimitiveType dtype,
                              DtypeToPrimitiveType(numpy_array.dtype()));
          if (!jax_enable_x64) {
            dtype = Squash64BitTypes(dtype);
          }
          // We use reinterpret_cast<> to defend against environments where
          // ssize_t may not be precisely the same type as int64_t, even if it
          // is the same size (long vs long long).
          static_assert(sizeof(int64_t) == sizeof(ssize_t),
                        "Code assumes ssize_t is the same as int64_t");
          return PyArgSignature(
              dtype,
              absl::MakeConstSpan(
                  reinterpret_cast<const int64_t*>(numpy_array.shape()),
                  numpy_array.ndim()),
              /*weak_type=*/false);
        };
        (*p)[reinterpret_cast<PyObject*>(&PyArray_Type)] = numpy_handler;

        ToPyArgSignatureHandler np_uint64_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::U64, {}, /*weak_type=*/false);
          } else {
            return PyArgSignature(PrimitiveType::U32, {}, /*weak_type=*/false);
          }
        };
        ToPyArgSignatureHandler np_int_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::S64, {}, /*weak_type=*/false);
          } else {
            return PyArgSignature(PrimitiveType::S32, {}, /*weak_type=*/false);
          }
        };
        ToPyArgSignatureHandler numpy_array_handler =
            [](nb::handle h,
               bool jax_enable_x64) -> absl::StatusOr<PyArgSignature> {
          // This block deals with all numpy scalar types, except for int64_dt,
          // float64_dt and complex128_dt which are taken care of in previous if
          // blocks.
          TF_ASSIGN_OR_RETURN(auto dtype,
                              DtypeToPrimitiveType(h.attr("dtype")));
          return PyArgSignature(dtype, {}, /*weak_type=*/false);
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
        // TODO: Uncomment once the minimum ml_dtypes in JAX is >= 0.5.0.
        // (*p)[dtypes.np_float8_e3m4.ptr()] = numpy_array_handler;
        // (*p)[dtypes.np_float8_e4m3.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e4m3fn.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e4m3b11fnuz.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e5m2.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e4m3fnuz.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float8_e5m2fnuz.ptr()] = numpy_array_handler;
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

  if (arg.type().ptr() == PyArray::type().ptr()) {
    auto array = nb::borrow<PyArray>(arg);
    ifrt::Array* ifrt_array = array.ifrt_array();
    if (ifrt_array == nullptr) {
      return xla::InvalidArgument("Array has been deleted.");
    }
    TF_ASSIGN_OR_RETURN(auto primitive_type,
                        ifrt::ToPrimitiveType(ifrt_array->dtype()));
    return PyArgSignature(primitive_type, array.shape(), array.weak_type());
  }

  auto res = handlers->find(arg.type().ptr());
  if (res == handlers->end()) {
    // We attempt to look at the MRO classes
    for (auto base_class : arg.type().attr("__mro__")) {
      res = handlers->find(base_class.ptr());
      if (res != handlers->end()) {
        return res->second(arg, jax_enable_x64);
      }
    }
    return InvalidArgument(
        "%s",
        absl::StrCat("Not supported: The C++ ToPyArgSignature only accepts "
                     "Buffer/DeviceArray, Numpy "
                     "arrays scalars of supported types "
                     "(see implementation), or Python scalars. Got type ",
                     nb::cast<absl::string_view>(nb::str(arg.type()))));
  }
  return res->second(arg, jax_enable_x64);
}

}  // namespace xla
