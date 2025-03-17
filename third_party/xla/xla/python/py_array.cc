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

#include "xla/python/py_array.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/lru_cache.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/primitive_util.h"
#include "xla/python/guard_lib.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device.h"
#include "xla/python/py_device_list.h"
#include "xla/python/py_values.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/sharding.h"
#include "xla/python/to_ifrt_sharding.h"
#include "xla/python/traceback.h"
#include "xla/python/types.h"
#include "xla/python/util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/python/lib/core/numpy.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace nb = nanobind;

PjRtBuffer* GetPjrtBuffer(ifrt::Array* ifrt_array) {
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  return arr->pjrt_buffers().front().get();
}

absl::StatusOr<const Shape*> XlaDynamicShape(ifrt::Array* ifrt_array,
                                             std::optional<Shape>& scratch) {
  auto* pjrt_buffer = GetPjrtBuffer(ifrt_array);

  if (!scratch) {
    absl::Span<const int64_t> dims;
    std::optional<std::vector<int64_t>> logical_dims_storage;
    if (pjrt_buffer->has_dynamic_dimensions()) {
      {
        nb::gil_scoped_release gil_release;
        TF_ASSIGN_OR_RETURN(std::vector<int64_t> logical_dims,
                            pjrt_buffer->logical_dimensions());
        logical_dims_storage.emplace(std::move(logical_dims));
      }
      dims = *logical_dims_storage;
    } else {
      dims = pjrt_buffer->dimensions();
    }
    Shape shape = ShapeUtil::MakeShape(pjrt_buffer->element_type(), dims);
    // TODO(b/327524065): fix this
    *shape.mutable_layout() = pjrt_buffer->layout()->xla_layout();
    scratch = std::move(shape);
  }
  return &scratch.value();
}

tsl::RCReference<ifrt::Array> CreateIfRtArrayFromSingleDeviceShardedPyArrays(
    nb_dtype dtype, absl::Span<const int64_t> shape,
    absl::Span<const PyArray> py_arrays, const nb::object& sharding) {
  const ifrt::MemoryKind dst_memory_kind = xla::GetMemoryKind(sharding);

  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(py_arrays.size());
  absl::InlinedVector<ifrt::Device*, 1> devices;
  devices.reserve(py_arrays.size());
  absl::flat_hash_set<ifrt::Device*> device_set;
  device_set.reserve(py_arrays.size());
  std::vector<ifrt::Shape> shapes;
  shapes.reserve(py_arrays.size());

  auto sharding_device_list = xla::GetIfrtDeviceList(sharding);
  if (!sharding_device_list.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(sharding_device_list.status().ToString().c_str());
  }
  ifrt::Device* device = sharding_device_list.value()->devices().front();

  // TODO(hyeontaek): Canonicalize every `ifrt::MemoryKind` at creation time to
  // skip canonicalization here once JAX begins to do it for JAX shardings.
  const ifrt::MemoryKind canonical_dst_memory_kind =
      ifrt::CanonicalizeMemoryKind(dst_memory_kind, device);
  for (const auto& py_array : py_arrays) {
    if (py_array.num_shards() != 1) {
      throw nb::value_error(
          absl::StrFormat(
              "When making an array from single-device arrays the input arrays "
              "must have one shard each. An argument array had %d shard(s).",
              py_array.num_shards())
              .c_str());
    }
    ifrt_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
    ifrt::Device* const device =
        ifrt_arrays.back()->sharding().devices()->devices().front();
    devices.push_back(device);
    device_set.insert(device);
    shapes.push_back(ifrt_arrays.back()->shape());
    if (canonical_dst_memory_kind !=
        ifrt::CanonicalizeMemoryKind(
            ifrt_arrays.back()->sharding().memory_kind(), device)) {
      throw nb::value_error(
          absl::StrFormat(
              "Memory kind mismatch with PjRtBuffers. Got sharding with "
              "memory kind '%v' and a buffer with memory_kind '%v'",
              dst_memory_kind, ifrt_arrays.back()->sharding().memory_kind())
              .c_str());
    }
  }
  ifrt::DeviceListRef device_list = device->client()->MakeDeviceList(devices);
  if (device_set.size() != device_list->size()) {
    throw nb::value_error(
        absl::StrFormat(
            "When making an array from single-device arrays, the input arrays "
            "must be from distinct devices, but got %v",
            *device_list)
            .c_str());
  }

  auto ifrt_dtype = DtypeToIfRtDType(dtype);
  if (!ifrt_dtype.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(ifrt_dtype.status().ToString().c_str());
  }

  absl::StatusOr<std::shared_ptr<const ifrt::Sharding>> ifrt_sharding =
      sharding.type().is(jax::PmapSharding::type())
          ? xla::GetIfrtConcreteSharding(sharding, ifrt::Shape(shape),
                                         std::move(shapes))
          : xla::GetIfrtHloSharding(sharding, ifrt::Shape(shape));
  if (!ifrt_sharding.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(ifrt_sharding.status().ToString().c_str());
  }
  // TODO(emilyaf): Always use `ifrt_dtype` once tokens are handled correctly.
  ifrt::DType array_dtype =
      ifrt_arrays.empty() ? ifrt_dtype.value() : ifrt_arrays[0]->dtype();
  absl::StatusOr<tsl::RCReference<ifrt::Array>> ifrt_array =
      device->client()->AssembleArrayFromSingleDeviceArrays(
          array_dtype, ifrt::Shape(shape), *std::move(ifrt_sharding),
          absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput,
          ifrt::SingleDeviceShardSemantics::kAddressableShards);
  if (!ifrt_array.ok()) {
    // TODO(hyeontaek): Return a absl::Status.
    throw nb::value_error(ifrt_array.status().ToString().c_str());
  }
  return *std::move(ifrt_array);
}

struct PyArrayObject {
  PyObject_HEAD;
#if PY_VERSION_HEX < 0x030C0000
  PyObject* weakrefs;
  PyObject* dict;
#endif  // PY_VERSION_HEX < 0x030B0000
  bool initialized;
  alignas(PyArray::Storage) char array_storage[sizeof(PyArray::Storage)];
};
static_assert(std::is_standard_layout<PyArrayObject>::value);

PyArray::Storage* GetPyArrayStorageFromObject(PyArrayObject* py_array_object) {
  return std::launder(
      reinterpret_cast<PyArray::Storage*>(py_array_object->array_storage));
}

extern "C" PyObject* PyArray_tp_new(PyTypeObject* type, PyObject*, PyObject*) {
  PyObject* self = type->tp_alloc(type, 0);
  auto* obj = reinterpret_cast<PyArrayObject*>(self);
  obj->initialized = false;
  return self;
}

extern "C" void PyArray_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  auto* obj = reinterpret_cast<PyArrayObject*>(self);

  if (obj->initialized) {
    GetPyArrayStorageFromObject(obj)->~PyArray_Storage();
  }

  PyObject_ClearWeakRefs(self);
#if PY_VERSION_HEX < 0x030C0000
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_CLEAR(dict);
#elif PY_VERSION_HEX < 0x030D0000
  _PyObject_ClearManagedDict(self);
#else
  PyObject_ClearManagedDict(self);
#endif  // PY_VERSION_HEX < 0x030C0000

  tp->tp_free(self);
  Py_DECREF(tp);
}

// dynamic_attr: Allow the garbage collector to traverse the internal instance
// `__dict__`.
extern "C" int PyArray_tp_traverse(PyObject* self, visitproc visit, void* arg) {
#if PY_VERSION_HEX < 0x030C0000
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_VISIT(dict);
#elif PY_VERSION_HEX < 0x030D0000
  _PyObject_VisitManagedDict(self, visit, arg);
#else
  PyObject_VisitManagedDict(self, visit, arg);
#endif  // PY_VERSION_HEX < 0x030C0000
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
  Py_VISIT(Py_TYPE(self));
  return 0;
}

// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" int PyArray_tp_clear(PyObject* self) {
  switch (auto guard_level = jax::GetGarbageCollectArrayGuard(); guard_level) {
    case jax::GarbageCollectionGuardLevel::kAllow:
      break;
    case jax::GarbageCollectionGuardLevel::kLog:
    case jax::GarbageCollectionGuardLevel::kFatal: {
      auto* obj = reinterpret_cast<PyArrayObject*>(self);
      std::string traceback_str;
      if (obj->initialized) {
        auto traceback = GetPyArrayStorageFromObject(obj)->traceback;
        if (traceback.has_value()) {
          traceback_str = traceback.value()->ToString();
        }
      }
      auto error_msg = absl::StrCat(
          "`jax.Array` was deleted by the Python garbage collector "
          "instead of reference counting. Break the reference cycle "
          "that delays the deletion of this `jax.Array` to avoid hogging "
          "memory. Traceback: \n",
          traceback_str.empty() ? "not available" : traceback_str);
      if (guard_level == jax::GarbageCollectionGuardLevel::kFatal) {
        Py_FatalError(error_msg.c_str());
      } else {
        PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        PyErr_Print();
        PyErr_Clear();
      }
      break;
    }
  }
#if PY_VERSION_HEX < 0x030C0000
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_CLEAR(dict);
#elif PY_VERSION_HEX < 0x030D0000
  _PyObject_ClearManagedDict(self);
#else
  PyObject_ClearManagedDict(self);
#endif  // PY_VERSION_HEX < 0x030C0000
  return 0;
}

template <typename... Args>
PyArray::Storage* Construct(PyArrayObject* self, Args&&... args) {
  PyArray::Storage* out =
      new (self->array_storage) PyArray::Storage(std::forward<Args>(args)...);
  self->initialized = true;
  return out;
}

struct ShapedArrayCacheKey {
  std::vector<int64_t> dims;
  ifrt::DType dtype{ifrt::DType::kInvalid};
  bool weak_type;

  template <typename H>
  friend H AbslHashValue(H h, const ShapedArrayCacheKey& value) {
    return H::combine(std::move(h), value.dims, value.dtype, value.weak_type);
  }
  bool operator==(const ShapedArrayCacheKey& other) const {
    return dims == other.dims && dtype == other.dtype &&
           weak_type == other.weak_type;
  }
};

// Constructing ShapedArrays has gotten slow. Cache it.
nb::object MakeShapedArrayCached(const ShapedArrayCacheKey& key) {
  using CacheT =
      LRUCache<ShapedArrayCacheKey, std::shared_ptr<std::optional<nb::object>>>;
  static nb::ft_mutex mu;
  static auto* lru_list = new CacheT::LRUList(4096);
  static auto* cache = new CacheT(lru_list);

  static const nb::object* shaped_array = []() -> nb::object* {
    nb::object jax_core;
    try {
      jax_core = nb::module_::import_("jax.core");
    } catch (nb::python_error& e) {
      return nullptr;
    }
    return new nb::object(jax_core.attr("ShapedArray"));
  }();
  if (!shaped_array) {
    return nb::none();
  }

  nb::ft_lock_guard lock(mu);
  auto value =
      cache->GetOrCreateIfAbsent(key, [](const ShapedArrayCacheKey& key) {
        return std::make_shared<std::optional<nb::object>>();
      });

  if (!value->has_value()) {
    nb_dtype dtype =
        IfrtDtypeToDtypeWithTokenCanonicalization(key.dtype).value();
    nb::object aval = (*shaped_array)(
        SpanToNbTuple(absl::Span<const int64_t>(
            key.dtype.kind() == ifrt::DType::kToken ? std::vector<int64_t>{0}
                                                    : key.dims)),
        dtype, key.weak_type);
    *value = aval;
    return aval;
  }
  return **value;
}

// Grouping key used by BatchedCopyToDeviceWithSharding.
// Defined outside of the function as required by templatized function
// `AbslHashValue`.
struct BatchedCopyToDeviceWithShardingKey {
  ifrt::DeviceListRef src_devices;
  ifrt::MemoryKind src_memory_kind;
  ifrt::DeviceListRef dst_devices;
  ifrt::MemoryKind dst_memory_kind;
  ifrt::ArrayCopySemantics array_copy_semantics;

  bool operator==(const BatchedCopyToDeviceWithShardingKey& other) const {
    return *src_devices == *other.src_devices &&
           src_memory_kind == other.src_memory_kind &&
           *dst_devices == *other.dst_devices &&
           dst_memory_kind == other.dst_memory_kind &&
           array_copy_semantics == other.array_copy_semantics;
  }

  template <typename H>
  friend H AbslHashValue(H h, const BatchedCopyToDeviceWithShardingKey& key) {
    return H::combine(std::move(h), key.src_devices, key.src_memory_kind,
                      key.dst_devices, key.dst_memory_kind,
                      key.array_copy_semantics);
  }
};

}  // namespace

PyArray_Storage::PyArray_Storage(
    nb::object aval, bool weak_type, xla::nb_dtype dtype,
    std::vector<int64_t> shape, nb::object sharding, bool committed,
    nb_class_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
    tsl::RCReference<ifrt::Array> ifrt_array, xla::PjRtFuture<> result_status)
    : aval(std::move(aval)),
      weak_type(weak_type),
      dtype(std::move(dtype)),
      shape(std::move(shape)),
      sharding(std::move(sharding)),
      committed(committed),
      py_client(std::move(py_client)),
      traceback(std::move(traceback)),
      ifrt_array(std::move(ifrt_array)),
      result_status(std::move(result_status)) {
  static_assert(PyClient::kNumArraysShards <
                std::numeric_limits<uint8_t>::max());
  thread_id_bucket = std::hash<std::thread::id>()(std::this_thread::get_id()) %
                     PyClient::kNumArraysShards;

  PyClient::ArraysShard& shard = this->py_client->arrays_[thread_id_bucket];
  nanobind::ft_lock_guard lock(shard.mutex);
  next = shard.arrays;
  shard.arrays = this;
  if (next) {
    next->prev = this;
  }
  prev = nullptr;
}

void PyInit_helper(PyArray self, nb::object aval, nb::object sharding,
                   absl::Span<const PyArray> py_arrays, bool committed) {
  auto dtype = nb::cast<nb_dtype>(aval.attr("dtype"));
  auto shape = nb::cast<std::vector<int64_t>>(aval.attr("shape"));
  auto py_device_list = nb::cast<const jax::PyDeviceList*>(
      sharding.attr("_internal_device_list"));
  nb_class_ptr<PyClient> py_client = py_device_list->py_client();
  auto ifrt_array = CreateIfRtArrayFromSingleDeviceShardedPyArrays(
      dtype, shape, py_arrays, sharding);
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()), aval,
            nb::cast<bool>(aval.attr("weak_type")), std::move(dtype),
            std::move(shape), std::move(sharding), committed, py_client,
            Traceback::Get(), std::move(ifrt_array), xla::PjRtFuture<>());
}

void PyArray::PyInit(PyArray self, nb::object aval, nb::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks) {
  if (skip_checks) {
    PyInit_helper(self, aval, sharding, py_arrays, committed);
  } else {
    nb::object rearranged_arrays =
        self.CheckAndRearrange(py_arrays, sharding, aval);
    auto rearranged_py_arrays =
        nb::cast<std::vector<PyArray>>(rearranged_arrays);
    PyInit_helper(self, aval, sharding, rearranged_py_arrays, committed);
  }
}

PyArray PyArray::MakeFromSingleDeviceArray(
    nb_class_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
    tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type, bool committed,
    xla::PjRtFuture<> result_status) {
  if (!llvm::isa<ifrt::SingleDeviceSharding>(ifrt_array->sharding())) {
    throw XlaRuntimeError(
        InvalidArgument("Constructing single device jax.Array from non-single "
                        "device ifrt array."));
  }
  auto shape_span = ifrt_array->shape().dims();
  ShapedArrayCacheKey key;
  key.dtype = ifrt_array->dtype();
  key.dims = key.dtype.kind() == ifrt::DType::kToken
                 ? std::vector<int64_t>{0}
                 : std::vector<int64_t>(shape_span.begin(), shape_span.end());
  key.weak_type = weak_type;
  auto aval = MakeShapedArrayCached(key);
  auto dtype = IfrtDtypeToDtypeWithTokenCanonicalization(key.dtype).value();
  const ifrt::MemoryKind memory_kind = ifrt_array->sharding().memory_kind();
  nb::object py_memory_kind =
      (memory_kind.memory_kind().has_value())
          ? nb::object(nb::str(memory_kind.memory_kind()->data(),
                               memory_kind.memory_kind()->size()))
          : nb::none();
  nb::object sharding = make_nb_class<jax::SingleDeviceSharding>(
      py_client, ifrt_array->sharding().devices(), std::move(py_memory_kind));
  return PyArray(std::move(aval), weak_type, dtype, std::move(key.dims),
                 std::move(sharding), std::move(py_client),
                 std::move(traceback), std::move(ifrt_array), committed,
                 /*skip_checks=*/true, std::move(result_status));
}

PyArray PyArray::MakeFromIfrtArrayAndSharding(
    nb_class_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
    tsl::RCReference<ifrt::Array> ifrt_array, nb::object sharding,
    bool weak_type, bool committed, bool skip_checks) {
  auto shape_span = ifrt_array->shape().dims();
  ShapedArrayCacheKey key;
  key.dtype = ifrt_array->dtype();
  key.dims = key.dtype.kind() == ifrt::DType::kToken
                 ? std::vector<int64_t>{0}
                 : std::vector<int64_t>(shape_span.begin(), shape_span.end());
  key.weak_type = weak_type;
  auto aval = MakeShapedArrayCached(key);
  auto dtype = IfrtDtypeToDtypeWithTokenCanonicalization(key.dtype).value();
  return PyArray(std::move(aval), weak_type, dtype, std::move(key.dims),
                 std::move(sharding), std::move(py_client),
                 std::move(traceback), std::move(ifrt_array), committed,
                 skip_checks);
}

PyArrayResultHandler::PyArrayResultHandler(nb::object aval, nb::object sharding,
                                           bool committed, bool skip_checks)
    : aval_(std::move(aval)),
      sharding_(std::move(sharding)),
      committed_(committed),
      skip_checks_(skip_checks) {
  weak_type_ = nb::cast<bool>(aval_.attr("weak_type"));
  dtype_ = nb::cast<nb_dtype>(aval_.attr("dtype"));
  shape_ = nb::cast<std::vector<int64_t>>(aval_.attr("shape"));
}

PyArray PyArrayResultHandler::Call(absl::Span<const PyArray> py_arrays) const {
  auto py_device_list = jax::GetPyDeviceList(sharding_);
  if (!py_device_list.ok()) {
    throw nb::value_error(
        absl::StrCat("Failed to get py device list from sharding: ",
                     py_device_list.status().ToString())
            .c_str());
  }
  return Call(py_device_list.value()->py_client(),
              CreateIfRtArrayFromSingleDeviceShardedPyArrays(
                  dtype_, shape_, py_arrays, sharding_),
              xla::PjRtFuture<>());
}

PyArray PyArrayResultHandler::Call(nb_class_ptr<PyClient> py_client,
                                   tsl::RCReference<ifrt::Array> ifrt_array,
                                   xla::PjRtFuture<> result_status) const {
  return PyArray(aval_, weak_type_, dtype_, shape_, sharding_,
                 std::move(py_client), Traceback::Get(), std::move(ifrt_array),
                 committed_, skip_checks_, std::move(result_status));
}

PyArray PyArrayResultHandler::Call(PyArray py_array) const {
  return Call(py_array.py_client(), tsl::FormRef(py_array.ifrt_array()),
              xla::PjRtFuture<>());
}

PyArray::PyArray(nb::object aval, bool weak_type, nb_dtype dtype,
                 std::vector<int64_t> shape, nb::object sharding,
                 nb_class_ptr<PyClient> py_client,
                 std::optional<nb_traceback> traceback,
                 tsl::RCReference<ifrt::Array> ifrt_array, bool committed,
                 bool skip_checks, xla::PjRtFuture<> result_status) {
  auto* self =
      PyArray_tp_new(reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr);
  m_ptr = self;
  Construct(reinterpret_cast<PyArrayObject*>(self), std::move(aval), weak_type,
            std::move(dtype), std::move(shape), std::move(sharding), committed,
            std::move(py_client), std::move(traceback), std::move(ifrt_array),
            std::move(result_status));

  if (!skip_checks) {
    this->attr("_arrays") = this->attr("_check_and_rearrange")(
        this->attr("_arrays"), this->attr("_sharding"), this->attr("aval"));
  }
}

PyArray::Storage& PyArray::GetStorage() {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

const PyArray::Storage& PyArray::GetStorage() const {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

nb::object PyArray::CheckAndRearrange(const absl::Span<const PyArray> py_arrays,
                                      const nb::object sharding,
                                      const nb::object aval) {
  return this->attr("_check_and_rearrange")(py_arrays, sharding, aval);
}

void PyArray::SetIfrtArray(tsl::RCReference<ifrt::Array> ifrt_array) {
  GetStorage().ifrt_array = std::move(ifrt_array);
}

const std::vector<PyArray>& PyArray::py_arrays_cached() {
  auto& py_arrays = this->py_arrays();

  if (py_arrays.empty()) {
    auto ifrt_arrays = ifrt_array()->DisassembleIntoSingleDeviceArrays(
        ifrt::ArrayCopySemantics::kReuseInput,
        ifrt::SingleDeviceShardSemantics::kAddressableShards);
    if (!ifrt_arrays.ok()) {
      throw nb::value_error(
          absl::StrCat("Failed to disassemble into single-device arrays: ",
                       ifrt_arrays.status().ToString())
              .c_str());
    }
    py_arrays.reserve(ifrt_arrays->size());
    for (auto& ifrt_array : *ifrt_arrays) {
      py_arrays.push_back(PyArray::MakeFromSingleDeviceArray(
          py_client(), traceback(), std::move(ifrt_array), weak_type(),
          committed(), result_status()));
    }
  }

  return py_arrays;
}

nb::object PyArray::arrays() {
  // For performance, we only keep pjrt buffers by default. But on python side
  // "_arrays" returns PyArrays instead, and subsequent calls to "_arrays"
  // should return the same PyArrays (to avoid duplicate device to host
  // transfers). So we create PyArrays the first time it is called and reuse
  // them later.
  if (ifrt_array() == nullptr || ifrt_array()->IsDeleted()) return nb::none();

  if (llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    std::vector<PyArray> py_arrays;
    py_arrays.push_back(*this);
    return nb::cast(py_arrays);
  }

  return nb::cast(py_arrays_cached());
}

absl::Status PyArray::set_arrays(nb::object obj) {
  if (obj.is_none()) {
    SetIfrtArray(tsl::RCReference<ifrt::Array>());
    py_arrays().clear();
    return absl::OkStatus();
  }

  if (!nb::isinstance<nb::list>(obj)) {
    return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                           nb::cast<absl::string_view>(nb::str(obj.type())));
  }

  nb::list list(obj);

  if (list.size() == 0) return absl::OkStatus();

  SetIfrtArray(tsl::RCReference<ifrt::Array>());
  py_arrays().clear();
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(list.size());
  absl::InlinedVector<ifrt::Device*, 1> devices;
  devices.reserve(list.size());
  std::vector<ifrt::Shape> shapes;
  shapes.reserve(list.size());
  for (nb::handle obj : list) {
    if (obj.type().is(PyArray::type())) {
      auto py_array = nb::borrow<PyArray>(obj);
      if (py_array.py_client().get() != py_client().get()) {
        return InvalidArgument("Client mismatch when assigning to _arrays.");
      }
      if (py_array.num_shards() != 1) {
        return InvalidArgument("Wrong number of shards: %d",
                               py_array.num_shards());
      }
      ifrt_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
      devices.push_back(
          ifrt_arrays.back()->sharding().devices()->devices().front());
      shapes.push_back(ifrt_arrays.back()->shape());
    } else {
      return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                             nb::cast<absl::string_view>(nb::str(obj.type())));
    }
  }
  const ifrt::MemoryKind first_memory_kind =
      ifrt_arrays.front()->sharding().memory_kind();
  // TODO(hyeontaek): Canonicalize every `ifrt::MemoryKind` at creation time to
  // skip canonicalization here once JAX begins to do it for JAX shardings.
  const ifrt::MemoryKind canonical_first_memory_kind =
      ifrt::CanonicalizeMemoryKind(
          first_memory_kind,
          ifrt_arrays.front()->sharding().devices()->devices().front());
  for (const auto& ifrt_array : ifrt_arrays) {
    if (canonical_first_memory_kind !=
        ifrt::CanonicalizeMemoryKind(
            ifrt_array->sharding().memory_kind(),
            ifrt_array->sharding().devices()->devices().front())) {
      throw nb::value_error(
          absl::StrFormat(
              "Memory kind mismatch between single-device arrays. Got one "
              "array with memory kind '%v' and another with memory_kind '%v'",
              first_memory_kind, ifrt_array->sharding().memory_kind())
              .c_str());
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto ifrt_sharding,
      sharding().type().is(jax::PmapSharding::type())
          ? xla::GetIfrtConcreteSharding(sharding(), ifrt::Shape(shape()),
                                         std::move(shapes))
          : xla::GetIfrtHloSharding(sharding(), ifrt::Shape(shape())));
  TF_ASSIGN_OR_RETURN(
      auto array,
      py_client()->ifrt_client()->AssembleArrayFromSingleDeviceArrays(
          ifrt::Shape(shape()), std::move(ifrt_sharding),
          absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput,
          ifrt::SingleDeviceShardSemantics::kAddressableShards));
  SetIfrtArray(std::move(array));
  return absl::OkStatus();
}

absl::StatusOr<PyArray> PyArray::FullyReplicatedShard() {
  auto& cached = GetStorage().fully_replicated_array;
  if (!cached.is_none()) {
    return nb::cast<PyArray>(cached);
  }

  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "FullyReplicatedShard() called on deleted or donated buffer");
  }

  TF_ASSIGN_OR_RETURN(auto fully_replicated_ifrt_shard,
                      ifrt_array()->FullyReplicatedShard(
                          ifrt::ArrayCopySemantics::kReuseInput));
  auto array = MakeFromSingleDeviceArray(
      py_client(), traceback(), std::move(fully_replicated_ifrt_shard),
      weak_type(), committed(), result_status());
  cached = array;
  return nb::cast<PyArray>(cached);
}

absl::Status PyArray::BlockUntilReady() const {
  nb::gil_scoped_release gil_release;
  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "BlockHostUntilReady() called on deleted or donated buffer");
  }
  ifrt::Array* ifrt_array = this->ifrt_array();
  return AwaitBuffersReady(absl::MakeConstSpan(&ifrt_array, 1));
}

absl::StatusOr<size_t> PyArray::GetOnDeviceSizeInBytes() {
  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "GetOnDeviceSizeInBytes() called on deleted or donated buffer");
  }

  TF_ASSIGN_OR_RETURN(size_t shard_size,
                      GetPjrtBuffer(ifrt_array())->GetOnDeviceSizeInBytes());
  return shard_size * nb::len(nb::object(sharding().attr("device_set")));
}

absl::Status PyArray::BlockUntilResultStatusIsReady() {
  auto& result_status = GetStorage().result_status;
  // If the result_status future is not valid, this result did not come directly
  // from a computation that returns tokens, so we don't wait for the status.
  if (!result_status.IsValid()) {
    return absl::OkStatus();
  }
  if (!result_status.IsReady()) {
    // Only release the gil if we need to Await().
    nb::gil_scoped_release release_gil;
    return result_status.Await();
  }
  return result_status.Await();
}

absl::StatusOr<std::pair<nb::object, bool>>
PyArray::SingleDeviceArrayToNumpyArrayDidCopy() {
  TF_ASSIGN_OR_RETURN(auto arr, FullyReplicatedShard());
  auto result = arr.GetStorage().host_value.AsNumPyArray(
      arr.GetStorage().dynamic_shape, arr.ifrt_array());
  TF_RETURN_IF_ERROR(arr.BlockUntilResultStatusIsReady());
  return result;
}

absl::StatusOr<nb::object> PyArray::SingleDeviceArrayToNumpyArray() {
  TF_ASSIGN_OR_RETURN(auto result, SingleDeviceArrayToNumpyArrayDidCopy());
  return result.first;
}

absl::Status PyArray::CopySingleDeviceArrayToHostAsync() {
  TF_ASSIGN_OR_RETURN(auto arr, FullyReplicatedShard());
  return arr.GetStorage().host_value.CopyToHostAsync(
      arr.GetStorage().dynamic_shape, arr.ifrt_array());
}

absl::StatusOr<PyArray> PyArray::AssertUnsharded(absl::string_view api) {
  if (ifrt_array() == nullptr) {
    return InvalidArgument("%s( called on deleted or donated buffer", api);
  }

  if (llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    return *this;
  }

  auto& py_arrays = py_arrays_cached();
  if (py_arrays.size() != 1) {
    return InvalidArgument("%s() is supported only for unsharded arrays.", api);
  }
  return py_arrays[0];
}

absl::StatusOr<std::uintptr_t> PyArray::UnsafeBufferPointer() {
  TF_ASSIGN_OR_RETURN(auto arr, AssertUnsharded("UnsafeBufferPointer"));

  return py_client()->pjrt_client()->UnsafeBufferPointer(
      GetPjrtBuffer(arr.ifrt_array()));
}

nb::dict PyArray::CudaArrayInterface() {
  auto arr_or_error = AssertUnsharded("UnsafeBufferPointer");
  if (!arr_or_error.ok()) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only supported for unsharded arrays.");
  }
  auto arr = *arr_or_error;

  ifrt::Array* ifrt_array = arr.ifrt_array();
  std::optional<Shape>& scratch = arr.GetStorage().dynamic_shape;
  auto* pjrt_buffer = GetPjrtBuffer(ifrt_array);
  if (pjrt_buffer->client()->platform_id() != CudaId()) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (pjrt_buffer->IsTuple()) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only defined for array buffers.");
  }

  switch (pjrt_buffer->element_type()) {
    case PrimitiveType::PRED:
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
    case PrimitiveType::F16:
    case PrimitiveType::F32:
    case PrimitiveType::F64:
    case PrimitiveType::C64:
    case PrimitiveType::C128:
      break;

    default:
      throw nb::attribute_error(
          absl::StrFormat(
              "__cuda_array_interface__ is not supported for %s buffers.",
              PrimitiveType_Name(pjrt_buffer->element_type()))
              .c_str());
  }

  nb::str typestr =
      ValueOrThrow(TypeDescriptorForPrimitiveType(pjrt_buffer->element_type()));

  // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
  Layout xla_layout = pjrt_buffer->layout()->xla_layout();
  if (!LayoutUtil::IsMonotonicWithDim0Major(xla_layout)) {
    throw nb::attribute_error(
        "__cuda_array_interface__ is only currently supported for "
        "buffers in row-major order.");
  }

  nb::dict result;
  const auto* dynamic_shape =
      ValueOrThrow(XlaDynamicShape(ifrt_array, scratch));
  result["shape"] = SpanToNbTuple(dynamic_shape->dimensions());
  result["typestr"] = std::move(typestr);
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold =
      ValueOrThrow(pjrt_buffer->AcquireExternalReference());
  const void* root_ptr =
      external_reference_hold->OpaqueDeviceMemoryDataPointer();
  nb::tuple data =
      nb::make_tuple(nb::int_(absl::bit_cast<std::uintptr_t>(root_ptr)),
                     nb::bool_(true) /* read-only */
      );
  result["data"] = std::move(data);
  result["version"] = nb::int_(2);
  return result;
}

absl::StatusOr<nb::object> CudaArrayInterfaceToBuffer(
    const nb::dict& cai, nb_class_ptr<PyClient> client,
    std::optional<int> device_id) {
  if (!cai.contains("data")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `data`");
  }
  if (!cai.contains("shape")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `shape`");
  }
  if (!cai.contains("typestr")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `typestr`");
  }
  if (!cai.contains("version")) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface does not define `version`");
  }
  auto version = nb::cast<int>(cai["version"]);
  if (version < 2 || version > 3) {
    LOG(WARNING) << "CUDA Array Interface version " << version
                 << " support is undefined";
  }
  auto data = nb::cast<nb::tuple>(cai["data"]);
  auto data_value = nb::cast<std::intptr_t>(data[0]);
  void* data_ptr = reinterpret_cast<void*>(data_value);
  auto dimensions = nb::cast<std::vector<int64_t>>(cai["shape"]);
  if (data_value == 0 && absl::c_find(dimensions, 0) == dimensions.end()) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface `data`(=NULL) and `shape`(no zero-valued "
        "dimensions) are inconsistent");
  }
  auto ndim = dimensions.size();
  TF_ASSIGN_OR_RETURN(
      PrimitiveType element_type,
      DtypeToPrimitiveType(nb_dtype::from_args(cai["typestr"])));

  if (!device_id.has_value()) {
    throw XlaRuntimeError(
        "This operation requires CUDA support from jaxlib or jax cuda plugin.");
  }
  TF_ASSIGN_OR_RETURN(auto device,
                      client->DeviceFromLocalHardwareId(*device_id));
  bool is_default_stream =
      data_value == 0 || version == 2 ||
      (version == 3 && (!cai.contains("stream") || cai["stream"].is_none()));
  TF_ASSIGN_OR_RETURN(
      std::intptr_t stream,
      ([is_default_stream, cai, device]() -> absl::StatusOr<std::intptr_t> {
        if (is_default_stream) {
          return device->GetStreamForExternalReadyEvents();
        } else {
          auto stream_ = nb::cast<std::intptr_t>(cai["stream"]);
          if (stream_ == 0) {
            return absl::InvalidArgumentError(
                "CUDA Array Interface does not allow zero stream value");
          }
          return stream_;
        }
      }()));

  std::vector<int64_t> minor_to_major(ndim);
  if (cai.contains("strides") && !cai["strides"].is_none() && data_value != 0) {
    std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
    auto strides = nb::cast<std::vector<int64_t>>(cai["strides"]);
    if (strides.size() != ndim) {
      return absl::InvalidArgumentError(
          "CUDA Array Interface `shape` and `strides` dimensionalities are "
          "inconsistent");
    }
    absl::c_sort(minor_to_major, [&](int a, int b) {
      // If two dimensions have the same stride, prefer the major-to-minor
      // interpretation of the ordering, since that's what JAX wants.
      return (strides[a] == strides[b] ? b < a : strides[a] < strides[b]);
    });
    int64_t stride = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
    for (int64_t d : minor_to_major) {
      if (dimensions[d] > 1 && strides[d] != stride) {
        return absl::UnimplementedError(absl::StrCat(
            "Only arrays with trivial (compact) striding are supported; "
            "i.e., arrays whose striding represents a transposition of the "
            "underlying buffer but not broadcasting. Dimensions were: [%s], "
            "strides were [%s].",
            absl::StrJoin(dimensions, ","), absl::StrJoin(strides, ",")));
      }
      stride *= dimensions[d];
    }
  } else {
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                                    minor_to_major);
  std::function<void()> on_delete_callback = []() {};
  auto* pjrt_device =
      llvm::dyn_cast_or_null<ifrt::PjRtDevice>(device->device());
  if (pjrt_device == nullptr) {
    return InvalidArgument(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  TF_RET_CHECK(pjrt_device->IsAddressable());
  TF_ASSIGN_OR_RETURN(
      auto pjrt_buffer,
      device->client()->pjrt_client()->CreateViewOfDeviceBuffer(
          static_cast<char*>(data_ptr), shape,
          *pjrt_device->pjrt_device()->default_memory_space(),
          on_delete_callback,
          stream <= 2 ? std::nullopt : std::make_optional(stream)));
  auto* ifrt_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(client->ifrt_client());
  if (ifrt_client == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  TF_ASSIGN_OR_RETURN(auto ifrt_array,
                      ifrt_client->CreatePjRtArray(std::move(pjrt_buffer)));
  return PyArray::MakeFromSingleDeviceArray(std::move(client), Traceback::Get(),
                                            std::move(ifrt_array), false, true);
}

absl::Status PyArray::Delete() {
  for (auto& arr : py_arrays()) {
    TF_RETURN_IF_ERROR(arr.Delete());
  }
  py_arrays().clear();
  if (ifrt_array() != nullptr) {
    // We do not wait for the deletion to complete here.
    //
    // (1) Skipping blocking does not affect the correctness of deletion as long
    // as the runtime preserves dispatch ordering of deletion w.r.t. other
    // operations.
    //
    // (2) Synchronously waiting for the deletion to complete is very expensive
    // when the deletion can return a status only after the underlying physical
    // buffer has been deleted or a request must be processed via RPC,
    // especially as this deletion is done per array.
    ifrt_array()->Delete();
    SetIfrtArray(tsl::RCReference<ifrt::Array>());
  }
  return absl::OkStatus();
}

bool PyArray::IsDeleted() const {
  if (ifrt_array() == nullptr) {
    return true;
  }

  return ifrt_array()->IsDeleted();
}

PyArray PyArray::Clone() const {
  auto array = tsl::FormRef(ifrt_array());
  auto* ifrt_client = py_client()->ifrt_client();
  tsl::RCReference<ifrt::Array> out =
      ifrt_client
          ->CopyArrays(absl::MakeSpan(&array, 1), /*devices=*/std::nullopt,
                       /*memory_kind=*/std::nullopt,
                       ifrt::ArrayCopySemantics::kReuseInput)
          .value()
          .front();
  return PyArray(aval(), weak_type(), dtype(),
                 std::vector<int64_t>(shape().begin(), shape().end()),
                 sharding(), py_client(), traceback(), std::move(out),
                 committed(), /*skip_checks=*/true, result_status());
}

nb::handle PyArray::Storage::AsHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(PyArrayObject, array_storage));
}

PyArray::Storage::~PyArray_Storage() {
  CHECK(PyGILState_Check());
  if (py_client) {
    PyClient::ArraysShard& shard = py_client->arrays_[thread_id_bucket];
    nanobind::ft_lock_guard lock(shard.mutex);
    if (shard.arrays == this) {
      shard.arrays = next;
    }
    if (prev) {
      prev->next = next;
    }
    if (next) {
      next->prev = prev;
    }
  }
  // Release GIL and then explicitly destroy `ifrt_array` to prevent deadlock on
  // CPU backend caused by interactions between argument donations and host
  // callbacks.
  nb::gil_scoped_release gil_release;
  ifrt_array.reset();
}

absl::StatusOr<std::vector<PyArray>> PyArray::BatchedCopyToDeviceWithSharding(
    absl::Span<const PyArray> py_arrays,
    absl::Span<const ifrt::DeviceListRef> dst_device_lists,
    absl::Span<const nb::object> dst_shardings,
    absl::Span<const ifrt::ArrayCopySemantics> array_copy_semantics) {
  if (py_arrays.empty()) {
    return std::vector<PyArray>();
  }

  TF_RET_CHECK(py_arrays.size() == dst_device_lists.size());
  TF_RET_CHECK(py_arrays.size() == dst_shardings.size());

  ifrt::Client* const client = py_arrays.front().ifrt_array()->client();
  std::vector<PyArray> results(py_arrays.size());

  // Arrays to be copied, grouped by source/destination devices and memory
  // kinds. The grouping is enforced by `ifrt::Client::CopyArrays()`.
  struct Batch {
    std::vector<int> indexes;
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  };
  absl::flat_hash_map<BatchedCopyToDeviceWithShardingKey, Batch> batches;

  for (int i = 0; i < py_arrays.size(); ++i) {
    const auto& py_array = py_arrays[i];
    const auto& dst_sharding = dst_shardings[i];
    const auto& array_cs = array_copy_semantics[i];

    auto* ifrt_array_ptr = py_array.ifrt_array();
    const ifrt::DeviceListRef& src_devices =
        ifrt_array_ptr->sharding().devices();
    const ifrt::DeviceListRef& dst_devices = dst_device_lists[i];

    ifrt::MemoryKind src_memory_kind =
        ifrt::CanonicalizeMemoryKind(ifrt_array_ptr->sharding().memory_kind(),
                                     src_devices->devices().front());
    ifrt::MemoryKind dst_memory_kind = ifrt::CanonicalizeMemoryKind(
        xla::GetMemoryKind(dst_sharding), dst_devices->devices().front());

    if (*src_devices == *dst_devices && src_memory_kind == dst_memory_kind &&
        array_cs == ifrt::ArrayCopySemantics::kReuseInput) {
      results[i] = py_arrays[i];
      continue;
    }

    auto transfer_guard_formatter = [&py_array, &dst_sharding] {
      return absl::StrCat(
          "aval=", nb::cast<absl::string_view>(nb::repr(py_array.aval())),
          ", sharding=",
          nb::cast<absl::string_view>(nb::repr(py_array.sharding())),
          ", dst_sharding=",
          nb::cast<absl::string_view>(nb::repr(dst_sharding)));
    };
    TF_RETURN_IF_ERROR(
        jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));

    Batch& batch = batches[BatchedCopyToDeviceWithShardingKey{
        src_devices, src_memory_kind, dst_devices, dst_memory_kind, array_cs}];
    batch.indexes.push_back(i);
    batch.ifrt_arrays.push_back(tsl::FormRef(ifrt_array_ptr));
  }

  std::vector<std::pair<int, tsl::RCReference<ifrt::Array>>> ifrt_arrays;
  {
    GlobalPyRefManager()->CollectGarbage();
    nb::gil_scoped_release gil_release;

    for (auto& [key, batch] : batches) {
      TF_ASSIGN_OR_RETURN(
          auto copied,
          client->CopyArrays(
              absl::MakeSpan(batch.ifrt_arrays),
              // All arrays in `batch` have the same `key.dst_devices` and
              // `key.dst_memory_kind` due to the grouping above.
              key.dst_devices, key.dst_memory_kind, key.array_copy_semantics));
      for (int i = 0; i < batch.indexes.size(); ++i) {
        ifrt_arrays.push_back(
            std::make_pair(batch.indexes[i], std::move(copied[i])));
      }
    }
  }

  auto traceback = Traceback::Get();
  for (auto& [i, ifrt_array] : ifrt_arrays) {
    const auto& py_array = py_arrays[i];
    absl::Span<const int64_t> shape_span = py_array.shape();
    results[i] =
        PyArray(py_array.aval(), py_array.weak_type(), py_array.dtype(),
                std::vector<int64_t>(shape_span.begin(), shape_span.end()),
                dst_shardings[i], py_array.py_client(), traceback,
                std::move(ifrt_array), py_array.committed(),
                /*skip_checks=*/true, py_array.result_status());
  }
  return results;
}

absl::StatusOr<PyArray> PyArray::BatchedDevicePut(
    nb::object aval, nb::object sharding, std::vector<nb::object> xs,
    absl::Span<const PyDevice* const> dst_devices, bool committed,
    bool force_copy, PjRtClient::HostBufferSemantics host_buffer_semantics,
    bool jax_enable_x64) {
  if (dst_devices.size() != xs.size()) {
    throw nb::value_error(
        absl::StrCat("Argument sizes (xs and devices) must match %zu vs %zu",
                     dst_devices.size(), xs.size())
            .c_str());
  }
  for (const PyDevice* device : dst_devices) {
    if (device->client().get() == nullptr) {
      return InvalidArgument("Cannot copy to unattached devices.");
    }
  }
  auto transfer_guard_formatter = [&aval, &sharding] {
    return absl::StrCat(
        "aval=", nb::cast<absl::string_view>(nb::repr(aval)),
        ", dst_sharding=", nb::cast<absl::string_view>(nb::repr(sharding)));
  };

  GlobalPyRefManager()->CollectGarbage();

  auto n_devices = dst_devices.size();

  DevicePutOptions options;
  options.squash_64bit_types = !jax_enable_x64;
  options.allow_zero_copy =
      (!force_copy && (host_buffer_semantics ==
                       ifrt::Client::HostBufferSemantics::kImmutableZeroCopy));

  nb::list owning_pylist;
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;

  absl::InlinedVector<ifrt::Device*, 1> devices;
  devices.reserve(n_devices);
  std::vector<xla::ifrt::Shape> shapes;
  shapes.reserve(n_devices);

  ifrt::MemoryKind dst_memory_kind = xla::GetMemoryKind(sharding);

  std::vector<DevicePutResultFn> device_put_fns;
  device_put_fns.reserve(xs.size());
  size_t i = 0;
  for (auto& x : xs) {
    if (PyArray::IsPyArray(x)) {
      TF_RETURN_IF_ERROR(
          jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));
    } else {
      TF_RETURN_IF_ERROR(
          jax::ApplyTransferGuardToHostToDevice(transfer_guard_formatter));
    }
    TF_ASSIGN_OR_RETURN(
        device_put_fns.emplace_back(),
        DevicePut(x, dst_devices[i]->client()->ifrt_client(),
                  dst_devices[i]->device(), options, dst_memory_kind));
    ++i;
  }
  std::vector<DevicePutResult> device_puts;
  device_puts.reserve(device_put_fns.size());
  {
    // TODO(b/318709106): This is a temporary solution to propagate a hint to
    // backends that the current traceback does not change within the scope.
    // This should be removed once context propagation from IFRT API is
    // implemented.
    TracebackCacheScope traceback_cache_scope;
    nb::gil_scoped_release gil_release;
    for (auto& device_put_fn : device_put_fns) {
      TF_ASSIGN_OR_RETURN(auto device_put, std::move(device_put_fn)());
      device_puts.push_back(std::move(device_put));
    }
  }
  for (auto& device_put : device_puts) {
    ifrt_arrays.push_back(std::move(device_put.ifrt_array));
    devices.push_back(
        ifrt_arrays.back()->sharding().devices()->devices().front());
    shapes.push_back(ifrt_arrays.back()->shape());
    if (device_put.owning_pybuffer) {
      owning_pylist.append(device_put.owning_pybuffer);
    }
  }

  // TODO(phawkins): it's highly suspicious to me that owning_pylist isn't
  // consumed here. Look into this.

  auto weak_type = nb::cast<bool>(aval.attr("weak_type"));
  auto dtype = aval.attr("dtype");
  auto shape = nb::cast<std::vector<int64_t>>(aval.attr("shape"));

  TF_ASSIGN_OR_RETURN(
      auto ifrt_sharding,
      sharding.type().is(jax::PmapSharding::type())
          ? xla::GetIfrtConcreteSharding(sharding, ifrt::Shape(shape),
                                         std::move(shapes))
          : xla::GetIfrtHloSharding(sharding, ifrt::Shape(shape)));
  TF_ASSIGN_OR_RETURN(auto ifrt_dtype, DtypeToIfRtDType(dtype));
  // TODO(emilyaf): Remove the following and just use ifrt_dtype when tokens are
  // supported.
  ifrt::DType array_dtype =
      ifrt_arrays.empty() ? ifrt_dtype : ifrt_arrays.front()->dtype();
  TF_ASSIGN_OR_RETURN(auto py_device_list, jax::GetPyDeviceList(sharding));
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      py_device_list->py_client()
          ->ifrt_client()
          ->AssembleArrayFromSingleDeviceArrays(
              array_dtype, ifrt::Shape(shape), std::move(ifrt_sharding),
              absl::MakeSpan(ifrt_arrays),
              xla::ifrt::ArrayCopySemantics::kReuseInput,
              xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  return PyArray(aval, weak_type, dtype, std::move(shape), sharding,
                 py_device_list->py_client(), Traceback::Get(),
                 std::move(ifrt_array), committed, /*skip_checks=*/true);
}

absl::StatusOr<PyArray> PyArray::ReorderShards(
    PyArray x, nanobind::object dst_sharding,
    ifrt::ArrayCopySemantics array_copy_semantics) {
  xla::ifrt::Array* ifrt_array_ptr = x.ifrt_array();
  if (ifrt_array_ptr == nullptr) {
    return absl::InvalidArgumentError(
        "Reorder() called on deleted or donated buffer");
  }

  ifrt::Client* const client = ifrt_array_ptr->client();

  const auto& device_list = ifrt_array_ptr->sharding().devices();
  TF_ASSIGN_OR_RETURN(auto dst_device_list, GetIfrtDeviceList(dst_sharding));
  if (device_list->AddressableDeviceList()->size() !=
      dst_device_list->AddressableDeviceList()->size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Array is expected to have ",
        dst_device_list->AddressableDeviceList()->size(),
        " addressable shards, but has ",
        device_list->AddressableDeviceList()->size(), " addressable shards"));
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const xla::ifrt::Sharding> dst_ifrt_sharding,
      GetIfrtConcreteEvenSharding(dst_sharding, ifrt_array_ptr->dtype(),
                                  ifrt_array_ptr->shape()));

  tsl::RCReference<xla::ifrt::Array> new_ifrt_array;
  {
    nb::gil_scoped_release gil_release;

    const absl::Span<xla::ifrt::Device* const> addressable_devices =
        device_list->AddressableDeviceList()->devices();
    const absl::Span<xla::ifrt::Device* const> dst_addressable_devices =
        dst_device_list->AddressableDeviceList()->devices();

    absl::flat_hash_map<int, int> device_id_to_array_shard_index;
    device_id_to_array_shard_index.reserve(dst_addressable_devices.size());
    for (int i = 0; i < dst_addressable_devices.size(); ++i) {
      const int device_id = dst_addressable_devices[i]->Id().value();
      const bool inserted =
          device_id_to_array_shard_index.insert({device_id, i}).second;
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrCat("Sharding contains duplicate device id=", device_id));
      }
    }

    std::vector<int64_t> from_shard_indices;
    from_shard_indices.reserve(addressable_devices.size());
    std::vector<int64_t> to_shard_indices;
    to_shard_indices.reserve(dst_addressable_devices.size());
    for (int i = 0; i < dst_addressable_devices.size(); ++i) {
      from_shard_indices.push_back(i);
      const int shard_device_id = addressable_devices[i]->Id().value();
      const auto it = device_id_to_array_shard_index.find(shard_device_id);
      if (it == device_id_to_array_shard_index.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Array shard ", i, " is on device id=", shard_device_id,
            ", but sharding does not have a shard on that device."));
      }
      to_shard_indices.push_back(it->second);
    }

    auto mappings =
        std::make_shared<std::vector<xla::ifrt::RemapPlan::Mapping>>();
    {
      auto& mapping = mappings->emplace_back();
      mapping.in_array = 0;
      mapping.out_array = 0;
      mapping.from.reserve(dst_addressable_devices.size());
      mapping.to.reserve(dst_addressable_devices.size());
      for (int64_t i = 0; i < dst_addressable_devices.size(); ++i) {
        mapping.from.push_back(xla::ifrt::RemapPlan::Interval{
            from_shard_indices[i], from_shard_indices[i] + 1, 1});
        mapping.to.push_back(xla::ifrt::RemapPlan::Interval{
            to_shard_indices[i], to_shard_indices[i] + 1, 1});
      }
    }

    xla::ifrt::RemapPlan plan = {
        /*input_specs=*/{xla::ifrt::ArraySpec{
            /*dtype=*/ifrt_array_ptr->dtype(),
            /*shape=*/ifrt_array_ptr->shape(),
            /*sharding=*/ifrt_array_ptr->shared_ptr_sharding()}},
        /*output_specs=*/
        {xla::ifrt::ArraySpec{/*dtype=*/ifrt_array_ptr->dtype(),
                              /*shape=*/ifrt_array_ptr->shape(),
                              /*sharding=*/std::move(dst_ifrt_sharding)}},
        /*mappings=*/std::move(mappings),
    };
    DCHECK_OK(plan.Validate());
    std::vector<tsl::RCReference<xla::ifrt::Array>> input;
    input.push_back(tsl::FormRef(ifrt_array_ptr));
    TF_ASSIGN_OR_RETURN(
        auto remapped,
        client->RemapArrays(plan, absl::MakeSpan(input), array_copy_semantics));

    TF_RET_CHECK(remapped.size() == 1);
    new_ifrt_array = std::move(remapped.front());
  }

  return xla::PyArray(nb::borrow<nb::object>(x.aval().ptr()), x.weak_type(),
                      nb::borrow<xla::nb_dtype>(x.dtype().ptr()),
                      std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                      std::move(dst_sharding), x.py_client(), x.traceback(),
                      std::move(new_ifrt_array),
                      /*committed=*/true,
                      /*skip_checks=*/true);
}

absl::Status PyArray::BatchedBlockUntilReady(std::vector<nb::object> objs) {
  // Create ready futures for all arrays before blocking on their readiness.
  // This helps reduce the latency in some backend implementations where
  // querying readiness of an array is not free.

  std::vector<ifrt::Array*> ifrt_arrays;
  ifrt_arrays.reserve(objs.size());
  for (nb::handle obj : objs) {
    if (obj.type().is(PyArray::type())) {
      auto py_array = nb::borrow<PyArray>(obj);
      ifrt::Array* const ifrt_array = py_array.ifrt_array();
      if (ifrt_array == nullptr) {
        return absl::InvalidArgumentError(
            "BlockHostUntilReady() called on deleted or donated buffer");
      }
      ifrt_arrays.push_back(ifrt_array);
    } else {
      return absl::InvalidArgumentError(
          "PyArray::BatchedBlockUntilReady can take PyArray only");
    }
  }

  GlobalPyRefManager()->CollectGarbage();
  nb::gil_scoped_release gil_release;
  return AwaitBuffersReady(absl::MakeConstSpan(ifrt_arrays));
}

std::vector<PyArray> PyClient::LiveArrays() const {
  std::vector<PyArray> result;
  for (auto& shard : arrays_) {
    nb::ft_lock_guard lock(shard.mutex);
    for (PyArray::Storage* array = shard.arrays; array; array = array->next) {
      bool all_deleted =
          (array->ifrt_array == nullptr || array->ifrt_array->IsDeleted());
      if (!all_deleted) {
        result.push_back(nb::borrow<PyArray>(array->AsHandle()));
      }
    }
  }
  return result;
}

// PEP 3118 buffer protocol implementation.

namespace {

// Extra data to be kept alive by the consumer of the buffer protocol.
struct ExtraBufferInfo {
  explicit ExtraBufferInfo(
      std::shared_ptr<PjRtBuffer> buffer,
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold)
      : buffer(std::move(buffer)),
        external_reference_hold(std::move(external_reference_hold)) {}

  std::vector<int64_t> strides;
  // We keep an external reference hold to the PjRtBuffer. This prevents a
  // use-after-free in the event that Delete() is called on a buffer with an
  // live buffer protocol view. It does however mean that Delete() sometimes
  // won't actually delete immediately.
  std::shared_ptr<PjRtBuffer> buffer;
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
};

// The default layout of a non-tuple array should have major-to-minor layout
// and no tiles.
bool HasDefaultLayout(const Layout& layout) {
  return LayoutUtil::IsMonotonicWithDim0Major(layout) && layout.tiles().empty();
}

int PyArray_bf_getbuffer(PyObject* exporter, Py_buffer* view, int flags) {
  absl::Status status = [&]() -> absl::Status {
    PyArray py_array = nb::borrow<PyArray>(exporter);
    if (py_array.ifrt_array() == nullptr) {
      // TODO(phawkins): why is this happening?
      return InvalidArgument("Array is null");
    }
    if (!llvm::isa<ifrt::PjRtCompatibleArray>(py_array.ifrt_array())) {
      return InvalidArgument("Only local arrays are supported, got %s",
                             py_array.ifrt_array()->DebugString());
    }
    auto* array =
        static_cast<ifrt::PjRtCompatibleArray*>(py_array.ifrt_array());
    absl::Span<const std::shared_ptr<PjRtBuffer>> buffers =
        array->pjrt_buffers();

    PjRtBuffer& buffer = *buffers.front();
    if (!buffer.IsOnCpu()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }

    if (buffers.size() != 1) {
      return InvalidArgument(
          "Python buffer protocol is only defined for buffers with a single "
          "shard.");
    }
    if (!py_array.sharding().type().is(jax::SingleDeviceSharding::type())) {
      return InvalidArgument(
          "Python buffer protocol is only defined for single-device sharded "
          "buffers.");
    }

    const char* format =
        PEP3118FormatDescriptorForPrimitiveType(buffer.element_type());
    // It isn't an option for us to export unknown types as, say, bytes. When
    // converting an object to an ndarray, NumPy tries the buffer protocol
    // first. We very much want NumPy to fail and fall back to using
    // __array__, which allows us to handle custom dtypes correctly.
    if (!format) {
      return InvalidArgument(
          "Buffers of type %s are not supported by the Python buffer protocol.",
          PrimitiveType_Name(buffer.element_type()));
    }

    std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
    {
      // We call BlockHostUntilReady() below, which may block.
      nb::gil_scoped_release gil_release;

      if (buffer.IsTuple()) {
        return InvalidArgument(
            "Python buffer protocol is only defined for array buffers.");
      }
      if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
        return InvalidArgument("XLA buffers are read-only.");
      }
      TF_ASSIGN_OR_RETURN(external_reference_hold,
                          buffer.AcquireExternalReference());
      if (buffer.IsDeleted()) {
        return InvalidArgument("Deleted buffer used in buffer protocol.");
      }

      // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
      Layout xla_layout = buffer.layout()->xla_layout();

      if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS ||
           (flags & PyBUF_STRIDES) == PyBUF_ND) &&
          !LayoutUtil::IsMonotonicWithDim0Major(xla_layout)) {
        return InvalidArgument("Buffer is not in C-contiguous layout.");
      } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
                 !LayoutUtil::IsMonotonicWithDim0Minor(xla_layout)) {
        return InvalidArgument("Buffer is not in F-contiguous layout.");
      } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
                 !LayoutUtil::IsMonotonicWithDim0Major(xla_layout) &&
                 !LayoutUtil::IsMonotonicWithDim0Minor(xla_layout)) {
        return InvalidArgument("Buffer is not in contiguous layout.");
      } else if (!HasDefaultLayout(xla_layout)) {
        // Fail and fall back to using __array__ if the CPU buffer has a device
        // specific layout. For instance, this happens for host buffers in
        // pinned memories of the TPU device.
        return InvalidArgument(
            "Buffer is potentially a device buffer with non default layout.");
      }
      TF_RETURN_IF_ERROR(buffer.GetReadyFuture().Await());
    }

    // We must hold the GIL (or at least prevent Python GC) while writing to the
    // view object, see https://github.com/python/cpython/issues/130409.
    std::memset(view, 0, sizeof(Py_buffer));
    const void* root_ptr =
        external_reference_hold->OpaqueDeviceMemoryDataPointer();
    view->buf = const_cast<void*>(root_ptr);
    auto extra = std::make_unique<ExtraBufferInfo>(
        buffers.front(), std::move(external_reference_hold));
    view->itemsize = ShapeUtil::ByteSizeOfPrimitiveType(buffer.element_type());
    TF_ASSIGN_OR_RETURN(view->len, buffer.GetOnDeviceSizeInBytes());
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
      view->format = const_cast<char*>(format);
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
      view->ndim = buffer.dimensions().size();
      static_assert(sizeof(int64_t) == sizeof(Py_ssize_t),
                    "Py_ssize_t must be 64 bits");
      if (view->ndim != 0) {
        view->shape = reinterpret_cast<Py_ssize_t*>(
            const_cast<int64_t*>(buffer.dimensions().data()));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          extra->strides =
              ByteStridesForShape(buffer.element_type(), buffer.dimensions(),
                                  buffer.layout()->xla_layout());
          view->strides = reinterpret_cast<Py_ssize_t*>(
              const_cast<int64_t*>(extra->strides.data()));
        }
      }
    }
    view->internal = extra.release();
    return absl::OkStatus();
  }();
  if (!status.ok()) {
    // numpy.asarray(...) eats the PyExc_BufferError. Adding a log here helps
    // debugging when the error really occurs.
    VLOG(1) << "Buffer Protocol Error: " << status;
    PyErr_SetString(PyExc_BufferError, status.ToString().c_str());
    return -1;
  }
  view->obj = exporter;
  Py_INCREF(view->obj);
  return 0;
}

void PyArray_bf_releasebuffer(PyObject*, Py_buffer* buffer) {
  auto extra = static_cast<ExtraBufferInfo*>(buffer->internal);
  delete extra;
}

// Returns if shape has a major-to-minor layout.
bool HasMajorToMinorLayout(const xla::Shape& shape) {
  if (shape.has_layout()) {
    for (int i = 0; i < shape.layout().minor_to_major_size(); ++i) {
      if (shape.layout().minor_to_major(i) !=
          shape.layout().minor_to_major_size() - 1 - i) {
        return false;
      }
    }
  }
  return true;
}

// Returns byte_strides if shape has a non-major-to-minor layout.
std::optional<std::vector<int64_t>> ByteStridesOrDefaultForShapeInt64(
    const Shape& shape) {
  if (!shape.has_layout() || HasMajorToMinorLayout(shape)) {
    return std::nullopt;
  }
  return ByteStridesForShape(shape);
}

bool IsZeroCopyableCpuBuffer(const PjRtBuffer* buf) {
  // For CPU buffers with device-specific layouts, we must delinearize
  // to unpack the array. This could happen for the host buffer
  // pre-mapped to the TPU device, a.k.a., pinned host buffers for the
  // device.
  bool has_default_layout =
      buf->layout() == nullptr || HasDefaultLayout(buf->layout()->xla_layout());
  // On CPU for values >= 8 bits, we can return the value in a zero-copy way.
  // For sub-byte values, we must copy in order to unpack the array.
  return buf->IsOnCpu() &&
         !primitive_util::IsSubByteNonPredType(buf->element_type()) &&
         has_default_layout;
}
}  // namespace

PyHostValue::PyHostValue() = default;
PyHostValue::~PyHostValue() = default;

absl::StatusOr<std::pair<nb::object, bool>> PyHostValue::AsNumPyArray(
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array) {
  if (ifrt_array->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  // The only `jax.Array` with token-shape buffer is the one wrapped by
  // `jax.core.Token`. Since it is an internal implementation detail, we
  // don't support converting it to a numpy array.
  if (ifrt_array->dtype().kind() == ifrt::DType::kToken) {
    return InvalidArgument(
        "Cannot convert a token-shape buffer to a numpy array.");
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    TF_RET_CHECK(!pjrt_buffer->IsTuple());
    // On CPU for values >= 8 bits, we can return the value in a zero-copy way.
    // For sub-byte values, we must copy in order to unpack the array.
    if (IsZeroCopyableCpuBuffer(pjrt_buffer)) {
      TF_ASSIGN_OR_RETURN(const auto* shape,
                          XlaDynamicShape(ifrt_array, dynamic_shape_holder));
      TF_ASSIGN_OR_RETURN(nb_dtype dtype,
                          PrimitiveTypeToNbDtype(shape->element_type()));
      // Objects that must be kept alive while the array is alive.
      struct Hold {
        tsl::RCReference<ifrt::Array> buffer;
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
      };
      auto hold = std::make_unique<Hold>();
      hold->buffer = tsl::FormRef(ifrt_array);
      auto* hold_ptr = hold.release();
      nb::capsule hold_capsule(
          hold_ptr, [](void* h) noexcept { delete static_cast<Hold*>(h); });
      {
        // Release the GIL as `AcquireExternalReference` may block.
        nb::gil_scoped_release gil;
        TF_ASSIGN_OR_RETURN(hold_ptr->external_reference_hold,
                            pjrt_buffer->AcquireExternalReference());
        TF_RETURN_IF_ERROR(ifrt_array->GetReadyFuture().Await());
      }
      void* data =
          hold_ptr->external_reference_hold->OpaqueDeviceMemoryDataPointer();
      nb_numpy_ndarray array(dtype, shape->dimensions(),
                             ByteStridesForShape(*shape), data, hold_capsule);
      array.attr("flags").attr("writeable") = nb::bool_(false);
      return std::make_pair(array, false);
    }
  }

  TF_RETURN_IF_ERROR(CopyToHostAsync(dynamic_shape_holder, ifrt_array));
  if (!ready_.IsReady()) {
    nb::gil_scoped_release gil;
    TF_RETURN_IF_ERROR(ready_.Await());
  } else {
    TF_RETURN_IF_ERROR(ready_.Await());
  }
  if (string_array_contents_ != nullptr) {
    TF_RETURN_IF_ERROR(ConvertStringArrayContentsToNumpyArray(ifrt_array));
  }
  return std::make_pair(value_, true);
}

absl::Status PyHostValue::ConvertStringArrayContentsToNumpyArray(
    ifrt::Array* ifrt_array) {
#ifdef NPY_2_0_API_VERSION
  if (PyArray_RUNTIME_VERSION < NPY_2_0_API_VERSION) {
    return absl::FailedPreconditionError(
        absl::StrCat("String arrays are not supported in NumPy version: ",
                     PyArray_RUNTIME_VERSION));
  }
  auto numpy_dtype = nb::steal<nb_dtype>(
      reinterpret_cast<PyObject*>(PyArray_DescrFromType(NPY_VSTRING)));
  value_ = nb_numpy_ndarray(numpy_dtype, ifrt_array->shape().dims(),
                            /*strides=*/std::nullopt);

  auto dst_py_array_obj = reinterpret_cast<::PyArrayObject*>(value_.ptr());
  auto iter =
      nb::steal(PyArray_IterNew(reinterpret_cast<PyObject*>(dst_py_array_obj)));
  for (auto& cord : *string_array_contents_) {
    absl::string_view input_str_view = cord.Flatten();
    auto py_unicode = nb::steal(PyUnicode_FromStringAndSize(
        input_str_view.data(), input_str_view.size()));
    if (py_unicode.ptr() == nullptr) {
      return absl::InternalError("PyUnicode_FromStringAndSize failed");
    }
    if (PyArray_SETITEM(dst_py_array_obj,
                        static_cast<char*>(PyArray_ITER_DATA(iter.ptr())),
                        py_unicode.ptr()) != 0) {
      return absl::InternalError("PyArray_SETITEM failed");
    }
    PyArray_ITER_NEXT(iter.ptr());
  }

  value_.attr("flags").attr("writeable") = nb::bool_(false);

  string_array_contents_.reset();

  return absl::OkStatus();
#else
  return absl::FailedPreconditionError(
      "String arrays are not supported in this NumPy version.");
#endif
}

absl::Status PyHostValue::CopyStringArrayToHostAsync(
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array) {
  auto transfer_guard_formatter = [ifrt_array] {
    return absl::StrCat(
        "shape=(", absl::StrJoin(ifrt_array->shape().dims(), ","),
        "), dtype=", ifrt_array->dtype().DebugString(), ", device=",
        ifrt_array->sharding().devices()->devices().front()->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

  TF_ASSIGN_OR_RETURN(nb_dtype dtype, IfrtDtypeToNbDtype(ifrt_array->dtype()));
  auto shape = ifrt_array->shape();

  // Allocate a vector of cords to hold the contents of the array until
  // they are until they are ultimately converted to a numpy array as part
  // of the `AsNumPyArray` call.
  string_array_contents_ =
      std::make_shared<std::vector<absl::Cord>>(shape.num_elements());
  ready_ = ifrt_array->CopyToHostBuffer(string_array_contents_->data(),
                                        /*byte_strides=*/std::nullopt,
                                        ifrt::ArrayCopySemantics::kAlwaysCopy);

  ready_.OnReady(
      [string_array_contents = string_array_contents_](absl::Status) {
      });  // Keeps the cords alive until the copy is done.

  return absl::OkStatus();
}

absl::Status PyHostValue::CopyToHostAsync(
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array) {
  if (ready_.IsValid()) {
    // The array value has been populated, so CopyToHostAsync has been called.
    return absl::OkStatus();
  }

  // Copying in Arrays of type kString requires some special handling
  if (ifrt_array->dtype().kind() == ifrt::DType::kString) {
    return CopyStringArrayToHostAsync(dynamic_shape_holder, ifrt_array);
  }

  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr && !arr->pjrt_buffers().front()->IsTuple() &&
      IsZeroCopyableCpuBuffer(arr->pjrt_buffers().front().get())) {
    return absl::OkStatus();
  }
  auto transfer_guard_formatter = [ifrt_array] {
    return absl::StrCat(
        "shape=(", absl::StrJoin(ifrt_array->shape().dims(), ","),
        "), dtype=", ifrt_array->dtype().DebugString(), ", device=",
        ifrt_array->sharding().devices()->devices().front()->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

  // TODO(b/182461453): This is a blocking call. If we further implemented
  // populating dynamic shape metadata while fetching the literal, we wouldn't
  // need this static approach.
  const xla::Shape* dynamic_shape;
  std::optional<xla::Shape> shape_holder;
  if (llvm::isa<ifrt::PjRtCompatibleArray>(ifrt_array)) {
    TF_ASSIGN_OR_RETURN(dynamic_shape,
                        XlaDynamicShape(ifrt_array, dynamic_shape_holder));
  } else {
    // Skip querying the dynamic shape for a non-PjRt Array.
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType type,
                        ifrt::ToPrimitiveType(ifrt_array->dtype()));
    shape_holder = ShapeUtil::MakeShapeWithDescendingLayout(
        type, ifrt_array->shape().dims());
    dynamic_shape = &*shape_holder;
  }

  xla::Shape host_shape = ShapeUtil::DeviceShapeToHostShape(*dynamic_shape);

  auto strides = ByteStridesOrDefaultForShapeInt64(host_shape);
  TF_ASSIGN_OR_RETURN(nb_dtype dtype,
                      PrimitiveTypeToNbDtype(host_shape.element_type()));
  value_ = nb_numpy_ndarray(dtype, host_shape.dimensions(), strides);
  // TODO(hyeontaek): Several PjRt runtimes assume that the host buffer uses
  // the same transposition as the device buffer. This is different from
  // PjRtBuffer::ToLiteral()'s semantics that the runtime respects the layout
  // of the host buffer literal. On the other hand, the runtime often knows
  // better about an efficient layout for the host buffer. It will be useful
  // to revisit the semantics of PjRtBuffer::ToLiteral() to see if it is
  // desirable for the runtime to choose the layout.
  ready_ = ifrt_array->CopyToHostBuffer(value_.mutable_data(), strides,
                                        ifrt::ArrayCopySemantics::kReuseInput);
  // Make sure the destination of the copy remains alive until the copy is done.
  value_.inc_ref();
  ready_.OnReady([array{value_.ptr()}](absl::Status status) {
    GlobalPyRefManager()->AddGarbage(nb::steal(array));
  });
  value_.attr("flags").attr("writeable") = nb::bool_(false);
  return absl::OkStatus();
}

namespace {
PyGetSetDef PyArray_tp_getset[] = {
    {"__dict__", PyObject_GenericGetDict, PyObject_GenericSetDict, nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMemberDef PyArray_members[] = {
#if PY_VERSION_HEX < 0x030C0000
    {"__weaklistoffset__", T_PYSSIZET,
     static_cast<Py_ssize_t>(offsetof(PyArrayObject, weakrefs)), READONLY,
     nullptr},
    {"__dictoffset__", T_PYSSIZET,
     static_cast<Py_ssize_t>(offsetof(PyArrayObject, dict)), READONLY, nullptr},
#endif  // PY_VERSION_HEX < 0x030C0000
    {nullptr, 0, 0, 0, nullptr},
};  // namespace xla

PyType_Slot PyArray_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyArray_tp_new)},
    {Py_tp_dealloc, reinterpret_cast<void*>(PyArray_tp_dealloc)},
    {Py_tp_members, reinterpret_cast<void*>(PyArray_members)},
    {Py_tp_traverse, reinterpret_cast<void*>(PyArray_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(PyArray_tp_clear)},
    {Py_tp_getset, reinterpret_cast<void*>(PyArray_tp_getset)},
    {Py_bf_getbuffer, reinterpret_cast<void*>(PyArray_bf_getbuffer)},
    {Py_bf_releasebuffer, reinterpret_cast<void*>(PyArray_bf_releasebuffer)},
    {0, nullptr},
};

}  // namespace

absl::Status PyArray::RegisterTypes(nb::module_& m) {
  std::string name =
      absl::StrCat(nb::cast<std::string>(m.attr("__name__")), ".ArrayImpl");

  PyType_Spec PyArray_spec = {
#if PY_VERSION_HEX < 0x030B0000
      // Work around for https://github.com/python/cpython/issues/89478
      // CPython 3.10 and earlier assume that the .name value remains alive
      // forever.
      /*.name=*/strdup(name.c_str()),
#else
      /*.name=*/name.c_str(),
#endif  // PY_VERSION_HEX < 0x030B0000
      /*.basicsize=*/static_cast<int>(sizeof(PyArrayObject)),
      /*.itemsize=*/0,
#if PY_VERSION_HEX < 0x030C0000
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
#else   // PY_VERSION_HEX >= 0x030C0000
      /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
          Py_TPFLAGS_MANAGED_DICT | Py_TPFLAGS_MANAGED_WEAKREF,
#endif  // PY_VERSION_HEX >= 0x030C0000
      /*.slots=*/PyArray_slots,
  };

  type_ = PyType_FromSpec(&PyArray_spec);
  if (!type_) {
    throw nb::python_error();
  }
  auto type = nb::borrow<nb::object>(type_);
  m.attr("ArrayImpl") = type;

  type.attr("__init__") = nb::cpp_function(
      [](PyArray self, nb::object aval, nb::object sharding, nb::list arrays,
         bool committed, bool skip_checks) {
        if (!(arrays.size() == 0 || arrays[0].type().is(PyArray::type()))) {
          throw nb::type_error(
              absl::StrCat(
                  "Unsupported type for elements in `arrays`: ",
                  nb::cast<absl::string_view>(nb::str(arrays[0].type())))
                  .c_str());
        }
        auto py_arrays = nb::cast<std::vector<PyArray>>(arrays);
        PyArray::PyInit(self, std::move(aval), std::move(sharding), py_arrays,
                        committed, skip_checks);
      },
      nb::is_method(), nb::arg("aval"), nb::arg("sharding"), nb::arg("arrays"),
      nb::arg("committed"), nb::arg("_skip_checks") = false);
  type.attr("delete") = nb::cpp_function(
      [](PyArray& self) { xla::ThrowIfError(self.Delete()); }, nb::is_method());
  type.attr("_sharding") = nb_property_readonly(&PyArray::sharding);
  type.attr("aval") = nb_property(&PyArray::aval, &PyArray::set_aval);
  type.attr("_arrays") =
      nb_property(&PyArray::arrays, [](PyArray& self, nb::object obj) {
        xla::ThrowIfError(self.set_arrays(obj));
      });
  type.attr("_fully_replicated_shard") = nb::cpp_function(
      [](PyArray self) {
        return xla::ValueOrThrow(self.FullyReplicatedShard());
      },
      nb::is_method());
  type.attr("_npy_value") =
      nb_property(&PyArray::npy_value, &PyArray::set_npy_value);
  type.attr("_committed") = nb_property_readonly(&PyArray::committed);
  type.attr("unsafe_buffer_pointer") = nb::cpp_function(
      [](PyArray self) {
        return xla::ValueOrThrow(self.UnsafeBufferPointer());
      },
      nb::is_method());
  type.attr("__cuda_array_interface__") = nb_property_readonly(
      [](PyArray self) { return self.CudaArrayInterface(); });
  type.attr("_pjrt_layout") =
      nb_property_readonly(xla::ValueOrThrowWrapper(&PyArray::layout));
  type.attr("on_device_size_in_bytes") = nb::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::GetOnDeviceSizeInBytes),
      nb::is_method());
  type.attr("_single_device_array_to_np_array_did_copy") = nb::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::SingleDeviceArrayToNumpyArrayDidCopy),
      nb::is_method());
  type.attr("_copy_single_device_array_to_host_async") = nb::cpp_function(
      [](PyArray& self) {
        xla::ThrowIfError(self.CopySingleDeviceArrayToHostAsync());
      },
      nb::is_method());
  type.attr("block_until_ready") = nb::cpp_function(
      [](PyArray self) -> nb::object {
        xla::ThrowIfError(self.BlockUntilReady());
        return self;
      },
      nb::is_method());
  type.attr("platform") = nb::cpp_function(
      [](PyArray self) {
        if (self.ifrt_array()->client()->platform_name() == "cuda" ||
            self.ifrt_array()->client()->platform_name() == "rocm") {
          return absl::string_view("gpu");
        } else {
          return self.ifrt_array()->client()->platform_name();
        }
      },
      nb::is_method());
  type.attr("is_ready") = nb::cpp_function(
      [](PyArray self) { return xla::ValueOrThrow(self.IsReady()); },
      nb::is_method());
  type.attr("is_deleted") =
      nb::cpp_function(&PyArray::IsDeleted, nb::is_method());
  type.attr("traceback") = nb_property_readonly(&PyArray::traceback);
  type.attr("clone") = nb::cpp_function(&PyArray::Clone, nb::is_method());
  type.attr("__module__") = m.attr("__name__");

  m.attr("batched_copy_array_to_devices_with_sharding") = nb::cpp_function(
      [](absl::Span<const PyArray> arrays,
         absl::Span<const std::vector<const PyDevice*>> dst_device_lists,
         absl::Span<const nb::object> shardings,
         absl::Span<const ifrt::ArrayCopySemantics> array_copy_semantics) {
        if (arrays.empty()) {
          return std::vector<PyArray>();
        }
        auto* client = arrays[0].ifrt_array()->client();
        std::vector<ifrt::DeviceListRef> device_lists;
        device_lists.reserve(dst_device_lists.size());
        for (const auto& dst_devices : dst_device_lists) {
          absl::InlinedVector<ifrt::Device*, 1> devices;
          devices.reserve(dst_devices.size());
          for (auto& d : dst_devices) {
            devices.push_back(d->device());
          }
          device_lists.push_back(client->MakeDeviceList(devices));
        }
        return xla::ValueOrThrow(PyArray::BatchedCopyToDeviceWithSharding(
            arrays, device_lists, shardings, array_copy_semantics));
      });
  m.attr("array_result_handler") = nb::cpp_function(
      [](nb::object aval, nb::object sharding, bool committed,
         bool skip_checks) -> nb_class_ptr<PyArrayResultHandler> {
        return make_nb_class<PyArrayResultHandler>(
            std::move(aval), std::move(sharding), committed, skip_checks);
      },
      nb::arg("aval"), nb::arg("sharding"), nb::arg("committed"),
      nb::arg("_skip_checks") = false);

  nb::class_<PyArrayResultHandler>(m, "ResultHandler")
      .def("__call__", [](const PyArrayResultHandler& self,
                          PyArray arg) { return self.Call(arg); })
      .def("__call__",
           [](const PyArrayResultHandler& self,
              std::vector<PyArray> py_arrays) { return self.Call(py_arrays); });

  return absl::OkStatus();
}

}  // namespace xla
