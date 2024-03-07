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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "xla/layout_util.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/lru_cache.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/python/py_values.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/python_utils.h"
#include "xla/python/sharding.h"
#include "xla/python/traceback.h"
#include "xla/python/transfer_guard_lib.h"
#include "xla/python/types.h"
#include "xla/python/util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#if GOOGLE_CUDA
#include "xla/stream_executor/cuda/cuda_driver.h"
#endif
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace py = pybind11;

PjRtBuffer* GetPjrtBuffer(ifrt::Array* ifrt_array) {
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  return arr->pjrt_buffers().front().get();
}

StatusOr<const Shape*> XlaDynamicShape(ifrt::Array* ifrt_array,
                                       std::optional<Shape>& scratch) {
  auto* pjrt_buffer = GetPjrtBuffer(ifrt_array);

  if (!scratch) {
    absl::Span<const int64_t> dims;
    std::optional<std::vector<int64_t>> logical_dims_storage;
    if (pjrt_buffer->has_dynamic_dimensions()) {
      {
        py::gil_scoped_release gil_release;
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
    *shape.mutable_layout() = GetXlaLayoutUnsafe(pjrt_buffer->layout());
    scratch = std::move(shape);
  }
  return &scratch.value();
}

tsl::RCReference<ifrt::Array> CreateIfRtArrayFromSingleDeviceShardedPyArrays(
    py::object dtype, absl::Span<const int64_t> shape,
    absl::Span<const PyArray> py_arrays) {
  if (py_arrays.empty()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error("At least one array must be provided.");
  }
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(py_arrays.size());
  ifrt::DeviceList::Devices devices;
  devices.reserve(py_arrays.size());
  std::vector<ifrt::Shape> shapes;
  shapes.reserve(py_arrays.size());

  const ifrt::MemoryKind first_memory_kind =
      py_arrays.front().ifrt_array()->sharding().memory_kind();
  // TODO(hyeontaek): Canonicalize every `ifrt::MemoryKind` at creation time to
  // skip canonicalization here once JAX begins to do it for JAX shardings.
  const ifrt::MemoryKind canonical_first_memory_kind =
      ifrt::CanonicalizeMemoryKind(
          first_memory_kind,
          py_arrays.front().ifrt_array()->sharding().devices().front());
  for (const auto& py_array : py_arrays) {
    DCHECK_EQ(py_array.num_shards(), 1);
    ifrt_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
    devices.push_back(ifrt_arrays.back()->sharding().devices().front());
    shapes.push_back(ifrt_arrays.back()->shape());
    if (canonical_first_memory_kind !=
        ifrt::CanonicalizeMemoryKind(
            ifrt_arrays.back()->sharding().memory_kind(), devices.back())) {
      throw py::value_error(absl::StrFormat(
          "Memory kind mismatch between PjRtBuffers. Got one buffer with "
          "memory kind '%s' and another with memory_kind '%s'",
          first_memory_kind.DebugString(),
          ifrt_arrays.back()->sharding().memory_kind().DebugString()));
    }
  }
  ifrt::Client* client = ifrt_arrays.front()->client();

  auto ifrt_dtype = DtypeToIfRtDType(dtype);
  if (!ifrt_dtype.ok()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error(ifrt_dtype.status().ToString());
  }
  auto ifrt_array = client->AssembleArrayFromSingleDeviceArrays(
      ifrt::Shape(shape),
      ifrt::ConcreteSharding::Create(ifrt::DeviceList(std::move(devices)),
                                     first_memory_kind,
                                     /*shape=*/ifrt::Shape(shape),
                                     /*shard_shapes=*/std::move(shapes)),
      absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput);
  if (!ifrt_array.ok()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error(ifrt_array.status().ToString());
  }
  return *std::move(ifrt_array);
}

// Creates an IFRT `MemoryKind` from a JAX `Sharding`.
ifrt::MemoryKind CreateIfRtMemoryKindFromSharding(const py::object& sharding) {
  py::object py_memory_kind = py::none();

  // sharding.attr("memory_kind") can crash if sharding was originally created
  // from C++ and casted into a Python Sharding object. Thus, we cast sharding
  // to a C++ type and use C++ `memory_kind()` method, which bypasses any Python
  // attribute access.
  auto type = sharding.get_type();
  if (type.is(jax::NamedSharding::type())) {
    py_memory_kind = py::cast<jax::NamedSharding>(sharding).memory_kind();
  } else if (type.is(jax::GSPMDSharding::type())) {
    py_memory_kind = py::cast<jax::GSPMDSharding>(sharding).memory_kind();
  } else if (type.is(jax::SingleDeviceSharding::type())) {
    py_memory_kind =
        py::cast<jax::SingleDeviceSharding>(sharding).memory_kind();
  } else {
    py_memory_kind = sharding.attr("memory_kind");
  }

  if (py_memory_kind.is_none()) {
    return ifrt::MemoryKind();
  }
  return ifrt::MemoryKind(py::cast<std::string>(py_memory_kind));
}

struct PyArrayObject {
  PyObject_HEAD;
  PyObject* weakrefs;
  alignas(PyArray::Storage) char array_storage[sizeof(PyArray::Storage)];
};
static_assert(std::is_standard_layout<PyArrayObject>::value);

PyArray::Storage* GetPyArrayStorageFromObject(PyArrayObject* py_array_object) {
  return std::launder(
      reinterpret_cast<PyArray::Storage*>(py_array_object->array_storage));
}

extern "C" PyObject* PyArray_tp_new(PyTypeObject* type, PyObject*, PyObject*) {
  PyObject* self = type->tp_alloc(type, 0);
  return self;
}

extern "C" void PyArray_tp_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  PyTypeObject* tp = Py_TYPE(self);
  auto* obj = reinterpret_cast<PyArrayObject*>(self);

  if (obj->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }

  GetPyArrayStorageFromObject(obj)->~PyArray_Storage();

  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_CLEAR(dict);

  tp->tp_free(self);
  Py_DECREF(tp);
}

// dynamic_attr: Allow the garbage collector to traverse the internal instance
// `__dict__`.
extern "C" int PyArray_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_VISIT(dict);
// https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
#if PY_VERSION_HEX >= 0x03090000
  Py_VISIT(Py_TYPE(self));
#endif
  return 0;
}

// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" int PyArray_tp_clear(PyObject* self) {
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_CLEAR(dict);
  return 0;
}

// Give instances of this type a `__dict__` and opt into garbage collection.
void EnableDynamicAttribute(PyHeapTypeObject* heap_type) {
  auto* type = &heap_type->ht_type;
  type->tp_flags |= Py_TPFLAGS_HAVE_GC;
#if PY_VERSION_HEX < 0x030B0000
  type->tp_dictoffset = type->tp_basicsize;  // place dict at the end
  type->tp_basicsize +=
      (ssize_t)sizeof(PyObject*);  // and allocate enough space for it
#else
  type->tp_flags |= Py_TPFLAGS_MANAGED_DICT;
#endif
  type->tp_traverse = PyArray_tp_traverse;
  type->tp_clear = PyArray_tp_clear;

  static PyGetSetDef getset[] = {{"__dict__", PyObject_GenericGetDict,
                                  PyObject_GenericSetDict, nullptr, nullptr},
                                 {nullptr, nullptr, nullptr, nullptr, nullptr}};
  type->tp_getset = getset;
}

template <typename... Args>
PyArray::Storage* Construct(PyArrayObject* self, Args&&... args) {
  return new (self->array_storage)
      PyArray::Storage(std::forward<Args>(args)...);
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
py::object MakeShapedArrayCached(const ShapedArrayCacheKey& key) {
  using CacheT =
      LRUCache<ShapedArrayCacheKey, std::shared_ptr<std::optional<py::object>>>;
  static auto* lru_list = new CacheT::LRUList(4096);
  static auto* cache = new CacheT(lru_list);

  static const py::handle* shaped_array = nullptr;
  if (shaped_array == nullptr) {
    auto* jax_core = PyImport_ImportModule("jax.core");
    if (jax_core != nullptr) {
      shaped_array = new py::handle(
          py::reinterpret_steal<py::module>(jax_core).attr("ShapedArray"));
    } else {
      PyErr_Clear();
      return py::none();
    }
  }

  auto value =
      cache->GetOrCreateIfAbsent(key, [](const ShapedArrayCacheKey& key) {
        return std::make_shared<std::optional<py::object>>();
      });

  if (!value->has_value()) {
    auto dtype = IfrtDtypeToDtype(key.dtype).value();
    py::object aval = (*shaped_array)(
        SpanToTuple(absl::Span<const int64_t>(key.dims)), dtype, key.weak_type);
    *value = aval;
    return aval;
  }
  return **value;
}

}  // namespace

PyArray_Storage::PyArray_Storage(pybind11::object aval, bool weak_type,
                                 pybind11::dtype dtype,
                                 std::vector<int64_t> shape,
                                 pybind11::object sharding, bool committed,
                                 std::shared_ptr<PyClient> py_client,
                                 std::optional<nb_traceback> traceback,
                                 tsl::RCReference<ifrt::Array> ifrt_array)
    : fastpath_enabled(true),
      aval(std::move(aval)),
      weak_type(weak_type),
      dtype(std::move(dtype)),
      shape(std::move(shape)),
      sharding(std::move(sharding)),
      committed(committed),
      py_client(std::move(py_client)),
      traceback(std::move(traceback)),
      ifrt_array(std::move(ifrt_array)) {
  next = this->py_client->arrays_;
  this->py_client->arrays_ = this;
  if (next) {
    next->prev = this;
  }
  prev = nullptr;
}

PyArray_Storage::PyArray_Storage(DisableFastpath) : fastpath_enabled(false) {}

void PyArray::PyInit(py::object self, py::object aval, py::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks) {
  auto dtype = aval.attr("dtype");
  auto shape = pybind11::cast<std::vector<int64_t>>(aval.attr("shape"));
  auto ifrt_array =
      CreateIfRtArrayFromSingleDeviceShardedPyArrays(dtype, shape, py_arrays);
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()), aval,
            pybind11::cast<bool>(aval.attr("weak_type")), std::move(dtype),
            std::move(shape), std::move(sharding), committed,
            py_arrays.at(0).py_client(), Traceback::Get(),
            std::move(ifrt_array));

  PyArray py_array = self;

  if (!skip_checks) {
    py_array.CheckAndRearrange();
  }
}

void PyArray::PyInit(py::object self, DisableFastpath) {
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()),
            PyArray_Storage::DisableFastpath());
}

PyArray PyArray::MakeFromSingleDeviceArray(
    std::shared_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
    tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type, bool committed) {
  if (!llvm::isa<ifrt::SingleDeviceSharding>(ifrt_array->sharding())) {
    throw XlaRuntimeError(
        InvalidArgument("Constructing single device jax.Array from non-single "
                        "device ifrt array."));
  }
  auto shape_span = ifrt_array->shape().dims();
  ShapedArrayCacheKey key;
  key.dims = std::vector<int64_t>(shape_span.begin(), shape_span.end());
  key.dtype = ifrt_array->dtype();
  key.weak_type = weak_type;
  auto aval = MakeShapedArrayCached(key);
  auto dtype = IfrtDtypeToDtype(key.dtype).value();
  const ifrt::MemoryKind memory_kind = ifrt_array->sharding().memory_kind();
  auto py_memory_kind =
      (jax::GetEnableMemories() && memory_kind.memory_kind().has_value())
          ? py::object(py::str(*memory_kind.memory_kind()))
          : py::none();
  auto sharding = py::cast(std::make_unique<jax::SingleDeviceSharding>(
      py_client, ifrt_array->sharding().devices(), std::move(py_memory_kind)));
  return PyArray(std::move(aval), weak_type, dtype, std::move(key.dims),
                 std::move(sharding), std::move(py_client),
                 std::move(traceback), std::move(ifrt_array), committed,
                 /*skip_checks=*/true);
}

PyArray PyArray::MakeFromIfrtArrayAndSharding(
    std::shared_ptr<PyClient> py_client, std::optional<nb_traceback> traceback,
    tsl::RCReference<ifrt::Array> ifrt_array, py::object sharding,
    bool weak_type, bool committed, bool skip_checks) {
  auto shape_span = ifrt_array->shape().dims();
  ShapedArrayCacheKey key;
  key.dims = std::vector<int64_t>(shape_span.begin(), shape_span.end());
  key.dtype = ifrt_array->dtype();
  key.weak_type = weak_type;
  auto aval = MakeShapedArrayCached(key);
  auto dtype = IfrtDtypeToDtype(key.dtype).value();
  return PyArray(std::move(aval), weak_type, dtype, std::move(key.dims),
                 std::move(sharding), std::move(py_client),
                 std::move(traceback), std::move(ifrt_array), committed,
                 skip_checks);
}

PyArrayResultHandler::PyArrayResultHandler(py::object aval, py::object sharding,
                                           bool committed, bool skip_checks)
    : aval_(std::move(aval)),
      sharding_(std::move(sharding)),
      committed_(committed),
      skip_checks_(skip_checks) {
  weak_type_ = pybind11::cast<bool>(aval_.attr("weak_type"));
  dtype_ = aval_.attr("dtype");
  shape_ = pybind11::cast<std::vector<int64_t>>(aval_.attr("shape"));
}

PyArray PyArrayResultHandler::Call(absl::Span<const PyArray> py_arrays) const {
  return Call(py_arrays.at(0).py_client(),
              CreateIfRtArrayFromSingleDeviceShardedPyArrays(dtype_, shape_,
                                                             py_arrays));
}

PyArray PyArrayResultHandler::Call(
    std::shared_ptr<PyClient> py_client,
    tsl::RCReference<ifrt::Array> ifrt_array) const {
  return PyArray(aval_, weak_type_, dtype_, shape_, sharding_,
                 std::move(py_client), Traceback::Get(), std::move(ifrt_array),
                 committed_, skip_checks_);
}

PyArray PyArrayResultHandler::Call(PyArray py_array) const {
  return Call(py_array.py_client(), tsl::FormRef(py_array.ifrt_array()));
}

PyArray::PyArray(py::object aval, bool weak_type, py::dtype dtype,
                 std::vector<int64_t> shape, py::object sharding,
                 std::shared_ptr<PyClient> py_client,
                 std::optional<nb_traceback> traceback,
                 tsl::RCReference<ifrt::Array> ifrt_array, bool committed,
                 bool skip_checks) {
  auto* self =
      PyArray_tp_new(reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr);
  ptr() = self;
  Construct(reinterpret_cast<PyArrayObject*>(self), std::move(aval), weak_type,
            std::move(dtype), std::move(shape), std::move(sharding), committed,
            std::move(py_client), std::move(traceback), std::move(ifrt_array));

  if (!skip_checks) {
    CheckAndRearrange();
  }
}

PyArray::Storage& PyArray::GetStorage() {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

const PyArray::Storage& PyArray::GetStorage() const {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

void PyArray::CheckAndRearrange() { this->attr("_check_and_rearrange")(); }

void PyArray::SetIfrtArray(tsl::RCReference<ifrt::Array> ifrt_array) {
  GetStorage().ifrt_array = std::move(ifrt_array);
}

const std::vector<PyArray>& PyArray::py_arrays_cached() {
  auto& py_arrays = this->py_arrays();

  if (py_arrays.empty()) {
    auto ifrt_arrays = ifrt_array()->DisassembleIntoSingleDeviceArrays(
        ifrt::ArrayCopySemantics::kReuseInput);
    if (!ifrt_arrays.ok()) {
      throw py::value_error(
          absl::StrCat("Failed to disassemble into single-device arrays: ",
                       ifrt_arrays.status().ToString()));
    }
    py_arrays.reserve(ifrt_arrays->size());
    for (auto& ifrt_array : *ifrt_arrays) {
      py_arrays.push_back(PyArray::MakeFromSingleDeviceArray(
          py_client(), traceback(), std::move(ifrt_array), weak_type(),
          committed()));
    }
  }

  return py_arrays;
}

py::object PyArray::arrays() {
  // For performance, we only keep pjrt buffers by default. But on python side
  // "_arrays" returns PyArrays instead, and subsequent calls to "_arrays"
  // should return the same PyArrays (to avoid duplicate device to host
  // transfers). So we create PyArrays the first time it is called and reuse
  // them later.
  if (ifrt_array() == nullptr || ifrt_array()->IsDeleted()) return py::none();

  if (llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    std::vector<PyArray> py_arrays;
    py_arrays.push_back(*this);
    return py::cast(py_arrays);
  }

  return py::cast(py_arrays_cached());
}

Status PyArray::set_arrays(py::object obj) {
  if (obj.is_none()) {
    SetIfrtArray(tsl::RCReference<ifrt::Array>());
    py_arrays().clear();
    return OkStatus();
  }

  if (!py::isinstance<py::list>(obj)) {
    return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                           py::cast<std::string>(py::str(obj.get_type())));
  }

  py::list list = obj;

  if (list.empty()) return OkStatus();

  SetIfrtArray(tsl::RCReference<ifrt::Array>());
  py_arrays().clear();
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(list.size());
  ifrt::DeviceList::Devices devices;
  devices.reserve(list.size());
  std::vector<ifrt::Shape> shapes;
  shapes.reserve(list.size());
  for (py::handle obj : list) {
    if (obj.get_type().is(PyArray::type())) {
      auto py_array = py::reinterpret_borrow<PyArray>(obj);
      if (py_array.py_client() != py_client()) {
        return InvalidArgument("Client mismatch when assigning to _arrays.");
      }
      if (py_array.num_shards() != 1) {
        return InvalidArgument("Wrong number of shards: %d",
                               py_array.num_shards());
      }
      ifrt_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
      devices.push_back(ifrt_arrays.back()->sharding().devices().front());
      shapes.push_back(ifrt_arrays.back()->shape());
    } else {
      return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                             py::cast<std::string>(py::str(obj.get_type())));
    }
  }
  const ifrt::MemoryKind first_memory_kind =
      ifrt_arrays.front()->sharding().memory_kind();
  // TODO(hyeontaek): Canonicalize every `ifrt::MemoryKind` at creation time to
  // skip canonicalization here once JAX begins to do it for JAX shardings.
  const ifrt::MemoryKind canonical_first_memory_kind =
      ifrt::CanonicalizeMemoryKind(
          first_memory_kind, ifrt_arrays.front()->sharding().devices().front());
  for (const auto& ifrt_array : ifrt_arrays) {
    if (canonical_first_memory_kind !=
        ifrt::CanonicalizeMemoryKind(
            ifrt_array->sharding().memory_kind(),
            ifrt_array->sharding().devices().front())) {
      throw py::value_error(absl::StrFormat(
          "Memory kind mismatch between single-device arrays. Got one array "
          "with memory kind '%s' and another with memory_kind '%s'",
          first_memory_kind.DebugString(),
          ifrt_array->sharding().memory_kind().DebugString()));
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto array,
      py_client()->ifrt_client()->AssembleArrayFromSingleDeviceArrays(
          ifrt::Shape(shape()),
          ifrt::ConcreteSharding::Create(ifrt::DeviceList(std::move(devices)),
                                         first_memory_kind,
                                         /*shape=*/ifrt::Shape(shape()),
                                         /*shard_shapes=*/std::move(shapes)),
          absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput));
  SetIfrtArray(std::move(array));
  return OkStatus();
}

StatusOr<PyArray> PyArray::FullyReplicatedShard() {
  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "FullyReplicatedShard() called on deleted or donated buffer");
  }

  TF_ASSIGN_OR_RETURN(auto fully_replicated_ifrt_shard,
                      ifrt_array()->FullyReplicatedShard(
                          ifrt::ArrayCopySemantics::kReuseInput));
  return MakeFromSingleDeviceArray(py_client(), traceback(),
                                   std::move(fully_replicated_ifrt_shard),
                                   weak_type(), committed());
}

Status PyArray::BlockUntilReady() const {
  pybind11::gil_scoped_release gil_release;
  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "BlockHostUntilReady() called on deleted or donated buffer");
  }
  return AwaitBuffersReady(ifrt_array());
}

StatusOr<size_t> PyArray::GetOnDeviceSizeInBytes() {
  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "GetOnDeviceSizeInBytes() called on deleted or donated buffer");
  }

  TF_ASSIGN_OR_RETURN(size_t shard_size,
                      GetPjrtBuffer(ifrt_array())->GetOnDeviceSizeInBytes());
  return shard_size * py::len(sharding().attr("device_set"));
}

StatusOr<PyArray> PyArray::FetchSingleShard(std::string_view api) {
  if (ifrt_array() == nullptr) {
    return InvalidArgument("%s( called on deleted or donated buffer", api);
  }

  if (llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    return *this;
  }

  auto& py_arrays = py_arrays_cached();
  if (py_arrays.empty() || py_arrays[0].shape() != shape()) {
    return InvalidArgument("%s() is supported only for unsharded arrays.", api);
  }
  return py_arrays[0];
}

StatusOr<pybind11::object> PyArray::SingleDeviceArrayToNumpyArray() {
  TF_ASSIGN_OR_RETURN(auto arr,
                      FetchSingleShard("SingleDeviceArrayToNumpyArray"));
  return arr.GetStorage().host_value.AsNumPyArray(
      arr.GetStorage().dynamic_shape, arr.ifrt_array());
}

Status PyArray::CopySingleDeviceArrayToHostAsync() {
  TF_ASSIGN_OR_RETURN(auto arr,
                      FetchSingleShard("CopySingleDeviceArrayToHostAsync"));
  return arr.GetStorage().host_value.CopyToHostAsync(
      arr.GetStorage().dynamic_shape, arr.ifrt_array());
}

StatusOr<PyArray> PyArray::AssertUnsharded(std::string_view api) {
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

StatusOr<std::uintptr_t> PyArray::UnsafeBufferPointer() {
  TF_ASSIGN_OR_RETURN(auto arr, AssertUnsharded("UnsafeBufferPointer"));

  return py_client()->pjrt_client()->UnsafeBufferPointer(
      GetPjrtBuffer(arr.ifrt_array()));
}

py::dict PyArray::CudaArrayInterface() {
  auto arr_or_error = AssertUnsharded("UnsafeBufferPointer");
  if (!arr_or_error.ok()) {
    throw py::attribute_error(
        "__cuda_array_interface__ is only supported for unsharded arrays.");
  }
  auto arr = *arr_or_error;

  ifrt::Array* ifrt_array = arr.ifrt_array();
  std::optional<Shape>& scratch = arr.GetStorage().dynamic_shape;
  auto* pjrt_buffer = GetPjrtBuffer(ifrt_array);
  if (pjrt_buffer->client()->platform_id() != CudaId()) {
    throw py::attribute_error(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (pjrt_buffer->IsTuple()) {
    throw py::attribute_error(
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
      throw py::attribute_error(absl::StrFormat(
          "__cuda_array_interface__ is not supported for %s buffers.",
          PrimitiveType_Name(pjrt_buffer->element_type())));
  }

  py::str typestr =
      ValueOrThrow(TypeDescriptorForPrimitiveType(pjrt_buffer->element_type()));

  // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
  Layout xla_layout = GetXlaLayoutUnsafe(pjrt_buffer->layout());
  if (!LayoutUtil::IsMonotonicWithDim0Major(xla_layout)) {
    throw py::attribute_error(
        "__cuda_array_interface__ is only currently supported for "
        "buffers in row-major order.");
  }

  py::dict result;
  const auto* dynamic_shape =
      ValueOrThrow(XlaDynamicShape(ifrt_array, scratch));
  result["shape"] = SpanToTuple(dynamic_shape->dimensions());
  result["typestr"] = std::move(typestr);
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold =
      ValueOrThrow(pjrt_buffer->AcquireExternalReference());
  const void* root_ptr =
      external_reference_hold->OpaqueDeviceMemoryDataPointer();
  py::tuple data(2);
  data[0] = py::int_(absl::bit_cast<std::uintptr_t>(root_ptr));
  data[1] = py::bool_(true);  // read-only
  result["data"] = std::move(data);
  result["version"] = py::int_(2);
  return result;
}

StatusOr<pybind11::object> CudaArrayInterfaceToBuffer(
    const pybind11::dict& cai, std::shared_ptr<PyClient> client) {
#ifndef GOOGLE_CUDA
  throw XlaRuntimeError("This operation requires CUDA support.");
#else
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
  auto version = py::cast<int>(cai["version"]);
  if (version < 2 || version > 3) {
    LOG(WARNING) << "CUDA Array Interface version " << version
                 << " support is undefined";
  }
  auto data = py::cast<pybind11::tuple>(cai["data"]);
  auto data_value = pybind11::cast<std::intptr_t>(data[0]);
  void* data_ptr = reinterpret_cast<void*>(data_value);
  auto dimensions = pybind11::cast<std::vector<int64_t>>(cai["shape"]);
  if (data_value == 0 && absl::c_find(dimensions, 0) == dimensions.end()) {
    return absl::InvalidArgumentError(
        "CUDA Array Interface `data`(=NULL) and `shape`(no zero-valued "
        "dimensions) are inconsistent");
  }
  auto ndim = dimensions.size();
  TF_ASSIGN_OR_RETURN(
      PrimitiveType element_type,
      DtypeToPrimitiveType(py::dtype::from_args(cai["typestr"])));

  // cannot determine device_id/stream when device pointer is NULL.
  int device_id =
      (data_value == 0
           ? 0
           : stream_executor::gpu::CreatedContexts::GetDeviceOrdinal(data_ptr));
  TF_ASSIGN_OR_RETURN(auto device,
                      client->DeviceFromLocalHardwareId(device_id));
  bool is_default_stream =
      data_value == 0 || version == 2 ||
      (version == 3 && (!cai.contains("stream") || cai["stream"].is_none()));
  TF_ASSIGN_OR_RETURN(
      std::intptr_t stream,
      ([is_default_stream, cai, device]() -> StatusOr<std::intptr_t> {
        if (is_default_stream) {
          return device->GetStreamForExternalReadyEvents();
        } else {
          auto stream_ = py::cast<std::intptr_t>(cai["stream"]);
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
    auto strides = pybind11::cast<std::vector<int64_t>>(cai["strides"]);
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
  TF_ASSIGN_OR_RETURN(
      auto pjrt_buffer,
      device->client()->CreateViewOfDeviceBuffer(
          static_cast<char*>(data_ptr), shape, device.get(), on_delete_callback,
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
#endif  // GOOGLE_CUDA
}

Status PyArray::Delete() {
  for (auto& arr : py_arrays()) {
    TF_RETURN_IF_ERROR(arr.Delete());
  }
  py_arrays().clear();
  if (ifrt_array() != nullptr) {
    TF_RETURN_IF_ERROR(ifrt_array()->Delete().Await());
    SetIfrtArray(tsl::RCReference<ifrt::Array>());
  }
  return OkStatus();
}

bool PyArray::IsDeleted() const {
  if (ifrt_array() == nullptr) {
    return true;
  }

  return ifrt_array()->IsDeleted();
}

PyArray PyArray::Clone() const {
  tsl::RCReference<ifrt::Array> out =
      ifrt_array()
          ->Reshard(ifrt_array()->shared_ptr_sharding(),
                    ifrt::ArrayCopySemantics::kReuseInput)
          .value();
  return PyArray(aval(), weak_type(), dtype(),
                 std::vector<int64_t>(shape().begin(), shape().end()),
                 sharding(), py_client(), traceback(), std::move(out),
                 committed(), /*skip_checks=*/true);
}

py::handle PyArray::Storage::AsHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(PyArrayObject, array_storage));
}

PyArray::Storage::~PyArray_Storage() {
  CHECK(PyGILState_Check());
  if (!fastpath_enabled) {
    return;
  }
  if (py_client->arrays_ == this) {
    py_client->arrays_ = next;
  }
  if (prev) {
    prev->next = next;
  }
  if (next) {
    next->prev = prev;
  }
}

StatusOr<PyArray> PyArray::CopyToDeviceWithSharding(
    ifrt::DeviceList devices, pybind11::object dst_sharding) {
  auto* ifrt_array_ptr = ifrt_array();
  ifrt::MemoryKind dst_memory_kind =
      CreateIfRtMemoryKindFromSharding(dst_sharding);
  if (ifrt_array_ptr->sharding().devices().devices() == devices.devices() &&
      (!dst_memory_kind.memory_kind().has_value() ||
       !ifrt_array_ptr->sharding().memory_kind().memory_kind().has_value() ||
       ifrt_array_ptr->sharding().memory_kind() == dst_memory_kind)) {
    return *this;
  }
  tsl::RCReference<ifrt::Array> out_array;
  {
    auto transfer_guard_formatter = [this, &dst_sharding] {
      return absl::StrCat(
          "aval=", py::cast<std::string>(py::repr(aval())),
          ", sharding=", py::cast<std::string>(py::repr(sharding())),
          ", dst_sharding=", py::cast<std::string>(py::repr(dst_sharding)));
    };
    TF_RETURN_IF_ERROR(
        jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));
    GlobalPyRefManager()->CollectGarbage();
    py::gil_scoped_release gil_release;
    std::shared_ptr<const ifrt::Sharding> ifrt_sharding;
    // The sharding conversions are tried in the order of narrowness (e.g.,
    // ShardingParamSharding is an IFRT-level sharding, whereas HloSharding is
    // a IFRT-XLA level sharding).
    if (llvm::isa<ifrt::SingleDeviceSharding>(ifrt_array_ptr->sharding())) {
      ifrt_sharding =
          ifrt::SingleDeviceSharding::Create(devices[0], dst_memory_kind);
    } else if (const auto* in_sharding = llvm::dyn_cast<ifrt::OpaqueSharding>(
                   &ifrt_array_ptr->sharding());
               in_sharding != nullptr) {
      ifrt_sharding =
          ifrt::OpaqueSharding::Create(std::move(devices), dst_memory_kind);
    } else if (const auto* in_sharding = llvm::dyn_cast<ifrt::ConcreteSharding>(
                   &ifrt_array_ptr->sharding());
               in_sharding != nullptr) {
      ifrt_sharding = ifrt::ConcreteSharding::Create(
          std::move(devices), dst_memory_kind, in_sharding->shape(),
          in_sharding->shard_shapes());
    } else if (const auto* in_sharding =
                   llvm::dyn_cast<ifrt::ConcreteEvenSharding>(
                       &ifrt_array_ptr->sharding());
               in_sharding != nullptr) {
      ifrt_sharding = ifrt::ConcreteEvenSharding::Create(
          std::move(devices), dst_memory_kind, in_sharding->shape(),
          in_sharding->shard_shape());
    } else if (const auto* in_sharding =
                   llvm::dyn_cast<ifrt::ShardingParamSharding>(
                       &ifrt_array_ptr->sharding());
               in_sharding != nullptr) {
      TF_ASSIGN_OR_RETURN(ifrt_sharding,
                          ifrt::ShardingParamSharding::Create(
                              in_sharding->sharding_param(), std::move(devices),
                              dst_memory_kind));
    } else if (const auto* in_sharding = llvm::dyn_cast<ifrt::HloSharding>(
                   &ifrt_array_ptr->sharding());
               in_sharding != nullptr) {
      ifrt_sharding = ifrt::HloSharding::Create(
          std::move(devices), dst_memory_kind, in_sharding->xla_hlo_sharding());
    } else {
      return InvalidArgument(
          "resharding only supported for ifrt::SingleDeviceSharding, "
          "ifrt::OpaqueSharding, ifrt::ConcreteSharding, "
          "ifrt::ConcreteEvenSharding, ifrt::ShardingParamSharding, and "
          "ifrt::HloSharding.");
    }
    TF_ASSIGN_OR_RETURN(out_array, ifrt_array_ptr->Reshard(
                                       std::move(ifrt_sharding),
                                       ifrt::ArrayCopySemantics::kReuseInput));
  }
  auto traceback = Traceback::Get();
  absl::Span<const int64_t> shape_span = shape();
  return PyArray(aval(), weak_type(), dtype(),
                 std::vector<int64_t>(shape_span.begin(), shape_span.end()),
                 dst_sharding, py_client(), std::move(traceback),
                 std::move(out_array), committed(), /*skip_checks=*/true);
}

StatusOr<PyArray> PyArray::BatchedDevicePut(
    py::object aval, py::object sharding, std::vector<py::object> xs,
    std::vector<ClientAndPtr<PjRtDevice>> dst_devices, bool committed,
    bool force_copy, PjRtClient::HostBufferSemantics host_buffer_semantics,
    bool jax_enable_x64) {
  if (dst_devices.size() != xs.size() || xs.empty()) {
    throw py::value_error(
        absl::StrCat("Argument sizes (xs and devices) must match %zu vs "
                     "%zu and be nonzero",
                     dst_devices.size(), xs.size()));
  }
  for (ClientAndPtr<PjRtDevice>& device : dst_devices) {
    if (device.get_client() == nullptr) {
      return InvalidArgument("Cannot copy to unattached devices.");
    }
  }
  auto transfer_guard_formatter = [&aval, &sharding] {
    return absl::StrCat(
        "aval=", py::cast<std::string>(py::repr(aval)),
        ", dst_sharding=", py::cast<std::string>(py::repr(sharding)));
  };

  GlobalPyRefManager()->CollectGarbage();

  auto n_devices = dst_devices.size();

  DevicePutOptions options;
  options.squash_64bit_types = !jax_enable_x64;
  options.allow_zero_copy =
      (!force_copy &&
       (host_buffer_semantics == ifrt::Client::HostBufferSemantics::kZeroCopy));

  py::list owning_pylist(dst_devices.size());
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;

  xla::ifrt::DeviceList::Devices devices;
  devices.reserve(n_devices);
  std::vector<xla::ifrt::Shape> shapes;
  shapes.reserve(n_devices);

  ifrt::MemoryKind dst_memory_kind = CreateIfRtMemoryKindFromSharding(sharding);

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
        DevicePutResult on_device,
        DevicePut(x, dst_devices[i].get_client()->ifrt_client(),
                  dst_devices[i].get(), options, dst_memory_kind));
    ifrt_arrays.push_back(std::move(on_device.ifrt_array));
    devices.push_back(ifrt_arrays.back()->sharding().devices().front());
    shapes.push_back(ifrt_arrays.back()->shape());
    if (on_device.owning_pybuffer) {
      owning_pylist.append(on_device.owning_pybuffer);
    }
    ++i;
  }

  auto weak_type = pybind11::cast<bool>(aval.attr("weak_type"));
  auto dtype = aval.attr("dtype");
  auto shape = pybind11::cast<std::vector<int64_t>>(aval.attr("shape"));
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      ifrt_arrays.front()->client()->AssembleArrayFromSingleDeviceArrays(
          ifrt::Shape(shape),
          xla::ifrt::ConcreteSharding::Create(
              xla::ifrt::DeviceList(std::move(devices)), dst_memory_kind,
              /*shape=*/ifrt::Shape(shape),
              /*shard_shapes=*/std::move(shapes)),
          absl::MakeSpan(ifrt_arrays),
          xla::ifrt::ArrayCopySemantics::kReuseInput));

  return PyArray(aval, weak_type, dtype, std::move(shape), sharding,
                 dst_devices[0].client(), Traceback::Get(), ifrt_array,
                 committed, /*skip_checks=*/true);
}

std::vector<py::object> PyClient::LiveArrays() {
  std::vector<py::object> result;
  for (PyArray::Storage* array = arrays_; array; array = array->next) {
    bool all_deleted =
        (array->ifrt_array == nullptr || array->ifrt_array->IsDeleted());
    if (!all_deleted) {
      result.push_back(py::reinterpret_borrow<py::object>(array->AsHandle()));
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

int PyArray_bf_getbuffer(PyObject* exporter, Py_buffer* view, int flags) {
  Status status = [&]() {
    PyArray py_array = py::reinterpret_borrow<PyArray>(exporter);
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
    if (!py_array.sharding().get_type().is(jax::SingleDeviceSharding::type())) {
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

    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

    if (buffer.IsTuple()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for array buffers.");
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
      return InvalidArgument("XLA buffers are read-only.");
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
        buffer.AcquireExternalReference());
    if (buffer.IsDeleted()) {
      return InvalidArgument("Deleted buffer used in buffer protocol.");
    }

    // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
    Layout xla_layout = GetXlaLayoutUnsafe(buffer.layout());

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
    }
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
          extra->strides = ByteStridesForShape(buffer.element_type(),
                                               buffer.dimensions(), xla_layout);
          view->strides = reinterpret_cast<Py_ssize_t*>(
              const_cast<int64_t*>(extra->strides.data()));
        }
      }
    }
    TF_RETURN_IF_ERROR(buffer.BlockHostUntilReady());
    view->internal = extra.release();
    return OkStatus();
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

PyBufferProcs PyArray_tp_as_buffer = []() {
  PyBufferProcs procs;
  procs.bf_getbuffer = &PyArray_bf_getbuffer;
  procs.bf_releasebuffer = &PyArray_bf_releasebuffer;
  return procs;
}();

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

}  // namespace

PyHostValue::PyHostValue() = default;
PyHostValue::~PyHostValue() = default;

StatusOr<pybind11::object> PyHostValue::AsNumPyArray(
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array) {
  if (ifrt_array->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    TF_RET_CHECK(!pjrt_buffer->IsTuple());
    // On CPU for non-int4 values, we can return the value in a zero-copy way.
    // For int4 values, we must copy in order to unpack the array.
    if (pjrt_buffer->IsOnCpu() &&
        !primitive_util::Is4BitType(pjrt_buffer->element_type())) {
      TF_ASSIGN_OR_RETURN(const auto* shape,
                          XlaDynamicShape(ifrt_array, dynamic_shape_holder));
      TF_ASSIGN_OR_RETURN(py::dtype dtype,
                          PrimitiveTypeToDtype(shape->element_type()));
      // Objects that must be kept alive while the array is alive.
      struct Hold {
        tsl::RCReference<ifrt::Array> buffer;
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
      };
      auto hold = std::make_unique<Hold>();
      TF_ASSIGN_OR_RETURN(hold->external_reference_hold,
                          pjrt_buffer->AcquireExternalReference());
      hold->buffer = tsl::FormRef(ifrt_array);
      void* data =
          hold->external_reference_hold->OpaqueDeviceMemoryDataPointer();
      py::capsule hold_capsule(hold.release(),
                               [](void* h) { delete static_cast<Hold*>(h); });
      py::array array(dtype, shape->dimensions(), ByteStridesForShape(*shape),
                      data, hold_capsule);
      array.attr("flags").attr("writeable") = Py_False;
      {
        py::gil_scoped_release gil;
        TF_RETURN_IF_ERROR(ifrt_array->GetReadyFuture().Await());
      }
      return array;
    }
  }

  TF_RETURN_IF_ERROR(CopyToHostAsync(dynamic_shape_holder, ifrt_array));
  if (!ready_.IsReady()) {
    py::gil_scoped_release gil;
    TF_RETURN_IF_ERROR(ready_.Await());
  } else {
    TF_RETURN_IF_ERROR(ready_.Await());
  }
  return value_;
}

Status PyHostValue::CopyToHostAsync(std::optional<Shape>& dynamic_shape_holder,
                                    ifrt::Array* ifrt_array) {
  if (ready_.IsValid()) {
    // The array value has been populated, so CopyToHostAsync has been called.
    return OkStatus();
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    if (pjrt_buffer->IsOnCpu() &&
        !primitive_util::Is4BitType(pjrt_buffer->element_type())) {
      return OkStatus();
    }
  }
  auto transfer_guard_formatter = [ifrt_array] {
    return absl::StrCat(
        "shape=()", absl::StrJoin(ifrt_array->shape().dims(), ","),
        "), dtype=", ifrt_array->dtype().DebugString(),
        ", device=", ifrt_array->sharding().devices().front()->DebugString());
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
  TF_ASSIGN_OR_RETURN(py::dtype dtype,
                      PrimitiveTypeToDtype(host_shape.element_type()));
  value_ = py::array(dtype, host_shape.dimensions(),
                     strides ? *strides : std::vector<int64_t>{});
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
  ready_.OnReady([array{value_.ptr()}](Status status) {
    GlobalPyRefManager()->AddGarbage(py::reinterpret_steal<py::object>(array));
  });
  value_.attr("flags").attr("writeable") = Py_False;
  return OkStatus();
}

Status PyArray::SetUpType() {
  static constexpr char kName[] = "ArrayImpl";

  py::str name = py::str(kName);
  py::str qualname = py::str(kName);

  auto* heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  // Caution: we must not call any functions that might invoke the GC until
  // PyType_Ready() is called below. Otherwise the GC might see a
  // half-constructed type object.
  if (!heap_type) {
    return Internal("Unable to create heap type object");
  }
  heap_type->ht_name = name.release().ptr();
  heap_type->ht_qualname = qualname.release().ptr();
  PyTypeObject* type = &heap_type->ht_type;
  type->tp_name = kName;
  type->tp_basicsize = sizeof(PyArrayObject);
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
  type->tp_new = PyArray_tp_new;
  type->tp_dealloc = PyArray_tp_dealloc;

  // Supported protocols
  type->tp_as_number = &heap_type->as_number;
  type->tp_as_sequence = &heap_type->as_sequence;
  type->tp_as_mapping = &heap_type->as_mapping;
  type->tp_as_buffer = &PyArray_tp_as_buffer;

  // Allow dynamic attributes.
  EnableDynamicAttribute(heap_type);

  // Allow weak references to DeviceArray objects.
  type->tp_weaklistoffset = offsetof(PyArrayObject, weakrefs);

  TF_RET_CHECK(PyType_Ready(type) == 0);

  PyArray::type_ = reinterpret_cast<PyObject*>(type);

  return OkStatus();
}

Status PyArray::RegisterTypes(py::module& m) {
  TF_RETURN_IF_ERROR(PyArray::SetUpType());
  auto type = py::reinterpret_borrow<py::object>(type_);
  m.attr("ArrayImpl") = type;

  type.attr("__init__") = py::cpp_function(
      [](py::object self, py::object aval, py::object sharding, py::list arrays,
         bool committed, bool skip_checks) {
        if (arrays[0].get_type().is(PyArray::type())) {
          auto py_arrays = py::cast<std::vector<PyArray>>(arrays);
          PyArray::PyInit(self, std::move(aval), std::move(sharding), py_arrays,
                          committed, skip_checks);
        } else {
          throw py::type_error(
              absl::StrCat("Unsupported type for elements in `arrays`: ",
                           std::string(py::str(arrays[0].get_type()))));
        }
      },
      py::is_method(type), py::arg("aval"), py::arg("sharding"),
      py::arg("arrays"), py::arg("committed"), py::arg("_skip_checks") = false);
  // TODO(yashkatariya): remove this once the transition completes.
  type.attr("_init_with_fastpath_disabled") = py::cpp_function(
      [](py::object self) {
        PyArray::PyInit(self, PyArray::DisableFastpath());
      },
      py::is_method(type));
  type.attr("delete") =
      py::cpp_function([](PyArray& self) { xla::ThrowIfError(self.Delete()); },
                       py::is_method(type));
  type.attr("_sharding") = jax::property_readonly(&PyArray::sharding);
  type.attr("aval") = jax::property(&PyArray::aval, &PyArray::set_aval);
  type.attr("_arrays") =
      jax::property(&PyArray::arrays, [](PyArray& self, py::object obj) {
        xla::ThrowIfError(self.set_arrays(obj));
      });
  type.attr("_fully_replicated_shard") = py::cpp_function(
      [](PyArray self) {
        return xla::ValueOrThrow(self.FullyReplicatedShard());
      },
      py::is_method(type));
  type.attr("_npy_value") =
      jax::property(&PyArray::npy_value, &PyArray::set_npy_value);
  type.attr("_committed") = jax::property_readonly(&PyArray::committed);
  type.attr("unsafe_buffer_pointer") = py::cpp_function(
      [](PyArray self) {
        return xla::ValueOrThrow(self.UnsafeBufferPointer());
      },
      py::is_method(type));
  type.attr("__cuda_array_interface__") = jax::property_readonly(
      [](PyArray self) { return self.CudaArrayInterface(); });
  type.attr("on_device_size_in_bytes") = py::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::GetOnDeviceSizeInBytes),
      py::is_method(type));
  type.attr("_single_device_array_to_np_array") = py::cpp_function(
      xla::ValueOrThrowWrapper(&PyArray::SingleDeviceArrayToNumpyArray),
      py::is_method(type));
  type.attr("_copy_single_device_array_to_host_async") = py::cpp_function(
      [](PyArray& self) {
        xla::ThrowIfError(self.CopySingleDeviceArrayToHostAsync());
      },
      py::is_method(type));
  type.attr("block_until_ready") = py::cpp_function(
      [](PyArray self) -> py::object {
        xla::ThrowIfError(self.BlockUntilReady());
        return self;
      },
      py::is_method(type));
  type.attr("platform") = py::cpp_function(
      [](PyArray self) {
        if (self.ifrt_array()->client()->platform_name() == "cuda" ||
            self.ifrt_array()->client()->platform_name() == "rocm") {
          return absl::string_view("gpu");
        } else {
          return self.ifrt_array()->client()->platform_name();
        }
      },
      py::is_method(type));
  type.attr("is_ready") = py::cpp_function(
      [](PyArray self) { return xla::ValueOrThrow(self.IsReady()); },
      py::is_method(type));
  type.attr("is_deleted") =
      py::cpp_function(&PyArray::IsDeleted, py::is_method(type));
  // TODO(phawkins): just use &PyArray::traceback when
  // nanobind port is complete.
  type.attr("traceback") =
      jax::property_readonly([](PyArray self) -> py::object {
        if (self.traceback()) {
          return py::reinterpret_borrow<py::object>(self.traceback()->ptr());
        } else {
          return py::none();
        }
      });
  type.attr("clone") = py::cpp_function(&PyArray::Clone, py::is_method(type));
  type.attr("__module__") = m.attr("__name__");

  m.attr("copy_array_to_devices_with_sharding") = py::cpp_function(
      [](PyArray self, std::vector<ClientAndPtr<PjRtDevice>> dst_devices,
         py::object sharding) {
        ifrt::DeviceList::Devices devices;
        devices.reserve(dst_devices.size());
        for (auto& d : dst_devices) {
          devices.push_back(d.get());
        }
        return xla::ValueOrThrow(self.CopyToDeviceWithSharding(
            ifrt::DeviceList(devices), std::move(sharding)));
      });
  m.attr("array_result_handler") = py::cpp_function(
      [](py::object aval, py::object sharding, bool committed,
         bool skip_checks) -> std::unique_ptr<PyArrayResultHandler> {
        return std::make_unique<PyArrayResultHandler>(
            std::move(aval), std::move(sharding), committed, skip_checks);
      },
      py::arg("aval"), py::arg("sharding"), py::arg("committed"),
      py::arg("_skip_checks") = false);

  py::class_<PyArrayResultHandler>(m, "ResultHandler")
      .def("__call__", [](const PyArrayResultHandler& self,
                          PyArray arg) { return self.Call(arg); })
      .def("__call__",
           [](const PyArrayResultHandler& self,
              std::vector<PyArray> py_arrays) { return self.Call(py_arrays); });

  return OkStatus();
}

}  // namespace xla
