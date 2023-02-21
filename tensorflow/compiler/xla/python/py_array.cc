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

#include "tensorflow/compiler/xla/python/py_array.h"

#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/Support/Casting.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/sharding.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/statusor.h"

#define PYBIND11_DETAILED_ERROR_MESSAGES
namespace xla {
namespace {

namespace py = pybind11;

tsl::RCReference<ifrt::Array> CreateIfRtArrayFromPyBuffers(
    py::dtype dtype, absl::Span<const int64_t> shape,
    absl::Span<const PyBuffer::object> py_buffers) {
  if (py_buffers.empty()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error("At least one buffer must be provided.");
  }

  auto* ifrt_client = py_buffers.front().buf()->client()->ifrt_client();

  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(py_buffers.size());
  ifrt::DeviceList::Devices devices;
  devices.reserve(py_buffers.size());
  std::vector<ifrt::Shape> shapes;
  shapes.reserve(py_buffers.size());

  for (const auto& py_buffer : py_buffers) {
    ifrt_arrays.push_back(tsl::FormRef(py_buffer.buf()->ifrt_array()));
    devices.push_back(ifrt_arrays.back()->sharding().devices()[0]);
    shapes.push_back(ifrt_arrays.back()->shape());
  }
  auto ifrt_array = ifrt_client->AssembleArrayFromSingleDeviceArrays(
      ifrt::Shape(shape),
      ifrt::OpaqueSharding::Create(
          ifrt::DeviceList(std::move(devices)),
          xla::ifrt::OpaqueSharding::MakeDisassembleFuncFromShapes(
              std::move(shapes))),
      absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput);
  if (!ifrt_array.ok()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error(ifrt_array.status().ToString());
  }
  return *std::move(ifrt_array);
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

  for (const auto& py_array : py_arrays) {
    DCHECK_EQ(py_array.num_shards(), 1);
    ifrt_arrays.push_back(tsl::FormRef(py_array.ifrt_array()));
    devices.push_back(ifrt_arrays.back()->sharding().devices().front());
    shapes.push_back(ifrt_arrays.back()->shape());
  }
  ifrt::Client* client = ifrt_arrays.front()->client();

  auto ifrt_dtype = ToIfRtDType(dtype);
  if (!ifrt_dtype.ok()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error(ifrt_dtype.status().ToString());
  }
  auto ifrt_array = client->AssembleArrayFromSingleDeviceArrays(
      ifrt::Shape(shape),
      ifrt::OpaqueSharding::Create(
          ifrt::DeviceList(std::move(devices)),
          xla::ifrt::OpaqueSharding::MakeDisassembleFuncFromShapes(
              std::move(shapes))),
      absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput);
  if (!ifrt_array.ok()) {
    // TODO(hyeontaek): Return a Status.
    throw py::value_error(ifrt_array.status().ToString());
  }
  return *std::move(ifrt_array);
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

}  // namespace

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

void PyArray::PyInit(py::object self, py::object aval, py::object sharding,
                     absl::Span<const PyBuffer::object> py_buffers,
                     bool committed, bool skip_checks) {
  auto dtype = aval.attr("dtype");
  auto shape = pybind11::cast<std::vector<int64_t>>(aval.attr("shape"));
  auto ifrt_array = CreateIfRtArrayFromPyBuffers(dtype, shape, py_buffers);
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()), aval,
            pybind11::cast<bool>(aval.attr("weak_type")), std::move(dtype),
            std::move(shape), std::move(sharding), committed,
            py_buffers.at(0).buf()->client(), Traceback::Get(),
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

PyArray::PyArray(py::object aval, bool weak_type, py::dtype dtype,
                 std::vector<int64_t> shape, py::object sharding,
                 std::shared_ptr<PyClient> py_client,
                 std::shared_ptr<Traceback> traceback,
                 tsl::RCReference<ifrt::Array> ifrt_array,
                 bool committed, bool skip_checks) {
  auto* self =
      PyArray_tp_new(reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr);
  ptr() = self;
  Construct(reinterpret_cast<PyArrayObject*>(self), std::move(aval), weak_type,
            std::move(dtype), std::move(shape), std::move(sharding), committed,
            std::move(py_client), std::move(traceback),
            std::move(ifrt_array)
  );

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

PyBuffer::object PyArray::ToPyBuffer() const {
  return PyBuffer::Make(py_client(),
                        ifrt_array()
                            ->Reshard(ifrt_array()->shared_ptr_sharding(),
                                      ifrt::ArrayCopySemantics::kReuseInput)
                            .value(),
                        traceback());
}

PyArray PyArray::MakeFromSingleDevice(std::shared_ptr<PyClient> py_client,
                                      std::shared_ptr<Traceback> traceback,
                                      tsl::RCReference<ifrt::Array> ifrt_array,
                                      bool weak_type, bool committed) {
  if (!llvm::isa<ifrt::SingleDeviceSharding>(ifrt_array->sharding())) {
    throw XlaRuntimeError(
        InvalidArgument("Constructing single device jax.Array from non-single "
                        "device ifrt array."));
  }
  static const py::handle* shaped_array = nullptr;
  if (shaped_array == nullptr) {
    auto* jax_core = PyImport_ImportModule("jax.core");
    if (jax_core != nullptr) {
      shaped_array = new py::handle(
          py::reinterpret_steal<py::module>(jax_core).attr("ShapedArray"));
    } else {
      PyErr_Clear();
    }
  }
  PrimitiveType primitive = ifrt::ToPrimitiveType(ifrt_array->dtype()).value();
  auto dtype = PrimitiveTypeToDtype(primitive).value();
  py::object aval;
  if (shaped_array != nullptr) {
    aval = (*shaped_array)(SpanToTuple(ifrt_array->shape().dims()), dtype,
                           weak_type);
  } else {
    aval = py::none();
  }
  auto shape_span = ifrt_array->shape().dims();
  auto shape = std::vector<int64_t>(shape_span.begin(), shape_span.end());
  auto sharding = py::cast(std::make_unique<jax::SingleDeviceSharding>(py::cast(
      WrapWithClient(py_client, ifrt_array->sharding().devices().front()))));
  return PyArray(std::move(aval), weak_type, dtype, std::move(shape),
                 std::move(sharding), std::move(py_client),
                 std::move(traceback), std::move(ifrt_array), committed);
}

py::object PyArray::arrays() {
  // For performance, we only keep pjrt buffers by default. But on python side
  // "_arrays" returns PyArrays instead, and subsequent calls to "_arrays"
  // should return the same PyArrays (to avoid duplicate device to host
  // transfers). So we create PyArrays the first time it is called and reuse
  // them later.
  if (ifrt_array() == nullptr) return py::none();

  if (llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    std::vector<PyArray> py_arrays;
    py_arrays.push_back(*this);
    return py::cast(py_arrays);
  }

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
      py_arrays.push_back(PyArray::MakeFromSingleDevice(
          py_client(), traceback(), std::move(ifrt_array), weak_type(),
          committed()));
    }
  }

  return py::cast(py_arrays);
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
    // TODO(chky): Currently only List[Buffer] is handled here. We need to
    // handle List[Array] as well.
    if (obj.get_type().ptr() == PyBuffer::type()) {
      auto* py_buffer = PyBuffer::AsPyBufferUnchecked(obj);
      DCHECK_EQ(py_buffer->client(), py_client());
      // TODO(hyeontaek): This should return an error instead of failing.
      CHECK(py_buffer->ifrt_array() != nullptr);
      ifrt_arrays.push_back(tsl::FormRef(py_buffer->ifrt_array()));
      devices.push_back(ifrt_arrays.back()->sharding().devices().front());
      shapes.push_back(ifrt_arrays.back()->shape());
    } else if (obj.get_type().is(PyArray::type())) {
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
  TF_ASSIGN_OR_RETURN(
      auto array,
      py_client()->ifrt_client()->AssembleArrayFromSingleDeviceArrays(
          ifrt::Shape(shape()),
          ifrt::OpaqueSharding::Create(
              ifrt::DeviceList(std::move(devices)),
              xla::ifrt::OpaqueSharding::MakeDisassembleFuncFromShapes(
                  std::move(shapes))),
          absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput));
  SetIfrtArray(std::move(array));
  return OkStatus();
}

Status PyArray::BlockUntilReady() const {
  pybind11::gil_scoped_release gil_release;
  Status status;
  if (ifrt_array() == nullptr) {
    return InvalidArgument(
        "BlockHostUntilReady() called on deleted or donated buffer");
  }
  return AwaitBuffersReady(ifrt_array());
  return status;
}

StatusOr<pybind11::object> PyArray::SingleDeviceArrayAsNumPyArray() {
  if (!llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    throw xla::XlaRuntimeError(
        InvalidArgument("_single_device_array_to_np_array only supported for "
                        "unsharded arrays."));
  }
  return PyHostValue::AsNumPyArray(
      GetStorage().host_value, GetStorage().dynamic_shape, ifrt_array(), *this);
}

Status PyArray::CopySingleDeviceArrayToHostAsync() {
  if (!llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    throw xla::XlaRuntimeError(
        InvalidArgument("_single_device_array_to_np_array only supported for "
                        "unsharded arrays."));
  }
  return PyHostValue::CopyToHostAsync(GetStorage().host_value,
                                      GetStorage().dynamic_shape, ifrt_array());
}

StatusOr<const Shape*> PyArray::xla_dynamic_shape() {
  return IfrtHelpers::xla_dynamic_shape(ifrt_array(),
                                        GetStorage().dynamic_shape);
}

StatusOr<py::object> PyArray::CopyToDevice(PjRtDevice* device) {
  if (!llvm::isa<ifrt::SingleDeviceSharding>(&ifrt_array()->sharding())) {
    throw xla::XlaRuntimeError(
        InvalidArgument("copy_to_device only supported for unsharded arrays."));
  }

  if (device == IfrtHelpers::pjrt_device(ifrt_array())) {
    if (committed()) { return *this; }
    tsl::RCReference<ifrt::Array> out =
        ifrt_array()
            ->Reshard(ifrt_array()->shared_ptr_sharding(),
                      ifrt::ArrayCopySemantics::kReuseInput)
            .value();
    return py::object(PyArray::MakeFromSingleDevice(
        py_client(), traceback(), std::move(out), weak_type(), true));
  }

  TF_ASSIGN_OR_RETURN(tsl::RCReference<ifrt::Array> out,
                      IfrtHelpers::CopyToDevice(ifrt_array(), device));
  return py::object(PyArray::MakeFromSingleDevice(
      py_client(), traceback(), std::move(out), weak_type(), true));
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
                 committed(), /* skip_checks= */ true);
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

std::vector<py::object> PyClient::LiveArrays() {
  absl::flat_hash_set<PyArray::Storage*> duplicate_arrays;
  std::vector<PyArray::Storage*> all_arrays;
  for (PyArray::Storage* array = arrays_; array; array = array->next) {
    bool all_deleted =
        (array->ifrt_array == nullptr || array->ifrt_array->IsDeleted());
    if (!all_deleted) {
      for (auto& shard_array : array->py_arrays) {
        duplicate_arrays.insert(GetPyArrayStorageFromObject(
            reinterpret_cast<PyArrayObject*>(shard_array.ptr())));
      }
      all_arrays.push_back(array);
    }
  }
  // TODO(parkers): This preserves the current behavior, but it should instead
  // populate py_arrays for all valid arrays and return a list of all single
  // device arrays deduplicated by their buffer ids.
  std::vector<py::object> result;
  for (PyArray::Storage* array : all_arrays) {
    if (!duplicate_arrays.contains(array)) {
      result.push_back(py::reinterpret_borrow<py::object>(array->AsHandle()));
    }
  }
  return result;
}

Status PyArray::SetUpType() {
  static constexpr char kName[] = "Array";

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
        } else if (arrays[0].get_type().ptr() == PyBuffer::type()) {
          auto py_buffers = py::cast<std::vector<PyBuffer::object>>(arrays);
          PyArray::PyInit(self, std::move(aval), std::move(sharding),
                          py_buffers, committed, skip_checks);
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
  type.attr("delete") = py::cpp_function(&PyArray::Delete, py::is_method(type));
  type.attr("_sharding") = jax::property_readonly(&PyArray::sharding);
  type.attr("aval") = jax::property(&PyArray::aval, &PyArray::set_aval);
  type.attr("_arrays") = jax::property(&PyArray::arrays, &PyArray::set_arrays);
  type.attr("_npy_value") =
      jax::property(&PyArray::npy_value, &PyArray::set_npy_value);
  type.attr("_committed") = jax::property_readonly(&PyArray::committed);
  type.attr("block_until_ready") = py::cpp_function(
      [](PyArray self) -> StatusOr<py::object> {
        TF_RETURN_IF_ERROR(self.BlockUntilReady());
        return self;
      },
      py::is_method(type));
  type.attr("platform") = py::cpp_function(
      [](PyArray self) { return self.ifrt_array()->client()->platform_name(); },
      py::is_method(type));
  type.attr("unsafe_buffer_pointer") = py::cpp_function(
      [](PyArray self) {
        return self.py_client()->pjrt_client()->UnsafeBufferPointer(
            IfrtHelpers::pjrt_buffer(self.ifrt_array()));
      },
      py::is_method(type));
  type.attr("_on_device_size_in_bytes") = py::cpp_function(
      [](PyArray self) -> StatusOr<size_t> {
        return IfrtHelpers::pjrt_buffer(self.ifrt_array())
            ->GetOnDeviceSizeInBytes();
      },
      py::is_method(type));
  type.attr("copy_to_device") = py::cpp_function(
      [](PyArray self, const ClientAndPtr<PjRtDevice>& dst_device) {
        return self.CopyToDevice(dst_device.get());
      },
      py::is_method(type));
  type.attr("_single_device_array_to_np_array") = py::cpp_function(
      &PyArray::SingleDeviceArrayAsNumPyArray, py::is_method(type));
  type.attr("_copy_single_device_array_to_host_async") = py::cpp_function(
      &PyArray::CopySingleDeviceArrayToHostAsync, py::is_method(type));
  type.attr("is_ready") = py::cpp_function(
      [](PyArray self) { return self.IsReady(); }, py::is_method(type));
  type.attr("is_deleted") =
      py::cpp_function(&PyArray::IsDeleted, py::is_method(type));
  type.attr("clone") = py::cpp_function(&PyArray::Clone, py::is_method(type));
  type.attr("traceback") = jax::property_readonly(&PyArray::traceback);
  type.attr("__module__") = m.attr("__name__");

  return OkStatus();
}

}  // namespace xla
