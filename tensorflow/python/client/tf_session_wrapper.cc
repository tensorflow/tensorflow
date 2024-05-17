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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "Python.h"
#include "absl/types/optional.h"
#include "Eigen/Core"  // from @eigen_archive
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/chrono.h"  // from @pybind11
#include "pybind11/complex.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/python_api.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/tf_datatype.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/version_info.h"
#include "tensorflow/python/client/tf_session_helper.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tsl/platform/mutex.h"

namespace pybind11 {
namespace detail {

// Convert between absl::optional and python.
//
// pybind11 supports std::optional, and absl::optional is meant to be a
// drop-in replacement for std::optional, so we can just use the built in
// implementation.
#ifndef ABSL_USES_STD_OPTIONAL
template <typename T>
struct type_caster<absl::optional<T>>
    : public optional_caster<absl::optional<T>> {};
template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

}  // namespace detail
}  // namespace pybind11

// TODO(amitpatankar): Consolidate Buffer methods into a separate header file.
TF_Buffer* ProtoStringToTFBuffer(PyObject* input) {
  // Convert a Python string object to TF_Buffer.
  char* c_string;
  Py_ssize_t py_size;
  // PyBytes_AsStringAndSize() does not copy but simply interprets the input
  if (PyBytes_AsStringAndSize(input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    throw py::error_already_set();
  }
  return TF_NewBufferFromString(static_cast<void*>(c_string),
                                static_cast<size_t>(py_size));
}

// Copied from tf_session.i
// We have to do convoluted logic of passing in a vector of py::bytes. If we
// pass in strings they are freed prior to the necessary function calls.
tensorflow::NameVector ConvertPyListToNameVector(
    const std::vector<py::bytes>& py_vector) {
  tensorflow::NameVector temp;
  for (size_t i = 0; i < py_vector.size(); ++i) {
    const char* string_elem = PyBytes_AsString(py_vector.at(i).ptr());
    temp.push_back(string_elem);
  }
  return temp;
}

namespace py = pybind11;

// TODO(power) -- share these with JAX (see python_utils.h)
template <typename Func, typename... Extra>
pybind11::object property_readonly(Func&& get, const char* doc = "") {
  pybind11::handle property_class(
      reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property_class(
      pybind11::cpp_function(std::forward<Func>(get),
                             py::return_value_policy::reference_internal),
      pybind11::none(), pybind11::none(), doc);
}

template <typename GetFunc, typename SetFunc>
pybind11::object property(GetFunc&& get, SetFunc&& set) {
  pybind11::handle property_class(
      reinterpret_cast<PyObject*>(&PyProperty_Type));
  return property_class(
      pybind11::cpp_function(std::forward<GetFunc>(get),
                             py::return_value_policy::reference_internal),
      pybind11::cpp_function(std::forward<SetFunc>(set)), pybind11::none(), "");
}

template <typename Constructor>
pybind11::object def_static(Constructor&& constructor) {
  return pybind11::staticmethod(
      pybind11::cpp_function(std::forward<Constructor>(constructor)));
}

template <typename Func, typename... Extra>
pybind11::object method(pybind11::object type, Func&& function,
                        const Extra&... extra) {
  return pybind11::cpp_function(std::forward<Func>(function),
                                pybind11::is_method(type), extra...);
}

// Construct a "TF" Python object. This covers the boiler-plate for Python type
// generation. The type is assumed to be a GC type (containing other types).
// To add the required Python type fields, classes definitions must start with
//
// TFObject_Head(classname, TfObjectDataType)
//
// Required attributes/methods for TfObjectDataType type:
//
// Constructor(PyObject* args, PyObject* kw)
// ~Destructor
// Clear()
// Visit(visitproc visit, void* arg)
//
// Individual methods/attributes are added to the type later, as seen below.
template <typename T>
void MakeTfObjectType(PyObject** py_type) {
  using TfObjectDataType = typename T::TfObjectDataType;

  py::str name = py::str(T::kTypeName);
  py::str qualname = py::str(T::kTypeName);
  PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));

  heap_type->ht_name = name.release().ptr();
  heap_type->ht_qualname = qualname.release().ptr();

  PyTypeObject* type = &heap_type->ht_type;
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE |
                   Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE;
  type->tp_name = T::kTypeName;

  // Allocation size for both Python object header and the TF data members.
  type->tp_basicsize = sizeof(T) + sizeof(TfObjectDataType);

  type->tp_new = [](PyTypeObject* subtype, PyObject* args,
                    PyObject* kwds) -> PyObject* {
    T* self = reinterpret_cast<T*>(subtype->tp_alloc(subtype, 0));
    TfObjectDataType* data = reinterpret_cast<TfObjectDataType*>(&self[1]);
    if (!self) return nullptr;

    // PyType_GenericAlloc (the default implementation of tp_alloc) by default
    // enables the garbage collector immediately for our object. This makes
    // initialization extremely tricky as we need to avoid having the object
    // in an invalid intermediate state.
    //
    // We disable the GC here until initialization is finished.
    PyObject_GC_UnTrack(self);
    new (data) TfObjectDataType(args, kwds);
    self->dict = PyDict_New();
    PyObject_GC_Track(self);

    if (PyErr_Occurred()) {
      return nullptr;
    }
    return reinterpret_cast<PyObject*>(self);
  };

  type->tp_dealloc = [](PyObject* self) {
    VLOG(3) << "Destroy: " << T::kTypeName;
    PyObject_GC_UnTrack(self);
    PyTypeObject* tp = Py_TYPE(self);
    PyObject_ClearWeakRefs(self);

    T* o = reinterpret_cast<T*>(self);
    TfObjectDataType* data = reinterpret_cast<TfObjectDataType*>(&o[1]);
    Py_CLEAR(o->dict);
    data->~TfObjectDataType();
    tp->tp_free(self);
    Py_DECREF(tp);
  };

  type->tp_traverse = [](PyObject* self, visitproc visit, void* arg) {
    VLOG(3) << "Visit: " << T::kTypeName;
    T* o = reinterpret_cast<T*>(self);
    TfObjectDataType* data = reinterpret_cast<TfObjectDataType*>(&o[1]);
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(o->dict);
    return data->Visit(visit, arg);
  };

  type->tp_clear = [](PyObject* self) {
    VLOG(3) << "Clear: " << T::kTypeName;
    T* o = reinterpret_cast<T*>(self);
    TfObjectDataType* data = reinterpret_cast<TfObjectDataType*>(&o[1]);
    Py_CLEAR(o->dict);
    data->Clear();
    return 0;
  };

  type->tp_weaklistoffset = offsetof(T, weakrefs);

  // All TF objects use a dictionary today, so we initialize it at construction.
  // If some types become fully C++ based or require only thin Python wrappers,
  // we can instead defer dictionary creation using a custom getter/setter.
  type->tp_dictoffset = offsetof(T, dict);

  // type->tp_getset = &tp_getset[0];
  type->tp_descr_get = nullptr;
  type->tp_descr_set = nullptr;
  type->tp_call = nullptr;
  type->tp_vectorcall_offset = 0;

  type->tp_repr = nullptr;

  if (PyType_Ready(type) != 0) {
    PyErr_Print();
    LOG(FATAL) << "Failed to build type.";  // Crash ok. In module init.
  }
  *py_type = reinterpret_cast<PyObject*>(type);
}

#define TFObject_HEAD(typename, datatypename) \
  using TfObjectDataType = datatypename;      \
  PyObject_HEAD;                              \
  PyObject* dict = nullptr;                   \
  PyObject* weakrefs = nullptr;               \
  TfObjectDataType data[0];                   \
  static PyObject* py_type;                   \
  static constexpr const char* kTypeName = #typename;

struct PyGraph;
struct PyOperation;
struct PyTensor;

// Bind operation maps opaquely to avoid copying.
typedef absl::flat_hash_map<int64_t, py::object> OpsByIdMap;
typedef absl::flat_hash_map<std::string, py::object> OpsByNameMap;

PYBIND11_MAKE_OPAQUE(TF_Operation);
PYBIND11_MAKE_OPAQUE(TF_Graph);
PYBIND11_MAKE_OPAQUE(TF_Session);
PYBIND11_MAKE_OPAQUE(TF_Buffer);
PYBIND11_MAKE_OPAQUE(TF_ImportGraphDefOptions);
PYBIND11_MAKE_OPAQUE(TF_ImportGraphDefResults);
PYBIND11_MAKE_OPAQUE(TF_DeprecatedSession);
PYBIND11_MAKE_OPAQUE(TF_OperationDescription);
PYBIND11_MAKE_OPAQUE(TF_Library);
PYBIND11_MAKE_OPAQUE(TF_SessionOptions);
PYBIND11_MAKE_OPAQUE(TF_ApiDefMap);
PYBIND11_MAKE_OPAQUE(TF_Server);
PYBIND11_MAKE_OPAQUE(TF_DeviceList);
PYBIND11_MAKE_OPAQUE(TF_Status);

PYBIND11_MAKE_OPAQUE(OpsByIdMap);
PYBIND11_MAKE_OPAQUE(OpsByNameMap);

// Convert the given handle to a TF object type.
template <typename T>
T* AsPyTfObject(py::handle handle) {
  if (handle.get_type() == T::py_type) {
    return reinterpret_cast<T*>(handle.ptr());
  }
  if (PyType_IsSubtype(Py_TYPE(handle.ptr()),
                       reinterpret_cast<PyTypeObject*>(T::py_type))) {
    return reinterpret_cast<T*>(handle.ptr());
  }
  // The tf_should_use wrapper masquerades as a base class, and forwards
  // attribute lookups to an underlying class. This should be removed (it is
  // slow, confusing, and not so relevant with TF2), or at least moved to the
  // C++ wrapper classes (it is only used on Tensor and Operation). In the
  // meantime, use a custom caster to handle the cases where we are passed a
  // `tf_should_use` instead of the original class.
  if (py::hasattr(handle, "_tf_should_use_wrapped_value")) {
    return AsPyTfObject<T>(py::getattr(handle, "_tf_should_use_wrapped_value"));
  }

  throw std::runtime_error(
      absl::StrCat("Expected a ", T::kTypeName, " got ",
                   py::cast<std::string>(py::str(handle))));
}

template <typename T>
py::object AsPyObject(T* obj) {
  return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(obj));
}

template <typename T>
typename T::TfObjectDataType* AsPyTfObjectData(py::handle handle) {
  return AsPyTfObject<T>(handle)->data;
}
// Reference counting helper for PyTfObjects.
//
// Similar to the pybind holder types, this manages the Python reference
// counting while allowing access to the underlying PyTfObject type.
//
// As a special case to support Dismantle(), this allows setting our underlying
// pointer to None when clearing the type. Direct access to attributes is not
// allowed after this point.
template <typename T>
class tf_handle {
 public:
  tf_handle() : obj_(nullptr) {}
  explicit tf_handle(PyObject* obj) : obj_(nullptr) {
    obj_ = AsPyTfObject<T>(obj);
    Py_INCREF(obj);
  }
  ~tf_handle() { Py_CLEAR(obj_); }

  tf_handle(const tf_handle<T>& other) { Reset(other.obj_); }

  tf_handle<T>& operator=(tf_handle<T>&& other) {
    if (this == &other) {
      return *this;
    }
    obj_ = other.obj_;
    other.obj_ = nullptr;
  }

  tf_handle<T>& operator=(const tf_handle<T>& other) {
    Reset(other.ptr());
    return *this;
  }

  tf_handle<T>& operator=(PyObject* obj) {
    Reset(obj);
    return *this;
  }

  void Destroy() {
    Py_INCREF(Py_None);
    Py_CLEAR(obj_);
    obj_ = reinterpret_cast<T*>(Py_None);
  }

  void Reset(PyObject* obj) {
    if (obj == reinterpret_cast<PyObject*>(obj_)) {
      return;
    }
    Py_INCREF(obj);
    Py_CLEAR(obj_);
    obj_ = AsPyTfObject<T>(obj);
  }

  void Clear() { Py_CLEAR(obj_); }

  T* operator->() {
    if (reinterpret_cast<PyObject*>(obj_) == Py_None) {
      throw std::runtime_error("Tried to deference None as a TF type.");
    }
    return obj_;
  }
  PyObject* ptr() const { return reinterpret_cast<PyObject*>(obj_); }

  py::handle borrow() { return py::reinterpret_borrow<py::object>(ptr()); }
  py::handle steal() { return py::reinterpret_steal<py::object>(ptr()); }

 private:
  T* obj_;
};

namespace pybind11 {
namespace detail {

#define TF_CASTER(TfObject)                                           \
  template <>                                                         \
  struct type_caster<TfObject> : public type_caster_base<TfObject> {  \
   public:                                                            \
    using base = type_caster_base<TfObject>;                          \
    bool load(py::handle src, bool convert) {                         \
      value = AsPyTfObject<TfObject>(src);                            \
      return true;                                                    \
    }                                                                 \
    static py::handle cast(TfObject* src, return_value_policy policy, \
                           py::handle parent) {                       \
      PyObject* src_obj = reinterpret_cast<PyObject*>(src);           \
      return py::reinterpret_borrow<py::object>(src_obj);             \
    }                                                                 \
  };

TF_CASTER(PyGraph);
TF_CASTER(PyOperation);
TF_CASTER(PyTensor);

}  // namespace detail
}  // namespace pybind11

// TF_Operation's are owned by their graph.
struct TF_OperationDeleter {
  void operator()(TF_Operation* op) {}
};

struct PyGraphData {
  TF_Graph* graph;

  // The C++ graph maintains an ID for every node, however our Python code has
  // _also_ previously assigned a node ID, which is independent and different
  // from the C++ ID. Moreover, the Python IDs are _dense_ and the Python
  // implementation relies on the `ops_by_id` map having "insertion order"
  // for the implementation of `get_operations` and auto control-deps.
  //
  // To keep compatibility and improve performance, we use 3 collections:
  //
  // * A py::list which tracks operations in insertion order.
  // * A flat-map from C++ ID to PyOperation.
  // * A flat-map from std::string to PyOperation.
  py::list op_list;

  // Operation ownership is maintained in ops_by_id.
  OpsByIdMap ops_by_id;
  OpsByNameMap ops_by_name;

  PyGraphData(PyObject* args, PyObject* kwds) {
    graph = TF_NewGraph();

    // By default shape inference functions are required, however this breaks
    // many custom ops. Disable this check for Python graphs.
    tsl::mutex_lock l(graph->mu);
    graph->refiner.set_require_shape_inference_fns(false);
  }

  ~PyGraphData() {
    Clear();
    TF_DeleteGraph(graph);
  }

  void Dismantle();

  void Clear() {
    Py_CLEAR(op_list.ptr());
    op_list.release();
    for (auto it = ops_by_id.begin(); it != ops_by_id.end(); ++it) {
      Py_CLEAR(it->second.ptr());
      it->second.release();
    }
    ops_by_id.clear();
    for (auto it = ops_by_name.begin(); it != ops_by_name.end(); ++it) {
      Py_CLEAR(it->second.ptr());
      it->second.release();
    }
    ops_by_name.clear();
  }

  int Visit(visitproc visit, void* arg) {
    Py_VISIT(op_list.ptr());
    for (auto it = ops_by_id.begin(); it != ops_by_id.end(); ++it) {
      Py_VISIT(it->second.ptr());
    }
    for (auto it = ops_by_name.begin(); it != ops_by_name.end(); ++it) {
      Py_VISIT(it->second.ptr());
    }
    return 0;
  }
};

struct PyGraph {
  TFObject_HEAD(PyGraph, PyGraphData);

  int64_t add_op(py::object obj);

  py::list operations() { return data->op_list; }
  int64_t num_operations() const { return data->op_list.size(); }

  // Return operations that are part of the Graph, but do not yet have
  // OperationHandle's. This logic is only invoked when importing an existing
  // GraphDef into Python. It should be removed once all logic moves to C++.
  std::vector<TF_Operation*> new_operations() {
    tsl::mutex_lock l(tf_graph()->mu);
    std::vector<TF_Operation*> ops;

    // SUBTLE: `op_nodes` skips the SOURCE and SINK nodes
    for (auto n : tf_graph()->graph.op_nodes()) {
      if (data->ops_by_name.find(n->name()) == data->ops_by_name.end()) {
        ops.push_back(reinterpret_cast<TF_Operation*>(n));
      }
    }
    return ops;
  }

  py::object get_operation_by_name(const std::string& name) {
    tsl::mutex_lock l(tf_graph()->mu);
    auto it = data->ops_by_name.find(name);
    if (it == data->ops_by_name.end()) {
      throw py::key_error();
    }
    return it->second;
  }

  int version() const { return data->ops_by_id.size(); }

  py::bytes version_def() const {
    // Potential deadlock:
    //
    // If different threads are building and executing the graph, there is a
    // potential for a deadlock. This can happen if one thread holds the GIL and
    // waits for the graph mutex, while another thread holds the graph mutex and
    // waits for the GIL.
    //
    // To avoid this, the GIL must be released before acquiring the graph mutex.
    // The graph mutex must then be held while getting the VersionDef. Finally,
    // the GIL must be reacquired.
    std::string versions;
    {
      py::gil_scoped_release release;
      tsl::mutex_lock l(tf_graph()->mu);
      versions = tf_graph()->graph.versions().SerializeAsString();
    }
    pybind11::gil_scoped_acquire acquire;
    return py::bytes(versions);
  }

  absl::StatusOr<py::bytes> _op_def_for_type(
      const std::string& kTypeName) const {
    tsl::mutex_lock l(tf_graph()->mu);
    const tensorflow::OpDef* op_def;
    TF_RETURN_IF_ERROR(
        tf_graph()->graph.op_registry()->LookUpOpDef(kTypeName, &op_def));
    return py::bytes(op_def->SerializeAsString());
  }

  void add_control_input(tensorflow::Node* src, tensorflow::Node* dst) {
    tsl::mutex_lock l(tf_graph()->mu);

    tf_graph()->graph.AddControlEdge(src, dst);
    record_mutation(*dst, "adding control edge");
  }

  void remove_all_control_inputs(const tensorflow::Node& node) {
    tsl::mutex_lock l(tf_graph()->mu);
    std::vector<const tensorflow::Edge*> control_edges;
    for (const tensorflow::Edge* edge : node.in_edges()) {
      if (!edge->IsControlEdge()) continue;
      control_edges.push_back(edge);
    }
    for (const tensorflow::Edge* edge : control_edges) {
      tf_graph()->graph.RemoveControlEdge(edge);
    }
  }

  void record_mutation(const tensorflow::Node& node, const std::string& reason)
      TF_EXCLUSIVE_LOCKS_REQUIRED(tf_graph()->mu) {
    tensorflow::RecordMutation(tf_graph(),
                               reinterpret_cast<const TF_Operation&>(node),
                               reason.c_str());
  }

  TF_Graph* tf_graph() const { return data->graph; }
};

struct PyOperationData {
  TF_Operation* tf_op = nullptr;

  py::list outputs;

  // N.B. initialized later by Python.
  tf_handle<PyGraph> graph;
  py::function tensor_fn;

  PyOperationData(PyObject* args, PyObject* kwds) {
    PyObject *py_op, *py_tensor_fn;
    if (!PyArg_ParseTuple(args, "OO", &py_op, &py_tensor_fn)) {
      return;
    }
    tf_op = py::cast<TF_Operation*>(py_op);
    tensor_fn = py::cast<py::function>(py_tensor_fn);
  }

  ~PyOperationData() { Clear(); }

  void Dismantle(PyOperation* py_op);

  void Clear() {
    Py_CLEAR(outputs.ptr());
    outputs.release();
    graph.Clear();
  }

  int Visit(visitproc visit, void* arg) {
    Py_VISIT(graph.ptr());
    Py_VISIT(outputs.ptr());
    return 0;
  }
};

struct PyOperation {
  TFObject_HEAD(PyOperation, PyOperationData);

  TF_Operation* tf_op() const { return data->tf_op; }

  void _init_outputs() {
    int num_outputs = TF_OperationNumOutputs(tf_op());
    for (int i = 0; i < num_outputs; ++i) {
      int dtype = TF_OperationOutputType(TF_Output{tf_op(), i});
      data->outputs.append(data->tensor_fn(AsPyObject(this), i, dtype));
    }
  }

  absl::Status _add_outputs(py::list dtypes, py::list shapes);

  TF_Output _tf_output(int idx) const { return TF_Output{tf_op(), idx}; }
  TF_Input _tf_input(int idx) const { return TF_Input{tf_op(), idx}; }

  py::bytes node_def() {
    return py::bytes(tf_op()->node.def().SerializeAsString());
  }

  py::bytes op_def() const {
    return py::bytes(tf_op()->node.op_def().SerializeAsString());
  }

  bool is_stateful() const { return tf_op()->node.op_def().is_stateful(); }

  const std::string& type() { return tf_op()->node.type_string(); }

  void add_control_input(PyOperation* input) {
    data->graph->add_control_input(&input->tf_op()->node, &tf_op()->node);
  }

  void add_control_inputs(py::iterable inputs);

  py::list control_inputs() {
    py::list output;
    for (const auto* edge : tf_op()->node.in_edges()) {
      if (edge->IsControlEdge() && !edge->src()->IsSource()) {
        output.append(data->graph->data->ops_by_id[edge->src()->id()]);
      }
    }
    return output;
  }
  py::list control_outputs() {
    py::list output;
    for (const auto* edge : tf_op()->node.out_edges()) {
      if (edge->IsControlEdge() && !edge->dst()->IsSink()) {
        output.append(data->graph->data->ops_by_id[edge->dst()->id()]);
      }
    }
    return output;
  }

  void remove_all_control_inputs() {
    data->graph->remove_all_control_inputs(tf_op()->node);
  }

  void set_device(const std::string& device) {
    tsl::mutex_lock l(data->graph->tf_graph()->mu);
    tf_op()->node.set_requested_device(device);
    data->graph->record_mutation(tf_op()->node, "setting device");
  }

  const std::string& device() { return tf_op()->node.requested_device(); }
  const std::string& name() { return tf_op()->node.name(); }
};

struct PyTensorData {
  py::object tf_output = py::none();
  py::object name = py::none();
  py::object dtype = py::none();
  py::object shape_val = py::none();
  py::object uid = py::none();

  tf_handle<PyOperation> op;
  tf_handle<PyGraph> graph;

  int value_index = -1;

  PyTensorData(PyObject* args, PyObject* kwds) {
    PyObject *py_op, *py_index, *py_dtype, *py_uid;
    if (!PyArg_ParseTuple(args, "OOOO", &py_op, &py_index, &py_dtype,
                          &py_uid)) {
      return;
    }
    dtype = py::reinterpret_borrow<py::object>(py_dtype);
    value_index = py::cast<int>(py::handle(py_index));
    op = py_op;
    graph = op->data->graph;
    name = py::str(absl::StrCat(op->name(), ":", value_index));
    tf_output = py::cast(TF_Output{op->tf_op(), value_index});
    uid = py::reinterpret_borrow<py::object>(py_uid);
  }

  ~PyTensorData() { Clear(); }

  void Clear() {
    Py_CLEAR(tf_output.ptr());
    tf_output.release();
    Py_CLEAR(name.ptr());
    name.release();
    Py_CLEAR(dtype.ptr());
    dtype.release();
    Py_CLEAR(shape_val.ptr());
    shape_val.release();
    Py_CLEAR(uid.ptr());
    uid.release();
    op.Clear();
    graph.Clear();
  }

  int Visit(visitproc visit, void* arg) {
    Py_VISIT(op.ptr());
    Py_VISIT(tf_output.ptr());
    Py_VISIT(graph.ptr());
    Py_VISIT(name.ptr());
    Py_VISIT(dtype.ptr());
    Py_VISIT(shape_val.ptr());
    Py_VISIT(uid.ptr());
    return 0;
  }
};

struct PyTensor {
  TFObject_HEAD(PyTensor, PyTensorData);

  int value_index() const { return data->value_index; }

  absl::StatusOr<py::object> shape() {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    bool unknown_shape = false;
    auto dims = tensorflow::TF_GraphGetTensorShapeHelper(
        data->graph->tf_graph(), TF_Output{data->op->tf_op(), value_index()},
        status.get(), &unknown_shape);
    if (!status.get()->status.ok()) {
      return status.get()->status;
    }

    py::list py_list;
    for (int64_t dim : dims) {
      py_list.append(dim == -1 ? py::none() : py::cast(dim));
    }

    return py::make_tuple(py_list, py::cast(unknown_shape));
  }

  absl::Status set_shape(py::iterable shape, bool unknown_shape) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    std::vector<int64_t> dims;
    if (!unknown_shape) {
      for (py::handle dim : shape) {
        if (dim.is_none()) {
          dims.push_back(-1);
        } else {
          dims.push_back(py::cast<int64_t>(dim));
        }
      }
    }
    tensorflow::TF_GraphSetTensorShape_wrapper(
        data->graph->tf_graph(), TF_Output{data->op->tf_op(), value_index()},
        dims, unknown_shape, status.get());
    return status.get()->status;
  }

  int64_t rank() {
    tsl::mutex_lock l(data->graph->tf_graph()->mu);
    tensorflow::shape_inference::InferenceContext* ic =
        data->graph->tf_graph()->refiner.GetContext(&data->op->tf_op()->node);

    tensorflow::shape_inference::ShapeHandle shape = ic->output(value_index());
    if (ic->RankKnown(shape)) {
      return ic->Rank(shape);
    }
    return -1;
  }

  py::list consumers() {
    py::list out;
    for (const auto* edge : data->op->tf_op()->node.out_edges()) {
      if (edge->src_output() != value_index()) {
        continue;
      }
      out.append(data->graph->data->ops_by_id[edge->dst()->id()]);
    }
    return out;
  }
};

PyObject* PyOperation::py_type = nullptr;
PyObject* PyTensor::py_type = nullptr;
PyObject* PyGraph::py_type = nullptr;

void PyOperationData::Dismantle(PyOperation* py_op) {
  outputs = py::list();
  graph.Destroy();
  PyDict_Clear(py_op->dict);
}

absl::Status PyOperation::_add_outputs(py::list dtypes, py::list shapes) {
  int orig_outputs = data->outputs.size();
  for (int i = 0; i < dtypes.size(); ++i) {
    py::object tensor =
        data->tensor_fn(AsPyObject(this), orig_outputs + i, dtypes[i]);

    // The passed in `shapes` may be TensorShapes, convert them to lists if
    // needed.
    bool unknown_shape;
    py::object dims;
    if (py::hasattr(shapes[i], "as_list")) {
      unknown_shape = shapes[i].attr("rank").is_none();
      if (!unknown_shape) {
        dims = shapes[i].attr("as_list")();
      } else {
        dims = py::list();
      }
    } else {
      unknown_shape = false;
      dims = shapes[i];
    }
    TF_RETURN_IF_ERROR(
        AsPyTfObject<PyTensor>(tensor)->set_shape(dims, unknown_shape));
    data->outputs.append(tensor);
  }
  return absl::OkStatus();
}

void PyOperation::add_control_inputs(py::iterable inputs) {
  tsl::mutex_lock l(data->graph->tf_graph()->mu);
  for (py::handle input : inputs) {
    auto* input_handle = py::cast<PyOperation*>(input);
    data->graph->tf_graph()->graph.AddControlEdge(&input_handle->tf_op()->node,
                                                  &tf_op()->node);
  }
  data->graph->record_mutation(tf_op()->node, "adding control input");
}

void PyGraphData::Dismantle() {
  for (auto& op : op_list) {
    AsPyTfObjectData<PyOperation>(op.ptr())->Dismantle(
        AsPyTfObject<PyOperation>(op.ptr()));
  }
  op_list = py::list();
  ops_by_id.clear();
  ops_by_name.clear();
}

int64_t PyGraph::add_op(py::object obj) {
  PyOperation* op_handle = AsPyTfObject<PyOperation>(obj);
  int64_t op_id = op_handle->tf_op()->node.id();
  data->op_list.append(obj);
  data->ops_by_id[op_id] = obj;
  data->ops_by_name[op_handle->name()] = obj;
  return op_id;
}

PYBIND11_MODULE(_pywrap_tf_session, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  // Numpy initialization code for array checks.
  tsl::ImportNumpy();

  py::bind_map<OpsByIdMap>(m, "OpsById");
  py::bind_map<OpsByNameMap>(m, "OpsByName");

  py::str module_name(m.attr("__name__"));

  MakeTfObjectType<PyGraph>(&PyGraph::py_type);
  py::object c_graph = py::reinterpret_borrow<py::object>(PyGraph::py_type);
  m.attr("PyGraph") = c_graph;
  c_graph.attr("__module__") = module_name;
  c_graph.attr("Dismantle") = method(c_graph, [](py::handle handle) {
    AsPyTfObjectData<PyGraph>(handle)->Dismantle();
  });
  c_graph.attr("_version_def") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyGraph>(handle)->version_def();
  });
  c_graph.attr("version") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyGraph>(handle)->version();
  });
  c_graph.attr("_op_def_for_type") =
      method(c_graph, [](py::handle handle, std::string type) {
        return AsPyTfObject<PyGraph>(handle)->_op_def_for_type(type);
      });
  c_graph.attr("_nodes_by_name") = property_readonly([](py::handle handle) {
    return AsPyTfObjectData<PyGraph>(handle)->ops_by_name;
  });
  c_graph.attr("_nodes_by_id") = property_readonly([](py::handle handle) {
    return AsPyTfObjectData<PyGraph>(handle)->ops_by_id;
  });
  c_graph.attr("_get_operation_by_name") =
      method(c_graph, [](py::handle handle, std::string name) {
        return AsPyTfObject<PyGraph>(handle)->get_operation_by_name(name);
      });
  c_graph.attr("get_operations") = method(c_graph, [](py::handle handle) {
    auto ops = AsPyTfObject<PyGraph>(handle)->operations();
    py::list copy;
    for (auto& op : ops) {
      copy.append(op);
    }
    return copy;
  });
  c_graph.attr("operations") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyGraph>(handle)->operations();
  });
  c_graph.attr("new_operations") = method(c_graph, [](py::handle handle) {
    return AsPyTfObject<PyGraph>(handle)->new_operations();
  });
  c_graph.attr("num_operations") = method(c_graph, [](py::handle handle) {
    return AsPyTfObject<PyGraph>(handle)->num_operations();
  });
  c_graph.attr("_add_op") =
      method(c_graph, [](py::handle handle, py::object op) {
        return AsPyTfObject<PyGraph>(handle)->add_op(op);
      });

  MakeTfObjectType<PyOperation>(&PyOperation::py_type);
  py::object c_op = py::reinterpret_borrow<py::object>(PyOperation::py_type);
  m.attr("PyOperation") = c_op;
  c_op.attr("__module__") = module_name;
  c_op.attr("_tf_output") = method(c_op, [](py::handle handle, int index) {
    return AsPyTfObject<PyOperation>(handle)->_tf_output(index);
  });
  c_op.attr("_tf_input") = method(c_op, [](py::handle handle, int index) {
    return AsPyTfObject<PyOperation>(handle)->_tf_input(index);
  });
  c_op.attr("_set_device_from_string") =
      method(c_op, [](py::handle handle, std::string device) {
        return AsPyTfObject<PyOperation>(handle)->set_device(device);
      });
  c_op.attr("_add_control_input") =
      method(c_op, [](py::handle handle, py::handle input) {
        return AsPyTfObject<PyOperation>(handle)->add_control_input(
            AsPyTfObject<PyOperation>(input));
      });
  c_op.attr("_add_control_inputs") =
      method(c_op, [](py::handle handle, py::iterable inputs) {
        return AsPyTfObject<PyOperation>(handle)->add_control_inputs(inputs);
      });
  c_op.attr("_remove_all_control_inputs") = method(c_op, [](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->remove_all_control_inputs();
  });
  c_op.attr("outputs") = property_readonly([](py::handle handle) {
    return AsPyTfObjectData<PyOperation>(handle)->outputs;
  });
  c_op.attr("graph") = property(
      [](py::handle handle) {
        return AsPyTfObjectData<PyOperation>(handle)->graph.borrow();
      },
      [](py::handle handle, py::handle graph) {
        auto op = AsPyTfObject<PyOperation>(handle);
        op->data->graph = graph.ptr();
      });
  c_op.attr("_c_op") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->tf_op();
  });
  c_op.attr("_is_stateful") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->is_stateful();
  });
  c_op.attr("_op_def") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->op_def();
  });
  c_op.attr("_node_def") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->node_def();
  });
  c_op.attr("type") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->type();
  });
  c_op.attr("name") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->name();
  });
  c_op.attr("device") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->device();
  });
  c_op.attr("_control_outputs") = property_readonly([](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->control_outputs();
  });
  c_op.attr("_init_outputs") = method(c_op, [](py::handle handle) {
    return AsPyTfObject<PyOperation>(handle)->_init_outputs();
  });
  c_op.attr("_add_outputs") =
      method(c_op, [](py::handle handle, py::list dtypes, py::list shapes) {
        return AsPyTfObject<PyOperation>(handle)->_add_outputs(dtypes, shapes);
      });
  c_op.attr("control_inputs") = property_readonly(
      [](py::handle handle) {
        return AsPyTfObject<PyOperation>(handle)->control_inputs();
      },
      R"doc(
    The `Operation` objects on which this op has a control dependency.

    Before this op is executed, TensorFlow will ensure that the
    operations in `self.control_inputs` have finished executing. This
    mechanism can be used to run ops sequentially for performance
    reasons, or to ensure that the side effects of an op are observed
    in the correct order.

    Returns:
      A list of `Operation` objects.
  )doc");

  [&m, &module_name]() {
    MakeTfObjectType<PyTensor>(&PyTensor::py_type);
    py::object c_tensor = py::reinterpret_borrow<py::object>(PyTensor::py_type);
    m.attr("PyTensor") = c_tensor;
    c_tensor.attr("__module__") = module_name;
    c_tensor.attr("device") = property_readonly([](py::handle handle) {
      return AsPyTfObjectData<PyTensor>(handle)->op->device();
    });
    c_tensor.attr("ndim") = property_readonly([](py::handle handle) {
      return AsPyTfObject<PyTensor>(handle)->rank();
    });
    c_tensor.attr("_rank") = method(c_tensor, [](py::handle handle) {
      return AsPyTfObject<PyTensor>(handle)->rank();
    });
    c_tensor.attr("_shape") = property_readonly([](py::handle handle) {
      return AsPyTfObject<PyTensor>(handle)->shape();
    });
    c_tensor.attr("_dtype") = property_readonly([](py::handle handle) {
      return AsPyTfObjectData<PyTensor>(handle)->dtype;
    });
    c_tensor.attr("_name") = property(
        [](py::handle handle) {
          return AsPyTfObjectData<PyTensor>(handle)->name;
        },
        [](py::handle handle, py::object name) {
          AsPyTfObjectData<PyTensor>(handle)->name = name;
        });
    c_tensor.attr("_shape_val") = property(
        [](py::handle handle) {
          auto py_tensor = AsPyTfObject<PyTensor>(handle);
          return py_tensor->data->shape_val;
        },
        [](py::handle handle, py::object shape) {
          AsPyTfObjectData<PyTensor>(handle)->shape_val = shape;
        });
    c_tensor.attr("_id") = property(
        [](py::handle handle) {
          return AsPyTfObjectData<PyTensor>(handle)->uid;
        },
        [](py::handle handle, py::object uid) {
          AsPyTfObjectData<PyTensor>(handle)->uid = uid;
        });
    c_tensor.attr("graph") =
        property_readonly([](py::handle handle) -> py::handle {
          auto& graph = AsPyTfObjectData<PyTensor>(handle)->graph;
          if (graph.ptr() != nullptr) {
            return graph.borrow();
          }
          return py::none();
        });
    c_tensor.attr("_as_tf_output") = method(c_tensor, [](py::handle handle) {
      return AsPyTfObjectData<PyTensor>(handle)->tf_output;
    });
    c_tensor.attr("_op") =
        property_readonly([](py::handle handle) -> py::handle {
          auto& op = AsPyTfObjectData<PyTensor>(handle)->op;
          if (op.ptr() != nullptr) {
            return op.borrow();
          }
          return py::none();
        });
    c_tensor.attr("op") =
        property_readonly([](py::handle handle) -> py::handle {
          auto& op = AsPyTfObjectData<PyTensor>(handle)->op;
          if (op.ptr() != nullptr) {
            return op.borrow();
          }
          return py::none();
        });
    c_tensor.attr("_set_shape") = method(c_tensor, [](py::handle handle,
                                                      py::iterable shape,
                                                      bool unknown_shape) {
      return AsPyTfObject<PyTensor>(handle)->set_shape(shape, unknown_shape);
    });
    c_tensor.attr("value_index") = property_readonly([](py::handle handle) {
      return AsPyTfObject<PyTensor>(handle)->value_index();
    });
    c_tensor.attr("consumers") = method(c_tensor, [](py::handle handle) {
      return AsPyTfObject<PyTensor>(handle)->consumers();
    });
  }();

  py::class_<TF_Operation, std::unique_ptr<TF_Operation, TF_OperationDeleter>>
      TF_Operation_class(m, "TF_Operation");

  py::class_<TF_Output>(m, "TF_Output")
      .def(py::init<>())
      .def_readwrite("oper", &TF_Output::oper)
      .def_readwrite("index", &TF_Output::index);

  py::class_<TF_Input>(m, "TF_Input")
      .def(py::init<>())
      .def_readwrite("oper", &TF_Input::oper)
      .def_readwrite("index", &TF_Input::index);

  py::class_<TF_ImportGraphDefOptions> TF_ImportGraphDefOptions_class(
      m, "TF_ImportGraphDefOptions");
  py::class_<TF_ImportGraphDefResults> TF_ImportGraphDefResults_class(
      m, "TF_ImportGraphDefResults");
  py::class_<TF_DeprecatedSession> TF_DeprecatedSession_class(
      m, "TF_DeprecatedSession");
  py::class_<TF_Session> TF_Session_class(m, "TF_Session");
  py::class_<TF_OperationDescription> TF_OperationDescription_class(
      m, "TF_OperationDescription");
  py::class_<TF_Library> TF_Library_class(m, "TF_Library");
  py::class_<TF_SessionOptions> TF_SessionOptions_class(m, "TF_SessionOptions");
  py::class_<TF_Buffer> TF_Buffer_class(m, "TF_Buffer");
  py::class_<TF_ApiDefMap> TF_ApiDefMap_class(m, "TF_ApiDefMap");
  py::class_<TF_Server> TF_Server_class(m, "TF_Server");
  py::class_<TF_Status> TF_Status_class(m, "TF_Status");

  // We only release the Python GIL for certain methods that are
  // not explicitly marked. We disable this behavior for some functions
  // because they uses Python method(s) that expect the GIL to be held
  // (at least PyArray_Return, maybe others).

  // Do not release GIL.
  m.def("TF_OperationGetControlOutputs_wrapper",
        tensorflow::TF_OperationGetControlOutputs_wrapper);
  // Do not release GIL.
  m.def("GetOperationInputs", tensorflow::GetOperationInputs);
  // Do not release GIL.
  m.def("TF_ImportGraphDefOptionsSetValidateColocationConstraints",
        TF_ImportGraphDefOptionsSetValidateColocationConstraints);
  // Do not release GIL.
  m.def("TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper",
        tensorflow::TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper);
  m.def("TF_SessionMakeCallable",
        [](TF_Session* session, const TF_Buffer* callable_options) {
          int64_t out_handle;
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());

          // Release GIL.
          py::gil_scoped_release release;
          tensorflow::TF_SessionMakeCallable(session, callable_options,
                                             &out_handle, status.get());

          // Acquire GIL for returning int conversion.
          pybind11::gil_scoped_acquire acquire;
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return out_handle;
        });
  m.def("_TF_SetTarget", TF_SetTarget);
  m.def("_TF_SetConfig", [](TF_SessionOptions* options, py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
    TF_SetConfig(options, buf.get()->data, buf.get()->length, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });
  m.def("_TF_NewSessionOptions", TF_NewSessionOptions,
        py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteSessionOptions", TF_DeleteSessionOptions,
        py::call_guard<py::gil_scoped_release>());

  m.def("EqualGraphDefWrapper", tensorflow::EqualGraphDefWrapper,
        py::call_guard<py::gil_scoped_release>());
  m.def("EqualAttrValueWrapper", tensorflow::EqualAttrValueWrapper,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_GraphToFunction_wrapper",
      [](PyGraph* fn_body, const char* fn_name, bool append_hash_to_fn_name,
         absl::optional<std::vector<TF_Operation*>> opers_opt,
         const std::vector<TF_Output>& inputs,
         const std::vector<TF_Output>& outputs,
         const std::vector<py::bytes> output_names,
         const std::vector<TF_Operation*> control_outputs,
         const std::vector<py::bytes> control_output_names, py::none opts,
         const char* description) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());

        // TODO(b/147674626): Use pybind11 list_caster instead.
        tensorflow::NameVector output_names_name_vector =
            ConvertPyListToNameVector(output_names);

        // TODO(b/147674626): Use pybind11 list_caster instead.
        tensorflow::NameVector control_output_names_name_vector =
            ConvertPyListToNameVector(control_output_names);

        // Release GIL.
        py::gil_scoped_release release;
        auto output = tensorflow::TF_GraphToFunction_wrapper(
            fn_body->tf_graph(), fn_name, append_hash_to_fn_name,
            opers_opt.has_value() ? &opers_opt.value() : nullptr, inputs,
            outputs, output_names_name_vector, &control_outputs,
            control_output_names_name_vector,
            /*opts=*/nullptr, description, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_GraphSetOutputHandleShapesAndTypes_wrapper",
        [](PyGraph* graph, TF_Output output,
           const std::vector<absl::optional<std::vector<int64_t>>>& shapes,
           const std::vector<int>& ranks, py::handle& types) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());

          // Cast types
          std::vector<TF_DataType> types_local;
          PyObject* seq =
              PySequence_Fast(types.ptr(), "$symname: expected list");
          if (seq == nullptr) {
            PyErr_SetString(PyExc_RuntimeError,
                            "$symname: PySequence_Fast returned NULL.");
            throw py::error_already_set();
          }

          int size = PySequence_Fast_GET_SIZE(seq);
          if (size == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "$symname: shapes list must be non-empty");
            throw py::error_already_set();
          }

          for (int i = 0; i < size; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
            types_local.push_back((TF_DataType)PyLong_AsLong(item));
          }

          // Convert shapes nested vector
          std::vector<std::vector<int64_t>> shapes_local;
          for (size_t i = 0; i < shapes.size(); ++i) {
            std::vector<int64_t> dims;
            std::vector<int64_t> item =
                shapes[i].has_value() ? shapes[i].value() : dims;
            shapes_local.push_back(item);
          }

          Py_DECREF(seq);

          tensorflow::TF_GraphSetOutputHandleShapesAndTypes_wrapper(
              graph->tf_graph(), output, shapes_local, ranks, types_local,
              status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  // Do not release GIL.
  m.def("TF_CreatePlaceholders",
        [](PyGraph* graph, py::handle& dtypes, const char* prefix) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          auto output = tensorflow::TF_CreatePlaceholders(
              graph->tf_graph(), dtypes.ptr(), prefix, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return output;
        });

  m.def(
      "TF_NewSession",
      [](PyGraph* graph, const TF_SessionOptions* opts) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_NewSession(graph->tf_graph(), opts, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_NewSessionRef",
      [](PyGraph* graph, const TF_SessionOptions* opts) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output =
            tensorflow::TF_NewSessionRef(graph->tf_graph(), opts, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_CloseSession", [](TF_Session* session) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());

    // Release GIL.
    py::gil_scoped_release release;
    TF_CloseSession(session, status.get());

    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_DeleteSession", [](TF_Session* session) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_DeleteSession(session, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  // Do not release GIL.
  m.def("TF_TryEvaluateConstant_wrapper",
        [](PyGraph* graph, const TF_Output output) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          auto result = tensorflow::TF_TryEvaluateConstant_wrapper(
              graph->tf_graph(), output, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return tensorflow::PyoOrThrow(result);
        });

  m.def("ExtendSession", [](TF_Session* session) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    pybind11::gil_scoped_release release;
    tensorflow::ExtendSession(session, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("GetHandleShapeAndType", [](PyGraph* graph, TF_Output output) {
    std::string output_string =
        tensorflow::GetHandleShapeAndType(graph->tf_graph(), output);
    // Override default py3 behavior of attempting to encode into Unicode as
    // the dependent functions expect bytes.
    return py::bytes(output_string);
  });

  m.def("SetHandleShapeAndType",
        [](PyGraph* graph, TF_Output output, py::bytes proto) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          tensorflow::Safe_TF_BufferPtr buf =
              tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
          tensorflow::SetHandleShapeAndType(graph->tf_graph(), output,
                                            buf.get()->data, buf.get()->length,
                                            status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  // Do not release GIL.
  m.def("TF_SessionRun_wrapper", [](TF_Session* session, TF_Buffer* run_options,
                                    const py::handle& input_dict,
                                    const std::vector<TF_Output>& outputs,
                                    const std::vector<TF_Operation*>& targets,
                                    TF_Buffer* run_metadata) {
    // Convert inputs dictionary
    std::vector<TF_Output> inputs;
    std::vector<PyObject*> input_ndarrays;
    if (!PyDict_Check(input_dict.ptr())) {
      PyErr_SetString(
          PyExc_TypeError,
          "Expected a dictionary as an argument to TF_SessionRun_wrapper.");
      throw py::error_already_set();
    }
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_dict.ptr(), &pos, &key, &value)) {
      TF_Output item = py::cast<TF_Output>(key);
      inputs.push_back(item);

      // TODO(amitpatankar): Fix this PyArray check. (b/147855599)

      // if (!PyArray_Check(value)) {
      //   PyErr_SetString(
      //       PyExc_TypeError,
      //       "$symname: Expected all values in input dict to be ndarray.");
      //   throw py::error_already_set();
      // }
      input_ndarrays.push_back(value);
    }

    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    std::vector<PyObject*> py_outputs;
    tensorflow::TF_SessionRun_wrapper(session, run_options, inputs,
                                      input_ndarrays, outputs, targets,
                                      run_metadata, status.get(), &py_outputs);
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    // Create a Python list using the C API rather than py::list. b/147855599
    PyObject* result = PyList_New(py_outputs.size());
    if (result == nullptr) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create a list.");
      throw py::error_already_set();
    }
    for (size_t i = 0; i < py_outputs.size(); ++i) {
      PyList_SET_ITEM(result, i, py_outputs.at(i));
    }

    return tensorflow::PyoOrThrow(result);
  });

  // Do not release GIL.
  m.def("TF_SessionPRun_wrapper", [](TF_Session* session, const char* handle,
                                     const py::handle& input_dict,
                                     const std::vector<TF_Output>& outputs) {
    // Convert inputs dictionary
    std::vector<TF_Output> inputs;
    std::vector<PyObject*> input_ndarrays;
    if (!PyDict_Check(input_dict.ptr())) {
      PyErr_SetString(
          PyExc_TypeError,
          "Expected a dictionary as an argument to TF_SessionPRun_wrapper.");
      throw py::error_already_set();
    }
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_dict.ptr(), &pos, &key, &value)) {
      TF_Output item = py::cast<TF_Output>(key);
      inputs.push_back(item);

      // TODO(amitpatankar): Fix this PyArray check. (b/147855599)

      // if (!PyArray_Check(value)) {
      //   PyErr_SetString(
      //       PyExc_TypeError,
      //       "$symname: Expected all values in input dict to be ndarray.");
      //   throw py::error_already_set();
      // }
      input_ndarrays.push_back(value);
    }

    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    std::vector<PyObject*> py_outputs;
    tensorflow::TF_SessionPRun_wrapper(session, handle, inputs, input_ndarrays,
                                       outputs, status.get(), &py_outputs);
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    PyObject* result = PyList_New(py_outputs.size());
    if (result == nullptr) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create a list.");
      throw py::error_already_set();
    }
    for (size_t i = 0; i < py_outputs.size(); ++i) {
      PyList_SET_ITEM(result, i, py_outputs.at(i));
    }

    return tensorflow::PyoOrThrow(result);
  });

  // Do not release GIL.
  m.def("TF_SessionPRunSetup_wrapper",
        [](TF_Session* session, const std::vector<TF_Output>& inputs,
           const std::vector<TF_Output>& outputs,
           const std::vector<TF_Operation*>& targets) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          const char* out_handle;
          tensorflow::TF_SessionPRunSetup_wrapper(
              session, inputs, outputs, targets, &out_handle, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
          return out_handle;
        });

  // Do not release GIL.
  m.def("TF_SessionRunCallable", [](TF_Session* session, int64_t handle,
                                    py::object feed_values,
                                    TF_Buffer* run_metadata) {
    tensorflow::PyObjectVector out_values;
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::TF_SessionRunCallable(session, handle, feed_values.ptr(),
                                      &out_values, run_metadata, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

    // Return out_values
    py::list py_list;
    for (size_t i = 0; i < out_values.size(); ++i) {
      py::object obj = tensorflow::Pyo(out_values.at(i));
      py_list.append(obj);
    }
    return py_list;
  });

  m.def("TF_SessionReleaseCallable", [](TF_Session* session, int64_t handle) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    tensorflow::TF_SessionReleaseCallable(session, handle, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def(
      "TF_NewOperation",
      [](PyGraph* graph, const char* op_type, const char* oper_name) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        TF_OperationDescription* output =
            TF_NewOperation(graph->tf_graph(), op_type, oper_name);
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_FinishOperation",
      [](TF_OperationDescription* desc) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        TF_Operation* output = TF_FinishOperation(desc, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_SetOpStackTrace",
        [](TF_Operation* op,
           std::shared_ptr<tensorflow::AbstractStackTrace> trace) {
          op->node.SetStackTrace(trace);
        });

  m.def("TF_OperationGetAttrInt",
        [](TF_Operation* oper, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          int64_t value;
          // Release GIL.
          py::gil_scoped_release release;
          TF_OperationGetAttrInt(oper, attr_name, &value, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
          // Convert TF_OperationGetAttrInt int64_t* out-argument to Python
          // bool.
          // Acquire GIL for returning output returning.
          pybind11::gil_scoped_acquire acquire;
          return tensorflow::Pyo(PyLong_FromLongLong(value));
        });

  m.def("TF_SetAttrValueProto", [](TF_OperationDescription* desc,
                                   const char* attr_name, py::bytes proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    tensorflow::Safe_TF_BufferPtr buf =
        tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
    TF_SetAttrValueProto(desc, attr_name, buf.get()->data, buf.get()->length,
                         status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });

  m.def("TF_OperationNumOutputs", TF_OperationNumOutputs,
        py::call_guard<py::gil_scoped_release>());

  // Convert types to ints
  m.def("TF_OperationInputType", TF_OperationInputType,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_OperationOutputType", TF_OperationOutputType,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_OperationName", TF_OperationName,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_OperationOpType", TF_OperationOpType,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_OperationDevice", TF_OperationDevice,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_AddInput", TF_AddInput);
  m.def(
      "TF_AddInputList", [](TF_OperationDescription* desc, py::handle& inputs) {
        std::vector<TF_Output> vec;
        size_t size = PyList_Size(inputs.ptr());
        for (size_t i = 0; i < size; ++i) {
          TF_Output item = py::cast<TF_Output>(PyList_GetItem(inputs.ptr(), i));
          vec.push_back(item);
        }
        TF_AddInputList(desc, vec.data(), vec.size());
      });

  m.def("TF_OperationToNodeDef",
        [](TF_Operation* oper, TF_Buffer* output_node_def) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TF_OperationToNodeDef(oper, output_node_def, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  m.def("TF_OperationGetAttrValueProto",
        [](TF_Operation* oper, const char* attr_name,
           TF_Buffer* output_attr_value) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TF_OperationGetAttrValueProto(oper, attr_name, output_attr_value,
                                        status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        });

  m.def("TF_OperationGetStackTrace", [](TF_Operation* oper) -> py::object {
    const std::shared_ptr<tensorflow::AbstractStackTrace> trace =
        oper->node.GetStackTrace();
    if (!trace) {
      return py::none();
    }
    return py::cast(*trace, py::return_value_policy::reference);
  });

  // TF_Buffer util methods
  // TODO(amitpatankar): Consolidate Buffer methods into a separate header
  // file.
  m.def("TF_NewBuffer", TF_NewBuffer, py::return_value_policy::reference);
  m.def("TF_GetBuffer", [](TF_Buffer* buf) {
    TF_Buffer buffer = TF_GetBuffer(buf);
    return tensorflow::PyoOrThrow(PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(buffer.data), buffer.length));
  });
  m.def("TF_DeleteBuffer", &TF_DeleteBuffer);
  m.def(
      "TF_NewBufferFromString",
      [](py::bytes buffer_as_string) {
        tensorflow::Safe_TF_BufferPtr buf = tensorflow::make_safe(
            ProtoStringToTFBuffer(buffer_as_string.ptr()));
        return TF_NewBufferFromString(buf.get()->data, buf.get()->length);
      },
      py::return_value_policy::reference);

  m.def("SetAttr", [](PyGraph* graph, TF_Operation* op, const char* attr_name,
                      TF_Buffer* attr_value_proto) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    tensorflow::SetAttr(graph->tf_graph(), op, attr_name, attr_value_proto,
                        status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("ClearAttr",
        [](PyGraph* graph, TF_Operation* op, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          tensorflow::ClearAttr(graph->tf_graph(), op, attr_name, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  // Note: users should prefer using tf.cast or equivalent, and only when
  // it's infeasible to set the type via OpDef's type constructor and
  // inference function.
  m.def("SetFullType",
        [](PyGraph* graph, TF_Operation* op, const TF_Buffer* full_type_proto) {
          tensorflow::SetFullType(graph->tf_graph(), op, full_type_proto);
        });

  m.def(
      "TF_LoadLibrary",
      [](const char* library_filename) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TF_LoadLibrary(library_filename, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_LoadPluggableDeviceLibrary",
      [](const char* library_filename) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output =
            TF_LoadPluggableDeviceLibrary(library_filename, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_GetOpList", [](TF_Library* lib_handle) {
    TF_Buffer output_buffer = TF_GetOpList(lib_handle);
    return tensorflow::PyoOrThrow(PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(output_buffer.data),
        output_buffer.length));
  });

  m.def("TF_DeleteLibraryHandle", TF_DeleteLibraryHandle,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_PluggableDeviceLibraryHandle",
        TF_DeletePluggableDeviceLibraryHandle,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_AddControlInput", TF_AddControlInput);

  m.def("UpdateEdge", [](PyGraph* graph, TF_Output new_src, TF_Input dst) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    tensorflow::UpdateEdge(graph->tf_graph(), new_src, dst, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_NewImportGraphDefOptions", TF_NewImportGraphDefOptions,
        py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsSetPrefix", TF_ImportGraphDefOptionsSetPrefix,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsSetUniquifyNames",
        TF_ImportGraphDefOptionsSetUniquifyNames,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsSetPropagateDeviceSpec",
        tensorflow::TF_ImportGraphDefOptionsSetPropagateDeviceSpec,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsRemapControlDependency",
        TF_ImportGraphDefOptionsRemapControlDependency,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsAddInputMapping",
        TF_ImportGraphDefOptionsAddInputMapping,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsAddReturnOperation",
        TF_ImportGraphDefOptionsAddReturnOperation,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_ImportGraphDefOptionsAddReturnOutput",
        TF_ImportGraphDefOptionsAddReturnOutput,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_GraphImportGraphDefWithResults",
      [](PyGraph* graph, const TF_Buffer* graph_def,
         const TF_ImportGraphDefOptions* options) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TF_GraphImportGraphDefWithResults(
            graph->tf_graph(), graph_def, options, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_GraphImportGraphDefWithResultsNoSerialization",
      [](PyGraph* graph, const tensorflow::GraphDef* graph_def,
         const TF_ImportGraphDefOptions* options) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        TF_ImportGraphDefResults* output;
        {
          TF_Buffer graph_def_buffer;
          graph_def_buffer.data = reinterpret_cast<const void*>(graph_def);
          graph_def_buffer.length = sizeof(tensorflow::GraphDef*);
          output = TF_GraphImportGraphDefWithResultsNoSerialization(
              graph->tf_graph(), &graph_def_buffer, options, status.get());
        }
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_GraphNextOperation",
      [](PyGraph* graph, size_t pos) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        auto output = TF_GraphNextOperation(graph->tf_graph(), &pos);
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());

        // Returns a (TF_Operation*, int pos) tuple.
        py::tuple result_tuple = py::make_tuple(
            py::cast(output), tensorflow::Pyo(PyLong_FromSize_t(pos)));
        return result_tuple;
      },
      py::return_value_policy::reference);

  // Python needs to own deletion of outputs
  m.def("TF_ImportGraphDefResultsReturnOutputs",
        [](TF_ImportGraphDefResults* results) {
          int num_outputs;
          TF_Output* outputs;
          TF_ImportGraphDefResultsReturnOutputs(results, &num_outputs,
                                                &outputs);
          py::list py_list;
          for (int i = 0; i < num_outputs; ++i) {
            TF_Output tf_output = TF_Output(outputs[i]);
            py_list.append(tf_output);
          }
          return py_list;
        });

  m.def(
      "TF_ImportGraphDefResultsReturnOperations",
      [](TF_ImportGraphDefResults* results) {
        int num_opers;
        TF_Operation** opers;
        TF_ImportGraphDefResultsReturnOperations(results, &num_opers, &opers);
        py::list py_list;
        for (int i = 0; i < num_opers; ++i) {
          py_list.append(opers[i]);
        }
        return py_list;
      },
      py::return_value_policy::reference);

  m.def("TF_GraphToGraphDefPybind", [](PyGraph* graph) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_Graph* tf_graph = graph->tf_graph();
    auto def = new tensorflow::GraphDef();
    {
      tensorflow::mutex_lock l(tf_graph->mu);
      tf_graph->graph.ToGraphDef(def);
    }
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
    return def;
  });

  m.def("TF_GraphToGraphDef", [](PyGraph* graph, TF_Buffer* output_graph_def) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_GraphToGraphDef(graph->tf_graph(), output_graph_def, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_OperationNumInputs", TF_OperationNumInputs,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_DeleteFunction", TF_DeleteFunction,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteImportGraphDefResults", TF_DeleteImportGraphDefResults,
        py::call_guard<py::gil_scoped_release>());
  m.def("TF_DeleteImportGraphDefOptions", TF_DeleteImportGraphDefOptions,
        py::call_guard<py::gil_scoped_release>());

  m.def("TF_FunctionSetAttrValueProto",
        [](TF_Function* func, const char* attr_name, py::bytes proto) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          tensorflow::Safe_TF_BufferPtr buf =
              tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
          // Release GIL.
          py::gil_scoped_release release;
          TF_FunctionSetAttrValueProto(func, attr_name, buf.get()->data,
                                       buf.get()->length, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_FunctionToFunctionDef",
        [](TF_Function* graph, TF_Buffer* output_func_def) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_FunctionToFunctionDef(graph, output_func_def, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_GraphCopyFunction",
        [](PyGraph* graph, const TF_Function* func, const TF_Function* grad) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_GraphCopyFunction(graph->tf_graph(), func, grad, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_GraphRemoveFunction", [](PyGraph* graph, const char* func_name) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_GraphRemoveFunction(graph->tf_graph(), func_name, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def(
      "TF_FunctionImportFunctionDef",
      [](py::bytes proto) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        tensorflow::Safe_TF_BufferPtr buf =
            tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));

        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_FunctionImportFunctionDef(
            buf.get()->data, buf.get()->length, status.get());

        // Acquire GIL for returning output returning.
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_FunctionImportFunctionDefNoSerialization",
      [](tensorflow::FunctionDef fdef) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());

        // Release GIL.
        py::gil_scoped_release release;
        TF_Function* func = new TF_Function();
        func->record =
            new tensorflow::FunctionRecord(std::move(fdef), {}, false);
        status.get()->status = absl::OkStatus();
        // Acquire GIL for returning output returning.
        pybind11::gil_scoped_acquire acquire;
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return func;
      },
      py::return_value_policy::reference);

  m.def("EqualAttrValueWrapper", tensorflow::EqualAttrValueWrapper,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_GetAllRegisteredKernels",
      []() {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_GetAllRegisteredKernels(status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_GetRegisteredKernelsForOp",
      [](const char* name) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_GetRegisteredKernelsForOp(name, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_GetAllOpList", TF_GetAllOpList, py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_NewApiDefMap",
      [](TF_Buffer* op_list_buffer) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output = TF_NewApiDefMap(op_list_buffer, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_DeleteApiDefMap", TF_DeleteApiDefMap,
        py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_ApiDefMapGet",
      [](TF_ApiDefMap* api_def_map, const char* name, size_t name_len) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // Release GIL.
        py::gil_scoped_release release;
        auto output =
            TF_ApiDefMapGet(api_def_map, name, name_len, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_ApiDefMapPut",
        [](TF_ApiDefMap* api_def_map, const char* name, size_t name_len) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL.
          py::gil_scoped_release release;
          TF_ApiDefMapPut(api_def_map, name, name_len, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_OperationGetAttrType",
        [](TF_Operation* oper, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          TF_DataType value;
          // Release GIL.
          py::gil_scoped_release release;
          TF_OperationGetAttrType(oper, attr_name, &value, status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
          return value;
        });

  m.def(
      "TF_NewServer",
      [](py::bytes proto) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        tensorflow::Safe_TF_BufferPtr buf =
            tensorflow::make_safe(ProtoStringToTFBuffer(proto.ptr()));
        TF_Server* output =
            TF_NewServer(buf.get()->data, buf.get()->length, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_ServerStart", [](TF_Server* server) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL.
    py::gil_scoped_release release;
    TF_ServerStart(server, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_ServerStop", [](TF_Server* server) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    py::gil_scoped_release release;
    TF_ServerStop(server, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def("TF_ServerJoin", [](TF_Server* server) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    py::gil_scoped_release release;
    TF_ServerJoin(server, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });

  m.def(
      "TF_ServerTarget",
      [](TF_Server* server) { return TF_ServerTarget(server); },
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "TF_SessionListDevices",
      [](TF_Session* session) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        TF_DeviceList* output = TF_SessionListDevices(session, status.get());
        tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
        return output;
      },
      py::return_value_policy::reference);

  m.def("TF_DeviceListCount",
        [](const TF_DeviceList* list) { return TF_DeviceListCount(list); });

  m.def("TF_DeviceListName", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    const char* output = TF_DeviceListName(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_DeviceListType", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    const char* output = TF_DeviceListType(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_DeviceListMemoryBytes", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    int64_t output = TF_DeviceListMemoryBytes(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_DeviceListIncarnation", [](const TF_DeviceList* list, int index) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    int64_t output = TF_DeviceListIncarnation(list, index, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
    return output;
  });

  m.def("TF_SetDevice", TF_SetDevice);

  m.def("TF_DeleteDeviceList", TF_DeleteDeviceList);

  m.def("TF_OperationGetAttrBool",
        [](TF_Operation* oper, const char* attr_name) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          unsigned char value;
          // Release GIL for threading.
          {
            py::gil_scoped_release release;
            TF_OperationGetAttrBool(oper, attr_name, &value, status.get());
            tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
          }
          return tensorflow::Pyo(PyBool_FromLong(value));
        });

  m.def("TF_NewStatus", TF_NewStatus, py::return_value_policy::reference);
  m.def("TF_DeleteStatus", TF_DeleteStatus);

  m.def("TF_DeleteDeviceList", TF_DeleteDeviceList);

  m.def("AddWhileInputHack",
        [](PyGraph* graph, TF_Output new_src, TF_Operation* dst) {
          tensorflow::Safe_TF_StatusPtr status =
              tensorflow::make_safe(TF_NewStatus());
          // Release GIL for threading.
          py::gil_scoped_release release;
          tensorflow::AddWhileInputHack(graph->tf_graph(), new_src, dst,
                                        status.get());
          tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
        });

  m.def("TF_Reset_wrapper", [](const TF_SessionOptions* opt,
                               const std::vector<py::bytes> containers) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    // Release GIL for threading.
    py::gil_scoped_release release;
    tensorflow::NameVector containers_name_vector =
        ConvertPyListToNameVector(containers);
    tensorflow::TF_Reset_wrapper(opt, containers_name_vector, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatusWithGIL(status.get());
  });
  m.def("TF_GetCode", TF_GetCode);

  m.def("TF_SetXlaAutoJitMode", TF_SetXlaAutoJitMode);
  m.def("TF_GetXlaAutoJitEnabled", TF_GetXlaAutoJitEnabled);
  m.def("TF_SetXlaEnableLazyCompilation", TF_SetXlaEnableLazyCompilation);
  m.def("TF_SetTfXlaCpuGlobalJit", TF_SetTfXlaCpuGlobalJit);
  m.def("TF_SetXlaMinClusterSize", TF_SetXlaMinClusterSize);
  m.def("TF_GetXlaConstantFoldingDisabled", TF_GetXlaConstantFoldingDisabled);
  m.def("TF_SetXlaConstantFoldingDisabled", TF_SetXlaConstantFoldingDisabled);

  // // Static constants are not working on Windows. b/145559202
  // // Creating getters instead.

  m.def("get_version", []() { return TF_VERSION_STRING; });
  m.def("get_git_version", []() { return TF_GIT_VERSION; });
  m.def("get_compiler_version", []() { return TF_COMPILER_VERSION; });
  m.def("get_cxx11_abi_flag", []() { return TF_CXX11_ABI_FLAG; });
  m.def("get_cxx_version", []() { return TF_CXX_VERSION; });
  m.def("get_eigen_max_align_bytes", []() { return EIGEN_MAX_ALIGN_BYTES; });
  m.def("get_monolithic_build", []() { return TF_MONOLITHIC_BUILD; });
  m.def("get_graph_def_version", []() { return TF_GRAPH_DEF_VERSION; });
  m.def("get_graph_def_version_min_consumer",
        []() { return TF_GRAPH_DEF_VERSION_MIN_CONSUMER; });
  m.def("get_graph_def_version_min_producer",
        []() { return TF_GRAPH_DEF_VERSION_MIN_PRODUCER; });
  m.def("get_tensor_handle_key", []() {
    // TODO(amitpatankar): Look into a more elegant solution.
    // Since this is a shared object we will hard code the value from
    // third_party/tensorflow/core/common_runtime/session_state.cc because
    // the Windows import will not load the libraries necessarily
    // in order. b/145559202
    return "TensorHandle";
  });

  m.def("TF_RegisterFilesystemPlugin", [](const char* plugin_filename) {
    tensorflow::Safe_TF_StatusPtr status =
        tensorflow::make_safe(TF_NewStatus());
    TF_RegisterFilesystemPlugin(plugin_filename, status.get());
    tensorflow::MaybeRaiseRegisteredFromTFStatus(status.get());
  });

  py::enum_<TF_DataType>(m, "TF_DataType")
      .value("TF_FLOAT", TF_FLOAT)
      .value("TF_DOUBLE", TF_DOUBLE)
      .value("TF_INT32", TF_INT32)
      .value("TF_UINT8", TF_UINT8)
      .value("TF_INT16", TF_INT16)
      .value("TF_INT8", TF_INT8)
      .value("TF_STRING", TF_STRING)
      .value("TF_COMPLEX64", TF_COMPLEX64)
      .value("TF_COMPLEX", TF_COMPLEX)
      .value("TF_INT64", TF_INT64)
      .value("TF_BOOL", TF_BOOL)
      .value("TF_QINT8", TF_QINT8)
      .value("TF_QUINT8", TF_QUINT8)
      .value("TF_QINT32", TF_QINT32)
      .value("TF_BFLOAT16", TF_BFLOAT16)
      .value("TF_QINT16", TF_QINT16)
      .value("TF_QUINT16", TF_QUINT16)
      .value("TF_UINT16", TF_UINT16)
      .value("TF_COMPLEX128", TF_COMPLEX128)
      .value("TF_HALF", TF_HALF)
      .value("TF_RESOURCE", TF_RESOURCE)
      .value("TF_VARIANT", TF_VARIANT)
      .value("TF_UINT32", TF_UINT32)
      .value("TF_UINT64", TF_UINT64)
      .export_values();

  py::enum_<TF_Code>(m, "TF_Code")
      .value("TF_OK", TF_OK)
      .value("TF_CANCELLED", TF_CANCELLED)
      .value("TF_UNKNOWN", TF_UNKNOWN)
      .value("TF_INVALID_ARGUMENT", TF_INVALID_ARGUMENT)
      .value("TF_DEADLINE_EXCEEDED", TF_DEADLINE_EXCEEDED)
      .value("TF_PERMISSION_DENIED", TF_PERMISSION_DENIED)
      .value("TF_UNAUTHENTICATED", TF_UNAUTHENTICATED)
      .value("TF_RESOURCE_EXHAUSTED", TF_RESOURCE_EXHAUSTED)
      .value("TF_FAILED_PRECONDITION", TF_FAILED_PRECONDITION)
      .value("TF_ABORTED", TF_ABORTED)
      .value("TF_OUT_OF_RANGE", TF_OUT_OF_RANGE)
      .value("TF_UNIMPLEMENTED", TF_UNIMPLEMENTED)
      .value("TF_INTERNAL", TF_INTERNAL)
      .value("TF_DATA_LOSS", TF_DATA_LOSS)
      .export_values();
};
