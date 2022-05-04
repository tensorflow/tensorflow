/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>
#include <cstring>
#include <unordered_map>

#include "absl/debugging/leak_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tape.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/compactptrset.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/python/eager/pywrap_gradient_exclusions.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
#include "tensorflow/python/util/stack_trace.h"
#include "tensorflow/python/util/util.h"

using tensorflow::Status;
using tensorflow::string;
using tensorflow::strings::Printf;

namespace {
// NOTE: Items are retrieved from and returned to these unique_ptrs, and they
// act as arenas. This is important if the same thread requests 2 items without
// releasing one.
// The following sequence of events on the same thread will still succeed:
// - GetOp <- Returns existing.
// - GetOp <- Allocates and returns a new pointer.
// - ReleaseOp <- Sets the item in the unique_ptr.
// - ReleaseOp <- Sets the item in the unique_ptr, deleting the old one.
// This occurs when a PyFunc kernel is run. This behavior makes it safe in that
// case, as well as the case where python decides to reuse the underlying
// C++ thread in 2 python threads case.
struct OpDeleter {
  void operator()(TFE_Op* op) const { TFE_DeleteOp(op); }
};
thread_local std::unordered_map<TFE_Context*,
                                std::unique_ptr<TFE_Op, OpDeleter>>
    thread_local_eager_operation_map;                             // NOLINT
thread_local std::unique_ptr<TF_Status> thread_local_tf_status =  // NOLINT
    nullptr;

std::unique_ptr<TFE_Op, OpDeleter> ReleaseThreadLocalOp(TFE_Context* ctx) {
  auto it = thread_local_eager_operation_map.find(ctx);
  if (it == thread_local_eager_operation_map.end()) {
    return nullptr;
  }
  return std::move(it->second);
}

TFE_Op* GetOp(TFE_Context* ctx, const char* op_or_function_name,
              const char* raw_device_name, TF_Status* status) {
  auto op = ReleaseThreadLocalOp(ctx);
  if (!op) {
    op.reset(tensorflow::wrap(tensorflow::unwrap(ctx)->CreateOperation()));
  }
  status->status =
      tensorflow::unwrap(op.get())->Reset(op_or_function_name, raw_device_name);
  if (!status->status.ok()) {
    op.reset();
  }
  return op.release();
}

void ReturnOp(TFE_Context* ctx, TFE_Op* op) {
  if (op) {
    tensorflow::unwrap(op)->Clear();
    thread_local_eager_operation_map[ctx].reset(op);
  }
}

TF_Status* ReleaseThreadLocalStatus() {
  if (thread_local_tf_status == nullptr) {
    return nullptr;
  }
  return thread_local_tf_status.release();
}

struct InputInfo {
  InputInfo(int i, bool is_list) : i(i), is_list(is_list) {}

  int i;
  bool is_list = false;
};

// Takes in output gradients, returns input gradients.
typedef std::function<PyObject*(PyObject*, const std::vector<int64_t>&)>
    PyBackwardFunction;

using AttrToInputsMap =
    tensorflow::gtl::FlatMap<string,
                             tensorflow::gtl::InlinedVector<InputInfo, 4>>;

tensorflow::gtl::FlatMap<string, AttrToInputsMap*>* GetAllAttrToInputsMaps() {
  static auto* all_attr_to_input_maps =
      new tensorflow::gtl::FlatMap<string, AttrToInputsMap*>;
  return all_attr_to_input_maps;
}

// This function doesn't use a lock, since we depend on the GIL directly.
AttrToInputsMap* GetAttrToInputsMapHoldingGIL(const tensorflow::OpDef& op_def) {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 4
  DCHECK(PyGILState_Check())
      << "This function needs to hold the GIL when called.";
#endif
  auto* all_attr_to_input_maps = GetAllAttrToInputsMaps();
  auto* output =
      tensorflow::gtl::FindPtrOrNull(*all_attr_to_input_maps, op_def.name());
  if (output != nullptr) {
    return output;
  }

  std::unique_ptr<AttrToInputsMap> m(new AttrToInputsMap);

  // Store a list of InputIndex -> List of corresponding inputs.
  for (int i = 0; i < op_def.input_arg_size(); i++) {
    if (!op_def.input_arg(i).type_attr().empty()) {
      auto it = m->find(op_def.input_arg(i).type_attr());
      if (it == m->end()) {
        it = m->insert({op_def.input_arg(i).type_attr(), {}}).first;
      }
      it->second.emplace_back(i, !op_def.input_arg(i).number_attr().empty());
    }
  }

  auto* retval = m.get();
  (*all_attr_to_input_maps)[op_def.name()] = m.release();

  return retval;
}

// This function doesn't use a lock, since we depend on the GIL directly.
tensorflow::gtl::FlatMap<
    string, tensorflow::gtl::FlatMap<string, tensorflow::DataType>*>*
GetAllAttrToDefaultsMaps() {
  static auto* all_attr_to_defaults_maps = new tensorflow::gtl::FlatMap<
      string, tensorflow::gtl::FlatMap<string, tensorflow::DataType>*>;
  return all_attr_to_defaults_maps;
}

tensorflow::gtl::FlatMap<string, tensorflow::DataType>*
GetAttrToDefaultsMapHoldingGIL(const tensorflow::OpDef& op_def) {
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 4
  DCHECK(PyGILState_Check())
      << "This function needs to hold the GIL when called.";
#endif
  auto* all_attr_to_defaults_maps = GetAllAttrToDefaultsMaps();
  auto* output =
      tensorflow::gtl::FindPtrOrNull(*all_attr_to_defaults_maps, op_def.name());
  if (output != nullptr) {
    return output;
  }

  auto* new_map = new tensorflow::gtl::FlatMap<string, tensorflow::DataType>;

  for (const auto& attr : op_def.attr()) {
    if (attr.type() == "type" && attr.has_default_value()) {
      new_map->insert({attr.name(), attr.default_value().type()});
    }
  }

  (*all_attr_to_defaults_maps)[op_def.name()] = new_map;

  return new_map;
}

struct FastPathOpExecInfo {
  TFE_Context* ctx;
  const char* device_name;

  bool run_callbacks;
  bool run_post_exec_callbacks;
  bool run_gradient_callback;

  // The op name of the main op being executed.
  PyObject* name;
  // The op type name of the main op being executed.
  PyObject* op_name;
  PyObject* callbacks;

  // All the args passed into the FastPathOpExecInfo.
  PyObject* args;

  // DTypes can come from another input that has the same attr. So build that
  // map.
  const AttrToInputsMap* attr_to_inputs_map;
  const tensorflow::gtl::FlatMap<string, tensorflow::DataType>* default_dtypes;
  tensorflow::gtl::FlatMap<string, tensorflow::DataType> cached_dtypes;
};

#define PARSE_VALUE(fn_name, type, check_fn, parse_fn)                       \
  bool fn_name(const string& key, PyObject* py_value, TF_Status* status,     \
               type* value) {                                                \
    if (check_fn(py_value)) {                                                \
      *value = static_cast<type>(parse_fn(py_value));                        \
      return true;                                                           \
    } else {                                                                 \
      TF_SetStatus(status, TF_INVALID_ARGUMENT,                              \
                   tensorflow::strings::StrCat(                              \
                       "Expecting " #type " value for attr ", key, ", got ", \
                       py_value->ob_type->tp_name)                           \
                       .c_str());                                            \
      return false;                                                          \
    }                                                                        \
  }

#if PY_MAJOR_VERSION >= 3
PARSE_VALUE(ParseIntValue, int, PyLong_Check, PyLong_AsLong)
PARSE_VALUE(ParseInt64Value, int64_t, PyLong_Check, PyLong_AsLongLong)
#else
PARSE_VALUE(ParseIntValue, int, PyInt_Check, PyInt_AsLong)
#endif
PARSE_VALUE(ParseFloatValue, float, PyFloat_Check, PyFloat_AsDouble)
#undef PARSE_VALUE

#if PY_MAJOR_VERSION < 3
bool ParseInt64Value(const string& key, PyObject* py_value, TF_Status* status,
                     int64_t* value) {
  if (PyInt_Check(py_value)) {
    *value = static_cast<int64_t>(PyInt_AsLong(py_value));
    return true;
  } else if (PyLong_Check(py_value)) {
    *value = static_cast<int64_t>(PyLong_AsLong(py_value));
    return true;
  }
  TF_SetStatus(
      status, TF_INVALID_ARGUMENT,
      tensorflow::strings::StrCat("Expecting int or long value for attr ", key,
                                  ", got ", py_value->ob_type->tp_name)
          .c_str());
  return false;
}
#endif

Py_ssize_t TensorShapeNumDims(PyObject* value) {
  const auto size = PySequence_Size(value);
  if (size == -1) {
    // TensorShape.__len__ raises an error in the scenario where the shape is an
    // unknown, which needs to be cleared.
    // TODO(nareshmodi): ensure that this is actually a TensorShape.
    PyErr_Clear();
  }
  return size;
}

bool IsInteger(PyObject* py_value) {
#if PY_MAJOR_VERSION >= 3
  return PyLong_Check(py_value);
#else
  return PyInt_Check(py_value) || PyLong_Check(py_value);
#endif
}

// This function considers a Dimension._value of None to be valid, and sets the
// value to be -1 in that case.
bool ParseDimensionValue(const string& key, PyObject* py_value,
                         TF_Status* status, int64_t* value) {
  if (IsInteger(py_value)) {
    return ParseInt64Value(key, py_value, status, value);
  }

  tensorflow::Safe_PyObjectPtr dimension_value(
      PyObject_GetAttrString(py_value, "_value"));
  if (dimension_value == nullptr) {
    PyErr_Clear();
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        tensorflow::strings::StrCat("Expecting a Dimension for attr ", key,
                                    ", got ", py_value->ob_type->tp_name)
            .c_str());
    return false;
  }

  if (dimension_value.get() == Py_None) {
    *value = -1;
    return true;
  }

  return ParseInt64Value(key, dimension_value.get(), status, value);
}

bool ParseStringValue(const string& key, PyObject* py_value, TF_Status* status,
                      tensorflow::StringPiece* value) {
  if (PyBytes_Check(py_value)) {
    Py_ssize_t size = 0;
    char* buf = nullptr;
    if (PyBytes_AsStringAndSize(py_value, &buf, &size) < 0) return false;
    *value = tensorflow::StringPiece(buf, size);
    return true;
  }
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(py_value)) {
    Py_ssize_t size = 0;
    const char* buf = PyUnicode_AsUTF8AndSize(py_value, &size);
    if (buf == nullptr) return false;
    *value = tensorflow::StringPiece(buf, size);
    return true;
  }
#endif
  TF_SetStatus(
      status, TF_INVALID_ARGUMENT,
      tensorflow::strings::StrCat("Expecting a string value for attr ", key,
                                  ", got ", py_value->ob_type->tp_name)
          .c_str());
  return false;
}

bool ParseBoolValue(const string& key, PyObject* py_value, TF_Status* status,
                    unsigned char* value) {
  if (PyBool_Check(py_value)) {
    *value = PyObject_IsTrue(py_value);
    return true;
  }
  TF_SetStatus(
      status, TF_INVALID_ARGUMENT,
      tensorflow::strings::StrCat("Expecting bool value for attr ", key,
                                  ", got ", py_value->ob_type->tp_name)
          .c_str());
  return false;
}

// The passed in py_value is expected to be an object of the python type
// dtypes.DType or an int.
bool ParseTypeValue(const string& key, PyObject* py_value, TF_Status* status,
                    int* value) {
  if (IsInteger(py_value)) {
    return ParseIntValue(key, py_value, status, value);
  }

  tensorflow::Safe_PyObjectPtr py_type_enum(
      PyObject_GetAttrString(py_value, "_type_enum"));
  if (py_type_enum == nullptr) {
    PyErr_Clear();
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        tensorflow::strings::StrCat("Expecting a DType.dtype for attr ", key,
                                    ", got ", py_value->ob_type->tp_name)
            .c_str());
    return false;
  }

  return ParseIntValue(key, py_type_enum.get(), status, value);
}

bool SetOpAttrList(TFE_Context* ctx, TFE_Op* op, const char* key,
                   PyObject* py_list, TF_AttrType type,
                   tensorflow::gtl::FlatMap<string, int64_t>* attr_list_sizes,
                   TF_Status* status) {
  if (!PySequence_Check(py_list)) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        tensorflow::strings::StrCat("Expecting sequence value for attr ", key,
                                    ", got ", py_list->ob_type->tp_name)
            .c_str());
    return false;
  }
  const int num_values = PySequence_Size(py_list);
  if (attr_list_sizes != nullptr) (*attr_list_sizes)[key] = num_values;

#define PARSE_LIST(c_type, parse_fn)                                      \
  std::unique_ptr<c_type[]> values(new c_type[num_values]);               \
  for (int i = 0; i < num_values; ++i) {                                  \
    tensorflow::Safe_PyObjectPtr py_value(PySequence_ITEM(py_list, i));   \
    if (!parse_fn(key, py_value.get(), status, &values[i])) return false; \
  }

  if (type == TF_ATTR_STRING) {
    std::unique_ptr<const void*[]> values(new const void*[num_values]);
    std::unique_ptr<size_t[]> lengths(new size_t[num_values]);
    for (int i = 0; i < num_values; ++i) {
      tensorflow::StringPiece value;
      tensorflow::Safe_PyObjectPtr py_value(PySequence_ITEM(py_list, i));
      if (!ParseStringValue(key, py_value.get(), status, &value)) return false;
      values[i] = value.data();
      lengths[i] = value.size();
    }
    TFE_OpSetAttrStringList(op, key, values.get(), lengths.get(), num_values);
  } else if (type == TF_ATTR_INT) {
    PARSE_LIST(int64_t, ParseInt64Value);
    TFE_OpSetAttrIntList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_FLOAT) {
    PARSE_LIST(float, ParseFloatValue);
    TFE_OpSetAttrFloatList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_BOOL) {
    PARSE_LIST(unsigned char, ParseBoolValue);
    TFE_OpSetAttrBoolList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_TYPE) {
    PARSE_LIST(int, ParseTypeValue);
    TFE_OpSetAttrTypeList(op, key,
                          reinterpret_cast<const TF_DataType*>(values.get()),
                          num_values);
  } else if (type == TF_ATTR_SHAPE) {
    // Make one pass through the input counting the total number of
    // dims across all the input lists.
    int total_dims = 0;
    for (int i = 0; i < num_values; ++i) {
      tensorflow::Safe_PyObjectPtr py_value(PySequence_ITEM(py_list, i));
      if (py_value.get() != Py_None) {
        if (!PySequence_Check(py_value.get())) {
          TF_SetStatus(
              status, TF_INVALID_ARGUMENT,
              tensorflow::strings::StrCat(
                  "Expecting None or sequence value for element", i,
                  " of attr ", key, ", got ", py_value->ob_type->tp_name)
                  .c_str());
          return false;
        }
        const auto size = TensorShapeNumDims(py_value.get());
        if (size >= 0) {
          total_dims += size;
        }
      }
    }
    // Allocate a buffer that can fit all of the dims together.
    std::unique_ptr<int64_t[]> buffer(new int64_t[total_dims]);
    // Copy the input dims into the buffer and set dims to point to
    // the start of each list's dims.
    std::unique_ptr<const int64_t*[]> dims(new const int64_t*[num_values]);
    std::unique_ptr<int[]> num_dims(new int[num_values]);
    int64_t* offset = buffer.get();
    for (int i = 0; i < num_values; ++i) {
      tensorflow::Safe_PyObjectPtr py_value(PySequence_ITEM(py_list, i));
      if (py_value.get() == Py_None) {
        dims[i] = nullptr;
        num_dims[i] = -1;
      } else {
        const auto size = TensorShapeNumDims(py_value.get());
        if (size == -1) {
          dims[i] = nullptr;
          num_dims[i] = -1;
          continue;
        }
        dims[i] = offset;
        num_dims[i] = size;
        for (int j = 0; j < size; ++j) {
          tensorflow::Safe_PyObjectPtr inner_py_value(
              PySequence_ITEM(py_value.get(), j));
          if (inner_py_value.get() == Py_None) {
            *offset = -1;
          } else if (!ParseDimensionValue(key, inner_py_value.get(), status,
                                          offset)) {
            return false;
          }
          ++offset;
        }
      }
    }
    TFE_OpSetAttrShapeList(op, key, dims.get(), num_dims.get(), num_values,
                           status);
    if (!status->status.ok()) return false;
  } else if (type == TF_ATTR_FUNC) {
    std::unique_ptr<const TFE_Op*[]> funcs(new const TFE_Op*[num_values]);
    for (int i = 0; i < num_values; ++i) {
      tensorflow::Safe_PyObjectPtr py_value(PySequence_ITEM(py_list, i));
      // Allow:
      // (1) String function name, OR
      // (2) A Python object with a .name attribute
      //     (A crude test for being a
      //     tensorflow.python.framework.function._DefinedFunction)
      //     (which is what the various "defun" or "Defun" decorators do).
      // And in the future also allow an object that can encapsulate
      // the function name and its attribute values.
      tensorflow::StringPiece func_name;
      if (!ParseStringValue(key, py_value.get(), status, &func_name)) {
        PyObject* name_attr = PyObject_GetAttrString(py_value.get(), "name");
        if (name_attr == nullptr ||
            !ParseStringValue(key, name_attr, status, &func_name)) {
          TF_SetStatus(
              status, TF_INVALID_ARGUMENT,
              tensorflow::strings::StrCat(
                  "unable to set function value attribute from a ",
                  py_value.get()->ob_type->tp_name,
                  " object. If you think this is an error, please file an "
                  "issue at "
                  "https://github.com/tensorflow/tensorflow/issues/new")
                  .c_str());
          return false;
        }
      }
      funcs[i] = TFE_NewOp(ctx, func_name.data(), status);
      if (!status->status.ok()) return false;
    }
    TFE_OpSetAttrFunctionList(op, key, funcs.get(), num_values);
    if (!status->status.ok()) return false;
  } else {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 tensorflow::strings::StrCat("Attr ", key,
                                             " has unhandled list type ", type)
                     .c_str());
    return false;
  }
#undef PARSE_LIST
  return true;
}

TFE_Op* GetFunc(TFE_Context* ctx, const tensorflow::NameAttrList& func,
                TF_Status* status) {
  TFE_Op* func_op = TFE_NewOp(ctx, func.name().data(), status);
  for (const auto& attr : func.attr()) {
    if (!status->status.ok()) return nullptr;
    SetOpAttrValueScalar(ctx, func_op, attr.second, attr.first.data(), status);
    if (!status->status.ok()) return nullptr;
  }
  return func_op;
}

void SetOpAttrListDefault(
    TFE_Context* ctx, TFE_Op* op, const tensorflow::OpDef::AttrDef& attr,
    const char* key, TF_AttrType type,
    tensorflow::gtl::FlatMap<string, int64_t>* attr_list_sizes,
    TF_Status* status) {
  if (type == TF_ATTR_STRING) {
    int num_values = attr.default_value().list().s_size();
    std::unique_ptr<const void*[]> values(new const void*[num_values]);
    std::unique_ptr<size_t[]> lengths(new size_t[num_values]);
    (*attr_list_sizes)[key] = num_values;
    for (int i = 0; i < num_values; i++) {
      const string& v = attr.default_value().list().s(i);
      values[i] = v.data();
      lengths[i] = v.size();
    }
    TFE_OpSetAttrStringList(op, key, values.get(), lengths.get(), num_values);
  } else if (type == TF_ATTR_INT) {
    int num_values = attr.default_value().list().i_size();
    std::unique_ptr<int64_t[]> values(new int64_t[num_values]);
    (*attr_list_sizes)[key] = num_values;
    for (int i = 0; i < num_values; i++) {
      values[i] = attr.default_value().list().i(i);
    }
    TFE_OpSetAttrIntList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_FLOAT) {
    int num_values = attr.default_value().list().f_size();
    std::unique_ptr<float[]> values(new float[num_values]);
    (*attr_list_sizes)[key] = num_values;
    for (int i = 0; i < num_values; i++) {
      values[i] = attr.default_value().list().f(i);
    }
    TFE_OpSetAttrFloatList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_BOOL) {
    int num_values = attr.default_value().list().b_size();
    std::unique_ptr<unsigned char[]> values(new unsigned char[num_values]);
    (*attr_list_sizes)[key] = num_values;
    for (int i = 0; i < num_values; i++) {
      values[i] = attr.default_value().list().b(i);
    }
    TFE_OpSetAttrBoolList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_TYPE) {
    int num_values = attr.default_value().list().type_size();
    std::unique_ptr<int[]> values(new int[num_values]);
    (*attr_list_sizes)[key] = num_values;
    for (int i = 0; i < num_values; i++) {
      values[i] = attr.default_value().list().type(i);
    }
    TFE_OpSetAttrTypeList(op, key,
                          reinterpret_cast<const TF_DataType*>(values.get()),
                          attr.default_value().list().type_size());
  } else if (type == TF_ATTR_SHAPE) {
    int num_values = attr.default_value().list().shape_size();
    (*attr_list_sizes)[key] = num_values;
    int total_dims = 0;
    for (int i = 0; i < num_values; ++i) {
      if (!attr.default_value().list().shape(i).unknown_rank()) {
        total_dims += attr.default_value().list().shape(i).dim_size();
      }
    }
    // Allocate a buffer that can fit all of the dims together.
    std::unique_ptr<int64_t[]> buffer(new int64_t[total_dims]);
    // Copy the input dims into the buffer and set dims to point to
    // the start of each list's dims.
    std::unique_ptr<const int64_t*[]> dims(new const int64_t*[num_values]);
    std::unique_ptr<int[]> num_dims(new int[num_values]);
    int64_t* offset = buffer.get();
    for (int i = 0; i < num_values; ++i) {
      const auto& shape = attr.default_value().list().shape(i);
      if (shape.unknown_rank()) {
        dims[i] = nullptr;
        num_dims[i] = -1;
      } else {
        for (int j = 0; j < shape.dim_size(); j++) {
          *offset = shape.dim(j).size();
          ++offset;
        }
      }
    }
    TFE_OpSetAttrShapeList(op, key, dims.get(), num_dims.get(), num_values,
                           status);
  } else if (type == TF_ATTR_FUNC) {
    int num_values = attr.default_value().list().func_size();
    (*attr_list_sizes)[key] = num_values;
    std::unique_ptr<const TFE_Op*[]> funcs(new const TFE_Op*[num_values]);
    for (int i = 0; i < num_values; i++) {
      funcs[i] = GetFunc(ctx, attr.default_value().list().func(i), status);
    }
    TFE_OpSetAttrFunctionList(op, key, funcs.get(), num_values);
  } else {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Lists of tensors are not yet implemented for default valued "
                 "attributes for an operation.");
  }
}

bool SetOpAttrScalar(TFE_Context* ctx, TFE_Op* op, const char* key,
                     PyObject* py_value, TF_AttrType type,
                     tensorflow::gtl::FlatMap<string, int64_t>* attr_list_sizes,
                     TF_Status* status) {
  if (type == TF_ATTR_STRING) {
    tensorflow::StringPiece value;
    if (!ParseStringValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrString(op, key, value.data(), value.size());
  } else if (type == TF_ATTR_INT) {
    int64_t value;
    if (!ParseInt64Value(key, py_value, status, &value)) return false;
    TFE_OpSetAttrInt(op, key, value);
    // attr_list_sizes is set for all int attributes (since at this point we are
    // not aware if that attribute might be used to calculate the size of an
    // output list or not).
    if (attr_list_sizes != nullptr) (*attr_list_sizes)[key] = value;
  } else if (type == TF_ATTR_FLOAT) {
    float value;
    if (!ParseFloatValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrFloat(op, key, value);
  } else if (type == TF_ATTR_BOOL) {
    unsigned char value;
    if (!ParseBoolValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrBool(op, key, value);
  } else if (type == TF_ATTR_TYPE) {
    int value;
    if (!ParseTypeValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrType(op, key, static_cast<TF_DataType>(value));
  } else if (type == TF_ATTR_SHAPE) {
    if (py_value == Py_None) {
      TFE_OpSetAttrShape(op, key, nullptr, -1, status);
    } else {
      if (!PySequence_Check(py_value)) {
        TF_SetStatus(status, TF_INVALID_ARGUMENT,
                     tensorflow::strings::StrCat(
                         "Expecting None or sequence value for attr", key,
                         ", got ", py_value->ob_type->tp_name)
                         .c_str());
        return false;
      }
      const auto num_dims = TensorShapeNumDims(py_value);
      if (num_dims == -1) {
        TFE_OpSetAttrShape(op, key, nullptr, -1, status);
        return true;
      }
      std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
      for (int i = 0; i < num_dims; ++i) {
        tensorflow::Safe_PyObjectPtr inner_py_value(
            PySequence_ITEM(py_value, i));
        // If an error is generated when iterating through object, we can
        // sometimes get a nullptr.
        if (inner_py_value.get() == Py_None) {
          dims[i] = -1;
        } else if (inner_py_value.get() == nullptr ||
                   !ParseDimensionValue(key, inner_py_value.get(), status,
                                        &dims[i])) {
          return false;
        }
      }
      TFE_OpSetAttrShape(op, key, dims.get(), num_dims, status);
    }
    if (!status->status.ok()) return false;
  } else if (type == TF_ATTR_FUNC) {
    // Allow:
    // (1) String function name, OR
    // (2) A Python object with a .name attribute
    //     (A crude test for being a
    //     tensorflow.python.framework.function._DefinedFunction)
    //     (which is what the various "defun" or "Defun" decorators do).
    // And in the future also allow an object that can encapsulate
    // the function name and its attribute values.
    tensorflow::StringPiece func_name;
    if (!ParseStringValue(key, py_value, status, &func_name)) {
      PyObject* name_attr = PyObject_GetAttrString(py_value, "name");
      if (name_attr == nullptr ||
          !ParseStringValue(key, name_attr, status, &func_name)) {
        TF_SetStatus(
            status, TF_INVALID_ARGUMENT,
            tensorflow::strings::StrCat(
                "unable to set function value attribute from a ",
                py_value->ob_type->tp_name,
                " object. If you think this is an error, please file an issue "
                "at https://github.com/tensorflow/tensorflow/issues/new")
                .c_str());
        return false;
      }
    }
    TF_SetStatus(status, TF_OK, "");
    TFE_OpSetAttrFunctionName(op, key, func_name.data(), func_name.size());
  } else {
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        tensorflow::strings::StrCat("Attr ", key, " has unhandled type ", type)
            .c_str());
    return false;
  }
  return true;
}

void SetOpAttrScalarDefault(
    TFE_Context* ctx, TFE_Op* op, const tensorflow::AttrValue& default_value,
    const char* attr_name,
    tensorflow::gtl::FlatMap<string, int64_t>* attr_list_sizes,
    TF_Status* status) {
  SetOpAttrValueScalar(ctx, op, default_value, attr_name, status);
  if (default_value.value_case() == tensorflow::AttrValue::kI) {
    (*attr_list_sizes)[attr_name] = default_value.i();
  }
}

// start_index is the index at which the Tuple/List attrs will start getting
// processed.
void SetOpAttrs(TFE_Context* ctx, TFE_Op* op, PyObject* attrs, int start_index,
                TF_Status* out_status) {
  if (attrs == Py_None) return;
  Py_ssize_t len = PyTuple_GET_SIZE(attrs) - start_index;
  if ((len & 1) != 0) {
    TF_SetStatus(out_status, TF_INVALID_ARGUMENT,
                 "Expecting attrs tuple to have even length.");
    return;
  }
  // Parse attrs
  for (Py_ssize_t i = 0; i < len; i += 2) {
    PyObject* py_key = PyTuple_GET_ITEM(attrs, start_index + i);
    PyObject* py_value = PyTuple_GET_ITEM(attrs, start_index + i + 1);
#if PY_MAJOR_VERSION >= 3
    const char* key = PyBytes_Check(py_key) ? PyBytes_AsString(py_key)
                                            : PyUnicode_AsUTF8(py_key);
#else
    const char* key = PyBytes_AsString(py_key);
#endif
    unsigned char is_list = 0;
    const TF_AttrType type = TFE_OpGetAttrType(op, key, &is_list, out_status);
    if (!out_status->status.ok()) return;
    if (is_list != 0) {
      if (!SetOpAttrList(ctx, op, key, py_value, type, nullptr, out_status))
        return;
    } else {
      if (!SetOpAttrScalar(ctx, op, key, py_value, type, nullptr, out_status))
        return;
    }
  }
}

// This function will set the op attrs required. If an attr has the value of
// None, then it will read the AttrDef to get the default value and set that
// instead. Any failure in this function will simply fall back to the slow
// path.
void SetOpAttrWithDefaults(
    TFE_Context* ctx, TFE_Op* op, const tensorflow::OpDef::AttrDef& attr,
    const char* attr_name, PyObject* attr_value,
    tensorflow::gtl::FlatMap<string, int64_t>* attr_list_sizes,
    TF_Status* status) {
  unsigned char is_list = 0;
  const TF_AttrType type = TFE_OpGetAttrType(op, attr_name, &is_list, status);
  if (!status->status.ok()) return;
  if (attr_value == Py_None) {
    if (is_list != 0) {
      SetOpAttrListDefault(ctx, op, attr, attr_name, type, attr_list_sizes,
                           status);
    } else {
      SetOpAttrScalarDefault(ctx, op, attr.default_value(), attr_name,
                             attr_list_sizes, status);
    }
  } else {
    if (is_list != 0) {
      SetOpAttrList(ctx, op, attr_name, attr_value, type, attr_list_sizes,
                    status);
    } else {
      SetOpAttrScalar(ctx, op, attr_name, attr_value, type, attr_list_sizes,
                      status);
    }
  }
}

PyObject* GetPythonObjectFromInt(int num) {
#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong(num);
#else
  return PyInt_FromLong(num);
#endif
}

// Python subclass of Exception that is created on not ok Status.
tensorflow::mutex exception_class_mutex(tensorflow::LINKER_INITIALIZED);
PyObject* exception_class TF_GUARDED_BY(exception_class_mutex) = nullptr;

// Python subclass of Exception that is created to signal fallback.
PyObject* fallback_exception_class = nullptr;

// Python function that returns input gradients given output gradients.
PyObject* gradient_function = nullptr;

// Python function that returns output gradients given input gradients.
PyObject* forward_gradient_function = nullptr;

static std::atomic<int64_t> _uid;

// This struct is responsible for marking thread_local storage as destroyed.
// Access to the `alive` field in already-destroyed ThreadLocalDestructionMarker
// is safe because it's a trivial type, so long as nobody creates a new
// thread_local in the space where now-destroyed marker used to be.
// Hopefully creating new thread_locals while destructing a thread is rare.
struct ThreadLocalDestructionMarker {
  ~ThreadLocalDestructionMarker() { alive = false; }
  bool alive = true;
};

}  // namespace

TF_Status* GetStatus() {
  TF_Status* maybe_status = ReleaseThreadLocalStatus();
  if (maybe_status) {
    TF_SetStatus(maybe_status, TF_OK, "");
    return maybe_status;
  } else {
    return TF_NewStatus();
  }
}

void ReturnStatus(TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  thread_local_tf_status.reset(status);
}

void TFE_Py_Execute(TFE_Context* ctx, const char* device_name,
                    const char* op_name, TFE_InputTensorHandles* inputs,
                    PyObject* attrs, TFE_OutputTensorHandles* outputs,
                    TF_Status* out_status) {
  TFE_Py_ExecuteCancelable(ctx, device_name, op_name, inputs, attrs,
                           /*cancellation_manager=*/nullptr, outputs,
                           out_status);
}

void TFE_Py_ExecuteCancelable(TFE_Context* ctx, const char* device_name,
                              const char* op_name,
                              TFE_InputTensorHandles* inputs, PyObject* attrs,
                              TFE_CancellationManager* cancellation_manager,
                              TFE_OutputTensorHandles* outputs,
                              TF_Status* out_status) {
  tensorflow::profiler::TraceMe activity(
      "TFE_Py_ExecuteCancelable", tensorflow::profiler::TraceMeLevel::kInfo);

  TFE_Op* op = GetOp(ctx, op_name, device_name, out_status);

  auto cleaner = tensorflow::gtl::MakeCleanup([ctx, op] { ReturnOp(ctx, op); });
  if (!out_status->status.ok()) return;

  tensorflow::unwrap(op)->SetStackTrace(tensorflow::GetStackTrace(
      tensorflow::StackTrace::kStackTraceInitialSize));

  for (int i = 0; i < inputs->size() && out_status->status.ok(); ++i) {
    TFE_OpAddInput(op, inputs->at(i), out_status);
  }
  if (cancellation_manager && out_status->status.ok()) {
    TFE_OpSetCancellationManager(op, cancellation_manager, out_status);
  }
  if (out_status->status.ok()) {
    SetOpAttrs(ctx, op, attrs, 0, out_status);
  }
  Py_BEGIN_ALLOW_THREADS;

  int num_outputs = outputs->size();

  if (out_status->status.ok()) {
    TFE_Execute(op, outputs->data(), &num_outputs, out_status);
  }

  if (out_status->status.ok()) {
    outputs->resize(num_outputs);
  } else {
    TF_SetStatus(out_status, TF_GetCode(out_status),
                 tensorflow::strings::StrCat(TF_Message(out_status),
                                             " [Op:", op_name, "]")
                     .c_str());
  }

  Py_END_ALLOW_THREADS;
}

PyObject* TFE_Py_RegisterExceptionClass(PyObject* e) {
  tensorflow::mutex_lock l(exception_class_mutex);
  if (exception_class != nullptr) {
    Py_DECREF(exception_class);
  }
  if (PyObject_IsSubclass(e, PyExc_Exception) <= 0) {
    exception_class = nullptr;
    PyErr_SetString(PyExc_TypeError,
                    "TFE_Py_RegisterExceptionClass: "
                    "Registered class should be subclass of Exception.");
    return nullptr;
  }

  Py_INCREF(e);
  exception_class = e;
  Py_RETURN_NONE;
}

PyObject* TFE_Py_RegisterFallbackExceptionClass(PyObject* e) {
  if (fallback_exception_class != nullptr) {
    Py_DECREF(fallback_exception_class);
  }
  if (PyObject_IsSubclass(e, PyExc_Exception) <= 0) {
    fallback_exception_class = nullptr;
    PyErr_SetString(PyExc_TypeError,
                    "TFE_Py_RegisterFallbackExceptionClass: "
                    "Registered class should be subclass of Exception.");
    return nullptr;
  } else {
    Py_INCREF(e);
    fallback_exception_class = e;
    Py_RETURN_NONE;
  }
}

PyObject* TFE_Py_RegisterGradientFunction(PyObject* e) {
  if (gradient_function != nullptr) {
    Py_DECREF(gradient_function);
  }
  if (!PyCallable_Check(e)) {
    gradient_function = nullptr;
    PyErr_SetString(PyExc_TypeError,
                    "TFE_Py_RegisterGradientFunction: "
                    "Registered object should be function.");
    return nullptr;
  } else {
    Py_INCREF(e);
    gradient_function = e;
    Py_RETURN_NONE;
  }
}

PyObject* TFE_Py_RegisterJVPFunction(PyObject* e) {
  if (forward_gradient_function != nullptr) {
    Py_DECREF(forward_gradient_function);
  }
  if (!PyCallable_Check(e)) {
    forward_gradient_function = nullptr;
    PyErr_SetString(PyExc_TypeError,
                    "TFE_Py_RegisterJVPFunction: "
                    "Registered object should be function.");
    return nullptr;
  } else {
    Py_INCREF(e);
    forward_gradient_function = e;
    Py_RETURN_NONE;
  }
}

void RaiseFallbackException(const char* message) {
  if (fallback_exception_class != nullptr) {
    PyErr_SetString(fallback_exception_class, message);
    return;
  }

  PyErr_SetString(
      PyExc_RuntimeError,
      tensorflow::strings::StrCat(
          "Fallback exception type not set, attempting to fallback due to ",
          message)
          .data());
}

// Format and return `status`' error message with the attached stack trace if
// available. `status` must have an error.
std::string FormatErrorStatusStackTrace(const tensorflow::Status& status) {
  tensorflow::DCheckPyGilState();
  DCHECK(!status.ok());

  if (status.stack_trace().empty()) return status.error_message();

  const std::vector<tensorflow::StackFrame>& stack_trace = status.stack_trace();

  PyObject* linecache = PyImport_ImportModule("linecache");
  PyObject* getline =
      PyObject_GetAttr(linecache, PyUnicode_FromString("getline"));
  DCHECK(getline);

  std::ostringstream result;
  result << "Exception originated from\n\n";

  for (const tensorflow::StackFrame& stack_frame : stack_trace) {
    PyObject* line_str_obj = PyObject_CallFunction(
        getline, const_cast<char*>("si"), stack_frame.file_name.c_str(),
        stack_frame.line_number);
    tensorflow::StringPiece line_str = TFE_GetPythonString(line_str_obj);
    tensorflow::str_util::RemoveWhitespaceContext(&line_str);
    result << "  File \"" << stack_frame.file_name << "\", line "
           << stack_frame.line_number << ", in " << stack_frame.function_name
           << '\n';

    if (!line_str.empty()) result << "    " << line_str << '\n';
    Py_XDECREF(line_str_obj);
  }

  Py_DecRef(getline);
  Py_DecRef(linecache);

  result << '\n' << status.error_message();
  return result.str();
}

namespace tensorflow {

int MaybeRaiseExceptionFromTFStatus(TF_Status* status, PyObject* exception) {
  if (status->status.ok()) return 0;
  const char* msg = TF_Message(status);
  if (exception == nullptr) {
    tensorflow::mutex_lock l(exception_class_mutex);
    if (exception_class != nullptr) {
      tensorflow::Safe_PyObjectPtr payloads(PyDict_New());
      for (const auto& payload :
           tensorflow::errors::GetPayloads(status->status)) {
        PyDict_SetItem(payloads.get(),
                       PyBytes_FromString(payload.first.c_str()),
                       PyBytes_FromString(payload.second.c_str()));
      }
      tensorflow::Safe_PyObjectPtr val(Py_BuildValue(
          "siO", FormatErrorStatusStackTrace(status->status).c_str(),
          TF_GetCode(status), payloads.get()));
      if (PyErr_Occurred()) {
        // NOTE: This hides the actual error (i.e. the reason `status` was not
        // TF_OK), but there is nothing we can do at this point since we can't
        // generate a reasonable error from the status.
        // Consider adding a message explaining this.
        return -1;
      }
      PyErr_SetObject(exception_class, val.get());
      return -1;
    } else {
      exception = PyExc_RuntimeError;
    }
  }
  // May be update already set exception.
  PyErr_SetString(exception, msg);
  return -1;
}

}  // namespace tensorflow

int MaybeRaiseExceptionFromStatus(const tensorflow::Status& status,
                                  PyObject* exception) {
  if (status.ok()) return 0;
  const char* msg = status.error_message().c_str();
  if (exception == nullptr) {
    tensorflow::mutex_lock l(exception_class_mutex);
    if (exception_class != nullptr) {
      tensorflow::Safe_PyObjectPtr payloads(PyDict_New());
      for (const auto& element : tensorflow::errors::GetPayloads(status)) {
        PyDict_SetItem(payloads.get(),
                       PyBytes_FromString(element.first.c_str()),
                       PyBytes_FromString(element.second.c_str()));
      }
      tensorflow::Safe_PyObjectPtr val(
          Py_BuildValue("siO", FormatErrorStatusStackTrace(status).c_str(),
                        status.code(), payloads.get()));
      PyErr_SetObject(exception_class, val.get());
      return -1;
    } else {
      exception = PyExc_RuntimeError;
    }
  }
  // May be update already set exception.
  PyErr_SetString(exception, msg);
  return -1;
}

const char* TFE_GetPythonString(PyObject* o) {
#if PY_MAJOR_VERSION >= 3
  if (PyBytes_Check(o)) {
    return PyBytes_AsString(o);
  } else {
    return PyUnicode_AsUTF8(o);
  }
#else
  return PyBytes_AsString(o);
#endif
}

int64_t get_uid() { return _uid++; }

PyObject* TFE_Py_UID() { return PyLong_FromLongLong(get_uid()); }

void TFE_DeleteContextCapsule(PyObject* context) {
  TFE_Context* ctx =
      reinterpret_cast<TFE_Context*>(PyCapsule_GetPointer(context, nullptr));
  auto op = ReleaseThreadLocalOp(ctx);
  op.reset();
  TFE_DeleteContext(ctx);
}

static int64_t MakeInt(PyObject* integer) {
#if PY_MAJOR_VERSION >= 3
  return PyLong_AsLong(integer);
#else
  return PyInt_AsLong(integer);
#endif
}

static int64_t FastTensorId(PyObject* tensor) {
  if (EagerTensor_CheckExact(tensor)) {
    return PyEagerTensor_ID(tensor);
  }
  PyObject* id_field = PyObject_GetAttrString(tensor, "_id");
  if (id_field == nullptr) {
    return -1;
  }
  int64_t id = MakeInt(id_field);
  Py_DECREF(id_field);
  return id;
}

namespace tensorflow {
DataType PyTensor_DataType(PyObject* tensor) {
  if (EagerTensor_CheckExact(tensor)) {
    return PyEagerTensor_Dtype(tensor);
  } else {
#if PY_MAJOR_VERSION < 3
    // Python 2.x:
    static PyObject* dtype_attr = PyString_InternFromString("dtype");
    static PyObject* type_enum_attr = PyString_InternFromString("_type_enum");
#else
    // Python 3.x:
    static PyObject* dtype_attr = PyUnicode_InternFromString("dtype");
    static PyObject* type_enum_attr = PyUnicode_InternFromString("_type_enum");
#endif
    Safe_PyObjectPtr dtype_field(PyObject_GetAttr(tensor, dtype_attr));
    if (!dtype_field) {
      return DT_INVALID;
    }

    Safe_PyObjectPtr enum_field(
        PyObject_GetAttr(dtype_field.get(), type_enum_attr));
    if (!enum_field) {
      return DT_INVALID;
    }

    return static_cast<DataType>(MakeInt(enum_field.get()));
  }
}
}  // namespace tensorflow

class PyTapeTensor {
 public:
  PyTapeTensor(int64_t id, tensorflow::DataType dtype,
               const tensorflow::TensorShape& shape)
      : id_(id), dtype_(dtype), shape_(shape) {}
  PyTapeTensor(int64_t id, tensorflow::DataType dtype, PyObject* shape)
      : id_(id), dtype_(dtype), shape_(shape) {
    Py_INCREF(absl::get<1>(shape_));
  }
  PyTapeTensor(const PyTapeTensor& other) {
    id_ = other.id_;
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    if (shape_.index() == 1) {
      Py_INCREF(absl::get<1>(shape_));
    }
  }

  ~PyTapeTensor() {
    if (shape_.index() == 1) {
      Py_DECREF(absl::get<1>(shape_));
    }
  }
  PyObject* GetShape() const;
  PyObject* GetPyDType() const { return PyLong_FromLong(dtype_); }
  int64_t GetID() const { return id_; }
  tensorflow::DataType GetDType() const { return dtype_; }

  PyObject* OnesLike() const;
  PyObject* ZerosLike() const;

 private:
  int64_t id_;
  tensorflow::DataType dtype_;

  // Note that if shape_.index() == 1, meaning shape_ contains a PyObject, that
  // PyObject is the tensor itself. This is used to support tf.shape(tensor) for
  // partially-defined shapes and tf.zeros_like(tensor) for variant-dtype
  // tensors.
  absl::variant<tensorflow::TensorShape, PyObject*> shape_;
};

static PyTapeTensor TapeTensorFromTensor(PyObject* tensor);

class PyVSpace : public tensorflow::eager::VSpace<PyObject, PyBackwardFunction,
                                                  PyTapeTensor> {
 public:
  explicit PyVSpace(PyObject* py_vspace) : py_vspace_(py_vspace) {
    Py_INCREF(py_vspace_);
  }

  tensorflow::Status Initialize() {
    num_elements_ = PyObject_GetAttrString(py_vspace_, "num_elements_fn");
    if (num_elements_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    aggregate_fn_ = PyObject_GetAttrString(py_vspace_, "aggregate_fn");
    if (aggregate_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    zeros_fn_ = PyObject_GetAttrString(py_vspace_, "zeros_fn");
    if (zeros_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    zeros_like_fn_ = PyObject_GetAttrString(py_vspace_, "zeros_like_fn");
    if (zeros_like_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    ones_fn_ = PyObject_GetAttrString(py_vspace_, "ones_fn");
    if (ones_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    ones_like_fn_ = PyObject_GetAttrString(py_vspace_, "ones_like_fn");
    if (ones_like_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    graph_shape_fn_ = PyObject_GetAttrString(py_vspace_, "graph_shape_fn");
    if (graph_shape_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    return tensorflow::Status::OK();
  }

  ~PyVSpace() override {
    Py_XDECREF(num_elements_);
    Py_XDECREF(aggregate_fn_);
    Py_XDECREF(zeros_fn_);
    Py_XDECREF(zeros_like_fn_);
    Py_XDECREF(ones_fn_);
    Py_XDECREF(ones_like_fn_);
    Py_XDECREF(graph_shape_fn_);

    Py_DECREF(py_vspace_);
  }

  int64_t NumElements(PyObject* tensor) const final {
    if (EagerTensor_CheckExact(tensor)) {
      return PyEagerTensor_NumElements(tensor);
    }
    PyObject* arglist =
        Py_BuildValue("(O)", reinterpret_cast<PyObject*>(tensor));
    PyObject* result = PyEval_CallObject(num_elements_, arglist);
    Py_DECREF(arglist);
    if (result == nullptr) {
      // The caller detects whether a python exception has been raised.
      return -1;
    }
    int64_t r = MakeInt(result);
    Py_DECREF(result);
    return r;
  }

  PyObject* AggregateGradients(
      tensorflow::gtl::ArraySlice<PyObject*> gradient_tensors) const final {
    PyObject* list = PyList_New(gradient_tensors.size());
    for (int i = 0; i < gradient_tensors.size(); ++i) {
      // Note: stealing a reference to the gradient tensors.
      CHECK(gradient_tensors[i] != nullptr);
      CHECK(gradient_tensors[i] != Py_None);
      PyList_SET_ITEM(list, i,
                      reinterpret_cast<PyObject*>(gradient_tensors[i]));
    }
    PyObject* arglist = Py_BuildValue("(O)", list);
    CHECK(arglist != nullptr);
    PyObject* result = PyEval_CallObject(aggregate_fn_, arglist);
    Py_DECREF(arglist);
    Py_DECREF(list);
    return result;
  }

  int64_t TensorId(PyObject* tensor) const final {
    return FastTensorId(tensor);
  }

  void MarkAsResult(PyObject* gradient) const final { Py_INCREF(gradient); }

  PyObject* Ones(PyObject* shape, PyObject* dtype) const {
    if (PyErr_Occurred()) {
      return nullptr;
    }
    PyObject* arg_list = Py_BuildValue("OO", shape, dtype);
    PyObject* result = PyEval_CallObject(ones_fn_, arg_list);
    Py_DECREF(arg_list);
    return result;
  }

  PyObject* OnesLike(PyObject* tensor) const {
    if (PyErr_Occurred()) {
      return nullptr;
    }
    return PyObject_CallFunctionObjArgs(ones_like_fn_, tensor, NULL);
  }

  // Builds a tensor filled with ones with the same shape and dtype as `t`.
  Status BuildOnesLike(const PyTapeTensor& t,
                       PyObject** result) const override {
    *result = t.OnesLike();
    return Status::OK();
  }

  PyObject* Zeros(PyObject* shape, PyObject* dtype) const {
    if (PyErr_Occurred()) {
      return nullptr;
    }
    PyObject* arg_list = Py_BuildValue("OO", shape, dtype);
    PyObject* result = PyEval_CallObject(zeros_fn_, arg_list);
    Py_DECREF(arg_list);
    return result;
  }

  PyObject* ZerosLike(PyObject* tensor) const {
    if (PyErr_Occurred()) {
      return nullptr;
    }
    return PyObject_CallFunctionObjArgs(zeros_like_fn_, tensor, NULL);
  }

  PyObject* GraphShape(PyObject* tensor) const {
    PyObject* arg_list = Py_BuildValue("(O)", tensor);
    PyObject* result = PyEval_CallObject(graph_shape_fn_, arg_list);
    Py_DECREF(arg_list);
    return result;
  }

  tensorflow::Status CallBackwardFunction(
      const string& op_type, PyBackwardFunction* backward_function,
      const std::vector<int64_t>& unneeded_gradients,
      tensorflow::gtl::ArraySlice<PyObject*> output_gradients,
      absl::Span<PyObject*> result) const final {
    PyObject* grads = PyTuple_New(output_gradients.size());
    for (int i = 0; i < output_gradients.size(); ++i) {
      if (output_gradients[i] == nullptr) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(grads, i, Py_None);
      } else {
        PyTuple_SET_ITEM(grads, i,
                         reinterpret_cast<PyObject*>(output_gradients[i]));
      }
    }
    PyObject* py_result = (*backward_function)(grads, unneeded_gradients);
    Py_DECREF(grads);
    if (py_result == nullptr) {
      return tensorflow::errors::Internal("gradient function threw exceptions");
    }
    PyObject* seq =
        PySequence_Fast(py_result, "expected a sequence of gradients");
    if (seq == nullptr) {
      return tensorflow::errors::InvalidArgument(
          "gradient function did not return a list");
    }
    int len = PySequence_Fast_GET_SIZE(seq);
    if (len != result.size()) {
      return tensorflow::errors::Internal(
          "Recorded operation '", op_type,
          "' returned too few gradients. Expected ", result.size(),
          " but received ", len);
    }
    PyObject** seq_array = PySequence_Fast_ITEMS(seq);
    VLOG(1) << "Gradient length is " << len;
    for (int i = 0; i < len; ++i) {
      PyObject* item = seq_array[i];
      if (item == Py_None) {
        result[i] = nullptr;
      } else {
        Py_INCREF(item);
        result[i] = item;
      }
    }
    Py_DECREF(seq);
    Py_DECREF(py_result);
    return tensorflow::Status::OK();
  }

  void DeleteGradient(PyObject* tensor) const final { Py_XDECREF(tensor); }

  PyTapeTensor TapeTensorFromGradient(PyObject* tensor) const final {
    return TapeTensorFromTensor(tensor);
  }

 private:
  PyObject* py_vspace_;

  PyObject* num_elements_;
  PyObject* aggregate_fn_;
  PyObject* zeros_fn_;
  PyObject* zeros_like_fn_;
  PyObject* ones_fn_;
  PyObject* ones_like_fn_;
  PyObject* graph_shape_fn_;
};
PyVSpace* py_vspace = nullptr;

bool HasAccumulator();

PyObject* TFE_Py_RegisterVSpace(PyObject* e) {
  if (py_vspace != nullptr) {
    if (HasAccumulator()) {
      // Accumulators reference py_vspace, so we can't swap it out while one is
      // active. This is unlikely to ever happen.
      MaybeRaiseExceptionFromStatus(
          tensorflow::errors::Internal(
              "Can't change the vspace implementation while a "
              "forward accumulator is active."),
          nullptr);
    }
    delete py_vspace;
  }

  py_vspace = new PyVSpace(e);
  auto status = py_vspace->Initialize();
  if (MaybeRaiseExceptionFromStatus(status, nullptr)) {
    delete py_vspace;
    return nullptr;
  }

  Py_RETURN_NONE;
}

PyObject* PyTapeTensor::GetShape() const {
  if (shape_.index() == 0) {
    auto& shape = absl::get<0>(shape_);
    PyObject* py_shape = PyTuple_New(shape.dims());
    for (int i = 0; i < shape.dims(); ++i) {
      PyTuple_SET_ITEM(py_shape, i, PyLong_FromLong(shape.dim_size(i)));
    }

    return py_shape;
  }

  return py_vspace->GraphShape(absl::get<1>(shape_));
}

PyObject* PyTapeTensor::OnesLike() const {
  if (shape_.index() == 1) {
    PyObject* tensor = absl::get<1>(shape_);
    return py_vspace->OnesLike(tensor);
  }
  PyObject* py_shape = GetShape();
  PyObject* dtype_field = GetPyDType();
  PyObject* result = py_vspace->Ones(py_shape, dtype_field);
  Py_DECREF(dtype_field);
  Py_DECREF(py_shape);
  return result;
}

PyObject* PyTapeTensor::ZerosLike() const {
  if (GetDType() == tensorflow::DT_RESOURCE) {
    // Gradient functions for ops which return resource tensors accept
    // None. This is the behavior of py_vspace->Zeros, but checking here avoids
    // issues with ZerosLike.
    Py_RETURN_NONE;
  }
  if (shape_.index() == 1) {
    PyObject* tensor = absl::get<1>(shape_);
    return py_vspace->ZerosLike(tensor);
  }
  PyObject* py_shape = GetShape();
  PyObject* dtype_field = GetPyDType();
  PyObject* result = py_vspace->Zeros(py_shape, dtype_field);
  Py_DECREF(dtype_field);
  Py_DECREF(py_shape);
  return result;
}

// Keeps track of all variables that have been accessed during execution.
class VariableWatcher {
 public:
  VariableWatcher() {}

  ~VariableWatcher() {
    for (const IdAndVariable& v : watched_variables_) {
      Py_DECREF(v.variable);
    }
  }

  int64_t WatchVariable(PyObject* v) {
    tensorflow::Safe_PyObjectPtr handle(PyObject_GetAttrString(v, "handle"));
    if (handle == nullptr) {
      return -1;
    }
    int64_t id = FastTensorId(handle.get());

    tensorflow::mutex_lock l(watched_variables_mu_);
    auto insert_result = watched_variables_.emplace(id, v);

    if (insert_result.second) {
      // Only increment the reference count if we aren't already watching this
      // variable.
      Py_INCREF(v);
    }

    return id;
  }

  PyObject* GetVariablesAsPyTuple() {
    tensorflow::mutex_lock l(watched_variables_mu_);
    PyObject* result = PyTuple_New(watched_variables_.size());
    Py_ssize_t pos = 0;
    for (const IdAndVariable& id_and_variable : watched_variables_) {
      PyTuple_SET_ITEM(result, pos++, id_and_variable.variable);
      Py_INCREF(id_and_variable.variable);
    }
    return result;
  }

 private:
  // We store an IdAndVariable in the map since the map needs to be locked
  // during insert, but should not call back into python during insert to avoid
  // deadlocking with the GIL.
  struct IdAndVariable {
    int64_t id;
    PyObject* variable;

    IdAndVariable(int64_t id, PyObject* variable)
        : id(id), variable(variable) {}
  };
  struct CompareById {
    bool operator()(const IdAndVariable& lhs, const IdAndVariable& rhs) const {
      return lhs.id < rhs.id;
    }
  };

  tensorflow::mutex watched_variables_mu_;
  std::set<IdAndVariable, CompareById> watched_variables_
      TF_GUARDED_BY(watched_variables_mu_);
};

class GradientTape
    : public tensorflow::eager::GradientTape<PyObject, PyBackwardFunction,
                                             PyTapeTensor> {
 public:
  explicit GradientTape(bool persistent, bool watch_accessed_variables)
      : tensorflow::eager::GradientTape<PyObject, PyBackwardFunction,
                                        PyTapeTensor>(persistent),
        watch_accessed_variables_(watch_accessed_variables) {}

  virtual ~GradientTape() {}

  void VariableAccessed(PyObject* v) {
    if (watch_accessed_variables_) {
      WatchVariable(v);
    }
  }

  void WatchVariable(PyObject* v) {
    int64_t id = variable_watcher_.WatchVariable(v);

    if (!PyErr_Occurred()) {
      this->Watch(id);
    }
  }

  PyObject* GetVariablesAsPyTuple() {
    return variable_watcher_.GetVariablesAsPyTuple();
  }

 private:
  bool watch_accessed_variables_;
  VariableWatcher variable_watcher_;
};

typedef tensorflow::eager::ForwardAccumulator<PyObject, PyBackwardFunction,
                                              PyTapeTensor>
    ForwardAccumulator;

// Incremented when a GradientTape or accumulator is newly added to a set, and
// used to enforce an ordering between them.
std::atomic_uint_fast64_t tape_nesting_id_counter(0);

typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      GradientTape* tape;
  // A nesting order between GradientTapes and ForwardAccumulators, used to
  // ensure that GradientTapes do not watch the products of outer
  // ForwardAccumulators.
  int64_t nesting_id;
} TFE_Py_Tape;

static void TFE_Py_Tape_Delete(PyObject* tape) {
  delete reinterpret_cast<TFE_Py_Tape*>(tape)->tape;
  Py_TYPE(tape)->tp_free(tape);
}

static PyTypeObject TFE_Py_Tape_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "tfe.Tape", /* tp_name */
    sizeof(TFE_Py_Tape),                          /* tp_basicsize */
    0,                                            /* tp_itemsize */
    &TFE_Py_Tape_Delete,                          /* tp_dealloc */
#if PY_VERSION_HEX < 0x03080000
    nullptr, /* tp_print */
#else
    0, /* tp_vectorcall_offset */
#endif
    nullptr,               /* tp_getattr */
    nullptr,               /* tp_setattr */
    nullptr,               /* tp_reserved */
    nullptr,               /* tp_repr */
    nullptr,               /* tp_as_number */
    nullptr,               /* tp_as_sequence */
    nullptr,               /* tp_as_mapping */
    nullptr,               /* tp_hash  */
    nullptr,               /* tp_call */
    nullptr,               /* tp_str */
    nullptr,               /* tp_getattro */
    nullptr,               /* tp_setattro */
    nullptr,               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,    /* tp_flags */
    "TFE_Py_Tape objects", /* tp_doc */
};

typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      ForwardAccumulator* accumulator;
  // A nesting order between GradientTapes and ForwardAccumulators, used to
  // ensure that GradientTapes do not watch the products of outer
  // ForwardAccumulators.
  int64_t nesting_id;
} TFE_Py_ForwardAccumulator;

static void TFE_Py_ForwardAccumulatorDelete(PyObject* accumulator) {
  delete reinterpret_cast<TFE_Py_ForwardAccumulator*>(accumulator)->accumulator;
  Py_TYPE(accumulator)->tp_free(accumulator);
}

static PyTypeObject TFE_Py_ForwardAccumulator_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "ForwardAccumulator", /* tp_name */
    sizeof(TFE_Py_ForwardAccumulator),                      /* tp_basicsize */
    0,                                                      /* tp_itemsize */
    &TFE_Py_ForwardAccumulatorDelete,                       /* tp_dealloc */
#if PY_VERSION_HEX < 0x03080000
    nullptr, /* tp_print */
#else
    0, /* tp_vectorcall_offset */
#endif
    nullptr,                             /* tp_getattr */
    nullptr,                             /* tp_setattr */
    nullptr,                             /* tp_reserved */
    nullptr,                             /* tp_repr */
    nullptr,                             /* tp_as_number */
    nullptr,                             /* tp_as_sequence */
    nullptr,                             /* tp_as_mapping */
    nullptr,                             /* tp_hash  */
    nullptr,                             /* tp_call */
    nullptr,                             /* tp_str */
    nullptr,                             /* tp_getattro */
    nullptr,                             /* tp_setattro */
    nullptr,                             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                  /* tp_flags */
    "TFE_Py_ForwardAccumulator objects", /* tp_doc */
};

typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      VariableWatcher* variable_watcher;
} TFE_Py_VariableWatcher;

static void TFE_Py_VariableWatcher_Delete(PyObject* variable_watcher) {
  delete reinterpret_cast<TFE_Py_VariableWatcher*>(variable_watcher)
      ->variable_watcher;
  Py_TYPE(variable_watcher)->tp_free(variable_watcher);
}

static PyTypeObject TFE_Py_VariableWatcher_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "tfe.VariableWatcher", /* tp_name */
    sizeof(TFE_Py_VariableWatcher),                          /* tp_basicsize */
    0,                                                       /* tp_itemsize */
    &TFE_Py_VariableWatcher_Delete,                          /* tp_dealloc */
#if PY_VERSION_HEX < 0x03080000
    nullptr, /* tp_print */
#else
    0, /* tp_vectorcall_offset */
#endif
    nullptr,                          /* tp_getattr */
    nullptr,                          /* tp_setattr */
    nullptr,                          /* tp_reserved */
    nullptr,                          /* tp_repr */
    nullptr,                          /* tp_as_number */
    nullptr,                          /* tp_as_sequence */
    nullptr,                          /* tp_as_mapping */
    nullptr,                          /* tp_hash  */
    nullptr,                          /* tp_call */
    nullptr,                          /* tp_str */
    nullptr,                          /* tp_getattro */
    nullptr,                          /* tp_setattro */
    nullptr,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    "TFE_Py_VariableWatcher objects", /* tp_doc */
};

// Note: in the current design no mutex is needed here because of the python
// GIL, which is always held when any TFE_Py_* methods are called. We should
// revisit this if/when decide to not hold the GIL while manipulating the tape
// stack.
tensorflow::gtl::CompactPointerSet<TFE_Py_Tape*>* GetTapeSet() {
  thread_local std::unique_ptr<tensorflow::gtl::CompactPointerSet<TFE_Py_Tape*>>
      tape_set;
  thread_local ThreadLocalDestructionMarker marker;
  if (!marker.alive) {
    // This thread is being destroyed. It is unsafe to access tape_set.
    return nullptr;
  }
  if (tape_set == nullptr) {
    tape_set.reset(new tensorflow::gtl::CompactPointerSet<TFE_Py_Tape*>);
  }
  return tape_set.get();
}

tensorflow::gtl::CompactPointerSet<TFE_Py_VariableWatcher*>*
GetVariableWatcherSet() {
  thread_local std::unique_ptr<
      tensorflow::gtl::CompactPointerSet<TFE_Py_VariableWatcher*>>
      variable_watcher_set;
  thread_local ThreadLocalDestructionMarker marker;
  if (!marker.alive) {
    // This thread is being destroyed. It is unsafe to access
    // variable_watcher_set.
    return nullptr;
  }
  if (variable_watcher_set == nullptr) {
    variable_watcher_set.reset(
        new tensorflow::gtl::CompactPointerSet<TFE_Py_VariableWatcher*>);
  }
  return variable_watcher_set.get();
}

// A linked hash set, where iteration is in insertion order.
//
// Nested accumulators rely on op recording happening in insertion order, so an
// unordered data structure like CompactPointerSet is not suitable. Outer
// accumulators need to observe operations first so they know to watch the inner
// accumulator's jvp computation.
//
// Not thread safe.
class AccumulatorSet {
 public:
  // Returns true if `element` was newly inserted, false if it already exists.
  bool insert(TFE_Py_ForwardAccumulator* element) {
    if (map_.find(element) != map_.end()) {
      return false;
    }
    ListType::iterator it = ordered_.insert(ordered_.end(), element);
    map_.insert(std::make_pair(element, it));
    return true;
  }

  void erase(TFE_Py_ForwardAccumulator* element) {
    MapType::iterator existing = map_.find(element);
    if (existing == map_.end()) {
      return;
    }
    ListType::iterator list_position = existing->second;
    map_.erase(existing);
    ordered_.erase(list_position);
  }

  bool empty() const { return ordered_.empty(); }

  size_t size() const { return ordered_.size(); }

 private:
  typedef std::list<TFE_Py_ForwardAccumulator*> ListType;
  typedef tensorflow::gtl::FlatMap<TFE_Py_ForwardAccumulator*,
                                   ListType::iterator>
      MapType;

 public:
  typedef ListType::const_iterator const_iterator;
  typedef ListType::const_reverse_iterator const_reverse_iterator;

  const_iterator begin() const { return ordered_.begin(); }
  const_iterator end() const { return ordered_.end(); }

  const_reverse_iterator rbegin() const { return ordered_.rbegin(); }
  const_reverse_iterator rend() const { return ordered_.rend(); }

 private:
  MapType map_;
  ListType ordered_;
};

AccumulatorSet* GetAccumulatorSet() {
  thread_local std::unique_ptr<AccumulatorSet> accumulator_set;
  thread_local ThreadLocalDestructionMarker marker;
  if (!marker.alive) {
    // This thread is being destroyed. It is unsafe to access accumulator_set.
    return nullptr;
  }
  if (accumulator_set == nullptr) {
    accumulator_set.reset(new AccumulatorSet);
  }
  return accumulator_set.get();
}

inline bool HasAccumulator() { return !GetAccumulatorSet()->empty(); }

inline bool HasGradientTape() { return !GetTapeSet()->empty(); }

inline bool HasAccumulatorOrTape() {
  return HasGradientTape() || HasAccumulator();
}

// A safe copy of a set, used for tapes and accumulators. The copy is not
// affected by other python threads changing the set of active tapes.
template <typename ContainerType>
class SafeSetCopy {
 public:
  explicit SafeSetCopy(const ContainerType& to_copy) : set_copy_(to_copy) {
    for (auto* member : set_copy_) {
      Py_INCREF(member);
    }
  }

  ~SafeSetCopy() {
    for (auto* member : set_copy_) {
      Py_DECREF(member);
    }
  }

  typename ContainerType::const_iterator begin() const {
    return set_copy_.begin();
  }

  typename ContainerType::const_iterator end() const { return set_copy_.end(); }

  bool empty() const { return set_copy_.empty(); }
  size_t size() const { return set_copy_.size(); }

 protected:
  ContainerType set_copy_;
};

class SafeTapeSet
    : public SafeSetCopy<tensorflow::gtl::CompactPointerSet<TFE_Py_Tape*>> {
 public:
  SafeTapeSet()
      : SafeSetCopy<tensorflow::gtl::CompactPointerSet<TFE_Py_Tape*>>(
            *GetTapeSet()) {}
};

class SafeAccumulatorSet : public SafeSetCopy<AccumulatorSet> {
 public:
  SafeAccumulatorSet() : SafeSetCopy<AccumulatorSet>(*GetAccumulatorSet()) {}

  typename AccumulatorSet::const_reverse_iterator rbegin() const {
    return set_copy_.rbegin();
  }

  typename AccumulatorSet::const_reverse_iterator rend() const {
    return set_copy_.rend();
  }
};

class SafeVariableWatcherSet
    : public SafeSetCopy<
          tensorflow::gtl::CompactPointerSet<TFE_Py_VariableWatcher*>> {
 public:
  SafeVariableWatcherSet()
      : SafeSetCopy<
            tensorflow::gtl::CompactPointerSet<TFE_Py_VariableWatcher*>>(
            *GetVariableWatcherSet()) {}
};

bool* ThreadTapeIsStopped() {
  thread_local bool thread_tape_is_stopped{false};
  return &thread_tape_is_stopped;
}

void TFE_Py_TapeSetStopOnThread() { *ThreadTapeIsStopped() = true; }

void TFE_Py_TapeSetRestartOnThread() { *ThreadTapeIsStopped() = false; }

PyObject* TFE_Py_TapeSetIsStopped() {
  if (*ThreadTapeIsStopped()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* TFE_Py_TapeSetNew(PyObject* persistent,
                            PyObject* watch_accessed_variables) {
  TFE_Py_Tape_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&TFE_Py_Tape_Type) < 0) return nullptr;
  TFE_Py_Tape* tape = PyObject_NEW(TFE_Py_Tape, &TFE_Py_Tape_Type);
  tape->tape = new GradientTape(persistent == Py_True,
                                watch_accessed_variables == Py_True);
  Py_INCREF(tape);
  tape->nesting_id = tape_nesting_id_counter.fetch_add(1);
  GetTapeSet()->insert(tape);
  return reinterpret_cast<PyObject*>(tape);
}

void TFE_Py_TapeSetAdd(PyObject* tape) {
  Py_INCREF(tape);
  TFE_Py_Tape* tfe_tape = reinterpret_cast<TFE_Py_Tape*>(tape);
  if (!GetTapeSet()->insert(tfe_tape).second) {
    // Already exists in the tape set.
    Py_DECREF(tape);
  } else {
    tfe_tape->nesting_id = tape_nesting_id_counter.fetch_add(1);
  }
}

PyObject* TFE_Py_TapeSetIsEmpty() {
  if (*ThreadTapeIsStopped() || !HasAccumulatorOrTape()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

void TFE_Py_TapeSetRemove(PyObject* tape) {
  auto* stack = GetTapeSet();
  if (stack != nullptr) {
    stack->erase(reinterpret_cast<TFE_Py_Tape*>(tape));
  }
  // We kept a reference to the tape in the set to ensure it wouldn't get
  // deleted under us; cleaning it up here.
  Py_DECREF(tape);
}

static std::vector<int64_t> MakeIntList(PyObject* list) {
  if (list == Py_None) {
    return {};
  }
  PyObject* seq = PySequence_Fast(list, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Size(list);
  PyObject** seq_array = PySequence_Fast_ITEMS(seq);
  std::vector<int64_t> tensor_ids;
  tensor_ids.reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* item = seq_array[i];
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check(item)) {
#else
    if (PyLong_Check(item) || PyInt_Check(item)) {
#endif
      int64_t id = MakeInt(item);
      tensor_ids.push_back(id);
    } else {
      tensor_ids.push_back(-1);
    }
  }
  Py_DECREF(seq);
  return tensor_ids;
}

// Fill `tensor_ids` and `dtypes` from `tensors`, none of which may be
// null. Returns true on success and false on a Python exception.
bool TensorShapesAndDtypes(PyObject* tensors, std::vector<int64_t>* tensor_ids,
                           std::vector<tensorflow::DataType>* dtypes) {
  tensorflow::Safe_PyObjectPtr seq(
      PySequence_Fast(tensors, "expected a sequence"));
  if (seq == nullptr) {
    return false;
  }
  int len = PySequence_Fast_GET_SIZE(seq.get());
  PyObject** seq_array = PySequence_Fast_ITEMS(seq.get());
  tensor_ids->reserve(len);
  dtypes->reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* item = seq_array[i];
    tensor_ids->push_back(FastTensorId(item));
    dtypes->push_back(tensorflow::PyTensor_DataType(item));
  }
  return true;
}

bool TapeCouldPossiblyRecord(PyObject* tensors) {
  if (tensors == Py_None) {
    return false;
  }
  if (*ThreadTapeIsStopped()) {
    return false;
  }
  if (!HasAccumulatorOrTape()) {
    return false;
  }
  return true;
}

bool CouldBackprop() { return !*ThreadTapeIsStopped() && HasGradientTape(); }

bool CouldForwardprop() { return !*ThreadTapeIsStopped() && HasAccumulator(); }

PyObject* TFE_Py_TapeSetShouldRecordBackprop(PyObject* tensors) {
  if (!TapeCouldPossiblyRecord(tensors) || !CouldBackprop()) {
    Py_RETURN_FALSE;
  }
  // TODO(apassos) consider not building a list and changing the API to check
  // each tensor individually.
  std::vector<int64_t> tensor_ids;
  std::vector<tensorflow::DataType> dtypes;
  if (!TensorShapesAndDtypes(tensors, &tensor_ids, &dtypes)) {
    return nullptr;
  }
  auto& tape_set = *GetTapeSet();
  for (TFE_Py_Tape* tape : tape_set) {
    if (tape->tape->ShouldRecord(tensor_ids, dtypes)) {
      Py_RETURN_TRUE;
    }
  }

  Py_RETURN_FALSE;
}

PyObject* TFE_Py_ForwardAccumulatorPushState() {
  auto& forward_accumulators = *GetAccumulatorSet();
  for (TFE_Py_ForwardAccumulator* accumulator : forward_accumulators) {
    accumulator->accumulator->PushState();
  }
  Py_RETURN_NONE;
}

PyObject* TFE_Py_ForwardAccumulatorPopState() {
  auto& forward_accumulators = *GetAccumulatorSet();
  for (TFE_Py_ForwardAccumulator* accumulator : forward_accumulators) {
    accumulator->accumulator->PopState();
  }
  Py_RETURN_NONE;
}

PyObject* TFE_Py_TapeSetPossibleGradientTypes(PyObject* tensors) {
  if (!TapeCouldPossiblyRecord(tensors)) {
    return GetPythonObjectFromInt(0);
  }
  std::vector<int64_t> tensor_ids;
  std::vector<tensorflow::DataType> dtypes;
  if (!TensorShapesAndDtypes(tensors, &tensor_ids, &dtypes)) {
    return nullptr;
  }

  // If there is a persistent tape watching, or if there are multiple tapes
  // watching, we'll return immediately indicating that higher-order tape
  // gradients are possible.
  bool some_tape_watching = false;
  if (CouldBackprop()) {
    auto& tape_set = *GetTapeSet();
    for (TFE_Py_Tape* tape : tape_set) {
      if (tape->tape->ShouldRecord(tensor_ids, dtypes)) {
        if (tape->tape->IsPersistent() || some_tape_watching) {
          // Either this is the second tape watching, or this tape is
          // persistent: higher-order gradients are possible.
          return GetPythonObjectFromInt(2);
        }
        some_tape_watching = true;
      }
    }
  }
  if (CouldForwardprop()) {
    auto& forward_accumulators = *GetAccumulatorSet();
    for (TFE_Py_ForwardAccumulator* accumulator : forward_accumulators) {
      if (accumulator->accumulator->ShouldRecord(tensor_ids, dtypes)) {
        if (some_tape_watching) {
          // This is the second tape watching: higher-order gradients are
          // possible. Note that there's no equivalent of persistence for
          // forward-mode.
          return GetPythonObjectFromInt(2);
        }
        some_tape_watching = true;
      }
    }
  }
  if (some_tape_watching) {
    // There's exactly one non-persistent tape. The user can request first-order
    // gradients but won't be able to get higher-order tape gradients.
    return GetPythonObjectFromInt(1);
  } else {
    // There are no tapes. The user can't request tape gradients.
    return GetPythonObjectFromInt(0);
  }
}

void TFE_Py_TapeWatch(PyObject* tape, PyObject* tensor) {
  if (!CouldBackprop()) {
    return;
  }
  int64_t tensor_id = FastTensorId(tensor);
  if (PyErr_Occurred()) {
    return;
  }
  reinterpret_cast<TFE_Py_Tape*>(tape)->tape->Watch(tensor_id);
}

bool ListContainsNone(PyObject* list) {
  if (list == Py_None) return true;
  tensorflow::Safe_PyObjectPtr seq(
      PySequence_Fast(list, "expected a sequence"));
  if (seq == nullptr) {
    return false;
  }

  int len = PySequence_Size(list);
  PyObject** seq_array = PySequence_Fast_ITEMS(seq.get());
  for (int i = 0; i < len; ++i) {
    PyObject* item = seq_array[i];
    if (item == Py_None) return true;
  }

  return false;
}

// As an optimization, the tape generally keeps only the shape and dtype of
// tensors, and uses this information to generate ones/zeros tensors. However,
// some tensors require OnesLike/ZerosLike because their gradients do not match
// their inference shape/dtype.
bool DTypeNeedsHandleData(tensorflow::DataType dtype) {
  return dtype == tensorflow::DT_VARIANT || dtype == tensorflow::DT_RESOURCE;
}

static PyTapeTensor TapeTensorFromTensor(PyObject* tensor) {
  if (EagerTensor_CheckExact(tensor)) {
    tensorflow::ImmediateExecutionTensorHandle* handle =
        tensorflow::unwrap(EagerTensor_Handle(tensor));
    int64_t id = PyEagerTensor_ID(tensor);
    tensorflow::DataType dtype =
        static_cast<tensorflow::DataType>(handle->DataType());
    if (DTypeNeedsHandleData(dtype)) {
      return PyTapeTensor(id, dtype, tensor);
    }

    tensorflow::TensorShape tensor_shape;
    int num_dims;
    tensorflow::Status status = handle->NumDims(&num_dims);
    if (status.ok()) {
      for (int i = 0; i < num_dims; ++i) {
        int64_t dim_size;
        status = handle->Dim(i, &dim_size);
        if (!status.ok()) break;
        tensor_shape.AddDim(dim_size);
      }
    }

    if (MaybeRaiseExceptionFromStatus(status, nullptr)) {
      return PyTapeTensor(id, static_cast<tensorflow::DataType>(0),
                          tensorflow::TensorShape({}));
    } else {
      return PyTapeTensor(id, dtype, tensor_shape);
    }
  }
  int64_t id = FastTensorId(tensor);
  if (PyErr_Occurred()) {
    return PyTapeTensor(id, static_cast<tensorflow::DataType>(0),
                        tensorflow::TensorShape({}));
  }
  PyObject* dtype_object = PyObject_GetAttrString(tensor, "dtype");
  PyObject* dtype_enum = PyObject_GetAttrString(dtype_object, "_type_enum");
  Py_DECREF(dtype_object);
  tensorflow::DataType dtype =
      static_cast<tensorflow::DataType>(MakeInt(dtype_enum));
  Py_DECREF(dtype_enum);
  if (PyErr_Occurred()) {
    return PyTapeTensor(id, static_cast<tensorflow::DataType>(0),
                        tensorflow::TensorShape({}));
  }
  static char _shape_tuple[] = "_shape_tuple";
  tensorflow::Safe_PyObjectPtr shape_tuple(
      PyObject_CallMethod(tensor, _shape_tuple, nullptr));
  if (PyErr_Occurred()) {
    return PyTapeTensor(id, static_cast<tensorflow::DataType>(0),
                        tensorflow::TensorShape({}));
  }

  if (ListContainsNone(shape_tuple.get()) || DTypeNeedsHandleData(dtype)) {
    return PyTapeTensor(id, dtype, tensor);
  }

  auto l = MakeIntList(shape_tuple.get());
  // Replace -1, which represents accidental Nones which can occur in graph mode
  // and can cause errors in shape construction with 0s.
  for (auto& c : l) {
    if (c < 0) {
      c = 0;
    }
  }
  tensorflow::TensorShape shape(l);
  return PyTapeTensor(id, dtype, shape);
}

// Populates output_info from output_seq, which must come from PySequence_Fast.
//
// Does not take ownership of output_seq. Returns true on success and false if a
// Python exception has been set.
bool TapeTensorsFromTensorSequence(PyObject* output_seq,
                                   std::vector<PyTapeTensor>* output_info) {
  Py_ssize_t output_len = PySequence_Fast_GET_SIZE(output_seq);
  PyObject** output_seq_array = PySequence_Fast_ITEMS(output_seq);
  output_info->reserve(output_len);
  for (Py_ssize_t i = 0; i < output_len; ++i) {
    output_info->push_back(TapeTensorFromTensor(output_seq_array[i]));
    if (PyErr_Occurred() != nullptr) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> MakeTensorIDList(PyObject* tensors) {
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Fast_GET_SIZE(seq);
  PyObject** seq_array = PySequence_Fast_ITEMS(seq);
  std::vector<int64_t> list;
  list.reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* tensor = seq_array[i];
    list.push_back(FastTensorId(tensor));
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return list;
    }
  }
  Py_DECREF(seq);
  return list;
}

void TFE_Py_TapeVariableAccessed(PyObject* variable) {
  if (!CouldBackprop()) {
    return;
  }
  for (TFE_Py_Tape* tape : SafeTapeSet()) {
    tape->tape->VariableAccessed(variable);
  }
}

void TFE_Py_TapeWatchVariable(PyObject* tape, PyObject* variable) {
  if (!CouldBackprop()) {
    return;
  }
  reinterpret_cast<TFE_Py_Tape*>(tape)->tape->WatchVariable(variable);
}

PyObject* TFE_Py_TapeWatchedVariables(PyObject* tape) {
  return reinterpret_cast<TFE_Py_Tape*>(tape)->tape->GetVariablesAsPyTuple();
}

PyObject* TFE_Py_VariableWatcherNew() {
  TFE_Py_VariableWatcher_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&TFE_Py_VariableWatcher_Type) < 0) return nullptr;
  TFE_Py_VariableWatcher* variable_watcher =
      PyObject_NEW(TFE_Py_VariableWatcher, &TFE_Py_VariableWatcher_Type);
  variable_watcher->variable_watcher = new VariableWatcher();
  Py_INCREF(variable_watcher);
  GetVariableWatcherSet()->insert(variable_watcher);
  return reinterpret_cast<PyObject*>(variable_watcher);
}

void TFE_Py_VariableWatcherRemove(PyObject* variable_watcher) {
  auto* stack = GetVariableWatcherSet();
  stack->erase(reinterpret_cast<TFE_Py_VariableWatcher*>(variable_watcher));
  // We kept a reference to the variable watcher in the set to ensure it
  // wouldn't get deleted under us; cleaning it up here.
  Py_DECREF(variable_watcher);
}

void TFE_Py_VariableWatcherVariableAccessed(PyObject* variable) {
  for (TFE_Py_VariableWatcher* variable_watcher : SafeVariableWatcherSet()) {
    variable_watcher->variable_watcher->WatchVariable(variable);
  }
}

PyObject* TFE_Py_VariableWatcherWatchedVariables(PyObject* variable_watcher) {
  return reinterpret_cast<TFE_Py_VariableWatcher*>(variable_watcher)
      ->variable_watcher->GetVariablesAsPyTuple();
}

namespace {
std::vector<tensorflow::DataType> MakeTensorDtypeList(PyObject* tensors) {
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Fast_GET_SIZE(seq);
  PyObject** seq_array = PySequence_Fast_ITEMS(seq);
  std::vector<tensorflow::DataType> list;
  list.reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* tensor = seq_array[i];
    list.push_back(tensorflow::PyTensor_DataType(tensor));
  }
  Py_DECREF(seq);
  return list;
}

PyObject* ForwardAccumulatorDeleteGradient(PyObject* tensor_id,
                                           PyObject* weak_tensor_ref) {
  auto* accumulator_set = GetAccumulatorSet();
  if (accumulator_set != nullptr) {
    int64_t parsed_tensor_id = MakeInt(tensor_id);
    for (TFE_Py_ForwardAccumulator* accumulator : *accumulator_set) {
      accumulator->accumulator->DeleteGradient(parsed_tensor_id);
    }
  }
  Py_DECREF(weak_tensor_ref);
  Py_DECREF(tensor_id);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef forward_accumulator_delete_gradient_method_def = {
    "ForwardAccumulatorDeleteGradient", ForwardAccumulatorDeleteGradient,
    METH_O, "ForwardAccumulatorDeleteGradient"};

void RegisterForwardAccumulatorCleanup(PyObject* tensor, int64_t tensor_id) {
  tensorflow::Safe_PyObjectPtr callback(
      PyCFunction_New(&forward_accumulator_delete_gradient_method_def,
                      PyLong_FromLong(tensor_id)));
  // We need to keep a reference to the weakref active if we want our callback
  // called. The callback itself now owns the weakref object and the tensor ID
  // object.
  PyWeakref_NewRef(tensor, callback.get());
}

void TapeSetRecordBackprop(
    const string& op_type, const std::vector<PyTapeTensor>& output_info,
    const std::vector<int64_t>& input_ids,
    const std::vector<tensorflow::DataType>& input_dtypes,
    const std::function<PyBackwardFunction*()>& backward_function_getter,
    const std::function<void(PyBackwardFunction*)>& backward_function_killer,
    tensorflow::uint64 max_gradient_tape_id) {
  if (!CouldBackprop()) {
    return;
  }
  for (TFE_Py_Tape* tape : SafeTapeSet()) {
    if (tape->nesting_id < max_gradient_tape_id) {
      tape->tape->RecordOperation(op_type, output_info, input_ids, input_dtypes,
                                  backward_function_getter,
                                  backward_function_killer);
    }
  }
}

bool TapeSetRecordForwardprop(
    const string& op_type, PyObject* output_seq,
    const std::vector<PyTapeTensor>& output_info, PyObject* input_tensors,
    const std::vector<int64_t>& input_ids,
    const std::vector<tensorflow::DataType>& input_dtypes,
    const std::function<PyBackwardFunction*()>& backward_function_getter,
    const std::function<void(PyBackwardFunction*)>& backward_function_killer,
    const tensorflow::eager::ForwardFunction<PyObject>* forward_function,
    PyObject* forwardprop_output_indices,
    tensorflow::uint64* max_gradient_tape_id) {
  *max_gradient_tape_id = std::numeric_limits<tensorflow::uint64>::max();
  if (!CouldForwardprop()) {
    return true;
  }
  auto accumulator_set = SafeAccumulatorSet();
  tensorflow::Safe_PyObjectPtr input_seq(
      PySequence_Fast(input_tensors, "expected a sequence of tensors"));
  if (input_seq == nullptr || PyErr_Occurred()) return false;
  Py_ssize_t input_len = PySequence_Fast_GET_SIZE(input_seq.get());
  PyObject** output_seq_array = PySequence_Fast_ITEMS(output_seq);
  for (int i = 0; i < output_info.size(); ++i) {
    RegisterForwardAccumulatorCleanup(output_seq_array[i],
                                      output_info[i].GetID());
  }
  if (forwardprop_output_indices != nullptr &&
      forwardprop_output_indices != Py_None) {
    tensorflow::Safe_PyObjectPtr indices_fast(PySequence_Fast(
        forwardprop_output_indices, "Expected a sequence of indices"));
    if (indices_fast == nullptr || PyErr_Occurred()) {
      return false;
    }
    if (PySequence_Fast_GET_SIZE(indices_fast.get()) !=
        accumulator_set.size()) {
      MaybeRaiseExceptionFromStatus(
          tensorflow::errors::Internal(
              "Accumulators were added or removed from the active set "
              "between packing and unpacking."),
          nullptr);
    }
    PyObject** indices_fast_array = PySequence_Fast_ITEMS(indices_fast.get());
    Py_ssize_t accumulator_index = 0;
    for (AccumulatorSet::const_reverse_iterator it = accumulator_set.rbegin();
         it != accumulator_set.rend(); ++it, ++accumulator_index) {
      tensorflow::Safe_PyObjectPtr jvp_index_seq(
          PySequence_Fast(indices_fast_array[accumulator_index],
                          "Expected a sequence of jvp indices."));
      if (jvp_index_seq == nullptr || PyErr_Occurred()) {
        return false;
      }
      Py_ssize_t num_jvps = PySequence_Fast_GET_SIZE(jvp_index_seq.get());
      PyObject** jvp_index_seq_array =
          PySequence_Fast_ITEMS(jvp_index_seq.get());
      for (Py_ssize_t jvp_index = 0; jvp_index < num_jvps; ++jvp_index) {
        PyObject* tuple = jvp_index_seq_array[jvp_index];
        int64_t primal_tensor_id =
            output_info[MakeInt(PyTuple_GetItem(tuple, 0))].GetID();
        (*it)->accumulator->Watch(
            primal_tensor_id,
            output_seq_array[MakeInt(PyTuple_GetItem(tuple, 1))]);
      }
    }
  } else {
    std::vector<PyTapeTensor> input_info;
    input_info.reserve(input_len);
    PyObject** input_seq_array = PySequence_Fast_ITEMS(input_seq.get());
    for (Py_ssize_t i = 0; i < input_len; ++i) {
      input_info.push_back(TapeTensorFromTensor(input_seq_array[i]));
    }
    for (TFE_Py_ForwardAccumulator* accumulator : accumulator_set) {
      tensorflow::Status status = accumulator->accumulator->Accumulate(
          op_type, input_info, output_info, input_ids, input_dtypes,
          forward_function, backward_function_getter, backward_function_killer);
      if (PyErr_Occurred()) return false;  // Don't swallow Python exceptions.
      if (MaybeRaiseExceptionFromStatus(status, nullptr)) {
        return false;
      }
      if (accumulator->accumulator->BusyAccumulating()) {
        // Ensure inner accumulators don't see outer accumulators' jvps. This
        // mostly happens on its own, with some potentially surprising
        // exceptions, so the blanket policy is for consistency.
        *max_gradient_tape_id = accumulator->nesting_id;
        break;
      }
    }
  }
  return true;
}

PyObject* TangentsAsPyTuple(const std::vector<PyObject*>& input_tangents) {
  PyObject* py_input_tangents = PyTuple_New(input_tangents.size());
  for (int i = 0; i < input_tangents.size(); ++i) {
    PyObject* element;
    if (input_tangents[i] == nullptr) {
      element = Py_None;
    } else {
      element = input_tangents[i];
    }
    Py_INCREF(element);
    PyTuple_SET_ITEM(py_input_tangents, i, element);
  }
  return py_input_tangents;
}

tensorflow::Status ParseTangentOutputs(
    PyObject* user_output, std::vector<PyObject*>* output_tangents) {
  if (user_output == Py_None) {
    // No connected gradients.
    return tensorflow::Status::OK();
  }
  tensorflow::Safe_PyObjectPtr fast_result(
      PySequence_Fast(user_output, "expected a sequence of forward gradients"));
  if (fast_result == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "forward gradient function did not return a sequence.");
  }
  int len = PySequence_Fast_GET_SIZE(fast_result.get());
  PyObject** fast_result_array = PySequence_Fast_ITEMS(fast_result.get());
  output_tangents->reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* item = fast_result_array[i];
    if (item == Py_None) {
      output_tangents->push_back(nullptr);
    } else {
      Py_INCREF(item);
      output_tangents->push_back(item);
    }
  }
  return tensorflow::Status::OK();
}

// Calls the registered forward_gradient_function, computing `output_tangents`
// from `input_tangents`. `output_tangents` must not be null.
//
// `op_name`, `attrs`, `inputs`, and `results` describe the operation for which
// the forward function is being called.
tensorflow::Status CallJVPFunction(PyObject* op_name, PyObject* attrs,
                                   PyObject* inputs, PyObject* results,
                                   const std::vector<PyObject*>& input_tangents,
                                   std::vector<PyObject*>* output_tangents,
                                   bool use_batch) {
  if (forward_gradient_function == nullptr) {
    return tensorflow::errors::Internal(
        "No forward gradient function registered.");
  }
  tensorflow::Safe_PyObjectPtr py_input_tangents(
      TangentsAsPyTuple(input_tangents));

  // Normalize the input sequence to a tuple so it works with function
  // caching; otherwise it may be an opaque _InputList object.
  tensorflow::Safe_PyObjectPtr input_tuple(PySequence_Tuple(inputs));
  PyObject* to_batch = (use_batch) ? Py_True : Py_False;
  tensorflow::Safe_PyObjectPtr callback_args(
      Py_BuildValue("OOOOOO", op_name, attrs, input_tuple.get(), results,
                    py_input_tangents.get(), to_batch));
  tensorflow::Safe_PyObjectPtr py_result(
      PyObject_CallObject(forward_gradient_function, callback_args.get()));
  if (py_result == nullptr || PyErr_Occurred()) {
    return tensorflow::errors::Internal(
        "forward gradient function threw exceptions");
  }
  return ParseTangentOutputs(py_result.get(), output_tangents);
}

// Like CallJVPFunction, but calls a pre-bound forward function.
// These are passed in from a record_gradient argument.
tensorflow::Status CallOpSpecificJVPFunction(
    PyObject* op_specific_forward_function,
    const std::vector<PyObject*>& input_tangents,
    std::vector<PyObject*>* output_tangents) {
  tensorflow::Safe_PyObjectPtr py_input_tangents(
      TangentsAsPyTuple(input_tangents));

  tensorflow::Safe_PyObjectPtr py_result(PyObject_CallObject(
      op_specific_forward_function, py_input_tangents.get()));
  if (py_result == nullptr || PyErr_Occurred()) {
    return tensorflow::errors::Internal(
        "forward gradient function threw exceptions");
  }
  return ParseTangentOutputs(py_result.get(), output_tangents);
}

bool ParseOpTypeString(PyObject* op_type, string* op_type_string) {
  if (PyBytes_Check(op_type)) {
    *op_type_string = PyBytes_AsString(op_type);
  } else if (PyUnicode_Check(op_type)) {
#if PY_MAJOR_VERSION >= 3
    *op_type_string = PyUnicode_AsUTF8(op_type);
#else
    PyObject* py_str = PyUnicode_AsUTF8String(op_type);
    if (py_str == nullptr) {
      return false;
    }
    *op_type_string = PyBytes_AS_STRING(py_str);
    Py_DECREF(py_str);
#endif
  } else {
    PyErr_SetString(PyExc_RuntimeError, "op_type should be a string.");
    return false;
  }
  return true;
}

bool TapeSetRecordOperation(
    PyObject* op_type, PyObject* input_tensors, PyObject* output_tensors,
    const std::vector<int64_t>& input_ids,
    const std::vector<tensorflow::DataType>& input_dtypes,
    const std::function<PyBackwardFunction*()>& backward_function_getter,
    const std::function<void(PyBackwardFunction*)>& backward_function_killer,
    const tensorflow::eager::ForwardFunction<PyObject>* forward_function) {
  std::vector<PyTapeTensor> output_info;
  tensorflow::Safe_PyObjectPtr output_seq(PySequence_Fast(
      output_tensors, "expected a sequence of integer tensor ids"));
  if (PyErr_Occurred() ||
      !TapeTensorsFromTensorSequence(output_seq.get(), &output_info)) {
    return false;
  }
  string op_type_str;
  if (!ParseOpTypeString(op_type, &op_type_str)) {
    return false;
  }
  tensorflow::uint64 max_gradient_tape_id;
  if (!TapeSetRecordForwardprop(
          op_type_str, output_seq.get(), output_info, input_tensors, input_ids,
          input_dtypes, backward_function_getter, backward_function_killer,
          forward_function, nullptr /* No special-cased jvps. */,
          &max_gradient_tape_id)) {
    return false;
  }
  TapeSetRecordBackprop(op_type_str, output_info, input_ids, input_dtypes,
                        backward_function_getter, backward_function_killer,
                        max_gradient_tape_id);
  return true;
}
}  // namespace

PyObject* TFE_Py_TapeSetRecordOperation(PyObject* op_type,
                                        PyObject* output_tensors,
                                        PyObject* input_tensors,
                                        PyObject* backward_function,
                                        PyObject* forward_function) {
  if (!HasAccumulatorOrTape() || *ThreadTapeIsStopped()) {
    Py_RETURN_NONE;
  }
  std::vector<int64_t> input_ids = MakeTensorIDList(input_tensors);
  if (PyErr_Occurred()) return nullptr;

  std::vector<tensorflow::DataType> input_dtypes =
      MakeTensorDtypeList(input_tensors);
  if (PyErr_Occurred()) return nullptr;

  std::function<PyBackwardFunction*()> backward_function_getter(
      [backward_function]() {
        Py_INCREF(backward_function);
        PyBackwardFunction* function = new PyBackwardFunction(
            [backward_function](PyObject* out_grads,
                                const std::vector<int64_t>& unused) {
              return PyObject_CallObject(backward_function, out_grads);
            });
        return function;
      });
  std::function<void(PyBackwardFunction*)> backward_function_killer(
      [backward_function](PyBackwardFunction* py_backward_function) {
        Py_DECREF(backward_function);
        delete py_backward_function;
      });

  if (forward_function == Py_None) {
    if (!TapeSetRecordOperation(
            op_type, input_tensors, output_tensors, input_ids, input_dtypes,
            backward_function_getter, backward_function_killer,
            nullptr /* No special-cased forward function */)) {
      return nullptr;
    }
  } else {
    tensorflow::eager::ForwardFunction<PyObject> wrapped_forward_function(
        [forward_function](const std::vector<PyObject*>& input_tangents,
                           std::vector<PyObject*>* output_tangents,
                           bool use_batch = false) {
          return CallOpSpecificJVPFunction(forward_function, input_tangents,
                                           output_tangents);
        });
    if (!TapeSetRecordOperation(
            op_type, input_tensors, output_tensors, input_ids, input_dtypes,
            backward_function_getter, backward_function_killer,
            &wrapped_forward_function)) {
      return nullptr;
    }
  }
  Py_RETURN_NONE;
}

PyObject* TFE_Py_TapeSetRecordOperationForwardprop(
    PyObject* op_type, PyObject* output_tensors, PyObject* input_tensors,
    PyObject* backward_function, PyObject* forwardprop_output_indices) {
  if (!HasAccumulator() || *ThreadTapeIsStopped()) {
    Py_RETURN_NONE;
  }
  std::vector<int64_t> input_ids = MakeTensorIDList(input_tensors);
  if (PyErr_Occurred()) return nullptr;

  std::vector<tensorflow::DataType> input_dtypes =
      MakeTensorDtypeList(input_tensors);
  if (PyErr_Occurred()) return nullptr;

  std::function<PyBackwardFunction*()> backward_function_getter(
      [backward_function]() {
        Py_INCREF(backward_function);
        PyBackwardFunction* function = new PyBackwardFunction(
            [backward_function](PyObject* out_grads,
                                const std::vector<int64_t>& unused) {
              return PyObject_CallObject(backward_function, out_grads);
            });
        return function;
      });
  std::function<void(PyBackwardFunction*)> backward_function_killer(
      [backward_function](PyBackwardFunction* py_backward_function) {
        Py_DECREF(backward_function);
        delete py_backward_function;
      });
  std::vector<PyTapeTensor> output_info;
  tensorflow::Safe_PyObjectPtr output_seq(PySequence_Fast(
      output_tensors, "expected a sequence of integer tensor ids"));
  if (PyErr_Occurred() ||
      !TapeTensorsFromTensorSequence(output_seq.get(), &output_info)) {
    return nullptr;
  }
  string op_type_str;
  if (!ParseOpTypeString(op_type, &op_type_str)) {
    return nullptr;
  }
  tensorflow::uint64 max_gradient_tape_id;
  if (!TapeSetRecordForwardprop(
          op_type_str, output_seq.get(), output_info, input_tensors, input_ids,
          input_dtypes, backward_function_getter, backward_function_killer,
          nullptr /* no special-cased forward function */,
          forwardprop_output_indices, &max_gradient_tape_id)) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* TFE_Py_TapeSetRecordOperationBackprop(PyObject* op_type,
                                                PyObject* output_tensors,
                                                PyObject* input_tensors,
                                                PyObject* backward_function) {
  if (!CouldBackprop()) {
    Py_RETURN_NONE;
  }
  std::vector<int64_t> input_ids = MakeTensorIDList(input_tensors);
  if (PyErr_Occurred()) return nullptr;

  std::vector<tensorflow::DataType> input_dtypes =
      MakeTensorDtypeList(input_tensors);
  if (PyErr_Occurred()) return nullptr;

  std::function<PyBackwardFunction*()> backward_function_getter(
      [backward_function]() {
        Py_INCREF(backward_function);
        PyBackwardFunction* function = new PyBackwardFunction(
            [backward_function](PyObject* out_grads,
                                const std::vector<int64_t>& unused) {
              return PyObject_CallObject(backward_function, out_grads);
            });
        return function;
      });
  std::function<void(PyBackwardFunction*)> backward_function_killer(
      [backward_function](PyBackwardFunction* py_backward_function) {
        Py_DECREF(backward_function);
        delete py_backward_function;
      });
  std::vector<PyTapeTensor> output_info;
  tensorflow::Safe_PyObjectPtr output_seq(PySequence_Fast(
      output_tensors, "expected a sequence of integer tensor ids"));
  if (PyErr_Occurred() ||
      !TapeTensorsFromTensorSequence(output_seq.get(), &output_info)) {
    return nullptr;
  }
  string op_type_str;
  if (!ParseOpTypeString(op_type, &op_type_str)) {
    return nullptr;
  }
  TapeSetRecordBackprop(op_type_str, output_info, input_ids, input_dtypes,
                        backward_function_getter, backward_function_killer,
                        // No filtering based on relative ordering with forward
                        // accumulators.
                        std::numeric_limits<tensorflow::uint64>::max());
  Py_RETURN_NONE;
}

void TFE_Py_TapeSetDeleteTrace(int64_t tensor_id) {
  auto* tape_set = GetTapeSet();
  if (tape_set == nullptr) {
    // Current thread is being destructed, and the tape set has already
    // been cleared.
    return;
  }
  for (TFE_Py_Tape* tape : *tape_set) {
    tape->tape->DeleteTrace(tensor_id);
  }
}

std::vector<PyObject*> MakeTensorList(PyObject* tensors) {
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Fast_GET_SIZE(seq);
  PyObject** seq_array = PySequence_Fast_ITEMS(seq);
  std::vector<PyObject*> list(seq_array, seq_array + len);
  Py_DECREF(seq);
  return list;
}

PyObject* TFE_Py_TapeGradient(PyObject* tape, PyObject* target,
                              PyObject* sources, PyObject* output_gradients,
                              PyObject* sources_raw,
                              PyObject* unconnected_gradients,
                              TF_Status* status) {
  TFE_Py_Tape* tape_obj = reinterpret_cast<TFE_Py_Tape*>(tape);
  if (!tape_obj->tape->IsPersistent()) {
    auto* tape_set = GetTapeSet();
    if (tape_set->find(tape_obj) != tape_set->end()) {
      PyErr_SetString(PyExc_RuntimeError,
                      "gradient() cannot be invoked within the "
                      "GradientTape context (i.e., while operations are being "
                      "recorded). Either move the call to gradient() to be "
                      "outside the 'with tf.GradientTape' block, or "
                      "use a persistent tape: "
                      "'with tf.GradientTape(persistent=true)'");
      return nullptr;
    }
  }

  std::vector<int64_t> target_vec = MakeTensorIDList(target);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  std::vector<int64_t> sources_vec = MakeTensorIDList(sources);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  tensorflow::gtl::FlatSet<int64_t> sources_set(sources_vec.begin(),
                                                sources_vec.end());

  tensorflow::Safe_PyObjectPtr seq =
      tensorflow::make_safe(PySequence_Fast(target, "expected a sequence"));
  int len = PySequence_Fast_GET_SIZE(seq.get());
  PyObject** seq_array = PySequence_Fast_ITEMS(seq.get());
  std::unordered_map<int64_t, PyTapeTensor> source_tensors_that_are_targets;
  for (int i = 0; i < len; ++i) {
    int64_t target_id = target_vec[i];
    if (sources_set.find(target_id) != sources_set.end()) {
      auto tensor = seq_array[i];
      source_tensors_that_are_targets.insert(
          std::make_pair(target_id, TapeTensorFromTensor(tensor)));
    }
    if (PyErr_Occurred()) {
      return nullptr;
    }
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }

  std::vector<PyObject*> outgrad_vec;
  if (output_gradients != Py_None) {
    outgrad_vec = MakeTensorList(output_gradients);
    if (PyErr_Occurred()) {
      return nullptr;
    }
    for (PyObject* tensor : outgrad_vec) {
      // Calling the backward function will eat a reference to the tensors in
      // outgrad_vec, so we need to increase their reference count.
      Py_INCREF(tensor);
    }
  }
  std::vector<PyObject*> result(sources_vec.size());
  status->status = tape_obj->tape->ComputeGradient(
      *py_vspace, target_vec, sources_vec, source_tensors_that_are_targets,
      outgrad_vec, absl::MakeSpan(result));
  if (!status->status.ok()) {
    if (PyErr_Occurred()) {
      // Do not propagate the erroneous status as that would swallow the
      // exception which caused the problem.
      status->status = tensorflow::Status::OK();
    }
    return nullptr;
  }

  bool unconnected_gradients_zero =
      strcmp(TFE_GetPythonString(unconnected_gradients), "zero") == 0;
  std::vector<PyObject*> sources_obj;
  if (unconnected_gradients_zero) {
    // Uses the "raw" sources here so it can properly make a zeros tensor even
    // if there are resource variables as sources.
    sources_obj = MakeTensorList(sources_raw);
  }

  if (!result.empty()) {
    PyObject* py_result = PyList_New(result.size());
    tensorflow::gtl::FlatSet<PyObject*> seen_results(result.size());
    for (int i = 0; i < result.size(); ++i) {
      if (result[i] == nullptr) {
        if (unconnected_gradients_zero) {
          // generate a zeros tensor in the shape of sources[i]
          tensorflow::DataType dtype =
              tensorflow::PyTensor_DataType(sources_obj[i]);
          PyTapeTensor tensor =
              PyTapeTensor(sources_vec[i], dtype, sources_obj[i]);
          result[i] = tensor.ZerosLike();
        } else {
          Py_INCREF(Py_None);
          result[i] = Py_None;
        }
      } else if (seen_results.find(result[i]) != seen_results.end()) {
        Py_INCREF(result[i]);
      }
      seen_results.insert(result[i]);
      PyList_SET_ITEM(py_result, i, reinterpret_cast<PyObject*>(result[i]));
    }
    return py_result;
  }
  return PyList_New(0);
}

PyObject* TFE_Py_ForwardAccumulatorNew(bool use_batch) {
  TFE_Py_ForwardAccumulator_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&TFE_Py_ForwardAccumulator_Type) < 0) return nullptr;
  TFE_Py_ForwardAccumulator* accumulator =
      PyObject_NEW(TFE_Py_ForwardAccumulator, &TFE_Py_ForwardAccumulator_Type);
  if (py_vspace == nullptr) {
    MaybeRaiseExceptionFromStatus(
        tensorflow::errors::Internal(
            "ForwardAccumulator requires a PyVSpace to be registered."),
        nullptr);
  }
  accumulator->accumulator = new ForwardAccumulator(*py_vspace, use_batch);
  return reinterpret_cast<PyObject*>(accumulator);
}

PyObject* TFE_Py_ForwardAccumulatorSetAdd(PyObject* accumulator) {
  TFE_Py_ForwardAccumulator* c_accumulator(
      reinterpret_cast<TFE_Py_ForwardAccumulator*>(accumulator));
  c_accumulator->nesting_id = tape_nesting_id_counter.fetch_add(1);
  if (GetAccumulatorSet()->insert(c_accumulator)) {
    Py_INCREF(accumulator);
    Py_RETURN_NONE;
  } else {
    MaybeRaiseExceptionFromStatus(
        tensorflow::errors::Internal(
            "A ForwardAccumulator was added to the active set twice."),
        nullptr);
    return nullptr;
  }
}

void TFE_Py_ForwardAccumulatorSetRemove(PyObject* accumulator) {
  auto* accumulator_set = GetAccumulatorSet();
  if (accumulator_set != nullptr) {
    accumulator_set->erase(
        reinterpret_cast<TFE_Py_ForwardAccumulator*>(accumulator));
  }
  Py_DECREF(accumulator);
}

void TFE_Py_ForwardAccumulatorWatch(PyObject* accumulator, PyObject* tensor,
                                    PyObject* tangent) {
  int64_t tensor_id = FastTensorId(tensor);
  reinterpret_cast<TFE_Py_ForwardAccumulator*>(accumulator)
      ->accumulator->Watch(tensor_id, tangent);
  RegisterForwardAccumulatorCleanup(tensor, tensor_id);
}

// Returns a new reference to the JVP Tensor.
PyObject* TFE_Py_ForwardAccumulatorJVP(PyObject* accumulator,
                                       PyObject* tensor) {
  PyObject* jvp = reinterpret_cast<TFE_Py_ForwardAccumulator*>(accumulator)
                      ->accumulator->FetchJVP(FastTensorId(tensor));
  if (jvp == nullptr) {
    jvp = Py_None;
  }
  Py_INCREF(jvp);
  return jvp;
}

PyObject* TFE_Py_PackJVPs(PyObject* tensors) {
  if (!TapeCouldPossiblyRecord(tensors)) {
    tensorflow::Safe_PyObjectPtr empty_tuple(PyTuple_New(0));
    tensorflow::Safe_PyObjectPtr empty_list(PyList_New(0));
    return PyTuple_Pack(2, empty_tuple.get(), empty_list.get());
  }
  auto& accumulators = *GetAccumulatorSet();
  tensorflow::Safe_PyObjectPtr tensors_fast(
      PySequence_Fast(tensors, "Expected a sequence of input Tensors."));
  if (tensors_fast == nullptr || PyErr_Occurred()) {
    return nullptr;
  }
  std::vector<int64_t> augmented_input_ids;
  int len = PySequence_Fast_GET_SIZE(tensors_fast.get());
  PyObject** tensors_fast_array = PySequence_Fast_ITEMS(tensors_fast.get());
  for (Py_ssize_t position = 0; position < len; ++position) {
    PyObject* input = tensors_fast_array[position];
    if (input == Py_None) {
      continue;
    }
    tensorflow::DataType input_dtype(tensorflow::PyTensor_DataType(input));
    if (input_dtype == tensorflow::DT_INVALID) {
      return nullptr;
    }
    augmented_input_ids.push_back(FastTensorId(input));
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }
  // Find the innermost accumulator such that all outer accumulators are
  // recording. Any more deeply nested accumulators will not have their JVPs
  // saved.
  AccumulatorSet::const_iterator innermost_all_recording = accumulators.begin();
  for (; innermost_all_recording != accumulators.end();
       ++innermost_all_recording) {
    if ((*innermost_all_recording)->accumulator->BusyAccumulating()) {
      break;
    }
  }
  AccumulatorSet::const_reverse_iterator reverse_innermost_all_recording(
      innermost_all_recording);

  bool saving_jvps = false;
  tensorflow::Safe_PyObjectPtr all_indices(PyTuple_New(accumulators.size()));
  std::vector<PyObject*> new_tensors;
  Py_ssize_t accumulator_index = 0;
  // Start with the innermost accumulators to give outer accumulators a chance
  // to find their higher-order JVPs.
  for (AccumulatorSet::const_reverse_iterator it = accumulators.rbegin();
       it != accumulators.rend(); ++it, ++accumulator_index) {
    std::vector<int64_t> new_input_ids;
    std::vector<std::pair<int64_t, int64_t>> accumulator_indices;
    if (it == reverse_innermost_all_recording) {
      saving_jvps = true;
    }
    if (saving_jvps) {
      for (int input_index = 0; input_index < augmented_input_ids.size();
           ++input_index) {
        int64_t existing_input = augmented_input_ids[input_index];
        PyObject* jvp = (*it)->accumulator->FetchJVP(existing_input);
        if (jvp != nullptr) {
          new_tensors.push_back(jvp);
          new_input_ids.push_back(FastTensorId(jvp));
          accumulator_indices.emplace_back(
              input_index,
              augmented_input_ids.size() + new_input_ids.size() - 1);
        }
      }
    }
    tensorflow::Safe_PyObjectPtr accumulator_indices_py(
        PyTuple_New(accumulator_indices.size()));
    for (int i = 0; i < accumulator_indices.size(); ++i) {
      tensorflow::Safe_PyObjectPtr from_index(
          GetPythonObjectFromInt(accumulator_indices[i].first));
      tensorflow::Safe_PyObjectPtr to_index(
          GetPythonObjectFromInt(accumulator_indices[i].second));
      PyTuple_SetItem(accumulator_indices_py.get(), i,
                      PyTuple_Pack(2, from_index.get(), to_index.get()));
    }
    PyTuple_SetItem(all_indices.get(), accumulator_index,
                    accumulator_indices_py.release());
    augmented_input_ids.insert(augmented_input_ids.end(), new_input_ids.begin(),
                               new_input_ids.end());
  }

  tensorflow::Safe_PyObjectPtr new_tensors_py(PyList_New(new_tensors.size()));
  for (int i = 0; i < new_tensors.size(); ++i) {
    PyObject* jvp = new_tensors[i];
    Py_INCREF(jvp);
    PyList_SET_ITEM(new_tensors_py.get(), i, jvp);
  }
  return PyTuple_Pack(2, all_indices.get(), new_tensors_py.get());
}

namespace {

// Indices for the "args" tuple that's passed to TFE_Py_FastPathExecute_C.
enum FastPathExecuteArgIndex {
  FAST_PATH_EXECUTE_ARG_CONTEXT = 0,
  FAST_PATH_EXECUTE_ARG_OP_NAME = 1,
  FAST_PATH_EXECUTE_ARG_NAME = 2,
  FAST_PATH_EXECUTE_ARG_INPUT_START = 3
};

PyObject* GetPythonObjectFromString(tensorflow::StringPiece s) {
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromStringAndSize(s.data(), s.size());
#else
  return PyBytes_FromStringAndSize(s.data(), s.size());
#endif
}

bool CheckResourceVariable(PyObject* item) {
  if (tensorflow::swig::IsResourceVariable(item)) {
    tensorflow::Safe_PyObjectPtr handle(
        PyObject_GetAttrString(item, "_handle"));
    return EagerTensor_CheckExact(handle.get());
  }

  return false;
}

bool IsNumberType(PyObject* item) {
#if PY_MAJOR_VERSION >= 3
  return PyFloat_Check(item) || PyLong_Check(item);
#else
  return PyFloat_Check(item) || PyInt_Check(item) || PyLong_Check(item);
#endif
}

bool CheckOneInput(PyObject* item) {
  if (EagerTensor_CheckExact(item) || CheckResourceVariable(item) ||
      PyArray_Check(item) || IsNumberType(item)) {
    return true;
  }

  // Sequences are not properly handled. Sequences with purely python numeric
  // types work, but sequences with mixes of EagerTensors and python numeric
  // types don't work.
  // TODO(nareshmodi): fix
  return false;
}

bool CheckInputsOk(PyObject* seq, int start_index,
                   const tensorflow::OpDef& op_def) {
  for (int i = 0; i < op_def.input_arg_size(); i++) {
    PyObject* item = PyTuple_GET_ITEM(seq, i + start_index);
    if (!op_def.input_arg(i).number_attr().empty() ||
        !op_def.input_arg(i).type_list_attr().empty()) {
      // This item should be a seq input.
      if (!PySequence_Check(item)) {
        VLOG(1) << "Falling back to slow path for Op \"" << op_def.name()
                << "\", Input \"" << op_def.input_arg(i).name()
                << "\" since we expected a sequence, but got "
                << item->ob_type->tp_name;
        return false;
      }
      tensorflow::Safe_PyObjectPtr fast_item(
          PySequence_Fast(item, "Could not parse sequence."));
      if (fast_item.get() == nullptr) {
        return false;
      }
      int len = PySequence_Fast_GET_SIZE(fast_item.get());
      PyObject** fast_item_array = PySequence_Fast_ITEMS(fast_item.get());
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* inner_item = fast_item_array[j];
        if (!CheckOneInput(inner_item)) {
          VLOG(1) << "Falling back to slow path for Op \"" << op_def.name()
                  << "\", Input \"" << op_def.input_arg(i).name()
                  << "\", Index " << j
                  << " since we expected an EagerTensor/ResourceVariable, "
                     "but got "
                  << inner_item->ob_type->tp_name;
          return false;
        }
      }
    } else if (!CheckOneInput(item)) {
      VLOG(1)
          << "Falling back to slow path for Op \"" << op_def.name()
          << "\", Input \"" << op_def.input_arg(i).name()
          << "\" since we expected an EagerTensor/ResourceVariable, but got "
          << item->ob_type->tp_name;
      return false;
    }
  }

  return true;
}

tensorflow::DataType MaybeGetDType(PyObject* item) {
  if (EagerTensor_CheckExact(item) || CheckResourceVariable(item)) {
    return tensorflow::PyTensor_DataType(item);
  }

  return tensorflow::DT_INVALID;
}

tensorflow::DataType MaybeGetDTypeForAttr(const string& attr,
                                          FastPathOpExecInfo* op_exec_info) {
  auto cached_it = op_exec_info->cached_dtypes.find(attr);
  if (cached_it != op_exec_info->cached_dtypes.end()) {
    return cached_it->second;
  }

  auto it = op_exec_info->attr_to_inputs_map->find(attr);
  if (it == op_exec_info->attr_to_inputs_map->end()) {
    // No other inputs - this should never happen.
    return tensorflow::DT_INVALID;
  }

  for (const auto& input_info : it->second) {
    PyObject* item = PyTuple_GET_ITEM(
        op_exec_info->args, FAST_PATH_EXECUTE_ARG_INPUT_START + input_info.i);
    if (input_info.is_list) {
      tensorflow::Safe_PyObjectPtr fast_item(
          PySequence_Fast(item, "Unable to allocate"));
      int len = PySequence_Fast_GET_SIZE(fast_item.get());
      PyObject** fast_item_array = PySequence_Fast_ITEMS(fast_item.get());
      for (int i = 0; i < len; i++) {
        auto dtype = MaybeGetDType(fast_item_array[i]);
        if (dtype != tensorflow::DT_INVALID) return dtype;
      }
    } else {
      auto dtype = MaybeGetDType(item);
      if (dtype != tensorflow::DT_INVALID) return dtype;
    }
  }

  auto default_it = op_exec_info->default_dtypes->find(attr);
  if (default_it != op_exec_info->default_dtypes->end()) {
    return default_it->second;
  }

  return tensorflow::DT_INVALID;
}

PyObject* CopySequenceSettingIndicesToNull(
    PyObject* seq, const tensorflow::gtl::FlatSet<int>& indices) {
  tensorflow::Safe_PyObjectPtr fast_seq(
      PySequence_Fast(seq, "unable to allocate"));
  int len = PySequence_Fast_GET_SIZE(fast_seq.get());
  PyObject** fast_seq_array = PySequence_Fast_ITEMS(fast_seq.get());
  PyObject* result = PyTuple_New(len);
  for (int i = 0; i < len; i++) {
    PyObject* item;
    if (indices.find(i) != indices.end()) {
      item = Py_None;
    } else {
      item = fast_seq_array[i];
    }
    Py_INCREF(item);
    PyTuple_SET_ITEM(result, i, item);
  }
  return result;
}

PyObject* RecordGradient(PyObject* op_name, PyObject* inputs, PyObject* attrs,
                         PyObject* results,
                         PyObject* forward_pass_name_scope = nullptr) {
  std::vector<int64_t> input_ids = MakeTensorIDList(inputs);
  if (PyErr_Occurred()) return nullptr;
  std::vector<tensorflow::DataType> input_dtypes = MakeTensorDtypeList(inputs);
  if (PyErr_Occurred()) return nullptr;

  bool should_record = false;
  for (TFE_Py_Tape* tape : SafeTapeSet()) {
    if (tape->tape->ShouldRecord(input_ids, input_dtypes)) {
      should_record = true;
      break;
    }
  }
  if (!should_record) {
    for (TFE_Py_ForwardAccumulator* accumulator : SafeAccumulatorSet()) {
      if (accumulator->accumulator->ShouldRecord(input_ids, input_dtypes)) {
        should_record = true;
        break;
      }
    }
  }
  if (!should_record) Py_RETURN_NONE;

  string c_op_name = TFE_GetPythonString(op_name);

  PyObject* op_outputs;
  bool op_outputs_tuple_created = false;

  if (const auto unused_output_indices =
          OpGradientUnusedOutputIndices(c_op_name)) {
    if (unused_output_indices->empty()) {
      op_outputs = Py_None;
    } else {
      op_outputs_tuple_created = true;
      op_outputs =
          CopySequenceSettingIndicesToNull(results, *unused_output_indices);
    }
  } else {
    op_outputs = results;
  }

  PyObject* op_inputs;
  bool op_inputs_tuple_created = false;

  if (const auto unused_input_indices =
          OpGradientUnusedInputIndices(c_op_name)) {
    if (unused_input_indices->empty()) {
      op_inputs = Py_None;
    } else {
      op_inputs_tuple_created = true;
      op_inputs =
          CopySequenceSettingIndicesToNull(inputs, *unused_input_indices);
    }
  } else {
    op_inputs = inputs;
  }

  tensorflow::eager::ForwardFunction<PyObject> py_forward_function(
      [op_name, attrs, inputs, results](
          const std::vector<PyObject*>& input_tangents,
          std::vector<PyObject*>* output_tangents, bool use_batch) {
        return CallJVPFunction(op_name, attrs, inputs, results, input_tangents,
                               output_tangents, use_batch);
      });
  tensorflow::eager::ForwardFunction<PyObject>* forward_function;
  if (c_op_name == "While" || c_op_name == "StatelessWhile" ||
      c_op_name == "If" || c_op_name == "StatelessIf") {
    // Control flow contains non-hashable attributes. Handling them in Python is
    // a headache, so instead we'll stay as close to GradientTape's handling as
    // possible (a null forward function means the accumulator forwards to a
    // tape).
    //
    // This is safe to do since we'll only see control flow when graph building,
    // in which case we can rely on pruning.
    forward_function = nullptr;
  } else {
    forward_function = &py_forward_function;
  }

  PyObject* num_inputs = PyLong_FromLong(PySequence_Size(inputs));

  if (!forward_pass_name_scope) forward_pass_name_scope = Py_None;

  TapeSetRecordOperation(
      op_name, inputs, results, input_ids, input_dtypes,
      [op_name, attrs, num_inputs, op_inputs, op_outputs,
       forward_pass_name_scope]() {
        Py_INCREF(op_name);
        Py_INCREF(attrs);
        Py_INCREF(num_inputs);
        Py_INCREF(op_inputs);
        Py_INCREF(op_outputs);
        Py_INCREF(forward_pass_name_scope);
        PyBackwardFunction* function = new PyBackwardFunction(
            [op_name, attrs, num_inputs, op_inputs, op_outputs,
             forward_pass_name_scope](
                PyObject* output_grads,
                const std::vector<int64_t>& unneeded_gradients) {
              if (PyErr_Occurred()) {
                return static_cast<PyObject*>(nullptr);
              }
              tensorflow::Safe_PyObjectPtr skip_input_indices;
              if (!unneeded_gradients.empty()) {
                skip_input_indices.reset(
                    PyTuple_New(unneeded_gradients.size()));
                for (int i = 0; i < unneeded_gradients.size(); i++) {
                  PyTuple_SET_ITEM(
                      skip_input_indices.get(), i,
                      GetPythonObjectFromInt(unneeded_gradients[i]));
                }
              } else {
                Py_INCREF(Py_None);
                skip_input_indices.reset(Py_None);
              }
              tensorflow::Safe_PyObjectPtr callback_args(Py_BuildValue(
                  "OOOOOOOO", op_name, attrs, num_inputs, op_inputs, op_outputs,
                  output_grads, skip_input_indices.get(),
                  forward_pass_name_scope));

              tensorflow::Safe_PyObjectPtr result(
                  PyObject_CallObject(gradient_function, callback_args.get()));

              if (PyErr_Occurred()) return static_cast<PyObject*>(nullptr);

              return tensorflow::swig::Flatten(result.get());
            });
        return function;
      },
      [op_name, attrs, num_inputs, op_inputs, op_outputs,
       forward_pass_name_scope](PyBackwardFunction* backward_function) {
        Py_DECREF(op_name);
        Py_DECREF(attrs);
        Py_DECREF(num_inputs);
        Py_DECREF(op_inputs);
        Py_DECREF(op_outputs);
        Py_DECREF(forward_pass_name_scope);

        delete backward_function;
      },
      forward_function);

  Py_DECREF(num_inputs);
  if (op_outputs_tuple_created) Py_DECREF(op_outputs);
  if (op_inputs_tuple_created) Py_DECREF(op_inputs);

  if (PyErr_Occurred()) {
    return nullptr;
  }

  Py_RETURN_NONE;
}

void MaybeNotifyVariableAccessed(PyObject* input) {
  DCHECK(CheckResourceVariable(input));
  DCHECK(PyObject_HasAttrString(input, "_trainable"));

  tensorflow::Safe_PyObjectPtr trainable(
      PyObject_GetAttrString(input, "_trainable"));
  if (trainable.get() == Py_False) return;
  TFE_Py_TapeVariableAccessed(input);
  TFE_Py_VariableWatcherVariableAccessed(input);
}

bool ReadVariableOp(const FastPathOpExecInfo& parent_op_exec_info,
                    PyObject* input, tensorflow::Safe_PyObjectPtr* output,
                    TF_Status* status) {
  MaybeNotifyVariableAccessed(input);

  TFE_Op* op = TFE_NewOp(parent_op_exec_info.ctx, "ReadVariableOp", status);
  auto cleaner = tensorflow::gtl::MakeCleanup([op] { TFE_DeleteOp(op); });
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr))
    return false;

  TFE_OpSetDevice(op, parent_op_exec_info.device_name, status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr))
    return false;

  // Set dtype
  DCHECK(PyObject_HasAttrString(input, "_dtype"));
  tensorflow::Safe_PyObjectPtr dtype(PyObject_GetAttrString(input, "_dtype"));
  int value;
  if (!ParseTypeValue("_dtype", dtype.get(), status, &value)) {
    return false;
  }
  TFE_OpSetAttrType(op, "dtype", static_cast<TF_DataType>(value));

  // Get handle
  tensorflow::Safe_PyObjectPtr handle(PyObject_GetAttrString(input, "_handle"));
  if (!EagerTensor_CheckExact(handle.get())) return false;
  TFE_OpAddInput(op, EagerTensor_Handle(handle.get()), status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr))
    return false;

  int num_retvals = 1;
  TFE_TensorHandle* output_handle;
  TFE_Execute(op, &output_handle, &num_retvals, status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr))
    return false;

  // Always create the py object (and correctly DECREF it) from the returned
  // value, else the data will leak.
  output->reset(EagerTensorFromHandle(output_handle));

  // TODO(nareshmodi): Should we run post exec callbacks here?
  if (parent_op_exec_info.run_gradient_callback) {
    tensorflow::Safe_PyObjectPtr inputs(PyTuple_New(1));
    PyTuple_SET_ITEM(inputs.get(), 0, handle.release());

    tensorflow::Safe_PyObjectPtr outputs(PyTuple_New(1));
    Py_INCREF(output->get());  // stay alive after since tuple steals.
    PyTuple_SET_ITEM(outputs.get(), 0, output->get());

    tensorflow::Safe_PyObjectPtr op_string(
        GetPythonObjectFromString("ReadVariableOp"));
    if (!RecordGradient(op_string.get(), inputs.get(), Py_None,
                        outputs.get())) {
      return false;
    }
  }

  return true;
}

// Supports 3 cases at the moment:
//  i) input is an EagerTensor.
//  ii) input is a ResourceVariable - in this case, the is_variable param is
//  set to true.
//  iii) input is an arbitrary python list/tuple (note, this handling doesn't
//  support packing).
//
//  NOTE: dtype_hint_getter must *always* return a PyObject that can be
//  decref'd. So if no hint is found, Py_RETURN_NONE (which correctly
//  increfs Py_None).
//
//  NOTE: This function sets a python error directly, and returns false.
//  TF_Status is only passed since we don't want to have to reallocate it.
bool ConvertToTensor(
    const FastPathOpExecInfo& op_exec_info, PyObject* input,
    tensorflow::Safe_PyObjectPtr* output_handle,
    // This gets a hint for this particular input.
    const std::function<tensorflow::DataType()>& dtype_hint_getter,
    // This sets the dtype after conversion is complete.
    const std::function<void(const tensorflow::DataType dtype)>& dtype_setter,
    TF_Status* status) {
  if (EagerTensor_CheckExact(input)) {
    Py_INCREF(input);
    output_handle->reset(input);
    return true;
  } else if (CheckResourceVariable(input)) {
    return ReadVariableOp(op_exec_info, input, output_handle, status);
  }

  // The hint comes from a supposedly similarly typed tensor.
  tensorflow::DataType dtype_hint = dtype_hint_getter();

  TFE_TensorHandle* handle = tensorflow::ConvertToEagerTensor(
      op_exec_info.ctx, input, dtype_hint, op_exec_info.device_name);
  if (handle == nullptr) {
    return tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr);
  }

  output_handle->reset(EagerTensorFromHandle(handle));
  dtype_setter(
      static_cast<tensorflow::DataType>(TFE_TensorHandleDataType(handle)));

  return true;
}

// Adds input and type attr to the op, and to the list of flattened
// inputs/attrs.
bool AddInputToOp(FastPathOpExecInfo* op_exec_info, PyObject* input,
                  const bool add_type_attr,
                  const tensorflow::OpDef::ArgDef& input_arg,
                  std::vector<tensorflow::Safe_PyObjectPtr>* flattened_attrs,
                  std::vector<tensorflow::Safe_PyObjectPtr>* flattened_inputs,
                  TFE_Op* op, TF_Status* status) {
  // py_eager_tensor's ownership is transferred to flattened_inputs if it is
  // required, else the object is destroyed and DECREF'd when the object goes
  // out of scope in this function.
  tensorflow::Safe_PyObjectPtr py_eager_tensor = nullptr;

  if (!ConvertToTensor(
          *op_exec_info, input, &py_eager_tensor,
          [&]() {
            if (input_arg.type() != tensorflow::DataType::DT_INVALID) {
              return input_arg.type();
            }
            return MaybeGetDTypeForAttr(input_arg.type_attr(), op_exec_info);
          },
          [&](const tensorflow::DataType dtype) {
            op_exec_info->cached_dtypes[input_arg.type_attr()] = dtype;
          },
          status)) {
    return false;
  }

  TFE_TensorHandle* input_handle = EagerTensor_Handle(py_eager_tensor.get());

  if (add_type_attr && !input_arg.type_attr().empty()) {
    auto dtype = TFE_TensorHandleDataType(input_handle);
    TFE_OpSetAttrType(op, input_arg.type_attr().data(), dtype);
    if (flattened_attrs != nullptr) {
      flattened_attrs->emplace_back(
          GetPythonObjectFromString(input_arg.type_attr()));
      flattened_attrs->emplace_back(PyLong_FromLong(dtype));
    }
  }

  if (flattened_inputs != nullptr) {
    flattened_inputs->emplace_back(std::move(py_eager_tensor));
  }

  TFE_OpAddInput(op, input_handle, status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr)) {
    return false;
  }

  return true;
}

const char* GetDeviceName(PyObject* py_device_name) {
  if (py_device_name != Py_None) {
    return TFE_GetPythonString(py_device_name);
  }
  return nullptr;
}

bool RaiseIfNotPySequence(PyObject* seq, const string& attr_name) {
  if (!PySequence_Check(seq)) {
    PyErr_SetString(PyExc_TypeError,
                    Printf("expected a sequence for attr %s, got %s instead",
                           attr_name.data(), seq->ob_type->tp_name)
                        .data());

    return false;
  }
  if (PyArray_Check(seq) &&
      PyArray_NDIM(reinterpret_cast<PyArrayObject*>(seq)) != 1) {
    PyErr_SetString(PyExc_ValueError,
                    Printf("expected a sequence for attr %s, got an ndarray "
                           "with rank %d instead",
                           attr_name.data(),
                           PyArray_NDIM(reinterpret_cast<PyArrayObject*>(seq)))
                        .data());
    return false;
  }
  return true;
}

bool RunCallbacks(
    const FastPathOpExecInfo& op_exec_info, PyObject* args,
    int num_inferred_attrs,
    const std::vector<tensorflow::Safe_PyObjectPtr>& flattened_inputs,
    const std::vector<tensorflow::Safe_PyObjectPtr>& flattened_attrs,
    PyObject* flattened_result) {
  DCHECK(op_exec_info.run_callbacks);

  tensorflow::Safe_PyObjectPtr inputs(PyTuple_New(flattened_inputs.size()));
  for (int i = 0; i < flattened_inputs.size(); i++) {
    PyObject* input = flattened_inputs[i].get();
    Py_INCREF(input);
    PyTuple_SET_ITEM(inputs.get(), i, input);
  }

  int num_non_inferred_attrs = PyTuple_GET_SIZE(args) - num_inferred_attrs;
  int num_attrs = flattened_attrs.size() + num_non_inferred_attrs;
  tensorflow::Safe_PyObjectPtr attrs(PyTuple_New(num_attrs));

  for (int i = 0; i < num_non_inferred_attrs; i++) {
    auto* attr = PyTuple_GET_ITEM(args, num_inferred_attrs + i);
    Py_INCREF(attr);
    PyTuple_SET_ITEM(attrs.get(), i, attr);
  }

  for (int i = num_non_inferred_attrs; i < num_attrs; i++) {
    PyObject* attr_or_name =
        flattened_attrs.at(i - num_non_inferred_attrs).get();
    Py_INCREF(attr_or_name);
    PyTuple_SET_ITEM(attrs.get(), i, attr_or_name);
  }

  if (op_exec_info.run_gradient_callback) {
    if (!RecordGradient(op_exec_info.op_name, inputs.get(), attrs.get(),
                        flattened_result)) {
      return false;
    }
  }

  if (op_exec_info.run_post_exec_callbacks) {
    tensorflow::Safe_PyObjectPtr callback_args(
        Py_BuildValue("OOOOO", op_exec_info.op_name, inputs.get(), attrs.get(),
                      flattened_result, op_exec_info.name));
    for (Py_ssize_t i = 0; i < PyList_Size(op_exec_info.callbacks); i++) {
      PyObject* callback_fn = PyList_GET_ITEM(op_exec_info.callbacks, i);
      if (!PyCallable_Check(callback_fn)) {
        PyErr_SetString(
            PyExc_TypeError,
            Printf("expected a function for "
                   "post execution callback in index %ld, got %s instead",
                   i, callback_fn->ob_type->tp_name)
                .c_str());
        return false;
      }
      PyObject* callback_result =
          PyObject_CallObject(callback_fn, callback_args.get());
      if (!callback_result) {
        return false;
      }
      Py_DECREF(callback_result);
    }
  }

  return true;
}

}  // namespace

PyObject* TFE_Py_FastPathExecute_C(PyObject* args) {
  tensorflow::profiler::TraceMe activity(
      "TFE_Py_FastPathExecute_C", tensorflow::profiler::TraceMeLevel::kInfo);
  Py_ssize_t args_size = PyTuple_GET_SIZE(args);
  if (args_size < FAST_PATH_EXECUTE_ARG_INPUT_START) {
    PyErr_SetString(
        PyExc_ValueError,
        Printf("There must be at least %d items in the input tuple.",
               FAST_PATH_EXECUTE_ARG_INPUT_START)
            .c_str());
    return nullptr;
  }

  FastPathOpExecInfo op_exec_info;

  PyObject* py_eager_context =
      PyTuple_GET_ITEM(args, FAST_PATH_EXECUTE_ARG_CONTEXT);

  // TODO(edoper): Use interned string here
  PyObject* eager_context_handle =
      PyObject_GetAttrString(py_eager_context, "_context_handle");

  TFE_Context* ctx = reinterpret_cast<TFE_Context*>(
      PyCapsule_GetPointer(eager_context_handle, nullptr));
  op_exec_info.ctx = ctx;
  op_exec_info.args = args;

  if (ctx == nullptr) {
    // The context hasn't been initialized. It will be in the slow path.
    RaiseFallbackException(
        "This function does not handle the case of the path where "
        "all inputs are not already EagerTensors.");
    return nullptr;
  }

  auto* tld = tensorflow::GetEagerContextThreadLocalData(py_eager_context);
  if (tld == nullptr) {
    return nullptr;
  }
  op_exec_info.device_name = GetDeviceName(tld->device_name.get());
  op_exec_info.callbacks = tld->op_callbacks.get();

  op_exec_info.op_name = PyTuple_GET_ITEM(args, FAST_PATH_EXECUTE_ARG_OP_NAME);
  op_exec_info.name = PyTuple_GET_ITEM(args, FAST_PATH_EXECUTE_ARG_NAME);

  // TODO(nareshmodi): Add a benchmark for the fast-path with gradient callbacks
  // (similar to benchmark_tf_gradient_function_*). Also consider using an
  // InlinedVector for flattened_attrs and flattened_inputs if the benchmarks
  // point out problems with heap allocs.
  op_exec_info.run_gradient_callback =
      !*ThreadTapeIsStopped() && HasAccumulatorOrTape();
  op_exec_info.run_post_exec_callbacks =
      op_exec_info.callbacks != Py_None &&
      PyList_Size(op_exec_info.callbacks) > 0;
  op_exec_info.run_callbacks = op_exec_info.run_gradient_callback ||
                               op_exec_info.run_post_exec_callbacks;

  TF_Status* status = GetStatus();
  const char* op_name = TFE_GetPythonString(op_exec_info.op_name);
  if (op_name == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    Printf("expected a string for op_name, got %s instead",
                           op_exec_info.op_name->ob_type->tp_name)
                        .c_str());
    return nullptr;
  }

  TFE_Op* op = GetOp(ctx, op_name, op_exec_info.device_name, status);

  auto cleaner = tensorflow::gtl::MakeCleanup([status, ctx, op] {
    ReturnStatus(status);
    ReturnOp(ctx, op);
  });

  if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr)) {
    return nullptr;
  }

  tensorflow::unwrap(op)->SetStackTrace(tensorflow::GetStackTrace(
      tensorflow::StackTrace::kStackTraceInitialSize));

  const tensorflow::OpDef* op_def = tensorflow::unwrap(op)->OpDef();
  if (op_def == nullptr) return nullptr;

  if (args_size <
      FAST_PATH_EXECUTE_ARG_INPUT_START + op_def->input_arg_size()) {
    PyErr_SetString(
        PyExc_ValueError,
        Printf("Tuple size smaller than intended. Expected to be at least %d, "
               "was %ld",
               FAST_PATH_EXECUTE_ARG_INPUT_START + op_def->input_arg_size(),
               args_size)
            .c_str());
    return nullptr;
  }

  if (!CheckInputsOk(args, FAST_PATH_EXECUTE_ARG_INPUT_START, *op_def)) {
    RaiseFallbackException(
        "This function does not handle the case of the path where "
        "all inputs are not already EagerTensors.");
    return nullptr;
  }

  op_exec_info.attr_to_inputs_map = GetAttrToInputsMapHoldingGIL(*op_def);
  op_exec_info.default_dtypes = GetAttrToDefaultsMapHoldingGIL(*op_def);

  // Mapping of attr name to size - used to calculate the number of values
  // to be expected by the TFE_Execute run.
  tensorflow::gtl::FlatMap<string, int64_t> attr_list_sizes;

  // Set non-inferred attrs, including setting defaults if the attr is passed in
  // as None.
  for (int i = FAST_PATH_EXECUTE_ARG_INPUT_START + op_def->input_arg_size();
       i < args_size; i += 2) {
    PyObject* py_attr_name = PyTuple_GET_ITEM(args, i);
    const char* attr_name = TFE_GetPythonString(py_attr_name);
    PyObject* py_attr_value = PyTuple_GET_ITEM(args, i + 1);

    // Not creating an index since most of the time there are not more than a
    // few attrs.
    // TODO(nareshmodi): Maybe include the index as part of the
    // OpRegistrationData.
    for (const auto& attr : op_def->attr()) {
      if (tensorflow::StringPiece(attr_name) == attr.name()) {
        SetOpAttrWithDefaults(ctx, op, attr, attr_name, py_attr_value,
                              &attr_list_sizes, status);

        if (!status->status.ok()) {
          VLOG(1) << "Falling back to slow path for Op \"" << op_def->name()
                  << "\" since we are unable to set the value for attr \""
                  << attr.name() << "\" due to: " << TF_Message(status);
          RaiseFallbackException(TF_Message(status));
          return nullptr;
        }

        break;
      }
    }
  }

  // Flat attrs and inputs as required by the record_gradient call. The attrs
  // here only contain inferred attrs (non-inferred attrs are added directly
  // from the input args).
  // All items in flattened_attrs and flattened_inputs contain
  // Safe_PyObjectPtr - any time something steals a reference to this, it must
  // INCREF.
  // TODO(nareshmodi): figure out why PyList_New/PyList_Append don't work
  // directly.
  std::unique_ptr<std::vector<tensorflow::Safe_PyObjectPtr>> flattened_attrs =
      nullptr;
  std::unique_ptr<std::vector<tensorflow::Safe_PyObjectPtr>> flattened_inputs =
      nullptr;

  // TODO(nareshmodi): Encapsulate callbacks information into a struct.
  if (op_exec_info.run_callbacks) {
    flattened_attrs.reset(new std::vector<tensorflow::Safe_PyObjectPtr>);
    flattened_inputs.reset(new std::vector<tensorflow::Safe_PyObjectPtr>);
  }

  // Add inferred attrs and inputs.
  // The following code might set duplicate type attrs. This will result in
  // the CacheKey for the generated AttrBuilder possibly differing from
  // those where the type attrs are correctly set. Inconsistent CacheKeys
  // for ops means that there might be unnecessarily duplicated kernels.
  // TODO(nareshmodi): Fix this.
  for (int i = 0; i < op_def->input_arg_size(); i++) {
    const auto& input_arg = op_def->input_arg(i);

    PyObject* input =
        PyTuple_GET_ITEM(args, FAST_PATH_EXECUTE_ARG_INPUT_START + i);
    if (!input_arg.number_attr().empty()) {
      // The item is a homogeneous list.
      if (!RaiseIfNotPySequence(input, input_arg.number_attr())) return nullptr;
      tensorflow::Safe_PyObjectPtr fast_input(
          PySequence_Fast(input, "Could not parse sequence."));
      if (fast_input.get() == nullptr) {
        return nullptr;
      }
      Py_ssize_t len = PySequence_Fast_GET_SIZE(fast_input.get());
      PyObject** fast_input_array = PySequence_Fast_ITEMS(fast_input.get());

      TFE_OpSetAttrInt(op, input_arg.number_attr().data(), len);
      if (op_exec_info.run_callbacks) {
        flattened_attrs->emplace_back(
            GetPythonObjectFromString(input_arg.number_attr()));
        flattened_attrs->emplace_back(PyLong_FromLong(len));
      }
      attr_list_sizes[input_arg.number_attr()] = len;

      if (len > 0) {
        // First item adds the type attr.
        if (!AddInputToOp(&op_exec_info, fast_input_array[0], true, input_arg,
                          flattened_attrs.get(), flattened_inputs.get(), op,
                          status)) {
          return nullptr;
        }

        for (Py_ssize_t j = 1; j < len; j++) {
          // Since the list is homogeneous, we don't need to re-add the attr.
          if (!AddInputToOp(&op_exec_info, fast_input_array[j], false,
                            input_arg, nullptr /* flattened_attrs */,
                            flattened_inputs.get(), op, status)) {
            return nullptr;
          }
        }
      }
    } else if (!input_arg.type_list_attr().empty()) {
      // The item is a heterogeneous list.
      if (!RaiseIfNotPySequence(input, input_arg.type_list_attr())) {
        return nullptr;
      }
      tensorflow::Safe_PyObjectPtr fast_input(
          PySequence_Fast(input, "Could not parse sequence."));
      if (fast_input.get() == nullptr) {
        return nullptr;
      }
      const string& attr_name = input_arg.type_list_attr();
      Py_ssize_t len = PySequence_Fast_GET_SIZE(fast_input.get());
      PyObject** fast_input_array = PySequence_Fast_ITEMS(fast_input.get());
      tensorflow::gtl::InlinedVector<TF_DataType, 4> attr_value(len);
      PyObject* py_attr_value = nullptr;
      if (op_exec_info.run_callbacks) {
        py_attr_value = PyTuple_New(len);
      }
      for (Py_ssize_t j = 0; j < len; j++) {
        PyObject* py_input = fast_input_array[j];
        tensorflow::Safe_PyObjectPtr py_eager_tensor;
        if (!ConvertToTensor(
                op_exec_info, py_input, &py_eager_tensor,
                []() { return tensorflow::DT_INVALID; },
                [](const tensorflow::DataType dtype) {}, status)) {
          return nullptr;
        }

        TFE_TensorHandle* input_handle =
            EagerTensor_Handle(py_eager_tensor.get());

        attr_value[j] = TFE_TensorHandleDataType(input_handle);

        TFE_OpAddInput(op, input_handle, status);
        if (tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr)) {
          return nullptr;
        }

        if (op_exec_info.run_callbacks) {
          flattened_inputs->emplace_back(std::move(py_eager_tensor));

          PyTuple_SET_ITEM(py_attr_value, j, PyLong_FromLong(attr_value[j]));
        }
      }
      if (op_exec_info.run_callbacks) {
        flattened_attrs->emplace_back(GetPythonObjectFromString(attr_name));
        flattened_attrs->emplace_back(py_attr_value);
      }
      TFE_OpSetAttrTypeList(op, attr_name.data(), attr_value.data(),
                            attr_value.size());
      attr_list_sizes[attr_name] = len;
    } else {
      // The item is a single item.
      if (!AddInputToOp(&op_exec_info, input, true, input_arg,
                        flattened_attrs.get(), flattened_inputs.get(), op,
                        status)) {
        return nullptr;
      }
    }
  }

  int64_t num_outputs = 0;
  for (int i = 0; i < op_def->output_arg_size(); i++) {
    const auto& output_arg = op_def->output_arg(i);
    int64_t delta = 1;
    if (!output_arg.number_attr().empty()) {
      delta = attr_list_sizes[output_arg.number_attr()];
    } else if (!output_arg.type_list_attr().empty()) {
      delta = attr_list_sizes[output_arg.type_list_attr()];
    }
    if (delta < 0) {
      RaiseFallbackException(
          "Attributes suggest that the size of an output list is less than 0");
      return nullptr;
    }
    num_outputs += delta;
  }

  // If number of retvals is larger than int32, we error out.
  if (static_cast<int64_t>(static_cast<int32_t>(num_outputs)) != num_outputs) {
    PyErr_SetString(
        PyExc_ValueError,
        Printf("Number of outputs is too big: %ld", num_outputs).c_str());
    return nullptr;
  }
  int num_retvals = num_outputs;

  tensorflow::gtl::InlinedVector<TFE_TensorHandle*, 2> retvals(num_retvals);

  Py_BEGIN_ALLOW_THREADS;
  TFE_Execute(op, retvals.data(), &num_retvals, status);
  Py_END_ALLOW_THREADS;

  if (!status->status.ok()) {
    // Augment the status with the op_name for easier debugging similar to
    // TFE_Py_Execute.
    status->status = tensorflow::errors::CreateWithUpdatedMessage(
        status->status, tensorflow::strings::StrCat(
                            TF_Message(status), " [Op:",
                            TFE_GetPythonString(op_exec_info.op_name), "]"));
    tensorflow::MaybeRaiseExceptionFromTFStatus(status, nullptr);
    return nullptr;
  }

  tensorflow::Safe_PyObjectPtr flat_result(PyList_New(num_retvals));
  for (int i = 0; i < num_retvals; ++i) {
    PyList_SET_ITEM(flat_result.get(), i, EagerTensorFromHandle(retvals[i]));
  }

  if (op_exec_info.run_callbacks) {
    if (!RunCallbacks(
            op_exec_info, args,
            FAST_PATH_EXECUTE_ARG_INPUT_START + op_def->input_arg_size(),
            *flattened_inputs, *flattened_attrs, flat_result.get())) {
      return nullptr;
    }
  }

  // Unflatten results.
  if (op_def->output_arg_size() == 0) {
    Py_RETURN_NONE;
  }

  if (op_def->output_arg_size() == 1) {
    if (!op_def->output_arg(0).number_attr().empty() ||
        !op_def->output_arg(0).type_list_attr().empty()) {
      return flat_result.release();
    } else {
      auto* result = PyList_GET_ITEM(flat_result.get(), 0);
      Py_INCREF(result);
      return result;
    }
  }

  // Correctly output the results that are made into a namedtuple.
  PyObject* result = PyList_New(op_def->output_arg_size());
  int flat_result_index = 0;
  for (int i = 0; i < op_def->output_arg_size(); i++) {
    if (!op_def->output_arg(i).number_attr().empty()) {
      int list_length = attr_list_sizes[op_def->output_arg(i).number_attr()];
      PyObject* inner_list = PyList_New(list_length);
      for (int j = 0; j < list_length; j++) {
        PyObject* obj = PyList_GET_ITEM(flat_result.get(), flat_result_index++);
        Py_INCREF(obj);
        PyList_SET_ITEM(inner_list, j, obj);
      }
      PyList_SET_ITEM(result, i, inner_list);
    } else if (!op_def->output_arg(i).type_list_attr().empty()) {
      int list_length = attr_list_sizes[op_def->output_arg(i).type_list_attr()];
      PyObject* inner_list = PyList_New(list_length);
      for (int j = 0; j < list_length; j++) {
        PyObject* obj = PyList_GET_ITEM(flat_result.get(), flat_result_index++);
        Py_INCREF(obj);
        PyList_SET_ITEM(inner_list, j, obj);
      }
      PyList_SET_ITEM(result, i, inner_list);
    } else {
      PyObject* obj = PyList_GET_ITEM(flat_result.get(), flat_result_index++);
      Py_INCREF(obj);
      PyList_SET_ITEM(result, i, obj);
    }
  }
  return result;
}

PyObject* TFE_Py_RecordGradient(PyObject* op_name, PyObject* inputs,
                                PyObject* attrs, PyObject* results,
                                PyObject* forward_pass_name_scope) {
  if (*ThreadTapeIsStopped() || !HasAccumulatorOrTape()) {
    Py_RETURN_NONE;
  }

  return RecordGradient(op_name, inputs, attrs, results,
                        forward_pass_name_scope);
}

// A method prints incoming messages directly to Python's
// stdout using Python's C API. This is necessary in Jupyter notebooks
// and colabs where messages to the C stdout don't go to the notebook
// cell outputs, but calls to Python's stdout do.
void PrintToPythonStdout(const char* msg) {
  if (Py_IsInitialized()) {
    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();

    string string_msg = msg;
    // PySys_WriteStdout truncates strings over 1000 bytes, so
    // we write the message in chunks small enough to not be truncated.
    int CHUNK_SIZE = 900;
    auto len = string_msg.length();
    for (int i = 0; i < len; i += CHUNK_SIZE) {
      PySys_WriteStdout("%s", string_msg.substr(i, CHUNK_SIZE).c_str());
    }

    // Force flushing to make sure print newlines aren't interleaved in
    // some colab environments
    PyRun_SimpleString("import sys; sys.stdout.flush()");

    PyGILState_Release(py_threadstate);
  }
}

// Register PrintToPythonStdout as a log listener, to allow
// printing in colabs and jupyter notebooks to work.
void TFE_Py_EnableInteractivePythonLogging() {
  static bool enabled_interactive_logging = false;
  if (!enabled_interactive_logging) {
    enabled_interactive_logging = true;
    TF_RegisterLogListener(PrintToPythonStdout);
  }
}

namespace {
// TODO(mdan): Clean this. Maybe by decoupling context lifetime from Python GC?
// Weak reference to the Python Context (see tensorflow/python/eager/context.py)
// object currently active. This object is opaque and wrapped inside a Python
// Capsule. However, the EagerContext object it holds is tracked by the
// global_c_eager_context object.
// Also see common_runtime/eager/context.cc.
PyObject* global_py_eager_context = nullptr;
}  // namespace

PyObject* TFE_Py_SetEagerContext(PyObject* py_context) {
  Py_XDECREF(global_py_eager_context);
  global_py_eager_context = PyWeakref_NewRef(py_context, nullptr);
  if (global_py_eager_context == nullptr) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* GetPyEagerContext() {
  if (global_py_eager_context == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Python eager context is not set");
    return nullptr;
  }
  PyObject* py_context = PyWeakref_GET_OBJECT(global_py_eager_context);
  if (py_context == Py_None) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Python eager context has been destroyed");
    return nullptr;
  }
  Py_INCREF(py_context);
  return py_context;
}

namespace {

// Default values for thread_local_data fields.
struct EagerContextThreadLocalDataDefaults {
  tensorflow::Safe_PyObjectPtr is_eager;
  tensorflow::Safe_PyObjectPtr device_spec;
};

// Maps each py_eager_context object to its thread_local_data.
//
// Note: we need to use the python Context object as the key here (and not
// its handle object), because the handle object isn't created until the
// context is initialized; but thread_local_data is potentially accessed
// before then.
using EagerContextThreadLocalDataMap = absl::flat_hash_map<
    PyObject*, std::unique_ptr<tensorflow::EagerContextThreadLocalData>>;
thread_local EagerContextThreadLocalDataMap*
    eager_context_thread_local_data_map = nullptr;

// Maps each py_eager_context object to default values.
using EagerContextThreadLocalDataDefaultsMap =
    absl::flat_hash_map<PyObject*, EagerContextThreadLocalDataDefaults>;
EagerContextThreadLocalDataDefaultsMap*
    eager_context_thread_local_data_defaults = nullptr;

}  // namespace

namespace tensorflow {

void MakeEagerContextThreadLocalData(PyObject* py_eager_context,
                                     PyObject* is_eager,
                                     PyObject* device_spec) {
  DCheckPyGilState();
  if (eager_context_thread_local_data_defaults == nullptr) {
    absl::LeakCheckDisabler disabler;
    eager_context_thread_local_data_defaults =
        new EagerContextThreadLocalDataDefaultsMap();
  }
  if (eager_context_thread_local_data_defaults->count(py_eager_context) > 0) {
    PyErr_SetString(PyExc_AssertionError,
                    "MakeEagerContextThreadLocalData may not be called "
                    "twice on the same eager Context object.");
  }

  auto& defaults =
      (*eager_context_thread_local_data_defaults)[py_eager_context];
  Py_INCREF(is_eager);
  defaults.is_eager.reset(is_eager);
  Py_INCREF(device_spec);
  defaults.device_spec.reset(device_spec);
}

EagerContextThreadLocalData* GetEagerContextThreadLocalData(
    PyObject* py_eager_context) {
  if (eager_context_thread_local_data_defaults == nullptr) {
    PyErr_SetString(PyExc_AssertionError,
                    "MakeEagerContextThreadLocalData must be called "
                    "before GetEagerContextThreadLocalData.");
    return nullptr;
  }
  auto defaults =
      eager_context_thread_local_data_defaults->find(py_eager_context);
  if (defaults == eager_context_thread_local_data_defaults->end()) {
    PyErr_SetString(PyExc_AssertionError,
                    "MakeEagerContextThreadLocalData must be called "
                    "before GetEagerContextThreadLocalData.");
    return nullptr;
  }

  if (eager_context_thread_local_data_map == nullptr) {
    absl::LeakCheckDisabler disabler;
    eager_context_thread_local_data_map = new EagerContextThreadLocalDataMap();
  }
  auto& thread_local_data =
      (*eager_context_thread_local_data_map)[py_eager_context];

  if (!thread_local_data) {
    thread_local_data.reset(new EagerContextThreadLocalData());

    Safe_PyObjectPtr is_eager(
        PyObject_CallFunctionObjArgs(defaults->second.is_eager.get(), nullptr));
    if (!is_eager) return nullptr;
    thread_local_data->is_eager = PyObject_IsTrue(is_eager.get());

#if PY_MAJOR_VERSION >= 3
    PyObject* scope_name = PyUnicode_FromString("");
#else
    PyObject* scope_name = PyString_FromString("");
#endif
    thread_local_data->scope_name.reset(scope_name);

#if PY_MAJOR_VERSION >= 3
    PyObject* device_name = PyUnicode_FromString("");
#else
    PyObject* device_name = PyString_FromString("");
#endif
    thread_local_data->device_name.reset(device_name);

    Py_INCREF(defaults->second.device_spec.get());
    thread_local_data->device_spec.reset(defaults->second.device_spec.get());

    Py_INCREF(Py_None);
    thread_local_data->function_call_options.reset(Py_None);

    Py_INCREF(Py_None);
    thread_local_data->executor.reset(Py_None);

    thread_local_data->op_callbacks.reset(PyList_New(0));
  }
  return thread_local_data.get();
}

void DestroyEagerContextThreadLocalData(PyObject* py_eager_context) {
  DCheckPyGilState();
  if (eager_context_thread_local_data_defaults) {
    eager_context_thread_local_data_defaults->erase(py_eager_context);
  }
  if (eager_context_thread_local_data_map) {
    eager_context_thread_local_data_map->erase(py_eager_context);
  }
}

}  // namespace tensorflow
