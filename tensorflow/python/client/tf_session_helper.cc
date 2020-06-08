/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/client/tf_session_helper.h"

#include <cstring>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/python/client/session_ref.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {

namespace {

static const char* kFeedDictErrorMsg =
    "feed_dict must be a dictionary mapping strings to NumPy arrays.";
}  // end namespace

TF_Session* TF_NewSessionRef(TF_Graph* graph, const TF_SessionOptions* opts,
                             TF_Status* status) {
  TF_Session* tf_session = TF_NewSession(graph, opts, status);
  if (tf_session == nullptr) {
    return nullptr;
  }

  Session* session = reinterpret_cast<Session*>(tf_session->session);
  SessionRef* session_ref = new SessionRef(session);
  tf_session->session = session_ref;
  return tf_session;
}

void TF_Run_wrapper_helper(TF_DeprecatedSession* session, const char* handle,
                           const TF_Buffer* run_options, PyObject* feed_dict,
                           const NameVector& output_names,
                           const NameVector& target_nodes,
                           TF_Status* out_status, PyObjectVector* out_values,
                           TF_Buffer* run_outputs) {
  // 1. Convert the feed inputs to the appropriate form for TF_Run.
  if (!PyDict_Check(feed_dict)) {
    Set_TF_Status_from_Status(out_status,
                              errors::InvalidArgument(kFeedDictErrorMsg));
    return;
  }

  NameVector input_names;
  std::vector<Safe_TF_TensorPtr> inputs_safe;  // Used to delete tensors.
  TF_TensorVector inputs_unsafe;  // Used to contain the arg to TF_Run.

  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int index = 0;
  Status s;

  while (PyDict_Next(feed_dict, &pos, &key, &value)) {
    char* key_string = PyBytes_AsString(key);
    if (!key_string) {
      Set_TF_Status_from_Status(out_status,
                                errors::InvalidArgument(kFeedDictErrorMsg));
      return;
    }
    input_names.push_back(key_string);

    inputs_safe.emplace_back(make_safe(static_cast<TF_Tensor*>(nullptr)));
    s = NdarrayToTensor(nullptr /*ctx*/, value, &inputs_safe.back(),
                        true /*convert_to_string*/);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    inputs_unsafe.push_back(inputs_safe.back().get());
    ++index;
  }

  // 2. Allocate a container for the output data.
  TF_TensorVector outputs(output_names.size());

  // In case any tensors were leftover from previous runs we might as well clear
  // them here.
  ClearDecrefCache();

  // 3. Actually call TF_Run().
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_Run(session, run_options, input_names.data(), inputs_unsafe.data(),
           input_names.size(), const_cast<const char**>(output_names.data()),
           outputs.data(), output_names.size(),
           const_cast<const char**>(target_nodes.data()), target_nodes.size(),
           run_outputs, out_status);
  } else {
    TF_PRun(session, handle, input_names.data(), inputs_unsafe.data(),
            input_names.size(), const_cast<const char**>(output_names.data()),
            outputs.data(), output_names.size(),
            const_cast<const char**>(target_nodes.data()), target_nodes.size(),
            out_status);
  }

  Py_END_ALLOW_THREADS;

  // Decref any numpy arrays we are not using anymore.
  ClearDecrefCache();

  if (TF_GetCode(out_status) != TF_OK) {
    return;
  }

  // 4. We now own the fetched tensors, so set up a safe container to
  // delete them when we exit this scope.
  std::vector<Safe_TF_TensorPtr> tf_outputs_safe;
  for (const auto& output : outputs) {
    tf_outputs_safe.emplace_back(make_safe(output));
  }

  // 5. Convert the fetched tensors into numpy ndarrays. Store them in a safe
  // container so that we do not leak
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  for (size_t i = 0; i < output_names.size(); ++i) {
    PyObject* py_array;
    s = TF_TensorToPyArray(std::move(tf_outputs_safe[i]), &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.emplace_back(
        make_safe(PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array))));
  }

  // 6. If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
}

// Wrapper for TF_Run that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_Run_wrapper(TF_DeprecatedSession* session, const TF_Buffer* run_options,
                    PyObject* feed_dict, const NameVector& output_names,
                    const NameVector& target_nodes, TF_Status* out_status,
                    PyObjectVector* out_values, TF_Buffer* run_outputs) {
  TF_Run_wrapper_helper(session, nullptr, run_options, feed_dict, output_names,
                        target_nodes, out_status, out_values, run_outputs);
  ClearDecrefCache();
}

namespace {
void MakeCallableHelper(tensorflow::Session* session,
                        const TF_Buffer* callable_options, int64_t* out_handle,
                        TF_Status* out_status) {
  tensorflow::CallableOptions callable_options_proto;
  if (callable_options != nullptr &&
      !callable_options_proto.ParseFromArray(callable_options->data,
                                             callable_options->length)) {
    Set_TF_Status_from_Status(
        out_status,
        errors::InvalidArgument("Unparseable CallableOptions proto"));
    return;
  }
  tensorflow::Session::CallableHandle handle;
  Status s = session->MakeCallable(callable_options_proto, &handle);
  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return;
  }
  *out_handle = handle;
}
}  // namespace

void TF_DeprecatedSessionMakeCallable(TF_DeprecatedSession* session,
                                      const TF_Buffer* callable_options,
                                      int64_t* out_handle, TF_Status* status) {
  MakeCallableHelper(session->session, callable_options, out_handle, status);
}
void TF_SessionMakeCallable(TF_Session* session,
                            const TF_Buffer* callable_options,
                            int64_t* out_handle, TF_Status* status) {
  MakeCallableHelper(session->session, callable_options, out_handle, status);
}

namespace {
void RunCallableHelper(tensorflow::Session* session, int64_t handle,
                       PyObject* feed_values, TF_Status* out_status,
                       PyObjectVector* out_values, TF_Buffer* run_metadata) {
  // Convert feed values to a vector of tensorflow::Tensor objects.
  std::vector<Tensor> input_tensors;
  Status s;
  {
    feed_values =
        PySequence_Fast(feed_values, "feed_values must be a sequence");
    if (feed_values == nullptr) return;
    Safe_PyObjectPtr feed_values_holder(make_safe(feed_values));
    Py_ssize_t len = PySequence_Fast_GET_SIZE(feed_values);
    input_tensors.reserve(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PySequence_Fast_GET_ITEM(feed_values, i);
      if (!elem) {
        Set_TF_Status_from_Status(
            out_status, errors::Internal("Could not get feed value ", i));
        return;
      }
      Tensor t;
      s = NdarrayToTensor(elem, &t);
      if (!s.ok()) {
        Set_TF_Status_from_Status(out_status, s);
        return;
      }
      input_tensors.push_back(std::move(t));
    }
  }

  // Allocate a RunMetadata protobuf object to receive the metadata,
  // if the caller is expecting any.
  std::unique_ptr<RunMetadata> run_metadata_proto;
  if (run_metadata != nullptr) {
    run_metadata_proto.reset(new RunMetadata);
  }

  // Run the callable.
  std::vector<Tensor> output_tensors;
  Py_BEGIN_ALLOW_THREADS;
  s = session->RunCallable(handle, input_tensors, &output_tensors,
                           run_metadata_proto.get());
  Py_END_ALLOW_THREADS;

  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return;
  }

  // If requested, serialize the RunMetadata to pass it back to the caller.
  if (run_metadata != nullptr) {
    s = MessageToBuffer(*run_metadata_proto, run_metadata);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
  }

  // Convert results to NumPy arrays. Since this can fail, stage the
  // results via a safe container that takes care of decreasing the
  // reference count on failure.
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  py_outputs_safe.reserve(output_tensors.size());
  for (const Tensor& output : output_tensors) {
    PyObject* py_array;
    s = TensorToNdarray(output, &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.push_back(
        make_safe(PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array))));
  }

  // If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  out_values->reserve(py_outputs_safe.size());
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
}
}  // namespace

void TF_DeprecatedSessionRunCallable(TF_DeprecatedSession* session,
                                     int64_t handle, PyObject* feed_values,
                                     PyObjectVector* out_values,
                                     TF_Buffer* run_metadata,
                                     TF_Status* status) {
  RunCallableHelper(session->session, handle, feed_values, status, out_values,
                    run_metadata);
  ClearDecrefCache();
}
void TF_SessionRunCallable(TF_Session* session, int64_t handle,
                           PyObject* feed_values, PyObjectVector* out_values,
                           TF_Buffer* run_metadata, TF_Status* status) {
  RunCallableHelper(session->session, handle, feed_values, status, out_values,
                    run_metadata);
  ClearDecrefCache();
}

void TF_DeprecatedSessionReleaseCallable(TF_DeprecatedSession* session,
                                         int64_t handle, TF_Status* status) {
  Set_TF_Status_from_Status(status, session->session->ReleaseCallable(handle));
}
void TF_SessionReleaseCallable(TF_Session* session, int64_t handle,
                               TF_Status* status) {
  Set_TF_Status_from_Status(status, session->session->ReleaseCallable(handle));
}

// Wrapper for TF_PRunSetup that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of *out_handle.
void TF_PRunSetup_wrapper(TF_DeprecatedSession* session,
                          const NameVector& input_names,
                          const NameVector& output_names,
                          const NameVector& target_nodes, TF_Status* out_status,
                          const char** out_handle) {
  Py_BEGIN_ALLOW_THREADS;
  TF_PRunSetup(
      session, const_cast<const char**>(input_names.data()), input_names.size(),
      const_cast<const char**>(output_names.data()), output_names.size(),
      const_cast<const char**>(target_nodes.data()), target_nodes.size(),
      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

// Wrapper for TF_PRun that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_PRun_wrapper(TF_DeprecatedSession* session, const char* handle,
                     PyObject* feed_dict, const NameVector& output_names,
                     TF_Status* out_status, PyObjectVector* out_values) {
  TF_Run_wrapper_helper(session, handle, nullptr, feed_dict, output_names,
                        NameVector(), out_status, out_values, nullptr);
  ClearDecrefCache();
}

// Wrapper for TF_Reset that converts the string vectors to character arrays.
void TF_Reset_wrapper(const TF_SessionOptions* opt,
                      const NameVector& containers, TF_Status* status) {
  TF_Reset(opt, const_cast<const char**>(containers.data()), containers.size(),
           status);
}

void TF_SessionRun_wrapper_helper(TF_Session* session, const char* handle,
                                  const TF_Buffer* run_options,
                                  const std::vector<TF_Output>& inputs,
                                  const std::vector<PyObject*>& input_ndarrays,
                                  const std::vector<TF_Output>& outputs,
                                  const std::vector<TF_Operation*>& targets,
                                  TF_Buffer* run_metadata,
                                  TF_Status* out_status,
                                  std::vector<PyObject*>* py_outputs) {
  DCHECK_EQ(inputs.size(), input_ndarrays.size());
  DCHECK(py_outputs != nullptr);
  DCHECK(py_outputs->empty());
  Status s;

  // Convert input ndarray PyObjects to TF_Tensors. We maintain a continuous
  // array of TF_Tensor*s as well as scoped containers to make sure they're
  // cleaned up properly.
  //
  // Memory management:
  // NdarrayToTensor() creates a new ndarray PyObject from the input
  // ndarray. We manage the new ndarray's lifetime in order to keep the
  // underlying data buffer alive (the new ndarray also guarantees a contiguous
  // data buffer). The new ndarray's data buffer is used to create the
  // corresponding TF_Tensor. The TF_Tensor's deallocator will queue the new
  // ndarray to be decref'd by the next ClearDecrefCache() call (we can't call
  // Py_DECREF in the deallocator directly because the GIL must be held).
  //
  // Note that TF_Tensor may directly delegate its data and deallocator to a
  // TensorBuffer, which may outlive the TF_Tensor (e.g. if the tensor gets
  // queued or assigned to a variable).
  TF_TensorVector input_vals;
  std::vector<Safe_TF_TensorPtr> input_vals_safe;
  for (PyObject* ndarray : input_ndarrays) {
    input_vals_safe.emplace_back(make_safe(static_cast<TF_Tensor*>(nullptr)));
    s = NdarrayToTensor(nullptr, ndarray, &input_vals_safe.back(), true);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    input_vals.push_back(input_vals_safe.back().get());
  }

  // Allocate space for output TF_Tensor*s
  TF_TensorVector output_vals(outputs.size());

  // Clear up any unused memory leftover from previous runs
  ClearDecrefCache();

  // Call TF_SessionRun() (and release GIL during execution)
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_SessionRun(session, run_options, inputs.data(), input_vals.data(),
                  inputs.size(), outputs.data(), output_vals.data(),
                  outputs.size(), targets.data(), targets.size(), run_metadata,
                  out_status);
  } else {
    TF_SessionPRun(session, handle, inputs.data(), input_vals.data(),
                   inputs.size(), outputs.data(), output_vals.data(),
                   outputs.size(), targets.data(), targets.size(), out_status);
  }
  Py_END_ALLOW_THREADS;

  // Create scoped containers for output tensors
  std::vector<Safe_TF_TensorPtr> output_vals_safe;
  for (TF_Tensor* output : output_vals) {
    output_vals_safe.emplace_back(make_safe(output));
  }

  // Convert outputs to ndarrays (in scoped containers)
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  for (size_t i = 0; i < outputs.size(); ++i) {
    PyObject* py_array;
    s = TF_TensorToPyArray(std::move(output_vals_safe[i]), &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.emplace_back(
        make_safe(PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array))));
  }

  // If we reach this point, we have successfully built a list of objects so we
  // can release them from the safe container into the return vector.
  for (size_t i = 0; i < outputs.size(); ++i) {
    py_outputs->push_back(py_outputs_safe[i].release());
  }
}

void TF_SessionRun_wrapper(TF_Session* session, const TF_Buffer* run_options,
                           const std::vector<TF_Output>& inputs,
                           const std::vector<PyObject*>& input_ndarrays,
                           const std::vector<TF_Output>& outputs,
                           const std::vector<TF_Operation*>& targets,
                           TF_Buffer* run_metadata, TF_Status* out_status,
                           std::vector<PyObject*>* py_outputs) {
  TF_SessionRun_wrapper_helper(session, nullptr, run_options, inputs,
                               input_ndarrays, outputs, targets, run_metadata,
                               out_status, py_outputs);
  // Release any unused ndarray references (see memory management comment in
  // TF_SessionRun_wrapper_helper)
  ClearDecrefCache();
}

string EqualGraphDefWrapper(const string& actual, const string& expected) {
  GraphDef actual_def;
  if (!actual_def.ParseFromString(actual)) {
    return "actual is not a valid serialized GraphDef";
  }
  GraphDef expected_def;
  if (!expected_def.ParseFromString(expected)) {
    return "expected is not a valid serialized GraphDef";
  }
  string diff;
  return EqualGraphDef(actual_def, expected_def, &diff) ? "" : diff;
}

string EqualAttrValueWrapper(const string& actual, const string& expected) {
  AttrValue actual_attr_value;
  if (!actual_attr_value.ParseFromString(actual)) {
    return "actual is not a valid serialized AttrValue";
  }

  AttrValue expected_attr_value;
  if (!expected_attr_value.ParseFromString(expected)) {
    return "expected is not a valid serialized AttrValue";
  }

  string diff;
  if (!AreAttrValuesEqual(actual_attr_value, expected_attr_value)) {
    diff = strings::Printf(
        "Actual AttrValue %s does not match Expected AttrValue %s.",
        SummarizeAttrValue(actual_attr_value).c_str(),
        SummarizeAttrValue(expected_attr_value).c_str());
  }
  return diff;
}

// Return value set to 6 inlined elements so it fits in a 64-byte cache line.
tensorflow::gtl::InlinedVector<int64_t, 6> TF_GraphGetTensorShapeHelper(
    TF_Graph* graph, TF_Output output, TF_Status* out_status,
    bool* unknown_shape) {
  // Allocate a single variable for holding the result for RVO.
  tensorflow::gtl::InlinedVector<int64_t, 6> result;
  *unknown_shape = false;
  int num_dims = TF_GraphGetTensorNumDims(graph, output, out_status);
  if (TF_GetCode(out_status) != TF_OK) {
    return result;
  }
  // If shape is unknown, set boolean and return.
  if (num_dims == -1) {
    *unknown_shape = true;
    return result;
  }

  // If shape is a scalar, avoid another C call and just return {}.
  if (num_dims == 0) {
    return result;
  }

  result.resize(num_dims);
  TF_GraphGetTensorShape(graph, output, result.data(), num_dims, out_status);
  return result;
}

void TF_SessionPRunSetup_wrapper(TF_Session* session,
                                 const std::vector<TF_Output>& inputs,
                                 const std::vector<TF_Output>& outputs,
                                 const std::vector<TF_Operation*>& targets,
                                 const char** out_handle,
                                 TF_Status* out_status) {
  // Call TF_SessionPRunSetup() (and release GIL during execution)
  Py_BEGIN_ALLOW_THREADS;
  TF_SessionPRunSetup(session, inputs.data(), inputs.size(), outputs.data(),
                      outputs.size(), targets.data(), targets.size(),
                      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

void TF_SessionPRun_wrapper(TF_Session* session, const char* handle,
                            const std::vector<TF_Output>& inputs,
                            const std::vector<PyObject*>& input_ndarrays,
                            const std::vector<TF_Output>& outputs,
                            TF_Status* out_status,
                            std::vector<PyObject*>* py_outputs) {
  const std::vector<TF_Operation*> targets;
  TF_SessionRun_wrapper_helper(session, handle,
                               nullptr,  // run_options
                               inputs, input_ndarrays, outputs, targets,
                               nullptr,  // run_metadata
                               out_status, py_outputs);
  // Release any unused ndarray references (see memory management comment in
  // TF_SessionRun_wrapper_helper)
  ClearDecrefCache();
}

std::vector<TF_Output> GetOperationInputs(TF_Operation* oper) {
  int num_inputs = TF_OperationNumInputs(oper);
  std::vector<TF_Output> inputs(num_inputs);
  TF_OperationAllInputs(oper, inputs.data(), inputs.size());
  return inputs;
}

std::vector<TF_Operation*> TF_OperationGetControlInputs_wrapper(
    TF_Operation* oper) {
  std::vector<TF_Operation*> control_inputs(TF_OperationNumControlInputs(oper));
  TF_OperationGetControlInputs(oper, control_inputs.data(),
                               control_inputs.size());
  return control_inputs;
}

std::vector<TF_Operation*> TF_OperationGetControlOutputs_wrapper(
    TF_Operation* oper) {
  std::vector<TF_Operation*> control_outputs(
      TF_OperationNumControlOutputs(oper));
  TF_OperationGetControlOutputs(oper, control_outputs.data(),
                                control_outputs.size());
  return control_outputs;
}

std::vector<const char*> TF_OperationOutputConsumers_wrapper(
    TF_Output oper_out) {
  int num_consumers = TF_OperationOutputNumConsumers(oper_out);
  std::vector<TF_Input> consumers(num_consumers);
  TF_OperationOutputConsumers(oper_out, consumers.data(), num_consumers);

  std::vector<const char*> consumer_names(num_consumers);
  for (int i = 0; i < num_consumers; ++i) {
    consumer_names[i] = TF_OperationName(consumers[i].oper);
  }
  return consumer_names;
}

TF_Function* TF_GraphToFunction_wrapper(
    const TF_Graph* fn_body, const char* fn_name, bool append_hash_to_fn_name,
    const std::vector<TF_Operation*>* opers,
    const std::vector<TF_Output>& inputs, const std::vector<TF_Output>& outputs,
    const NameVector& output_names,
    const std::vector<TF_Operation*>* control_outputs,
    const NameVector& control_output_names, const TF_FunctionOptions* opts,
    const char* description, TF_Status* out_status) {
  if (!output_names.empty() && output_names.size() != outputs.size()) {
    Set_TF_Status_from_Status(
        out_status,
        errors::InvalidArgument(
            "output names must be either empty or equal in size to outputs. ",
            "output names size = ", output_names.size(),
            " outputs size = ", outputs.size()));
    return nullptr;
  }

  int nopers = -1;
  const TF_Operation* const* opers_array = nullptr;
  if (opers != nullptr) {
    nopers = opers->size();
    opers_array = opers->data();
  }

  const char** output_names_ptr =
      output_names.empty() ? nullptr
                           : const_cast<const char**>(output_names.data());

  const char** control_output_names_ptr =
      control_output_names.empty()
          ? nullptr
          : const_cast<const char**>(control_output_names.data());

  return TF_GraphToFunctionWithControlOutputs(
      fn_body, fn_name, append_hash_to_fn_name, nopers, opers_array,
      inputs.size(), inputs.data(), outputs.size(), outputs.data(),
      output_names_ptr,
      control_outputs == nullptr ? 0 : control_outputs->size(),
      control_outputs == nullptr ? nullptr : control_outputs->data(),
      control_output_names_ptr, opts, description, out_status);
}

void TF_GraphSetOutputHandleShapesAndTypes_wrapper(
    TF_Graph* graph, TF_Output output,
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<int>& ranks, const std::vector<TF_DataType>& types,
    TF_Status* status) {
  std::vector<const int64_t*> shapes_pointers(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    shapes_pointers[i] = ranks[i] <= 0 ? nullptr : &shapes[i][0];
  }
  TF_GraphSetOutputHandleShapesAndTypes(graph, output, shapes.size(),
                                        shapes_pointers.data(), ranks.data(),
                                        types.data(), status);
}

void CreatePlaceholder(TF_Graph* graph, TF_Status* s, string&& name,
                       TF_DataType dtype, TF_Output* output) {
  TF_OperationDescription* desc =
      TF_NewOperation(graph, "Placeholder", name.data());
  TF_SetAttrType(desc, "dtype", dtype);
  TF_Operation* op = TF_FinishOperation(desc, s);
  output->oper = op;
  output->index = 0;
}

std::vector<TF_Output> TF_CreatePlaceholders(TF_Graph* graph, PyObject* dtypes,
                                             const char* prefix,
                                             TF_Status* status) {
  std::vector<TF_Output> outputs;
  dtypes = PySequence_Fast(dtypes, "dtypes must be a sequence");
  if (dtypes == nullptr) {
    Set_TF_Status_from_Status(status, errors::Internal("dtypes is nullptr"));
    return outputs;
  }
  Safe_PyObjectPtr dtypes_holder(make_safe(dtypes));
  Py_ssize_t len = PySequence_Fast_GET_SIZE(dtypes);
  outputs.reserve(len);
  for (size_t i = 0; i < len; i++) {
    PyObject* dtype = PySequence_Fast_GET_ITEM(dtypes, i);
    if (!dtype) {
      Set_TF_Status_from_Status(status,
                                errors::Internal("Could not get dtype ", i));
      return outputs;
    }
#if PY_MAJOR_VERSION >= 3
    TF_DataType tf_datatype = static_cast<TF_DataType>(PyLong_AsLong(dtype));
#else
    TF_DataType tf_datatype = static_cast<TF_DataType>(PyInt_AsLong(dtype));
#endif
    outputs.push_back(TF_Output());
    CreatePlaceholder(graph, status, strings::StrCat(prefix, i), tf_datatype,
                      &outputs.back());
    if (!status->status.ok()) break;
  }
  return outputs;
}

void TF_GraphSetTensorShape_wrapper(TF_Graph* graph, TF_Output output,
                                    const std::vector<int64_t>& dims,
                                    bool unknown_shape, TF_Status* status) {
  if (unknown_shape) {
    TF_GraphSetTensorShape(graph, output, nullptr, -1, status);
    return;
  }
  TF_GraphSetTensorShape(graph, output, dims.data(), dims.size(), status);
}

std::vector<string> TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
    TF_ImportGraphDefResults* results) {
  int num_missing_unused_input_mappings;
  const char** src_names;
  int* src_indexes;
  TF_ImportGraphDefResultsMissingUnusedInputMappings(
      results, &num_missing_unused_input_mappings, &src_names, &src_indexes);
  std::vector<string> input_strs(num_missing_unused_input_mappings);
  for (int i = 0; i < num_missing_unused_input_mappings; ++i) {
    input_strs[i] = TensorId(src_names[i], src_indexes[i]).ToString();
  }
  return input_strs;
}

PyObject* TF_TryEvaluateConstant_wrapper(TF_Graph* graph, TF_Output output,
                                         TF_Status* status) {
  TF_Tensor* result_tensor;
  bool evaluated =
      TF_TryEvaluateConstant(graph, output, &result_tensor, status);
  if (!evaluated || TF_GetCode(status) != TF_OK) Py_RETURN_NONE;

  Safe_TF_TensorPtr safe_result_tensor(result_tensor);
  PyObject* out;
  Status s = TF_TensorToPyArray(std::move(safe_result_tensor), &out);
  Set_TF_Status_from_Status(status, s);
  if (!s.ok()) Py_RETURN_NONE;
  return PyArray_Return(reinterpret_cast<PyArrayObject*>(out));
}

}  // namespace tensorflow
