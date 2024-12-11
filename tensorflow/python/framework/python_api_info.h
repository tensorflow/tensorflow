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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_INFO_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_INFO_H_

#include <Python.h>

#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/framework/python_tensor_converter.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Precomputed information about a TensorFlow Python API.
//
// PythonAPIInfo records information about a single TensorFlow Python API,
// in order to allow calls to the API to be executed more efficiently.  This
// information includes:
//
// * The name of the API.  (E.g. "tf.math.add")
//
// * The name of the registered op that implements the API, if applicable
//   (e.g. "AddV2").
//
// * Information about the API's parameters.  Parameters are divided into two
//   "kinds": inputs and attributes.  An *input* is a parameter that
//   expects a Tensor or list of Tensors, and it is described by an `ArgDef`.
//   An *attribute* is a parameter that expects any other value type, and it is
//   described by an `AttrDef`.
//
// * Default values for the API's attribute parameters.
//
// * Information about "inferred attributes" -- attributes whose values are
//   inferred from `input` parameters.  There are two kinds of inferred
//   attributes: Tensor dtypes, which are inferred from tensor and list(tensor)
//   parameters; and list lengths, which are inferred from list(tensor)
//   parameters.
class PythonAPIInfo {
 public:
  // The index of a parameter in the canonicalized parameter list.  The
  // canonicalized parameter list includes inputs and attributes (but does
  // not include inferred attributes).  `-1` is used for inferred attributes.
  using ParamIndex = int;

  // Information about a parameter that expects a non-Tensor value.
  struct Attribute {
    ParamIndex index;  // -1 if this is an inferred attribute
    AttributeType type;
    const char* name;    // Interned python string
    int inferred_index;  // index to store attribute in InferredAttributes
  };

  // Information about a parameter that expects a Tensor or list(Tensor).
  // Additional information about tensor parameters is stored in types
  // defined below, in order to simplify dtype/length inference:
  //   * FixedDTypeInput: inputs with fixed dtypes.
  //   * InputsWithTypeAttr: groups inputs that use a type_attr for dtype.
  //   * InputsWithTypeListAttr: groups inputs that use a type_list_attr.
  //   * InputsWithNumberAttr: groups inputs by a number_attr for length.
  struct Input {
    ParamIndex index;
    bool is_list;
  };

  // Information about a Tensor parameter w/ fixed dtype.
  struct InputWithFixedDType {
    DataType dtype;
    ParamIndex index;
    bool is_list;
  };

  // Information about Tensor parameters whose DType is specified by a single
  // `type_attr` attribute.
  struct InputsWithTypeAttr {
    Attribute* type_attr;                        // not owned.
    DataType default_dtype;                      // DT_INVALID if no default.
    std::vector<ParamIndex> tensor_params;       // single-tensor inputs.
    std::vector<ParamIndex> tensor_list_params;  // list(tensor) inputs.
    std::vector<DataType> ok_dtypes;
  };

  // Information about Tensor parameters whose DType is specified by a single
  // `type_list_attr` attribute.
  struct InputsWithTypeListAttr {
    Attribute* type_list_attr;                   // not owned.
    std::vector<DataType> default_dtypes;        // empty if no default.
    std::vector<ParamIndex> tensor_list_params;  // list(tensor) inputs.
    std::vector<DataType> ok_dtypes;
  };

  // Information about Tensor-list parameters whose length is specified by a
  // single `int` attribute.
  struct InputsWithNumberAttr {
    Attribute* number_attr;                      // not owned.
    int64_t default_length;                      // -1 for no default.
    std::vector<ParamIndex> tensor_list_params;  // list(tensor) inputs.
  };

  // Structure used to return inferred attribute values.
  //   * types[i] is the inferred value for inferred_type_attrs()[i]
  //   * type_lists[i] is the inferred value for inferred_type_list_attrs()[i]
  //   * lengths[i] is the inferred value for inferred_length_attrs()[i]
  struct InferredAttributes {
    std::vector<DataType> types;
    std::vector<std::vector<DataType>> type_lists;
    std::vector<int64_t> lengths;
  };

  // Constructs a new PythonAPIInfo.
  //
  // Note: One of the `Initialize()` functions must be called before the
  // `PythonAPIInfo` is used.
  //
  // Args:
  //   api_name: The fully-qualified name of the python API (e.g., tf.math.sum).
  explicit PythonAPIInfo(const std::string& api_name);

  // Initializes this PythonAPIInfo.
  //
  // Args:
  //   op_def: Contains information about the parameters.
  //   param_names: The argument names for the python API, in canonical order.
  //   defaults_tuple: Tuple containing default values for the parameters,
  //     right-aligned with `param_names` -- i.e., `defaults[-i]` is the default
  //     for `param_names[-i]`.
  absl::Status Initialize(const OpDef& op_def,
                          const std::vector<string> param_names,
                          PyObject* defaults_tuple);

  // Initialize this PythonAPIInfo based on the registered OpDef for the given
  // operation.
  //
  // Args:
  //   op_name: The registered name of the operation (e.g. "AddV2").
  absl::Status InitializeFromRegisteredOp(const std::string& op_name);

  // Initializes this PythonAPIInfo based on a set of parameter specifications.
  //
  // Args:
  //   input_specs: Mapping from parameter name to specification string for
  //     each input (parameter that expects a tensor value).
  //   attr_specs: Mapping from parameter name to specification string for
  //     each attribute (parameter that expects a non-tensor value).
  //   param_names: The argument names for the python API, in canonical order.
  //   defaults_tuple: Tuple containing default values for the parameters,
  //     right-aligned with `param_names` -- i.e., `defaults[-i]` is the default
  //     for `param_names[-i]`.
  //
  // Note: the `name` parameter should not be included in `input_specs` or
  // `attr_specs`.
  absl::Status InitializeFromParamSpecs(
      const std::map<std::string, std::string>& input_specs,
      const std::map<std::string, std::string>& attr_specs,
      const std::vector<string> param_names, PyObject* defaults_tuple);

  // The name of the API that is described by this PythonAPIInfo.
  const char* api_name() const { return api_name_; }

  // The ordered names of the canononical parameters that this API expects.
  const std::vector<const char*>& param_names() const { return param_names_; }

  // A Python tuple containing the default values for parameters.  This is
  // right-aligned with `param_name` -- i.e., `defaults[-i]` is the default
  // for `param_names[-i]`.
  const PyObject* defaults_tuple() const { return defaults_tuple_.get(); }

  // Information about the attribute (non-tensor) parameters for this API.
  const std::vector<Attribute>& attributes() const { return attributes_; }

  // Information about the input (tensor) parameters for this API.
  const std::vector<Input>& inputs() const { return inputs_; }
  const std::vector<InputWithFixedDType>& inputs_with_fixed_dtype() const {
    return inputs_with_fixed_dtype_;
  }
  const std::vector<InputsWithTypeAttr>& inputs_with_type_attrs() const {
    return inputs_with_type_attrs_;
  }
  const std::vector<InputsWithTypeListAttr>& inputs_with_type_list_attrs()
      const {
    return inputs_with_type_list_attrs_;
  }
  const std::vector<InputsWithNumberAttr>& inputs_with_number_attrs() const {
    return inputs_with_number_attrs_;
  }

  // Names of inferred attributes.
  const std::vector<const char*>& inferred_type_attrs() const {
    return inferred_type_attrs_;
  }
  const std::vector<const char*>& inferred_type_list_attrs() const {
    return inferred_type_list_attrs_;
  }
  const std::vector<const char*>& inferred_length_attrs() const {
    return inferred_length_attrs_;
  }

  // Returns a string summarizing the internal state of this type converter.
  string DebugInfo() const;

 private:
  // Adds an entry to the attributes_ vector based on the given `AttrDef`.
  //
  // If `attr_def` describes a type attribute, then adds a value to
  // inputs_with_type_attrs_ or inputs_with_type_list_attrs_ (to record any
  // tensor inputs that use this dtype).
  //
  // If `attr_def` describes an int attribute, then adds a value to
  // inputs_with_number_attrs_ (to record any tensor inputs that use this
  // value as a list length).
  absl::Status InitializeAttribute(
      const OpDef::AttrDef& attr_def,
      const std::map<std::string, ParamIndex>& param_name_to_index);

  // Adds an entry to the inputs_ vector based on the given `ArgDef`.
  //
  // If `arg_def` has a fixed dtype, then adds a value to `fixed_dtype_inputs`.
  //
  // If `arg_def`'s dtype is described by a `type` attr, then updates the
  // appropriate value in `inputs_with_type_attrs_` with information about the
  // `arg_def`.
  //
  // If `arg_def`'s dtype is described by a `list(type)` attr, then updates the
  // appropriate value in `inputs_with_type_list_attrs_` with information about
  // the `arg_def`.
  absl::Status InitializeInput(
      const OpDef::ArgDef& arg_def,
      const std::map<std::string, int>& param_name_to_index);

  // Checks that the OpDef used to initialize this PythonAPIInfo
  // had an AttrDef or ArgDef specification for each parameter.
  absl::Status CheckParamNames() const;

  // Searches inputs_with_type_attrs_ for an input with the given name.
  InputsWithTypeAttr* FindInputsWithTypeAttr(const string& name);

  // Searches inputs_with_type_list_attrs_ for an input with the given name.
  InputsWithTypeListAttr* FindInputsWithTypeListAttr(const string& name);

  // Searches inputs_with_type_list_attrs_ for an input with the given name.
  InputsWithNumberAttr* FindInputsWithNumberAttr(const string& name);

  ABSL_MUST_USE_RESULT
  bool InferLengthAttributes(const absl::Span<PyObject*> params,
                             std::vector<int64_t>& inferred_length_attrs) const;

  // ==========================================================================
  // Member Variables
  // ==========================================================================

  // The name of the API that is described by this PythonAPIInfo.
  // (Interned python string).
  const char* api_name_;

  // The names of the parameters that this API expects.
  // (Interned python strings.)
  std::vector<const char*> param_names_;

  // Tuple containing default values for the parameters, right-aligned with
  // `param_names` -- i.e., `defaults[-i]` is the default for `param_names[-i]`.
  Safe_PyObjectPtr defaults_tuple_;

  // Information about the non-tensor-valued parameters that this API expects.
  std::vector<Attribute> attributes_;

  // Information about the tensor-valued parameters that this API expects.
  std::vector<Input> inputs_;
  std::vector<InputWithFixedDType> inputs_with_fixed_dtype_;
  std::vector<InputsWithTypeAttr> inputs_with_type_attrs_;
  std::vector<InputsWithTypeListAttr> inputs_with_type_list_attrs_;
  std::vector<InputsWithNumberAttr> inputs_with_number_attrs_;

  // Names of inferred attributes.  (Interned python strings.)
  std::vector<const char*> inferred_type_attrs_;
  std::vector<const char*> inferred_type_list_attrs_;
  std::vector<const char*> inferred_length_attrs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_INFO_H_
