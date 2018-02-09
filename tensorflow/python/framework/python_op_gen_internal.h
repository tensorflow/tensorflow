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

#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_INTERNAL_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_INTERNAL_H_

#include <unordered_map>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace python_op_gen_internal {

// Returns true if s is a Python keyword or built-in.
bool IsPythonReserved(const string& s);

// Add a _ to the end of s if necessary to avoid a Python keyword or built-in.
string AvoidPythonReserved(const string& s);

// Convert an AttrValue with type `type` to the Python representation for
// that value.
string AttrValueToPython(const string& type, const AttrValue& value,
                         const string& dtype_module = "tf.");

void GenerateLowerCaseOpName(const string& str, string* result);

string DataTypeToPython(DataType dtype, const string& dtype_module);

// Names that corresponds to a single input parameter.
class ParamNames {
 public:
  // Create param based on Arg.
  ParamNames(const string& name, const string& rename_to) : name_(name) {
    rename_to_ = AvoidPythonReserved(rename_to);
  }

  // Get original parameter name.
  string GetName() const { return name_; }

  // Get the name to rename the parameter to. Note that AvoidPythonReserved
  // has already been applied.
  string GetRenameTo() const { return rename_to_; }

 private:
  // Original parameter name.
  string name_;
  // API name for this parameter.
  string rename_to_;
};

class GenPythonOp {
 public:
  GenPythonOp(const OpDef& op_def, const ApiDef& api_def,
              const string& function_name);
  virtual ~GenPythonOp();

  virtual string Code();

 protected:
  // Print: def Function(parameters):
  void AddDefLine(const string& parameters);

  // Format the Op's descriptions so that it can be a Python docstring.
  void AddDocStringDescription();

  void AddDocStringArgs();
  void AddDocStringInputs();
  void AddDocStringAttrs();
  void AddDocStringNameArg();
  void AddOutputGlobals();
  void AddDocStringOutputs();
  void AddBody(const string& prefix);
  void AddBodyNoReturn(const string& apply_prefix);
  void AddExport();

  // From constructor arguments
  const OpDef& op_def_;
  const ApiDef& api_def_;
  const string function_name_;
  const int num_outs_;

  // Return value from Code() is prelude_ + result_.
  string prelude_;  // Code before function definition
  string result_;   // Function definition

  // Map from attr name to the first input arg it is inferred from
  std::unordered_map<string, string> inferred_attrs_;

  // The names of the non-inferred attrs, in parameter order
  std::vector<string> attrs_;

  // All parameters, including inputs & non-inferred attrs, required and those
  // with defaults, except "name"
  std::vector<ParamNames> param_names_;
};

}  // namespace python_op_gen_internal
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_INTERNAL_H_
