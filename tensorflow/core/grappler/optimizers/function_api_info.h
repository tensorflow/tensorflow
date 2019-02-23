/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
class FunctionApiInfo {
 public:
  FunctionApiInfo();
  virtual ~FunctionApiInfo();

  enum FunctionType {
    INFERENCE,  // Default type.
    FORWARD,
    BACKWARD,
  };

  Status Init(const FunctionDef& function_def);

  const string& interface_name() const;
  const string& preferred_device() const;
  const FunctionType function_type() const;
  const string& pairing_function_name() const;
  const DataTypeVector& input_arg_dtypes() const;
  const DataTypeVector& output_arg_dtypes() const;

 private:
  string interface_name_;
  string preferred_device_;
  FunctionType function_type_;
  // The pairing function is used to pair between forward and backward function,
  // which will be useful during function swapping. Inference function won't
  // have pairing function.
  string pairing_function_name_;
  // The following two attributes are useful for forward and backward functions.
  DataTypeVector input_arg_dtypes_;
  DataTypeVector output_arg_dtypes_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionApiInfo);
};

// A collection of information for function and the interface it implements.
// A interface is a well defined math operation, eg I1 = 2 * x + y. Multiple
// functions could implement the same interface with different behavior based on
// different hardware condition and limits,
// eg F1 = math_ops.add(math_ops.add(x, x), y), or
//    F2 = math_ops.add(math_ops.matmul(x, 2), y).
class FunctionLibraryApiInfo {
 public:
  FunctionLibraryApiInfo();
  virtual ~FunctionLibraryApiInfo();
  // Populate the internal field for the functions within the function_library.
  Status Init(const FunctionDefLibrary& function_library);

  Status GetEquivalentImplementations(
      const string& function_name, std::vector<string>* other_functions) const;

  const FunctionApiInfo* GetApiInfo(const string& function_name) const;
  bool empty() const { return func_info_.empty(); }
  std::size_t size() const { return func_info_.size(); }

 private:
  // Map between function name to function details.
  std::unordered_map<string, std::unique_ptr<FunctionApiInfo>> func_info_;

  // Map between interface name to function names.
  // Forward/backward function pair usually have different signatures between
  // each other since forward function could produce extra internal state as
  // output, and backward will take those extra state as inputs.
  absl::flat_hash_map<string, std::vector<string>> intf_to_inference_funcs_;
  absl::flat_hash_map<string, std::vector<string>> intf_to_forward_funcs_;
  absl::flat_hash_map<string, std::vector<string>> intf_to_backward_funcs_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionLibraryApiInfo);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_
