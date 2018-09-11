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

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
class FunctionApiInfo {
 public:
  FunctionApiInfo();
  virtual ~FunctionApiInfo();

  Status Init(const FunctionDef& function_def);

  const string& interface_name() const;
  const string& preferred_device() const;

 private:
  string interface_name_;
  string preferred_device_;

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

  void GetEquivalentImplementations(const string& function_name,
                                    std::vector<string>* other_names) const;

  void GetBestImplementation(const string& function_name, const string& device,
                             string* best_func_name) const;

  const FunctionApiInfo* GetApiInfo(const string& function_name) const;

 private:
  // Map between function name to function details.
  std::unordered_map<string, std::unique_ptr<FunctionApiInfo>> func_info_;
  // Map between function name to interface name.
  std::unordered_map<string, string> func_to_intf_;
  // Map between interface name to function names.
  std::unordered_map<string, std::vector<string>> intf_to_funcs_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionLibraryApiInfo);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_
