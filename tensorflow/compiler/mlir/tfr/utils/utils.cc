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

#include "tensorflow/compiler/mlir/tfr/utils/utils.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFR {

std::string GetComposeFuncName(StringRef tf_op_name) {
  std::string compose_func_name;
  for (int i = 0; i < tf_op_name.size(); ++i) {
    if (tf_op_name[i] == '_') {
      // The field name must not contain "_"s. "_Arg" and "_RetVal" are special
      // op names and we can return empty string to skip the decomposition.
      return {};
    }
    if (tf_op_name[i] == '.') {
      compose_func_name.push_back('_');
    } else if (tf_op_name[i] >= 'A' && tf_op_name[i] <= 'Z') {
      compose_func_name.push_back('_');
      compose_func_name.push_back(tf_op_name[i] + 'a' - 'A');
    } else {
      compose_func_name.push_back(tf_op_name[i]);
    }
  }
  return compose_func_name;
}

std::string GetTFOpName(StringRef compose_func_name) {
  std::string tf_op_name;
  bool after_underscore = false;
  for (int i = 0; i < compose_func_name.size(); ++i) {
    if (compose_func_name[i] >= 'A' && compose_func_name[i] <= 'Z') {
      // The field name must not contain uppercase letters.
      return {};
    }
    if (after_underscore) {
      if (compose_func_name[i] >= 'a' && compose_func_name[i] <= 'z') {
        tf_op_name.push_back(compose_func_name[i] + 'A' - 'a');
        after_underscore = false;
      } else {
        // The character after a "_" must be a lowercase letter.
        return {};
      }
    } else if (compose_func_name[i] == '_') {  // first time visit '_'
      if (i + 1 < compose_func_name.size() && compose_func_name[i + 1] == '_') {
        tf_op_name.push_back('.');
        i++;
      }
      after_underscore = true;
    } else {
      tf_op_name.push_back(compose_func_name[i]);
    }
  }
  if (after_underscore) {
    // Trailing "_".
    return {};
  }
  return tf_op_name;
}

}  // namespace TFR
}  // namespace mlir
