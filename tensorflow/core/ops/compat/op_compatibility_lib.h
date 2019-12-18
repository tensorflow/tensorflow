/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_OPS_COMPAT_OP_COMPATIBILITY_LIB_H_
#define TENSORFLOW_CORE_OPS_COMPAT_OP_COMPATIBILITY_LIB_H_

#include <set>

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpCompatibilityLib {
 public:
  // `ops_prefix` is a filename prefix indicating where to find the
  //   ops files.
  // `history_version` is used to construct the ops history file name.
  // `*stable_ops` has an optional list of ops that we care about.
  //   If stable_ops == nullptr, we use all registered ops.
  //   Otherwise ValidateCompatible() ignores ops not in *stable_ops
  //   and require all ops in *stable_ops to exist.
  OpCompatibilityLib(const string& ops_prefix, const string& history_version,
                     const std::set<string>* stable_ops);

  // Name of the file that contains the checked-in versions of *all*
  // ops, with docs.
  const string& ops_file() const { return ops_file_; }

  // Name of the file that contains all versions of *stable* ops,
  // without docs.  Op history is in (alphabetical, oldest-first)
  // order.
  const string& op_history_file() const { return op_history_file_; }

  // Name of the directory that contains all versions of *stable* ops,
  // without docs.  Op history is one file per op, in oldest-first
  // order within the file.
  const string& op_history_directory() const { return op_history_directory_; }

  // Should match the contents of ops_file().  Run before calling
  // ValidateCompatible().
  string OpsString() const { return op_list_.DebugString(); }

  // Returns the number of ops in OpsString(), includes all ops, not
  // just stable ops.
  int num_all_ops() const { return op_list_.op_size(); }

  // <file name, file contents> pairs representing op history.
  typedef std::vector<std::pair<string, OpList>> OpHistory;

  // Make sure the current version of the *stable* ops are compatible
  // with the historical versions, and if out_op_history != nullptr,
  // generate a new history adding all changed ops.  Sets
  // *changed_ops/*added_ops to the number of changed/added ops
  // (ignoring doc changes).
  Status ValidateCompatible(Env* env, int* changed_ops, int* added_ops,
                            OpHistory* out_op_history);

 private:
  const string ops_file_;
  const string op_history_file_;
  const string op_history_directory_;
  const std::set<string>* stable_ops_;
  OpList op_list_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_OPS_COMPAT_OP_COMPATIBILITY_LIB_H_
