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
#ifndef THIRD_PARTY_TENSORFLOW_PYTHON_EAGER_PYTHON_EAGER_OP_GEN_H_
#define THIRD_PARTY_TENSORFLOW_PYTHON_EAGER_PYTHON_EAGER_OP_GEN_H_

#include <string>
#include <vector>
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// hidden_ops should be a list of Op names that should get a leading _
// in the output. Prints the output to stdout.
void PrintEagerPythonOps(const OpList& ops,
                         const std::vector<string>& hidden_ops,
                         bool require_shapes);

// Get the python wrappers for a list of ops in a OpList.
// `op_list_buf` should be a pointer to a buffer containing
// the binary encoded OpList proto, and `op_list_len` should be the
// length of that buffer.
string GetEagerPythonWrappers(const char* op_list_buf, size_t op_list_len);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_PYTHON_EAGER_PYTHON_EAGER_OP_GEN_H_
