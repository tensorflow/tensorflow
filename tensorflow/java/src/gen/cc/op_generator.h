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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_OP_GENERATOR_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_OP_GENERATOR_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/java/src/gen/cc/op_specs.h"

namespace tensorflow {
namespace java {

// A generator of Java operation wrappers.
//
// This generator takes a list of ops definitions in input and outputs
// a Java Op wrapper for each of them in the provided directory. The same
// generator instance can be invoked multiple times with a different list of
// ops definitions.
class OpGenerator {
 public:
  explicit OpGenerator(const std::vector<string>& api_dirs,
                       Env* env = Env::Default())
      : api_dirs_(api_dirs), env_(env) {}

  // Generates wrappers for the given list of 'ops'.
  //
  // Output files are generated in <output_dir>/<base_package>/<op_package>,
  // where 'op_package' is derived from ops endpoints.
  Status Run(const OpList& op_list, const string& base_package,
             const string& output_dir);

 private:
  const std::vector<string> api_dirs_;
  Env* env_;
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_GENERATOR_H_
