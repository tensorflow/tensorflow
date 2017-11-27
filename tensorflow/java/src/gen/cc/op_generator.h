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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

/// \brief A generator of Java operation wrappers.
///
/// Such generator is normally ran only once per executable, outputting
/// wrappers for the all registered operations it has been compiled with.
/// Nonetheless, it is designed to support multiple runs, giving a different
/// list of operations on each cycle.
class OpGenerator {
 public:
  OpGenerator();
  virtual ~OpGenerator();

  /// \brief Generates wrappers for the given list of 'ops'.
  ///
  /// Output files are generated in <output_dir>/<base_package>/<lib_package>,
  /// where 'lib_package' is derived from 'lib_name'.
  Status Run(const OpList& ops, const string& lib_name,
             const string& base_package, const string& output_dir);

 private:
  Env* env;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_GENERATOR_H_
