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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

/// \brief A generator of Java operation wrappers.
///
/// Such generator is normally ran only once per executable, outputting
/// wrappers for the ops library it has been linked with. Nonetheless,
/// it is designed to support multiple runs, giving a different list of
/// operations on each cycle.
class OpGenerator {
 public:
  /// \brief Create a new generator, giving an environment and an
  /// output directory path.
  explicit OpGenerator(Env* env, const string& output_dir);
  virtual ~OpGenerator();

  /// \brief Generates wrappers for the given list of 'ops'.
  ///
  /// The list of operations should be issued from the library whose
  /// file name starts with 'ops_file' (see /core/ops/*.cc).
  ///
  /// Generated files are output under this directory:
  ///   <output_dir>/src/main/java/org/tensorflow/java/op/<group>
  /// where
  ///   'output_dir' is the directory passed in the constructor and
  ///   'group' is extracted from the 'ops_file' name
  Status Run(const string& ops_file, const OpList& ops);

 private:
  Env* env;
  const string output_path;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_GENERATOR_H_
