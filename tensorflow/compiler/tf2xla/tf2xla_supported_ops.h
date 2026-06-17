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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_SUPPORTED_OPS_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_SUPPORTED_OPS_H_

namespace tensorflow {
namespace tf2xla {

// The implementation of a main function for a binary that prints a table of
// supported tf2xla operators for a given device, along with their type
// constraints, to stdout.
//
// Pass the argc and argv from main, unmodified.  Use regen_run to specify the
// command used to regenerate the table.
void SupportedOpsMain(int argc, char** argv, const char* regen_run);

}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_SUPPORTED_OPS_H_
