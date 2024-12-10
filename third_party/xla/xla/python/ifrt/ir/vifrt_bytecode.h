/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_BYTECODE_H_
#define XLA_PYTHON_IFRT_IR_VIFRT_BYTECODE_H_

namespace xla {
namespace ifrt {

class VifrtDialect;

// Add the interface necessary for encoding and decoding VIFRT dialect
// types and attributes in bytecode.
void addBytecodeInterface(VifrtDialect *dialect);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_BYTECODE_H_
