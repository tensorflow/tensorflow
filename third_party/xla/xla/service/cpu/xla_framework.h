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

#ifndef XLA_SERVICE_CPU_XLA_FRAMEWORK_H_
#define XLA_SERVICE_CPU_XLA_FRAMEWORK_H_

#include <cstdint>
#include <vector>

#include "xla/service/cpu/xla_framework.pb.h"

namespace xla {
namespace cpu {

// Maps the descriptor table with inputs/outputs. Note that flattened_outputs
// and result are mutually exclusive -- see below.
//
// Contains the same info as "xla_framework" MLIR annotations. That is:
// - inputs: indices in the descriptor table of the input arguments.
// - output_is_tuple: if set, the output is a tuple.
// - flattened_outputs: if the output is a tuple, this contains the indices
//   (if any) in the descriptor table that correspond to the expanded tuple.
// - result: if the output is NOT a tuple, contains the index in the descriptor
//   table of the result.
struct XlaFrameworkMapping {
  std::vector<int64_t> inputs;
  std::vector<int64_t> flattened_outputs;
  int64_t result = -1;
  bool output_is_tuple = false;

  XlaFrameworkMappingProto ToProto() const {
    XlaFrameworkMappingProto proto;
    *proto.mutable_inputs() = {inputs.begin(), inputs.end()};
    *proto.mutable_flattened_outputs() = {flattened_outputs.begin(),
                                          flattened_outputs.end()};
    proto.set_result(result);
    proto.set_output_is_tuple(output_is_tuple);
    return proto;
  }

  void FromProto(const XlaFrameworkMappingProto& proto) {
    inputs = {proto.inputs().begin(), proto.inputs().end()};
    flattened_outputs = {proto.flattened_outputs().begin(),
                         proto.flattened_outputs().end()};
    result = proto.result();
    output_is_tuple = proto.output_is_tuple();
  }
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_XLA_FRAMEWORK_H_
