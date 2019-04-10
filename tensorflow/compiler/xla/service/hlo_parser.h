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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PARSER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PARSER_H_

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_lexer.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// For details about the syntax accepted by this parser, see
// g3doc/hlo_parser.md.

// Given a string in the HloModule::ToString() format, parses the string and
// creates a HloModule with the given config.
StatusOr<std::unique_ptr<HloModule>> ParseHloString(
    absl::string_view str, const HloModuleConfig& config);

// Given a string in the HloModule::ToString() format, parses the string and
// builds the HloModule in place at the given module pointer. 'module' must
// point to an empty module (no computations).
Status ParseHloString(absl::string_view str, HloModule* module);

// Given a string in the HloModule::ToString() format, parses the string and
// creates a HloModule with default config.
StatusOr<std::unique_ptr<HloModule>> ParseHloString(absl::string_view str);

// Parses sharding from str. str is supposed to contain the body of the
// sharding, i.e. just the rhs of the "sharding={...}" attribute string, e.g.,
// "{replicated}".
StatusOr<HloSharding> ParseSharding(absl::string_view str);

// Parses parameter replication from str. str is supposed to contain the body of
// the parameter replication, i.e. just the rhs of the
// "parameter_replication={...}" attribute string, e.g., "{true, false}".
StatusOr<std::vector<bool>> ParseParameterReplication(absl::string_view str);

// Parses the result of window_util::ToString(const Window&).
StatusOr<Window> ParseWindow(absl::string_view str);

// Parses the result of ConvolutionDimensionNumbersToString(), e.g.
// "b0f_0io->b0f".
StatusOr<ConvolutionDimensionNumbers> ParseConvolutionDimensionNumbers(
    absl::string_view str);

// Parses the result of PaddingConfigToString(), e.g. "0_0x1_1".
StatusOr<PaddingConfig> ParsePaddingConfig(absl::string_view str);

// Parses and returns a Shape::ToString-format string.
StatusOr<Shape> ParseShape(absl::string_view str);

// Parses and returns a std::vector<ReplicaGroup> from str. str is supposed to
// contain a list of the replica groups, i.e. just the rhs of the
// "replica_groups={...}" attribute string, e.g., "{{0,1}, {2,3}}".
StatusOr<std::vector<ReplicaGroup>> ParseReplicaGroupsOnly(
    absl::string_view str);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PARSER_H_
