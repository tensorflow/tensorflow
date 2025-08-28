/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_MLIR_HLO_UTILS_UNREGISTERED_ATTRIBUTES_H_
#define XLA_MLIR_HLO_UTILS_UNREGISTERED_ATTRIBUTES_H_

#include <string_view>

namespace xla {

// Module level attributes require namespacing.
inline constexpr char kMhloCrossProgramPrefetches[] =
    "mhlo.cross_program_prefetches";
inline constexpr char kMhloInputOutputAlias[] = "mhlo.input_output_alias";
inline constexpr char kMhloIsDynamic[] = "mhlo.is_dynamic";
inline constexpr char kMhloLiteral[] = "mhlo.literal";
inline constexpr char kMhloReplication[] = "mhlo.is_same_data_across_replicas";
inline constexpr char kMhloSpmdOutputSharding[] = "mhlo.spmd_output_sharding";
inline constexpr char kMhloSpmdParametersShardings[] =
    "mhlo.spmd_parameters_shardings";
inline constexpr char kMhloUseAutoSpmdPartitioning[] =
    "mhlo.use_auto_spmd_partitioning";
inline constexpr char kMhloXlaEntryComputationParameterLayouts[] =
    "mhlo.xla_entry_computation_parameter_layouts";
inline constexpr char kMhloXlaEntryComputationParameterTiles[] =
    "mhlo.xla_entry_computation_parameter_tiles";
inline constexpr char kMhloXlaEntryComputationResultLayout[] =
    "mhlo.xla_entry_computation_result_layout";
inline constexpr char kMhloXlaEntryComputationResultTiles[] =
    "mhlo.xla_entry_computation_result_tiles";
inline constexpr char kMhloNumPartitions[] = "mhlo.num_partitions";
inline constexpr char kMhloNumReplicas[] = "mhlo.num_replicas";

// Function attributes.
inline constexpr char kExecutionThread[] = "execution_thread";
inline constexpr char kJaxBufferDonor[] = "jax.buffer_donor";

// Op attributes.
inline constexpr char kMhloFrontendAttributes[] = "mhlo.frontend_attributes";
inline constexpr char kMhloSharding[] = "mhlo.sharding";

// Returns true if the given attribute name is a known XLA discardable module
// attribute.
bool IsKnownDiscardableModuleAttribute(std::string_view attr_name);

// Returns true if the given attribute name is a known XLA discardable function
// attribute.
bool IsKnownDiscardableFuncAttribute(std::string_view attr_name);

// Returns true if the given attribute name is a known XLA discardable op
// attribute, this applies to all ops except for module and func ops.
bool IsKnownDiscardableOpAttribute(std::string_view attr_name);

}  // namespace xla

#endif  // XLA_MLIR_HLO_UTILS_UNREGISTERED_ATTRIBUTES_H_
