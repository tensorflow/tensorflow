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

#include <array>
#include <string_view>
namespace xla {

// Module level attributes require namespacing.
constexpr char kMhloCrossProgramPrefetches[] = "mhlo.cross_program_prefetches";
constexpr char kMhloInputOutputAlias[] = "mhlo.input_output_alias";
constexpr char kMhloIsDynamic[] = "mhlo.is_dynamic";
constexpr char kMhloLiteral[] = "mhlo.literal";
constexpr char kMhloReplication[] = "mhlo.is_same_data_across_replicas";
constexpr char kMhloSpmdOutputSharding[] = "mhlo.spmd_output_sharding";
constexpr char kMhloSpmdParametersShardings[] =
    "mhlo.spmd_parameters_shardings";
constexpr char kMhloUseAutoSpmdPartitioning[] =
    "mhlo.use_auto_spmd_partitioning";
constexpr char kMhloXlaEntryComputationParameterLayouts[] =
    "mhlo.xla_entry_computation_parameter_layouts";
constexpr char kMhloXlaEntryComputationParameterTiles[] =
    "mhlo.xla_entry_computation_parameter_tiles";
constexpr char kMhloXlaEntryComputationResultLayout[] =
    "mhlo.xla_entry_computation_result_layout";
constexpr char kMhloXlaEntryComputationResultTiles[] =
    "mhlo.xla_entry_computation_result_tiles";
constexpr char kMhloNumPartitions[] = "mhlo.num_partitions";
constexpr char kMhloNumReplicas[] = "mhlo.num_replicas";

// Function attributes.
constexpr char kExecutionThread[] = "execution_thread";
constexpr char kJaxBufferDonor[] = "jax.buffer_donor";

// Op attributes.
constexpr char kMhloFrontendAttributes[] = "mhlo.frontend_attributes";
constexpr char kMhloSharding[] = "mhlo.sharding";

bool IsKnownUnregisteredAttribute(std::string_view attr_name);

}  // namespace xla

#endif  // XLA_MLIR_HLO_UTILS_UNREGISTERED_ATTRIBUTES_H_
