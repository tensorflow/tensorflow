/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_UTILS_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Returns the num_replicas, num_partitions and device assignment given a
// ExecutableBuildOptions and whether we want a portable executable.
Status ParseDeviceAssignmentCompileOptions(
    bool compile_portable_executable, ExecutableBuildOptions* build_options,
    std::function<StatusOr<DeviceAssignment>(int, int)>
        GetDefaultDeviceAssignmentFunction,
    int* num_replicas, int* num_partitions,
    std::shared_ptr<DeviceAssignment>* device_assignment);

// Returns pointers to the argument layouts given an XlaComputation and
// ExecutableBuildOptions.
Status DetermineArgumentLayoutsFromCompileOptions(
    const XlaComputation& computation,
    std::function<StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    std::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers);

// Executables can donate buffers so that buffers can be aliased from inputs
// to outputs. This function returns a sorted vector of parameters that must be
// donated when executable is run. tuple_inputs reflects the option that
// executable was compiled with.
StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    const HloModule& hlo_module, bool tuple_inputs);

// Return max parallelism level.
int DefaultThreadPoolSize();

// Returns true if the striding of an array corresponds to a major-to-minor
// layout.
bool HasMajorToMinorLayout(PrimitiveType type, absl::Span<int64_t const> dims,
                           absl::Span<int64_t const> byte_strides);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_UTILS_H_
