/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALL_REDUCE_BLUECONNECT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALL_REDUCE_BLUECONNECT_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Decomposes all-reduce operations using the BlueConnect algorithm.
//
// Paper: "BLUECONNECT: DECOMPOSING ALL-REDUCE FOR DEEP LEARNING ON
// HETEROGENEOUS NETWORK HIERARCHY"
// https://mlsys.org/Conferences/2019/doc/2019/130.pdf
//
// This algorithm attempts to minimize the number of levels of network hierarchy
// traversed for as much data transfer as possible. This implementation assumes
// that host IDs are ordered corresponding to network hierarchy.
class AllReduceBlueConnect : public HloModulePass {
 public:
  explicit AllReduceBlueConnect(size_t num_devices_per_host)
      : num_devices_per_host_(num_devices_per_host) {}

  absl::string_view name() const override { return "all-reduce-blueconnect"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  size_t num_devices_per_host_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALL_REDUCE_BLUECONNECT_H_
