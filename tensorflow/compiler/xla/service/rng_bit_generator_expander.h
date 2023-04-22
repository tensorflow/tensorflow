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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_RNG_BIT_GENERATOR_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_RNG_BIT_GENERATOR_EXPANDER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

class RngBitGeneratorExpander : public OpExpanderPass {
 public:
  explicit RngBitGeneratorExpander(RandomAlgorithm default_algorithm)
      : default_algorithm_(default_algorithm) {
    CHECK_NE(default_algorithm_, RandomAlgorithm::RNG_DEFAULT);
  }

  absl::string_view name() const override {
    return "rng-bit-generator-expander";
  }

 protected:
  struct RngGeneratorKey {
    Shape data_shape;
    Shape state_shape;
    RandomAlgorithm algorithm;
    HloModule* module;

    template <typename H>
    friend H AbslHashValue(H h, const RngGeneratorKey& c) {
      return H::combine(std::move(h), c.state_shape, c.data_shape, c.algorithm,
                        c.module);
    }

    bool operator==(const RngGeneratorKey& o) const {
      return data_shape == o.data_shape && state_shape == o.state_shape &&
             algorithm == o.algorithm && module == o.module;
    }
  };

  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  StatusOr<HloInstruction*> ExpandInstruction(HloInstruction* hlo) override;
  StatusOr<HloComputation*> GetGeneratorComputation(const Shape& data_shape,
                                                    const Shape& state_shape,
                                                    RandomAlgorithm algorithm,
                                                    HloModule* module);

  const RandomAlgorithm default_algorithm_;
  absl::flat_hash_map<RngGeneratorKey, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_RNG_BIT_GENERATOR_EXPANDER_H_
