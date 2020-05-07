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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_QUEUE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_QUEUE_H_

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

// Fusion configuration.
using FusionConfig = std::vector<std::vector<bool>>;

// Converts fusion config to string format.
static std::string FusionConfigToString(const FusionConfig& config) {
  std::string s;
  for (const auto& edge_list : config) {
    for (bool edge : edge_list) {
      absl::StrAppend(&s, edge ? "1" : "0");
    }
  }
  return s;
}

// A queue interface that allows implementations to choose fusion candidates in
// custom order.
class FusionQueue {
 public:
  FusionQueue() = default;
  virtual ~FusionQueue() = default;

  // Dequeues the next fusion candidates: a consumer and the list of producers
  // as operand indices.
  virtual std::pair<HloInstruction*, std::vector<int64>>
  DequeueNextInstructionAndOperandsToFuseInOrder() = 0;

  // A callback passed to the queue implementation right before the producer is
  // fused into the consumer.
  virtual void PreFusion(HloInstruction* producer, HloInstruction* consumer) {}

  // A callback passed to the queue implementation right after the fusion is
  // created. Note that original_producer could have been destroyed.
  virtual void OnFusingInstruction(HloInstruction* fusion,
                                   HloInstruction* original_producer,
                                   HloInstruction* original_consumer) {}

  // A callback passed to the queue implementation to notify the removal of an
  // instruction.
  virtual void RemoveInstruction(HloInstruction* instruction) = 0;

  // Returns the fusion configuration.
  virtual const std::vector<bool>* FusionConfiguration() = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_QUEUE_H_
