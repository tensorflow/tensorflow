/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RING_REDUCER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RING_REDUCER_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/ring_alg.h"
#include "tensorflow/core/framework/collective.h"

namespace tensorflow {
class Device;

// Ring-algorithm implementation of collective all-reduce.
class RingReducer : public RingAlg {
 public:
  RingReducer() : RingAlg(REDUCTION_COLLECTIVE, "Reduce") {}
  ~RingReducer() override;

  // Begins async execution of the ring reduce algorithm.
  // Must be called in a blockable thread.
  // TODO(b/80529858): remove the previous warning when we have a dedicated
  // collective threadpool.
  void Run(StatusCallback done) override;

  Status InitializeCollectiveParams(CollectiveParams* col_params) override;

 protected:
  void InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                     int field_idx) override;

 private:
  void ContinueAfterInputCopy();
  bool RunAsyncParts();

  Tensor group_size_tensor_;
  Notification group_size_tensor_ready_;

  friend class RingReducerTest;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RING_REDUCER_H_
