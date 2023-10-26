/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_ID_ASSIGNER_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_ID_ASSIGNER_H_

#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"

namespace tensorflow::quantization {

// An interface that assigns UUIDs to CustomAggregator ops.
class CustomAggregatorIdAssigner {
 public:
  virtual ~CustomAggregatorIdAssigner() = default;

  // Assigns UUIDs to each CustomAggregator op found in each GraphDef in
  // `exported_model`. The UUIDs are set to the `id` attributes. The UUIDs will
  // be used during calibration step to identify the collected quantization
  // statistics for each CustsomAggregator op.
  virtual ExportedModel AssignIds(
      const ExportedModel& exported_model) const = 0;
};

}  // namespace tensorflow::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_CALIBRATOR_ID_ASSIGNER_H_
