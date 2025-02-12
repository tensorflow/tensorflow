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
#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

template <ArrayDataType Type>
bool ComputeRandomUniformArray(Model* model, RandomUniformOperator* op) {
  typedef tensorflow::random::UniformDistribution<
      tensorflow::random::PhiloxRandom, DataType<Type>>
      Distribution;

  // Allocate output
  auto& output_array = model->GetArray(op->outputs[0]);
  CHECK(output_array.data_type == Type);
  std::vector<DataType<Type>>& data =
      output_array.GetMutableBuffer<Type>().data;
  data.resize(RequiredBufferSizeForShape(output_array.shape()));

  // We use the same random number generator and distribution as TensorFlow to
  // produce the exact same values given the same seeds. See
  // tensorflow::functor::FillPhiloxRandomTask<Distribution, false> in
  // //tensorflow/core/kernels/random_op.cc for the implementation.
  tensorflow::random::PhiloxRandom generator(op->seed, op->seed2);
  Distribution dist;

  // The generator creates Distribution::kResultElementCount samples at a time.
  size_t offset = 0;
  size_t num_samples = Distribution::kResultElementCount;
  while (offset < data.size()) {
    const typename Distribution::ResultType samples = dist(&generator);
    std::copy(&samples[0],
              &samples[0] + std::min(num_samples, data.size() - offset),
              &data[0] + offset);
    offset += num_samples;
  }

  return true;
}

::tensorflow::Status ResolveConstantRandomUniform::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* base_op = it->get();
  if (base_op->type != OperatorType::kRandomUniform) {
    return absl::OkStatus();
  }
  auto* op = static_cast<RandomUniformOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return absl::OkStatus();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return absl::OkStatus();
  }

  if ((op->seed == 0) && (op->seed2 == 0)) {
    LOG(WARNING) << "RandomUniform op outputting \"" << op->outputs[0]
                 << "\" is truly random (using /dev/random system entropy). "
                    "Therefore, cannot resolve as constant. Set \"seed\" or "
                    "\"seed2\" attr non-zero to fix this";
    return absl::OkStatus();
  }

  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      if (!ComputeRandomUniformArray<ArrayDataType::kFloat>(model, op)) {
        return absl::OkStatus();
      }
      break;
    // For future support of double or half.
    // case ArrayDataType::kDouble...
    default:
      LOG(FATAL)
          << "Unsupported data type given to RandomUniform op with output \""
          << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
