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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool ApplyAttrsToArray(GraphTransformation* transformation, Model* model,
                       const FakeQuantOperator& fq_op,
                       const string& array_name) {
  bool changed = false;
  auto& annotated_array = model->GetArray(array_name);
  if (!annotated_array.minmax) {
    const MinMax& minmax = *fq_op.minmax;
    annotated_array.GetOrCreateMinMax() = minmax;
    transformation->AddMessageF(
        "Read min/max annotation for array %s: min=%g, max=%g", array_name,
        minmax.min, minmax.max);
    changed = true;
  }
  if (fq_op.narrow_range && !annotated_array.narrow_range) {
    annotated_array.narrow_range = true;
    transformation->AddMessageF("Read narrow_range annotation for array %s",
                                array_name);
    changed = true;
  }
  return changed;
}

}  // end namespace

::tensorflow::Status ReadArrayMinmaxAndNarrowRangeFromFakeQuant::Run(
    Model* model, std::size_t op_index, bool* modified) {
  *modified = false;
  const auto fakequant_it = model->operators.begin() + op_index;
  auto* fakequant_base_op = fakequant_it->get();
  if (fakequant_base_op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::Status::OK();
  }
  auto* fq_op = static_cast<FakeQuantOperator*>(fakequant_base_op);

  if (!fq_op->minmax) {
    // Need to be resolved first by ResolveFakeQuantArgsFromVars.
    return ::tensorflow::Status::OK();
  }

  // At this point, this FakeQuantOperator should have a MinMax
  // attached to it, and should only have 1 input (it should not have
  // 2nd and 3rd input arrays giving min and max anymore).
  CHECK(fq_op->minmax);
  CHECK_EQ(1, fq_op->inputs.size());

  bool changed = false;
  changed |= ApplyAttrsToArray(this, model, *fq_op, fq_op->inputs[0]);
  changed |= ApplyAttrsToArray(this, model, *fq_op, fq_op->outputs[0]);
  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
