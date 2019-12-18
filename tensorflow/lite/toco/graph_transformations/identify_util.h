/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_IDENTIFY_UTIL_H_
#define TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_IDENTIFY_UTIL_H_
#include <string>

#include "tensorflow/lite/toco/model.h"

namespace toco {

namespace util {

bool IsBinaryOp(
    const Operator* op, OperatorType optype,
    FusedActivationFunctionType act = FusedActivationFunctionType::kNone);

// Returns true if given array is a scalar and is val.
bool CheckArrayIsScalarFloat(Model* model, const std::string& name, float val);

// Returns index of scalar input that is equal to val, returns -1 otherwise.
int GetSingleScalarInputIndexOfBinaryOp(Model* model, const Operator* op,
                                        float val);
}  // namespace util
}  // namespace toco
#endif  // TENSORFLOW_LITE_TOCO_GRAPH_TRANSFORMATIONS_IDENTIFY_UTIL_H_
