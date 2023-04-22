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
#ifndef TENSORFLOW_PYTHON_EAGER_PYWRAP_GRADIENT_EXCLUSIONS_H_
#define TENSORFLOW_PYTHON_EAGER_PYWRAP_GRADIENT_EXCLUSIONS_H_

#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

// Lookup whether the Op with the given op_name has unused input indices.
// Returns absl::nullopt if all inputs are used, set of unused indices
// otherwise. Empty set indicates that all indices are unused. The latter is
// necessary because sometimes it may not be possible to enumerate all indices
// just using OpDef e.g. when there are `list(T)` or `N * T` type inputs.
absl::optional<tensorflow::gtl::FlatSet<int>> OpGradientUnusedInputIndices(
    const tensorflow::string& op_name);

// Lookup whether the Op with the given op_name has unused output indices.
// Returns absl::nullopt if all outputs are used, set of unused indices
// otherwise. Empty set indicates that all indices are unused. The latter is
// necessary because sometimes it may not be possible to enumerate all indices
// just using OpDef e.g. when there are `list(T)` or `N * T` type outputs.
absl::optional<tensorflow::gtl::FlatSet<int>> OpGradientUnusedOutputIndices(
    const tensorflow::string& op_name);

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_GRADIENT_EXCLUSIONS_H_
