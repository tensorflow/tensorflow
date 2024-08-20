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

#include "pybind11/pybind11.h"  // from @pybind11
#include "stablehlo/integrations/python/PortableApi.h"  // from @stablehlo

namespace mlir {
namespace stablehlo {

PYBIND11_MODULE(stablehlo_extension, m) { mlir::stablehlo::AddPortableApi(m); }

}  // namespace stablehlo
}  // namespace mlir
