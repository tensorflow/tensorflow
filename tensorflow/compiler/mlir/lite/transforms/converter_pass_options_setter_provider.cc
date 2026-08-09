/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/converter_pass_options_setter.h"

#include <cstdint>

#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/variable_freezing_pipeline_options.h"

// This file is intentionally tiny and never called at runtime. Its sole
// purpose is to force the linker to retain the `ConverterPassOptionsSetter`
// vtable and its `SetOptions` overloads (defined in
// `converter_pass_options_setter.cc`) when this library is linked into a
// shared library / DLL.
//
// On Windows, MSVC lazily drops object files from static libraries if their
// symbols are not directly referenced from the final link. The
// `_pywrap_tensorflow_common.dll` target references the
// `ConverterPassOptionsSetter` vtable from `tf_tfl_passes.obj`, but the
// object file that *defines* the vtable and its `SetOptions` methods is a
// transitive, non-`alwayslink` dependency, so it is stripped at link time.
// That produces LNK2019 (unresolved external symbol).
//
// This provider target is marked `alwayslink = 1` and is added to the
// `_pywrap_tensorflow` deps (Windows-only). By constructing a
// `ConverterPassOptionsSetter` and invoking every virtual `SetOptions`
// overload, the vtable and all its method definitions in
// `converter_pass_options_setter.cc` are anchored, so they are always present
// in the final DLL.

namespace mlir {
namespace TFL {
namespace {

[[maybe_unused]] uint64_t
AnchorConverterPassOptionsSetterSymbols(const tflite::ConverterFlags& flags) {
  QuantizationSpecs quant_specs;
  PassConfig pass_config(quant_specs);
  ConverterPassOptionsSetter setter(flags, pass_config);

  OptimizePassOptions optimize_options;
  setter.SetOptions(optimize_options);

  VariableFreezingPipelineOptions variable_freezing_options;
  setter.SetOptions(variable_freezing_options);

  EmptyPassOptions empty_options;
  setter.SetOptions(empty_options);

  OptimizeBroadcastLikePassOptions broadcast_like_options;
  setter.SetOptions(broadcast_like_options);

  // Return a value derived from the object to prevent the compiler from
  // optimizing away the calls above.
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&setter));
}

}  // namespace
}  // namespace TFL
}  // namespace mlir
