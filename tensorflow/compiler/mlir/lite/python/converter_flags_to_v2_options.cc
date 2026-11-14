/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/python/converter_flags_to_v2_options.h"

#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/converter_v2_options.pb.h"
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"

namespace tflite {

ConverterV2Options ToConverterV2Options(const ConverterFlags& converter_flags) {
  ConverterV2Options options;
  if (converter_flags.has_unfold_batchmatmul()) {
    options.set_unfold_batchmatmul(converter_flags.unfold_batchmatmul());
  }
  if (converter_flags.has_fold_fp16_casts_into_fully_connected()) {
    options.set_fold_fp16_casts_into_fully_connected(
        converter_flags.fold_fp16_casts_into_fully_connected());
  }
  if (converter_flags.has_enable_x64()) {
    options.set_enable_x64(converter_flags.enable_x64());
  }
  if (converter_flags.has_enable_debug()) {
    options.set_enable_debug(converter_flags.enable_debug());
  }
  if (converter_flags.has_debug_dir()) {
    options.set_debug_dir(converter_flags.debug_dir());
  }
  if (converter_flags.has_debug_options()) {
    *options.mutable_debug_options() = converter_flags.debug_options();
  }
  return options;
}

}  // namespace tflite
