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

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/converter_v2_options.pb.h"
#include "tensorflow/compiler/mlir/lite/debug/debug_options.pb.h"

namespace tflite {
namespace {

TEST(ConverterFlagsToV2OptionsTest, DefaultFlagsGiveDefaultOptions) {
  ConverterFlags converter_flags;

  ConverterV2Options options = ToConverterV2Options(converter_flags);

  EXPECT_EQ(options.ByteSizeLong(), 0u);
  // The defaults of both protos agree.
  EXPECT_EQ(options.unfold_batchmatmul(), converter_flags.unfold_batchmatmul());
  EXPECT_EQ(options.fold_fp16_casts_into_fully_connected(),
            converter_flags.fold_fp16_casts_into_fully_connected());
  EXPECT_EQ(options.enable_x64(), converter_flags.enable_x64());
  EXPECT_EQ(options.enable_debug(), converter_flags.enable_debug());
  EXPECT_EQ(options.debug_dir(), converter_flags.debug_dir());
}

TEST(ConverterFlagsToV2OptionsTest, MapsSupportedFlags) {
  ConverterFlags converter_flags;
  converter_flags.set_unfold_batchmatmul(true);
  converter_flags.set_fold_fp16_casts_into_fully_connected(true);
  converter_flags.set_enable_x64(false);
  converter_flags.set_enable_debug(true);
  converter_flags.set_debug_dir("/tmp/debug");
  converter_flags.mutable_debug_options()->set_print_ir_before("FooPass");
  converter_flags.mutable_debug_options()->set_print_ir_after("BarPass");
  converter_flags.mutable_debug_options()->set_elide_elementsattrs_if_larger(
      16);
  converter_flags.mutable_debug_options()->set_enable_timing(true);
  converter_flags.mutable_debug_options()->set_ir_dump_dir("/tmp/dump");

  ConverterV2Options options = ToConverterV2Options(converter_flags);

  EXPECT_TRUE(options.unfold_batchmatmul());
  EXPECT_TRUE(options.fold_fp16_casts_into_fully_connected());
  EXPECT_FALSE(options.enable_x64());
  EXPECT_TRUE(options.enable_debug());
  EXPECT_EQ(options.debug_dir(), "/tmp/debug");
  EXPECT_EQ(options.debug_options().SerializeAsString(),
            converter_flags.debug_options().SerializeAsString());
}

TEST(ConverterFlagsToV2OptionsTest, IgnoresUnsupportedFlags) {
  ConverterFlags converter_flags;
  converter_flags.set_model_origin_framework(ConverterFlags::PYTORCH);
  converter_flags.set_fold_fp16_resource_casts(false);
  converter_flags.set_enable_composite_direct_lowering(false);
  converter_flags.set_strict_qdq_mode(false);
  converter_flags.set_unsafe_fuse_dynamic_shaped_broadcast(true);
  converter_flags.set_allow_custom_ops(true);

  ConverterV2Options options = ToConverterV2Options(converter_flags);

  EXPECT_EQ(options.ByteSizeLong(), 0u);
}

}  // namespace
}  // namespace tflite
