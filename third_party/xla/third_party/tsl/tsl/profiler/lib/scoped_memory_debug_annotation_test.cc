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

#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

#include <cstddef>
#include <functional>
#include <string>

#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

TEST(ScopedMemoryDebugAnnotationTest, IndexedAndZeroArgShapeCallbacks) {
  EXPECT_EQ(ScopedMemoryDebugAnnotation::CurrentAnnotation().pending_shape_func(
                /*index=*/0),
            "");

  {
    ScopedMemoryDebugAnnotation batch_anno(
        "batch_op", /*step_id=*/7, "output", /*data_type=*/1,
        [](size_t index) { return absl::StrCat("shape_", index); });
    const MemoryDebugAnnotation& current =
        ScopedMemoryDebugAnnotation::CurrentAnnotation();
    EXPECT_EQ(current.pending_op_name, "batch_op");
    EXPECT_EQ(current.pending_step_id, 7);
    EXPECT_EQ(current.pending_region_type, "output");
    EXPECT_EQ(current.pending_data_type, 1);
    EXPECT_EQ(current.pending_shape_func(0), "shape_0");
    EXPECT_EQ(current.pending_shape_func(2), "shape_2");

    {
      ScopedMemoryDebugAnnotation single_anno("ignored_op", "temp",
                                              /*data_type=*/3,
                                              []() { return "scalar_shape"; });
      const MemoryDebugAnnotation& nested =
          ScopedMemoryDebugAnnotation::CurrentAnnotation();
      EXPECT_EQ(nested.pending_op_name, "batch_op");
      EXPECT_EQ(nested.pending_step_id, 7);
      EXPECT_EQ(nested.pending_region_type, "temp");
      EXPECT_EQ(nested.pending_data_type, 3);
      EXPECT_EQ(nested.pending_shape_func(0), "scalar_shape");
      EXPECT_EQ(nested.pending_shape_func(5), "scalar_shape");
    }

    EXPECT_EQ(
        ScopedMemoryDebugAnnotation::CurrentAnnotation().pending_shape_func(1),
        "shape_1");
  }

  EXPECT_EQ(ScopedMemoryDebugAnnotation::CurrentAnnotation().pending_shape_func(
                /*index=*/0),
            "");
}

TEST(ScopedMemoryDebugAnnotationTest, EmptyZeroArgStdFunctionPreservesNull) {
  std::function<std::string()> empty_fn;
  ScopedMemoryDebugAnnotation anno("op", "output", /*data_type=*/0,
                                   std::move(empty_fn));
  EXPECT_EQ(ScopedMemoryDebugAnnotation::CurrentAnnotation().pending_shape_func,
            nullptr);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
