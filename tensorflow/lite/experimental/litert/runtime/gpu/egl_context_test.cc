// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/gpu/egl_context.h"

#include <gtest/gtest.h>

namespace {

TEST(EGLContextTest, CreateContext) {
  // Create EGL display and context.
  auto display = GetDisplay();
  ASSERT_TRUE(display);
  EXPECT_NE(display.Value(), EGL_NO_DISPLAY);

  auto context = CreateContext(display.Value(), EGL_NO_CONTEXT);
  ASSERT_TRUE(context);
  EXPECT_NE(context.Value(), EGL_NO_CONTEXT);
}

}  // namespace
