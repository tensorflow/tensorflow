
// Copyright 2025 Google LLC.
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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"

namespace {

using ::testing::HasSubstr;

TEST(AngleTest, CheckAngle) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
  ABSL_CHECK_OK(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env));

  EXPECT_THAT((const char *)glGetString(GL_VENDOR), HasSubstr("Google"));
  EXPECT_THAT((const char *)glGetString(GL_VERSION), HasSubstr("ANGLE"));
  EXPECT_THAT((const char *)glGetString(GL_RENDERER), HasSubstr("ANGLE"));
}

}  // namespace
