/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/microfrontend/lib/memory_util.h"

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MemoryUtil_CheckAlloc) {
  TF_LITE_MICRO_EXPECT_NE(nullptr, microfrontend_alloc(256));
}

TF_LITE_MICRO_TEST(MemoryUtil_CheckFree) {
  void* ptr = microfrontend_alloc(128);
  TF_LITE_MICRO_EXPECT_NE(nullptr, ptr);
  microfrontend_free(ptr);
}

TF_LITE_MICRO_TESTS_END
