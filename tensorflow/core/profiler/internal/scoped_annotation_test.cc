/* Copyright 2019 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/lib/scoped_annotation.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/profiler/internal/annotation_stack.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(ScopedAnnotation, Simple) {
  {
    ScopedAnnotation trace("blah");
    EXPECT_EQ(AnnotationStack::Get(), "");  // not enabled
  }

  {
    AnnotationStack::Enable(true);
    ScopedAnnotation trace("blah");
    EXPECT_EQ(AnnotationStack::Get(), "blah");  // enabled
    AnnotationStack::Enable(false);
  }
  {
    AnnotationStack::Enable(true);
    ScopedAnnotation outer("foo");
    ScopedAnnotation inner("bar");
    EXPECT_EQ(AnnotationStack::Get(), "foo::bar");  // enabled
    AnnotationStack::Enable(false);
  }

  EXPECT_EQ(AnnotationStack::Get(), "");  // not enabled
}

std::string GenerateRandomString(int length) {
  return std::string(length, 'a');
}

void BM_ScopedAnnotationDisabled(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    ScopedAnnotation trace(annotation);
  }
  testing::StopTiming();
}

BENCHMARK(BM_ScopedAnnotationDisabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  AnnotationStack::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    ScopedAnnotation trace(annotation);
  }
  testing::StopTiming();
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Nested(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  AnnotationStack::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    ScopedAnnotation trace(annotation);
    { ScopedAnnotation trace(annotation); }
  }
  testing::StopTiming();
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Nested)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Adhoc(int iters, int annotation_size) {
  testing::StopTiming();
  AnnotationStack::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    // generate the annotation on the fly.
    ScopedAnnotation trace(absl::StrCat(i, "-", i * i));
  }
  testing::StopTiming();
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationDisabled_Lambda(int iters, int annotation_size) {
  for (int i = 0; i < iters; i++) {
    ScopedAnnotation trace([&]() { return absl::StrCat(i, "-", i * i); });
  }
}

BENCHMARK(BM_ScopedAnnotationDisabled_Lambda)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Adhoc_Lambda(int iters, int annotation_size) {
  AnnotationStack::Enable(true);
  for (int i = 0; i < iters; i++) {
    ScopedAnnotation trace([&]() { return absl::StrCat(i, "-", i * i); });
  }
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc_Lambda)->Arg(8)->Arg(32)->Arg(128);

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
