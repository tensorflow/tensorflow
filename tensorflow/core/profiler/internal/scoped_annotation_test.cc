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

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/annotation.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace {

TEST(ScopedAnnotation, Simple) {
  {
    tracing::ScopedAnnotation trace("blah");
    EXPECT_EQ(Annotation::CurrentAnnotation(), "");  // not enabled
  }

  {
    tracing::ScopedAnnotation::Enable(true);
    tracing::ScopedAnnotation trace("blah");
    EXPECT_EQ(Annotation::CurrentAnnotation(), "blah");  // enabled
    tracing::ScopedAnnotation::Enable(false);
  }
  {
    tracing::ScopedAnnotation::Enable(true);
    tracing::ScopedAnnotation outer("foo");
    tracing::ScopedAnnotation inner("bar");
    EXPECT_EQ(Annotation::CurrentAnnotation(), "foo::bar");  // enabled
    tracing::ScopedAnnotation::Enable(false);
  }

  EXPECT_EQ(Annotation::CurrentAnnotation(), "");  // not enabled
}

std::string GenerateRandomString(int length) {
  return std::string(length, 'a');
}

void BM_ScopedAnnotationDisabled(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation);
  }
  testing::StopTiming();
}

BENCHMARK(BM_ScopedAnnotationDisabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  tracing::ScopedAnnotation::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation);
  }
  testing::StopTiming();
  tracing::ScopedAnnotation::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_TwoParts(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  tracing::ScopedAnnotation::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation, annotation);
  }
  testing::StopTiming();
  tracing::ScopedAnnotation::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_TwoParts)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Nested(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  tracing::ScopedAnnotation::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation);
    { tracing::ScopedAnnotation trace(annotation); }
  }
  testing::StopTiming();
  tracing::ScopedAnnotation::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Nested)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Adhoc(int iters, int annotation_size) {
  testing::StopTiming();
  tracing::ScopedAnnotation::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    // generate the annotation on the fly.
    tracing::ScopedAnnotation trace(absl::StrCat(i, "-", i * i));
  }
  testing::StopTiming();
  tracing::ScopedAnnotation::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationDisabled_Lambda(int iters, int annotation_size) {
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(
        [&]() { return absl::StrCat(i, "-", i * i); });
  }
}

BENCHMARK(BM_ScopedAnnotationDisabled_Lambda)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Adhoc_Lambda(int iters, int annotation_size) {
  tracing::ScopedAnnotation::Enable(true);
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(
        [&]() { return absl::StrCat(i, "-", i * i); });
  }
  tracing::ScopedAnnotation::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc_Lambda)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_TwoPartsLambda(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  tracing::ScopedAnnotation::Enable(true);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(
        [&]() { return absl::StrCat(annotation, ":", annotation); });
  }
  testing::StopTiming();
  tracing::ScopedAnnotation::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_TwoPartsLambda)->Arg(8)->Arg(32)->Arg(128);

}  // namespace
}  // namespace tensorflow
