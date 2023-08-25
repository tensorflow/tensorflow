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

#include "tensorflow/tsl/profiler/lib/scoped_annotation.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"
#include "tensorflow/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tensorflow/tsl/profiler/lib/scoped_annotation_stack.h"

namespace tsl {
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

  {
    AnnotationStack::Enable(true);
    int64_t id0 = ScopedAnnotationStack::ActivityStart("foo");
    int64_t id1 = ScopedAnnotationStack::ActivityStart("bar");
    EXPECT_EQ(AnnotationStack::Get(), "foo::bar");  // enabled
    ScopedAnnotationStack::ActivityEnd(id1);
    ScopedAnnotationStack::ActivityEnd(id0);
    AnnotationStack::Enable(false);
  }

  EXPECT_EQ(AnnotationStack::Get(), "");  // not enabled
}

std::string GenerateRandomString(int length) {
  return std::string(length, 'a');
}

void BM_ScopedAnnotationDisabled(::testing::benchmark::State& state) {
  const int annotation_size = state.range(0);

  std::string annotation = GenerateRandomString(annotation_size);
  for (auto s : state) {
    ScopedAnnotation trace(annotation);
  }
}

BENCHMARK(BM_ScopedAnnotationDisabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled(::testing::benchmark::State& state) {
  const int annotation_size = state.range(0);

  std::string annotation = GenerateRandomString(annotation_size);
  AnnotationStack::Enable(true);
  for (auto s : state) {
    ScopedAnnotation trace(annotation);
  }
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Nested(::testing::benchmark::State& state) {
  const int annotation_size = state.range(0);

  std::string annotation = GenerateRandomString(annotation_size);
  AnnotationStack::Enable(true);
  for (auto s : state) {
    ScopedAnnotation trace(annotation);
    { ScopedAnnotation trace(annotation); }
  }
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Nested)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Adhoc(::testing::benchmark::State& state) {
  AnnotationStack::Enable(true);
  int i = 0;
  for (auto s : state) {
    // generate the annotation on the fly.
    ScopedAnnotation trace(absl::StrCat(i, "-", i * i));
    ++i;
  }
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc);

void BM_ScopedAnnotationDisabled_Lambda(::testing::benchmark::State& state) {
  int i = 0;
  for (auto s : state) {
    ScopedAnnotation trace([&]() { return absl::StrCat(i, "-", i * i); });
    ++i;
  }
}

BENCHMARK(BM_ScopedAnnotationDisabled_Lambda);

void BM_ScopedAnnotationEnabled_Adhoc_Lambda(
    ::testing::benchmark::State& state) {
  AnnotationStack::Enable(true);
  int i = 0;
  for (auto s : state) {
    ScopedAnnotation trace([&]() { return absl::StrCat(i, "-", i * i); });
    ++i;
  }
  AnnotationStack::Enable(false);
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc_Lambda);

}  // namespace
}  // namespace profiler
}  // namespace tsl
