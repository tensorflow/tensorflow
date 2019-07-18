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

/*
 * bazel run -c opt --config cuda --dynamic_mode=off \
 * --define tf_use_oss_timeline_nonprod=1 \
 * third_party/tensorflow/core/profiler/internal:scoped_annotation_test \
 * -- --benchmarks=all
 */

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"

namespace tensorflow {
namespace {

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
  auto profiler_session =
      tensorflow::ProfilerSession::Create(/*ProfilerContext*/ nullptr);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation);
  }
  testing::StopTiming();
}

BENCHMARK(BM_ScopedAnnotationEnabled)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_TwoParts(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  auto profiler_session =
      tensorflow::ProfilerSession::Create(/*ProfilerContext*/ nullptr);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation, annotation);
  }
  testing::StopTiming();
}

BENCHMARK(BM_ScopedAnnotationEnabled_TwoParts)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Nested(int iters, int annotation_size) {
  testing::StopTiming();
  std::string annotation = GenerateRandomString(annotation_size);
  auto profiler_session =
      tensorflow::ProfilerSession::Create(/*ProfilerContext*/ nullptr);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    tracing::ScopedAnnotation trace(annotation);
    { tracing::ScopedAnnotation trace(annotation); }
  }
  testing::StopTiming();
}

BENCHMARK(BM_ScopedAnnotationEnabled_Nested)->Arg(8)->Arg(32)->Arg(128);

void BM_ScopedAnnotationEnabled_Adhoc(int iters, int annotation_size) {
  testing::StopTiming();
  auto profiler_session =
      tensorflow::ProfilerSession::Create(/*ProfilerContext*/ nullptr);
  testing::StartTiming();
  for (int i = 0; i < iters; i++) {
    // generate the annotation on the fly.
    tracing::ScopedAnnotation trace(absl::StrCat(i, "-", i * i));
  }
  testing::StopTiming();
}

BENCHMARK(BM_ScopedAnnotationEnabled_Adhoc)->Arg(8)->Arg(32)->Arg(128);

}  // namespace
}  // namespace tensorflow
