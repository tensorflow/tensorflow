/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_HISTOGRAM_HISTOGRAM_H_
#define TENSORFLOW_CORE_LIB_HISTOGRAM_HISTOGRAM_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/lib/histogram/histogram.h"

namespace tensorflow {

using tsl::HistogramProto;  // NOLINT

namespace histogram {

using tsl::histogram::Histogram;            // NOLINT
using tsl::histogram::ThreadSafeHistogram;  // NOLINT

}  // namespace histogram
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_HISTOGRAM_HISTOGRAM_H_
