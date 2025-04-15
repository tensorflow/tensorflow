/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_STATS_CALCULATOR_H_
#define TENSORFLOW_CORE_UTIL_STATS_CALCULATOR_H_

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "xla/tsl/util/stats_calculator.h"
#include "tensorflow/core/util/stat_summarizer_options.h"

namespace tensorflow {

using tsl::Stat;
using tsl::StatsCalculator;
using tsl::StatWithPercentiles;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STATS_CALCULATOR_H_
