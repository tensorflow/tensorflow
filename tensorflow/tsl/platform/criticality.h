/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_CRITICALITY_H_
#define TENSORFLOW_TSL_PLATFORM_CRITICALITY_H_

#include "tensorflow/tsl/platform/platform.h"

namespace tsl {

namespace criticality {

enum class Criticality {
  // Frequent full and paritial unavailability is expected and not a cause for
  // concern.
  kSheddable = 0,
  // Partial unavailability is expected and not necessarily a cause for concern.
  kSheddablePlus = 1,
  // Any outage is a serious concern. This is the default priority for RPCs
  // sent from production jobs.
  kCritical = 2,
  // Any outage is a serious concern.  Less than 50% of requests to a service
  // can be in this band. During an outage, this band will be prioritized above
  // all others.
  kCriticalPlus = 3,
};

}  // namespace criticality

}  // namespace tsl

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/tsl/platform/google/criticality.h"  // IWYU pragma: export
#else
#include "tensorflow/tsl/platform/default/criticality.h"  // IWYU pragma: export
#endif

#endif  // TENSORFLOW_TSL_PLATFORM_CRITICALITY_H_
