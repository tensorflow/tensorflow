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

#include "tensorflow/tsl/platform/setround.h"

#include "tensorflow/tsl/platform/logging.h"

namespace tsl {
namespace port {

#if defined(TF_BROKEN_CFENV)

ScopedSetRound::ScopedSetRound(const int mode) : original_mode_(mode) {
  // If cfenv usage is broken, assume support only for TONEAREST.
  DCHECK_EQ(mode, FE_TONEAREST);
}

ScopedSetRound::~ScopedSetRound() {}

#else

ScopedSetRound::ScopedSetRound(const int mode) {
  original_mode_ = std::fegetround();
  if (original_mode_ < 0) {
    // Failed to get current mode, assume ROUND TO NEAREST.
    original_mode_ = FE_TONEAREST;
  }
  std::fesetround(mode);
}

ScopedSetRound::~ScopedSetRound() { std::fesetround(original_mode_); }

#endif  // TF_BROKEN_CFENV

}  // namespace port
}  // namespace tsl
