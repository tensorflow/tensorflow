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

#include "tensorflow/core/platform/setround.h"


namespace tensorflow {
namespace port {

ScopedSetRound::ScopedSetRound(const int mode) {
  original_mode_ = std::fegetround();
  if (original_mode_ < 0) {
    // Failed to get current mode, assume ROUND TO NEAREST.
    original_mode_ = FE_TONEAREST;
  }
  std::fesetround(mode);
}

ScopedSetRound::~ScopedSetRound() { std::fesetround(original_mode_); }

}  // namespace port
}  // namespace tensorflow
