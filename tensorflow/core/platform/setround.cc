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

#ifdef __STDC_IEC_559__
#include <fenv.h> // fesetround, FE_*
#endif

namespace tensorflow {
namespace port {

ScopedSetRound::ScopedSetRound() {
#ifdef __STDC_IEC_559__
   std::fesetround(FE_TONEAREST);
#endif
}

ScopedSetRound::~ScopedSetRound() {
}

}  // namespace port
}  // namespace tensorflow
