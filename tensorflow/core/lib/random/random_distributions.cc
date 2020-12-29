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

#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {
namespace random {
template <>
void SingleSampleAdapter<PhiloxRandom>::SkipFromGenerator(uint64 num_skips) {
  // Use the O(1) PhiloxRandom::Skip instead of the default O(N) impl.
  generator_->Skip(num_skips);
}
}  // namespace random
}  // namespace tensorflow
