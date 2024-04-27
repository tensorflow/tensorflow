/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/array.h"

#include <complex>
#include <random>

namespace xla {

// Specialization of FillRandom() method for complex64 type. Uses real part of
// the stddev parameter as the standard deviation value.
template <>
void Array<complex64>::FillRandom(const complex64& stddev, const double mean,
                                  const int seed) {
  std::mt19937 g(seed);
  std::normal_distribution<double> distribution(mean, std::real(stddev));
  for (int64_t i = 0; i < num_elements(); ++i) {
    values_[i] = complex64(distribution(g), distribution(g));
  }
}

}  // namespace xla
