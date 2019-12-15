/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_ROBUST_STATS_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_ROBUST_STATS_H_

#include <vector>
namespace tensorflow {
namespace grappler {
class RobustStats {
 public:
  explicit RobustStats(const std::vector<double>& values);
  explicit RobustStats(std::vector<double>&& values);

  double lo() const { return lo_; }
  double hi() const { return hi_; }
  double mean() const { return mean_; }

 private:
  void HuberMAD(const std::vector<double>& values);

  double lo_;
  double hi_;
  double mean_;
  double stddev_;
};
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_ROBUST_STATS_H_
