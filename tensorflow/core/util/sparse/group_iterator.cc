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

#include "tensorflow/core/util/sparse/group_iterator.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
namespace tensorflow {
namespace sparse {

void GroupIterable::IteratorStep::UpdateEndOfGroup() {
  ++next_loc_;
  const auto& ix_t = iter_->ix_matrix_;
  const int64_t N = ix_t.dimension(0);
  while (next_loc_ < N && iter_->GroupMatches(ix_t, loc_, next_loc_)) {
    ++next_loc_;
  }
}

bool GroupIterable::IteratorStep::operator!=(const IteratorStep& rhs) const {
  CHECK_EQ(rhs.iter_, iter_) << "Can't compare steps from different iterators";
  return (rhs.loc_ != loc_);
}

bool GroupIterable::IteratorStep::operator==(const IteratorStep& rhs) const {
  CHECK_EQ(rhs.iter_, iter_) << "Can't compare steps from different iterators";
  return (rhs.loc_ == loc_);
}

GroupIterable::IteratorStep& GroupIterable::IteratorStep::
operator++() {  // prefix ++
  loc_ = next_loc_;
  UpdateEndOfGroup();
  return *this;
}

GroupIterable::IteratorStep GroupIterable::IteratorStep::operator++(
    int) {  // postfix ++
  IteratorStep lhs(*this);
  ++(*this);
  return lhs;
}

std::vector<int64_t> Group::group() const {
  std::vector<int64_t> g;
  const auto& ix_t = iter_->ix_matrix_;
  for (const int d : iter_->group_dims_) {
    g.push_back(ix_t(loc_, d));
  }
  return g;
}

TTypes<int64_t>::UnalignedConstMatrix Group::indices() const {
  return TTypes<int64_t>::UnalignedConstMatrix(&(iter_->ix_matrix_(loc_, 0)),
                                               next_loc_ - loc_, iter_->dims_);
}

}  // namespace sparse
}  // namespace tensorflow
