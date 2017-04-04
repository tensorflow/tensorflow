// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_STAMPED_RESOURCE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_STAMPED_RESOURCE_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace boosted_trees {

// A StampedResource is a resource that has a stamp token associated with it.
// Before reading from or applying updates to the resource, the stamp should
// be checked to verify that the update is not stale.
class StampedResource : public ResourceBase {
 public:
  StampedResource() : stamp_(-1) {}

  bool is_stamp_valid(int64 stamp) const { return stamp_ == stamp; }

  int64 stamp() const { return stamp_; }
  void set_stamp(int64 stamp) { stamp_ = stamp; }

 private:
  int64 stamp_;
};

}  // namespace boosted_trees
}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_STAMPED_RESOURCE_H_
