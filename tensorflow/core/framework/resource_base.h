/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_BASE_H_

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

// Forward declaration to avoid introducing a dependency on headers in
// "tensorflow/core/graph/...".
class GraphDefBuilder;
class Node;

// This is the base class of all resource classes. Each resource must be
// represented as a sub-class of ResourceBase (which is reference counted) to be
// able to work with resource facilities such ResourceHandle and ResourceMgr.
class ResourceBase : public core::RefCounted {
 public:
  // Returns a debug string for *this.
  virtual std::string DebugString() const = 0;

  // Returns memory used by this resource.
  virtual int64_t MemoryUsed() const { return 0; }

  // Writes a representation of this resource into `builder`, so that executing
  // `*out` will recreate this resource.
  virtual Status AsGraphDef(GraphDefBuilder* builder, Node** out) const {
    return errors::Unimplemented("AsGraphDef not implemented for resource ",
                                 DebugString());
  }
};

}  //  end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RESOURCE_BASE_H_
