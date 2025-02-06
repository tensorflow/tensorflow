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

#ifndef TENSORFLOW_CORE_GRAPH_TENSOR_ID_H_
#define TENSORFLOW_CORE_GRAPH_TENSOR_ID_H_

#include <string>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

struct SafeTensorId;

// Identifier for a tensor within a step.
// first == operation_name, second == output_index
// Note: does not own backing storage for name.
struct TensorId : public std::pair<absl::string_view, int> {
  typedef std::pair<absl::string_view, int> Base;

  // Inherit the set of constructors.
  using Base::pair;

  // NOTE(skyewm): this is required on some platforms. I'm not sure why the
  // using statement above isn't always sufficient.
  TensorId() : Base() {}
  TensorId(const SafeTensorId& id);

  const absl::string_view node() const { return first; }
  int index() const { return second; }

  string ToString() const {
    if (second == Graph::kControlSlot) return strings::StrCat("^", first);
    return strings::StrCat(first, ":", second);
  }

  struct Hasher {
   public:
    std::size_t operator()(const TensorId& x) const {
      return Hash32(x.first.data(), x.first.size(), x.second);
    }
  };
};

TensorId ParseTensorName(const string& name);
TensorId ParseTensorName(absl::string_view name);

bool IsTensorIdControl(const TensorId& tensor_id);

// Same as TensorId, except owns the backing storage for the op name. This makes
// the memory management simpler at the expense of a copy.
struct SafeTensorId : public std::pair<string, int> {
  typedef std::pair<string, int> Base;

  // NOTE(skyewm): this is required on some platforms. I'm not sure why the
  // using "using Base::pair;" isn't always sufficient.
  SafeTensorId() : Base() {}
  SafeTensorId(const string& str, int idx) : Base(str, idx) {}
  SafeTensorId(const TensorId& id);

  const string& node() const { return first; }
  int index() const { return second; }

  string ToString() const {
    if (second == Graph::kControlSlot) return strings::StrCat("^", first);
    return strings::StrCat(first, ":", second);
  }

  struct Hasher {
   public:
    std::size_t operator()(const TensorId& x) const {
      return Hash32(x.first.data(), x.first.size(), x.second);
    }
  };
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_TENSOR_ID_H_
