/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_STACK_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_STACK_H_

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

template <typename OpInfo>
class OpStack {
 public:
  // Pushes an Op onto the stack.
  void Push(uint32 op_id, std::unique_ptr<OpInfo> op_info) {
    stack_.emplace_back(op_id, std::move(op_info));
  }

  // Pops the Op with the given op_id from the stack.
  std::unique_ptr<OpInfo> Pop(uint32 op_id) {
    // Pop until match or stack_ is empty.
    std::unique_ptr<OpInfo> result;
    while (!stack_.empty()) {
      auto back = std::move(stack_.back());
      stack_.pop_back();
      if (op_id == back.first) {
        result = std::move(back.second);
        break;
      }
    }
    return result;
  }

  // Returns the Op at the top of the stack.
  OpInfo* Top() const {
    return stack_.empty() ? nullptr : stack_.back().second.get();
  }

  // Returns true if the stack is empty.
  bool Empty() const { return stack_.empty(); }

  // Clears the stack.
  void Clear() { stack_.clear(); }

 private:
  std::vector<std::pair<uint32 /*op_id*/, std::unique_ptr<OpInfo>>> stack_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_STACK_H_
