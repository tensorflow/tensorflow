/* Copyright 2020 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

#ifdef RUY_PROFILER

namespace ruy {
namespace profiler {

void Label::operator=(const Label& other) {
  format_ = other.format_;
  args_count_ = other.args_count_;
  for (int i = 0; i < args_count_; i++) {
    args_[i] = other.args_[i];
  }
}

bool Label::operator==(const Label& other) const {
  if (std::string(format_) != std::string(other.format_)) {
    return false;
  }
  if (args_count_ != other.args_count_) {
    return false;
  }
  for (int i = 0; i < args_count_; i++) {
    if (args_[i] != other.args_[i]) {
      return false;
    }
  }
  return true;
}

std::string Label::Formatted() const {
  static constexpr int kBufSize = 256;
  char buf[kBufSize];
  if (args_count_ == 0) {
    return format_;
  }
  if (args_count_ == 1) {
    snprintf(buf, kBufSize, format_, args_[0]);
  } else if (args_count_ == 2) {
    snprintf(buf, kBufSize, format_, args_[0], args_[1]);
  } else if (args_count_ == 3) {
    snprintf(buf, kBufSize, format_, args_[0], args_[1], args_[2]);
  } else if (args_count_ == 4) {
    snprintf(buf, kBufSize, format_, args_[0], args_[1], args_[2], args_[3]);
  } else {
    abort();
  }
  return buf;
}

namespace detail {

std::mutex* GlobalsMutex() {
  static std::mutex mutex;
  return &mutex;
}

bool& GlobalIsProfilerRunning() {
  static bool b;
  return b;
}

std::vector<ThreadStack*>* GlobalAllThreadStacks() {
  static std::vector<ThreadStack*> all_stacks;
  return &all_stacks;
}

ThreadStack* ThreadLocalThreadStack() {
  thread_local static ThreadStack thread_stack;
  return &thread_stack;
}

ThreadStack::ThreadStack() {
  std::lock_guard<std::mutex> lock(*GlobalsMutex());
  static std::uint32_t global_next_thread_stack_id = 0;
  stack_.id = global_next_thread_stack_id++;
  GlobalAllThreadStacks()->push_back(this);
}

ThreadStack::~ThreadStack() {
  std::lock_guard<std::mutex> lock(*GlobalsMutex());
  std::vector<ThreadStack*>* all_stacks = GlobalAllThreadStacks();
  for (auto it = all_stacks->begin(); it != all_stacks->end(); ++it) {
    if (*it == this) {
      all_stacks->erase(it);
      return;
    }
  }
}
int GetBufferSize(const Stack& stack) {
  return sizeof(stack.id) + sizeof(stack.size) +
         stack.size * sizeof(stack.labels[0]);
}

void CopyToBuffer(const Stack& stack, char* dst) {
  memcpy(dst, &stack.id, sizeof(stack.id));
  dst += sizeof(stack.id);
  memcpy(dst, &stack.size, sizeof(stack.size));
  dst += sizeof(stack.size);
  memcpy(dst, stack.labels, stack.size * sizeof(stack.labels[0]));
}

void ReadFromBuffer(const char* src, Stack* stack) {
  memcpy(&stack->id, src, sizeof(stack->id));
  src += sizeof(stack->id);
  memcpy(&stack->size, src, sizeof(stack->size));
  src += sizeof(stack->size);
  memcpy(stack->labels, src, stack->size * sizeof(stack->labels[0]));
}

}  // namespace detail
}  // namespace profiler
}  // namespace ruy

#endif
