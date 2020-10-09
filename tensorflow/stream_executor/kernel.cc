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

// Implementation of the pointer-to-implementation wrapper for the data-parallel
// kernel abstraction. KernelBase just delegates to the internal
// platform-specific implementation instance.

#include "tensorflow/stream_executor/kernel.h"

#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/stream_executor/lib/demangle.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {

bool KernelMetadata::registers_per_thread(int *registers_per_thread) const {
  if (has_registers_per_thread_) {
    *registers_per_thread = registers_per_thread_;
    return true;
  }

  return false;
}

void KernelMetadata::set_registers_per_thread(int registers_per_thread) {
  registers_per_thread_ = registers_per_thread;
  has_registers_per_thread_ = true;
}

bool KernelMetadata::shared_memory_bytes(int *shared_memory_bytes) const {
  if (has_shared_memory_bytes_) {
    *shared_memory_bytes = shared_memory_bytes_;
    return true;
  }

  return false;
}

void KernelMetadata::set_shared_memory_bytes(int shared_memory_bytes) {
  shared_memory_bytes_ = shared_memory_bytes;
  has_shared_memory_bytes_ = true;
}

KernelBase::KernelBase(KernelBase &&from)
    : parent_(from.parent_),
      implementation_(std::move(from.implementation_)),
      name_(std::move(from.name_)),
      demangled_name_(std::move(from.demangled_name_)),
      metadata_(from.metadata_) {
  from.parent_ = nullptr;
}

KernelBase::KernelBase(StreamExecutor *parent)
    : parent_(parent),
      implementation_(parent->implementation()->CreateKernelImplementation()) {}

KernelBase::KernelBase(StreamExecutor *parent,
                       internal::KernelInterface *implementation)
    : parent_(parent), implementation_(implementation) {}

KernelBase::~KernelBase() {
  if (parent_) {
    parent_->UnloadKernel(this);
  }
}

unsigned KernelBase::Arity() const { return implementation_->Arity(); }

void KernelBase::SetPreferredCacheConfig(KernelCacheConfig config) {
  return implementation_->SetPreferredCacheConfig(config);
}

KernelCacheConfig KernelBase::GetPreferredCacheConfig() const {
  return implementation_->GetPreferredCacheConfig();
}

void KernelBase::set_name(absl::string_view name) {
  name_ = std::string(name);

  // CUDA splitter prefixes stub functions with __device_stub_.
  demangled_name_ =
      port::Demangle(absl::StripPrefix(name, "__device_stub_").data());
}

}  // namespace stream_executor
