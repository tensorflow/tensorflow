/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/host/host_stream_factory.h"

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace {

ABSL_CONST_INIT absl::Mutex host_stream_factory_lock(absl::kConstInit);

struct FactoryItem {
  std::shared_ptr<stream_executor::host::HostStreamFactory> factory
      ABSL_GUARDED_BY(host_stream_factory_lock);
  int priority ABSL_GUARDED_BY(host_stream_factory_lock) = -1;
};

FactoryItem& host_stream_factory() {
  static absl::NoDestructor<FactoryItem> factory;
  return *factory;
}

class HostStreamDefaultFactory
    : public stream_executor::host::HostStreamFactory {
 public:
  ~HostStreamDefaultFactory() override = default;
  std::unique_ptr<stream_executor::host::HostStream> CreateStream(
      stream_executor::StreamExecutor* executor) const override {
    return std::make_unique<stream_executor::host::HostStream>(executor);
  }
};
}  // namespace

namespace stream_executor {
namespace host {

// static
void HostStreamFactory::Register(std::unique_ptr<HostStreamFactory> factory,
                                 int priority) {
  absl::MutexLock lock(host_stream_factory_lock);
  FactoryItem& factory_item = host_stream_factory();
  if (factory_item.factory == nullptr || factory_item.priority < priority) {
    factory_item.factory = std::move(factory);
    factory_item.priority = priority;
  }
}

// static
std::shared_ptr<HostStreamFactory> HostStreamFactory::GetFactory() {
  absl::ReaderMutexLock lock(host_stream_factory_lock);
  FactoryItem& factory_item = host_stream_factory();
  return factory_item.factory;
}

// static
void HostStreamFactory::ResetForTesting() {
  absl::MutexLock lock(host_stream_factory_lock);
  FactoryItem& factory_item = host_stream_factory();
  factory_item.factory = std::make_shared<HostStreamDefaultFactory>();
  factory_item.priority = 100;
}

}  // namespace host
}  // namespace stream_executor

REGISTER_HOST_STREAM_FACTORY(HostStreamDefaultFactory, 100);
