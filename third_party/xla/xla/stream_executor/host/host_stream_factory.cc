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

#include "tsl/platform/mutex.h"

namespace {

static tsl::mutex* get_host_stream_factory_lock() {
  static tsl::mutex host_stream_factory_lock(tsl::LINKER_INITIALIZED);
  return &host_stream_factory_lock;
}

struct FactoryItem {
  std::unique_ptr<stream_executor::host::HostStreamFactory> factory;
  int priority = -1;
};

FactoryItem& host_stream_factory() {
  static FactoryItem* factory = new FactoryItem();
  return *factory;
}

}  // namespace

namespace stream_executor {
namespace host {

// static
void HostStreamFactory::Register(std::unique_ptr<HostStreamFactory> factory,
                                 int priority) {
  tsl::mutex_lock l(*get_host_stream_factory_lock());
  FactoryItem& factory_item = host_stream_factory();
  if (factory_item.factory == nullptr) {
    factory_item.factory = std::move(factory);
    factory_item.priority = priority;
    return;
  }
  if (factory_item.priority < priority) {
    factory_item.factory = std::move(factory);
    factory_item.priority = priority;
  }
}

// static
const HostStreamFactory* HostStreamFactory::GetFactory() {
  tsl::tf_shared_lock l(*get_host_stream_factory_lock());
  FactoryItem& factory_item = host_stream_factory();
  return factory_item.factory.get();
}

}  // namespace host
}  // namespace stream_executor

REGISTER_HOST_STREAM_FACTORY(stream_executor::host::HostStreamDefaultFactory,
                             100);
