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

#ifndef XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_FACTORY_H_
#define XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_FACTORY_H_

#include <memory>

#include "absl/base/attributes.h"

namespace stream_executor {

class StreamExecutor;

namespace host {

class HostStream;

// Factory interface for creating HostStream instances. Supports static
// registration of custom factory implementations with priority ranking.
class HostStreamFactory {
 public:
  virtual ~HostStreamFactory() = default;

  // Creates a new HostStream associated with the given executor.
  virtual std::unique_ptr<HostStream> CreateStream(
      StreamExecutor* executor) const = 0;
  // Registers a custom HostStreamFactory instance with a specified priority.
  // Higher priority values override lower priority registrations.
  static void Register(std::unique_ptr<HostStreamFactory> factory,
                       int priority);

  // Returns a shared pointer to the highest-priority registered factory.
  static std::shared_ptr<HostStreamFactory> GetFactory();

  // Resets the registered factory to the default factory (for testing).
  static void ResetForTesting();
};

template <class Factory>
class HostStreamFactoryRegistrar {
 public:
  explicit HostStreamFactoryRegistrar(int priority) {
    HostStreamFactory::Register(std::make_unique<Factory>(), priority);
  }
};

}  // namespace host
}  // namespace stream_executor

#define REGISTER_HOST_STREAM_FACTORY(factory, priority) \
  INTERNAL_REGISTER_HOST_STREAM_FACTORY(factory, priority, __COUNTER__)

#define INTERNAL_REGISTER_HOST_STREAM_FACTORY(factory, priority, ctr) \
  ABSL_ATTRIBUTE_UNUSED static ::stream_executor::host::              \
      HostStreamFactoryRegistrar<factory>                             \
      INTERNAL_REGISTER_LOCAL_HOST_STREAM_FACTORY_NAME(ctr) {         \
    priority                                                          \
  }

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_HOST_STREAM_FACTORY_NAME(ctr) \
  register_host_stream_factory_##ctr##_object_

#endif  // XLA_STREAM_EXECUTOR_HOST_HOST_STREAM_FACTORY_H_
