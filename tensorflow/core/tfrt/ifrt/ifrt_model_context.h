/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

inline constexpr absl::string_view kIfrtModelContextName = "IfrtModelContext";

// Device specific configuration not available through ifrt. This should be
// rare.
struct DeviceConfig {
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn =
      tensorflow::IdentityShapeRepresentationFn();
};

// The runtime context for ifrt to be used in TFRT serving.
//
// This class is thread compatible.
class IfrtModelContext {
 public:
  explicit IfrtModelContext(std::shared_ptr<xla::ifrt::Client> client,
                            const Eigen::ThreadPoolDevice* thread_pool_device)
      : client_(std::move(client)), thread_pool_device_(*thread_pool_device) {}
  IfrtModelContext(
      std::shared_ptr<xla::ifrt::Client> client,
      const Eigen::ThreadPoolDevice* thread_pool_device,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn)
      : client_(std::move(client)),
        thread_pool_device_(*thread_pool_device),
        shape_representation_fn_(shape_representation_fn) {}

  void RegisterHandle(ServingExecutableRegistry::Handle handle) {
    handles_.push_back(std::move(handle));
  }

  std::shared_ptr<xla::ifrt::Client> GetClient() const { return client_; }

  const tensorflow::XlaHelpers::ShapeRepresentationFn&
  GetShapeRepresentationFn() const {
    return shape_representation_fn_;
  }

  const Eigen::ThreadPoolDevice& GetThreadPoolDevice() const;

  absl::Status RegisterLoadedVariable(
      absl::string_view name,
      tsl::RCReference<xla::ifrt::Array> loaded_variable)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> GetLoadedVariable(
      absl::string_view name) const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  std::shared_ptr<xla::ifrt::Client> client_;
  const Eigen::ThreadPoolDevice& thread_pool_device_;
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn_ =
      tensorflow::IdentityShapeRepresentationFn();

  std::vector<ServingExecutableRegistry::Handle> handles_;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, tsl::RCReference<xla::ifrt::Array>>
      loaded_variable_map_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_
