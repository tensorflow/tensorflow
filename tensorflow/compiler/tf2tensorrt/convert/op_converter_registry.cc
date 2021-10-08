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
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"

#include <set>
#include <utility>

#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

struct OpConverterRegistration {
  OpConverter converter;
  int priority;
};
class OpConverterRegistry::Impl {
 public:
  ~Impl() = default;

  InitOnStartupMarker Register(const string& name, const int priority,
                               OpConverter converter) {
    mutex_lock lock(mu_);
    auto item = registry_.find(name);
    if (item != registry_.end()) {
      const int existing_priority = item->second.priority;
      if (priority <= existing_priority) {
        LOG(WARNING) << absl::StrCat(
            "Ignoring TF->TRT ", name, " op converter with priority ",
            existing_priority, " due to another converter with priority ",
            priority);
        return {};
      } else {
        LOG(WARNING) << absl::StrCat(
            "Overwriting TF->TRT ", name, " op converter with priority ",
            existing_priority, " using another converter with priority ",
            priority);
        registry_.erase(item);
      }
    }
    registry_.insert({name, OpConverterRegistration{converter, priority}});
    return {};
  }

  StatusOr<OpConverter> LookUp(const string& name) {
    mutex_lock lock(mu_);
    auto found = registry_.find(name);
    if (found != registry_.end()) {
      return found->second.converter;
    }
    return errors::NotFound("No converter for op ", name);
  }

  void Clear(const std::string& name) {
    mutex_lock lock(mu_);
    auto itr = registry_.find(name);
    if (itr == registry_.end()) {
      return;
    }
    registry_.erase(itr);
  }

 private:
  mutable mutex mu_;
  mutable std::unordered_map<std::string, OpConverterRegistration> registry_
      TF_GUARDED_BY(mu_);
};

OpConverterRegistry::OpConverterRegistry() : impl_(std::make_unique<Impl>()) {}

StatusOr<OpConverter> OpConverterRegistry::LookUp(const string& name) {
  return impl_->LookUp(name);
}

InitOnStartupMarker OpConverterRegistry::Register(const string& name,
                                                  const int priority,
                                                  OpConverter converter) {
  return impl_->Register(name, priority, converter);
}

void OpConverterRegistry::Clear(const std::string& name) { impl_->Clear(name); }

OpConverterRegistry* GetOpConverterRegistry() {
  static OpConverterRegistry* registry = new OpConverterRegistry();
  return registry;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
