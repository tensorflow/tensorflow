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
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

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

  StatusOr<OpConverter> LookUp(string name) {
    // Fetch the user-provide TF operations denylisted for conversion by TF-TRT.
    static const absl::flat_hash_set<string> tftrt_op_fakelist = [] {
      string tftrt_op_fakelist_str;
      TF_CHECK_OK(ReadStringFromEnvVar("TF_TRT_OP_FAKELIST",
                                       /*default_value=*/"",
                                       &tftrt_op_fakelist_str));
      absl::flat_hash_set<string> tftrt_op_fakelist{};
      for (const auto& x : str_util::Split(tftrt_op_fakelist_str, ",")) {
        tftrt_op_fakelist.insert(x);
      }
      // Force a rehash of the flat hash set
      tftrt_op_fakelist.rehash(0);
      return tftrt_op_fakelist;
    }();

    // In case the TensorFlow OP `name` matches any of the names passed to
    // TF_TRT_OP_FAKELIST environment variable, force ::LookUp to resolves to
    // ConvertFake OP converter.
    if (tftrt_op_fakelist.contains(name)) {
      LOG_FIRST_N(INFO, 2) << "Emulating OP Converter: `" << name << "`. It "
                           << "will cause TRT engine building to fail. This "
                           << "feature is only intended to be used for "
                           << "TF-TRT graph segmentation experiments. This "
                           << "feature is controlled using: "
                           << "`TF_TRT_OP_FAKELIST=OpName1,OpName2`.";
      // Forces ::LookUp to resolve to `ConvertFake` registered to `FakeOp`.
      mutex_lock lock(mu_);
      return registry_.find("FakeOp")->second.converter;
    }

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

  std::vector<std::string> ListRegisteredOps() const {
    mutex_lock lock(mu_);
    std::vector<std::string> result;
    result.reserve(registry_.size());
    for (const auto& item : registry_) {
      result.push_back(item.first);
    }
    return result;
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

std::vector<std::string> OpConverterRegistry::ListRegisteredOps() const {
  return impl_->ListRegisteredOps();
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
