/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_resource_manager.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tensorrt {

std::shared_ptr<TRTResourceManager>
tensorflow::tensorrt::TRTResourceManager::instance() {
  static std::shared_ptr<TRTResourceManager> instance_(new TRTResourceManager);
  return instance_;
}

std::shared_ptr<tensorflow::ResourceMgr>
tensorflow::tensorrt::TRTResourceManager::getManager(const string& op_name) {
  // mutex is held for lookup only. Most instantiations where mutex will be held
  // longer will be during op creation and should be ok.
  tensorflow::mutex_lock lock(map_mutex_);
  auto s = managers_.find(op_name);
  if (s == managers_.end()) {
    auto it = managers_.emplace(
        op_name, std::make_shared<tensorflow::ResourceMgr>(op_name));
    VLOG(1) << "Returning a new manager " << op_name;
    return it.first->second;
  }
  VLOG(1) << "Returning old manager " << op_name;
  return s->second;
}

}  // namespace tensorrt
}  // namespace tensorflow
