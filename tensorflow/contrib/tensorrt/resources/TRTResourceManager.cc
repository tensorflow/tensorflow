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

#include "tensorflow/contrib/tensorrt/resources/TRTResourceManager.h"
#include "tensorflow/core/platform/default/logging.h"

std::shared_ptr<tensorflow::ResourceMgr>
tensorflow::trt::TRTResourceManager::getManager(const std::string& mgr_name) {
  // mutex is held for lookup only. Most instantiations where mutex will be held
  // longer will be during op creation and should be ok.
  tensorflow::mutex_lock lock(map_mutex_);
  auto s = managers_.find(mgr_name);
  if (s == managers_.end()) {
    auto it = managers_.emplace(
        mgr_name, std::make_shared<tensorflow::ResourceMgr>(mgr_name));
    VLOG(0) << "Returning a new manager " << mgr_name;
    return it.first->second;
  }
  VLOG(1) << "Returning old manager " << mgr_name;
  return s->second;
}
