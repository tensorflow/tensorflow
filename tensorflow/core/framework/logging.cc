/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/logging.h"

#include <iostream>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace logging {

typedef std::vector<void (*)(const char*)> Listeners;

Listeners* GetListeners() {
  static Listeners* listeners = new Listeners;
  return listeners;
}

bool RegisterListener(void (*listener)(const char*)) {
  GetListeners()->push_back(listener);
  return true;
}

bool LogToListeners(string msg, string end) {
  auto listeners = logging::GetListeners();
  if (listeners->empty()) {
    return false;
  }

  string ended_msg = strings::StrCat(msg, end);

  for (auto& listener : *listeners) {
    listener(ended_msg.c_str());
  }

  return true;
}

}  // end namespace logging

}  // end namespace tensorflow
