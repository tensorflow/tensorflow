/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/tracing.h"

#include <atomic>
#include <map>
#include <string>
#include <vector>
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace port {

int32 Tracing::category_id_[kEventCategoryMax];
uint64 Tracing::event_mask_ = 0;
std::map<string, int32>* Tracing::name_map_ = new std::map<string, int32>;

// This needs to be kept in sync with the EventCategory enumeration.
const char* Tracing::EventCategoryString(EventCategory category) {
  switch (category) {
    case EventCategory::kScheduleClosure:
      return "ScheduleClosure";
    case EventCategory::kRunClosure:
      return "RunClosure";
    case EventCategory::kCompute:
      return "Compute";
    case EventCategory::kEventCategoryMax:
      return "EventCategoryMax";
  }
  return "Unknown";
}

// This function allows the user to specify arbitrary subsets of the
// supported Threadscape events and activities.
bool Tracing::ParseEventMask(const char* flagname, const string& value) {
  VLOG(1) << flagname << " set to " << value;
  int64 new_mask = 0;
  std::vector<string> events =
      str_util::Split(value, ',', str_util::SkipEmpty());
  for (string name : events) {
    bool clear = false;
    int64 mask = 0;
    if (name[0] == '!') {
      // invert the sense of the flag
      clear = true;
      name = name.substr(1);
    }
    if (name == "ALL") {
      mask = ~0;
    } else {
      auto it = name_map_->find(name);
      int32 id;
      if (it == name_map_->end()) {
        id = -1;
      } else {
        id = it->second;
      }
      if (id < 0) {
        LOG(ERROR) << "Can't parse event mask name " << name;
        return false;
      }
      mask = 1 << id;
    }
    if (clear) {
      new_mask &= ~mask;
    } else {
      new_mask |= mask;
    }
  }
  // parsing was successful; set the permanent event mask
  event_mask_ = new_mask;
  return true;
}

/*static*/ std::atomic<Tracing::Engine*> Tracing::tracing_engine_;

void Tracing::RegisterEngine(Engine* e) {
  tracing_engine_.store(e, std::memory_order_release);
}

Tracing::Engine::~Engine() {}
Tracing::Engine::Annotation::~Annotation() {}
Tracing::Engine::Tracer::~Tracer() {}

}  // namespace port
}  // namespace tensorflow
