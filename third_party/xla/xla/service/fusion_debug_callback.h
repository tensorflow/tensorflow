/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FUSION_DEBUG_CALLBACK_H_
#define XLA_SERVICE_FUSION_DEBUG_CALLBACK_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "xla/literal.h"

namespace xla {

struct FusionDebugCallbackAttributes {
  std::string user_tag;
  std::string fusion_hlo_name;
  std::string fused_hlo_name;
  std::string core_location;
};

using FusionDebugCallbackId = int64_t;
using FusionDebugCallbackFunction = std::function<void(
    std::shared_ptr<const xla::Literal>, const FusionDebugCallbackAttributes&)>;

FusionDebugCallbackId RegisterFusionDebugCallback(
    FusionDebugCallbackFunction cb);

void UnregisterFusionDebugCallback(FusionDebugCallbackId callback_id);

void TriggerFusionDebugCallback(
    const std::shared_ptr<const xla::Literal>& literal,
    const FusionDebugCallbackAttributes& attributes);

}  // namespace xla

#endif  // XLA_SERVICE_FUSION_DEBUG_CALLBACK_H_
