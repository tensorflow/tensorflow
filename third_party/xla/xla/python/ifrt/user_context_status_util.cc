/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/user_context_status_util.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_registry.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ifrt {

namespace {

constexpr absl::string_view kIfrtUserContextPayloadUrl =
    "type.googleapis.com/ifrt.UserContext";

}  // namespace

absl::Status AttachUserContextId(absl::Status status, UserContextId id) {
  if (status.ok()) {
    return status;
  }
  status.SetPayload(kIfrtUserContextPayloadUrl,
                    absl::Cord(absl::StrCat(id.value())));
  return status;
}

absl::Status AttachUserContextRef(absl::Status status,
                                  UserContextRef user_context) {
  if (status.ok()) {
    return status;
  }
  auto user_context_id_str_holder =
      std::make_unique<std::string>(absl::StrCat(user_context->Id().value()));
  absl::string_view user_context_id_str_view = *user_context_id_str_holder;
  TrackedUserContextRef tracked_user_context =
      UserContextRegistry::Get().Register(std::move(user_context));
  status.SetPayload(
      kIfrtUserContextPayloadUrl,
      absl::MakeCordFromExternal(
          user_context_id_str_view,
          [user_context_id_str_holder = std::move(user_context_id_str_holder),
           tracked_user_context = std::move(tracked_user_context)]() {}));
  return status;
}

absl::Status ReattachUserContextRefs(absl::Status status) {
  if (status.ok()) {
    return status;
  }
  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  if (!payload.has_value()) {
    return status;
  }
  uint64_t user_context_id;
  if (!absl::SimpleAtoi(payload->Flatten(), &user_context_id)) {
    return status;
  }
  TrackedUserContextRef tracked_user_context =
      UserContextRegistry::Get().Lookup(UserContextId(user_context_id));
  if (tracked_user_context == nullptr) {
    return status;
  }
  auto user_context_id_str_holder =
      std::make_unique<std::string>(payload->Flatten());
  absl::string_view user_context_id_str_view = *user_context_id_str_holder;
  status.SetPayload(
      kIfrtUserContextPayloadUrl,
      absl::MakeCordFromExternal(
          user_context_id_str_view,
          [user_context_id_str_holder = std::move(user_context_id_str_holder),
           tracked_user_context = std::move(tracked_user_context)]() {}));
  return status;
}

static void ExpandStandardUserContext(absl::Status& status) {
  if (status.ok()) {
    return;
  }

  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  if (!payload.has_value()) {
    return;
  }

  status.ErasePayload(kIfrtUserContextPayloadUrl);

  uint64_t user_context_id;
  if (!absl::SimpleAtoi(payload->Flatten(), &user_context_id)) {
    tsl::errors::AppendToMessage(
        &status, "\n(failed to parse a user context ID: ", payload->Flatten(),
        ")");
    return;
  }
  TrackedUserContextRef user_context =
      UserContextRegistry::Get().Lookup(UserContextId(user_context_id));
  if (user_context == nullptr) {
    tsl::errors::AppendToMessage(
        &status, "\n(failed to find a user context for ID: ", user_context_id,
        ")");
    return;
  }
  tsl::errors::AppendToMessage(&status, "\n",
                               user_context->user_context()->DebugString());
}

absl::Status ExpandUserContexts(absl::Status status) {
  CustomStatusExpanderRegistry::Get().Process(status);
  return status;
}

static const bool register_standard_user_context = []() {
  xla::ifrt::CustomStatusExpanderRegistry::Get().Register(
      kIfrtUserContextPayloadUrl, ExpandStandardUserContext,
      /*process_order=*/-1);
  return true;
}();

}  // namespace ifrt
}  // namespace xla
