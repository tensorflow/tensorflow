/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/gloo_collectives.h"

#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "gloo/context.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"
#include "xla/backends/cpu/collectives/gloo_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/global_device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

GlooCollectives::GlooCollectives(
    std::unique_ptr<gloo::rendezvous::Store> store,
    std::shared_ptr<gloo::transport::Device> device)
    : store_(std::move(store)), device_(std::move(device)) {}

GlooCollectives::~GlooCollectives() = default;

absl::StatusOr<std::shared_ptr<Communicator>> GlooCollectives::GetCommunicator(
    absl::Span<GlobalDeviceId const> global_devices, int rank) {
  Context* context;
  {
    absl::MutexLock lock(&mu_);
    auto& context_ref = contexts_[std::make_tuple(
        std::vector<GlobalDeviceId>(global_devices.begin(),
                                    global_devices.end()),
        rank)];
    if (!context_ref) {
      context_ref = std::make_unique<Context>();
    }
    context = context_ref.get();
  }
  absl::MutexLock context_lock(&context->mu);
  if (context->communicator) {
    return context->communicator;
  }
  auto gloo_context =
      std::make_shared<gloo::rendezvous::Context>(rank, global_devices.size());
  auto prefix_store = gloo::rendezvous::PrefixStore(
      absl::StrCat("gloo/",
                   absl::StrJoin(global_devices, ",",
                                 [](std::string* out, GlobalDeviceId id) {
                                   absl::StrAppend(out, id.value());
                                 })),
      *store_);
  try {
    gloo_context->connectFullMesh(prefix_store, device_);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo context initialization failed: ", e.what()));
  }
  context->communicator = std::make_shared<GlooCommunicator>(
      std::move(gloo_context), rank, global_devices.size());
  return context->communicator;
}

}  // namespace xla::cpu
