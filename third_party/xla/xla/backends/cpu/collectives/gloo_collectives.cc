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

#include "xla/backends/cpu/collectives/gloo_collectives.h"

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "gloo/context.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"
#include "xla/backends/cpu/collectives/gloo_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/global_device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

GlooCollectives::GlooCollectives(
    std::unique_ptr<gloo::rendezvous::Store> store,
    std::shared_ptr<gloo::transport::Device> device)
    : store_(std::move(store)), device_(std::move(device)) {}

GlooCollectives::~GlooCollectives() = default;

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
GlooCollectives::CreateCommunicators(const CliqueKey& clique_key,
                                     const std::optional<CliqueIds>& clique_ids,
                                     absl::Span<const DeviceRank> ranks,
                                     const Config& config) {
  std::vector<std::unique_ptr<Communicator>> communicators;
  for (auto& device_rank : ranks) {
    size_t rank = device_rank.rank.value();

    auto gloo_context = std::make_shared<gloo::rendezvous::Context>(
        rank, clique_key.num_devices());

#ifdef GLOO_SHARED_STORE
    auto store_pointer = std::shared_ptr<gloo::rendezvous::Store>(
        store_.get(), [](gloo::rendezvous::Store*) {});
#else
    auto& store_pointer = *store_;
#endif  // GLOO_SHARED_STORE

    auto prefix_store = std::make_shared<gloo::rendezvous::PrefixStore>(
        absl::StrCat("gloo/",
                     absl::StrJoin(clique_key.devices(), ",",
                                   [](std::string* out, GlobalDeviceId id) {
                                     absl::StrAppend(out, id.value());
                                   })),
        store_pointer);

    try {
#ifdef GLOO_SHARED_STORE
      auto prefix_store_pointer = prefix_store;
#else
      auto& prefix_store_pointer = *prefix_store;
#endif  // GLOO_SHARED_STORE
      gloo_context->connectFullMesh(prefix_store_pointer, device_);
    } catch (std::exception& e) {
      return absl::UnknownError(
          absl::StrCat("Gloo context initialization failed: ", e.what()));
    }

    communicators.push_back(std::make_unique<GlooCommunicator>(
        std::move(gloo_context), rank, clique_key.num_devices()));
  }

  return communicators;
}

}  // namespace xla::cpu
