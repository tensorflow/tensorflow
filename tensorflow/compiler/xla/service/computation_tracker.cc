/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/computation_tracker.h"

#include <list>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

using ::tensorflow::strings::Appendf;

namespace xla {

ComputationTracker::ComputationTracker() : next_computation_(1) {}

ComputationHandle ComputationTracker::NewComputation(
    const string& computation_name) {
  tensorflow::mutex_lock lock(computation_mutex_);
  ComputationHandle computation_handle;
  int64 handle_value = next_computation_++;
  computation_handle.set_handle(handle_value);
  opaque_to_computation_[handle_value] =
      MakeUnique<UserComputation>(computation_name, computation_handle);
  return computation_handle;
}

StatusOr<ComputationHandle> ComputationTracker::LoadSessionModule(
    const SessionModule& session_module) {
  tensorflow::mutex_lock lock(computation_mutex_);

  // For each embedded computation, create a new computation based on its
  // serialized data, and place the mapping from the old computation handle to
  // the new computation handle.

  // Build a mapping from old embedded computation handles to new computation
  // handles. We build the ID mapping first since the embedded computations are
  // in no particular order and may refer to each other.
  std::map<int64, ComputationHandle> old_to_new;
  for (const SessionComputation& computation :
       session_module.embedded_computations()) {
    const int64 old_handle = computation.computation_handle().handle();
    if (!old_to_new.emplace(old_handle, AllocateHandle()).second) {
      return InvalidArgument("Duplicate embedded computation handle %lld",
                             old_handle);
    }
  }

  // Create a new computation from each serialized embedded computation.
  for (const SessionComputation& computation :
       session_module.embedded_computations()) {
    const int64 old_handle = computation.computation_handle().handle();
    const ComputationHandle& new_handle = old_to_new[old_handle];
    TF_ASSIGN_OR_RETURN(opaque_to_computation_[new_handle.handle()],
                        UserComputation::MakeWithRemapping(
                            computation, new_handle, old_to_new));
  }

  // Finally, place the entry computation in the tracker with all of the
  // remappings populated from the above.
  const int64 old_handle = session_module.entry().computation_handle().handle();
  TF_ASSIGN_OR_RETURN(
      old_to_new[old_handle],
      LoadSessionComputation(session_module.entry(), &old_to_new));
  return old_to_new[old_handle];
}

StatusOr<std::unique_ptr<SessionModule>>
ComputationTracker::SnapshotComputation(const ComputationHandle& computation) {
  TF_ASSIGN_OR_RETURN(UserComputation * user_computation, Resolve(computation));
  const VersionedComputationHandle entry_versioned_handle =
      user_computation->GetVersionedHandle();
  std::set<VersionedComputationHandle> visited;
  std::list<VersionedComputationHandle> post_order;
  {
    tensorflow::mutex_lock lock(computation_mutex_);
    ComputeComputationPostOrder(entry_versioned_handle, &visited, &post_order);
  }
  auto session_module = MakeUnique<SessionModule>();
  *session_module->mutable_entry() =
      Resolve(entry_versioned_handle.handle)
          .ValueOrDie()
          ->CloneSessionComputation(entry_versioned_handle.version);
  for (auto it = ++post_order.rbegin(); it != post_order.rend(); ++it) {
    *session_module->add_embedded_computations() =
        Resolve(it->handle).ValueOrDie()->CloneSessionComputation(it->version);
  }
  return std::move(session_module);
}

StatusOr<UserComputation*> ComputationTracker::Resolve(
    const ComputationHandle& computation) const {
  tensorflow::mutex_lock lock(computation_mutex_);
  return ResolveInternal(computation);
}

ComputationHandle ComputationTracker::AllocateHandle() {
  int64 handle_value = next_computation_++;
  ComputationHandle result;
  result.set_handle(handle_value);
  return result;
}

StatusOr<ComputationHandle> ComputationTracker::LoadSessionComputation(
    const SessionComputation& session_computation,
    std::map<int64, ComputationHandle>* old_to_new) {
  TF_RET_CHECK(old_to_new != nullptr);
  const ComputationHandle new_handle = AllocateHandle();
  (*old_to_new)[session_computation.computation_handle().handle()] = new_handle;
  TF_ASSIGN_OR_RETURN(opaque_to_computation_[new_handle.handle()],
                      UserComputation::MakeWithRemapping(
                          session_computation, new_handle, *old_to_new));
  return new_handle;
}

StatusOr<UserComputation*> ComputationTracker::ResolveInternal(
    const ComputationHandle& computation) const {
  auto it = opaque_to_computation_.find(computation.handle());
  if (it == opaque_to_computation_.end()) {
    return NotFound("computation handle not found: %lld", computation.handle());
  }
  UserComputation* user_computation = it->second.get();
  return user_computation;
}

void ComputationTracker::ComputeComputationPostOrder(
    const VersionedComputationHandle& versioned_handle,
    std::set<VersionedComputationHandle>* visited,
    std::list<VersionedComputationHandle>* post_order) const {
  if (visited->count(versioned_handle) > 0) {
    CHECK_EQ(1, visited->count(versioned_handle));
    return;
  }

  UserComputation* computation =
      ResolveInternal(versioned_handle.handle).ValueOrDie();
  std::vector<VersionedComputationHandle> embedded_handles =
      computation->GetEmbeddedComputations(versioned_handle.version);

  for (const auto& embedded_handle : embedded_handles) {
    ComputeComputationPostOrder(embedded_handle, visited, post_order);
  }

  visited->insert(versioned_handle);
  post_order->push_back(versioned_handle);
  return;
}

StatusOr<std::unique_ptr<HloModule>> ComputationTracker::BuildHloModule(
    const VersionedComputationHandle& entry_handle,
    bool include_unreachable_instructions) const {
  tensorflow::mutex_lock lock(computation_mutex_);

  VLOG(1) << "BuildHloModule(" << entry_handle
          << ", include_unreachable_instructions="
          << include_unreachable_instructions << ")";
  XLA_VLOG_LINES(1, ToStringInternal());

  TF_ASSIGN_OR_RETURN(UserComputation * entry_computation,
                      ResolveInternal(entry_handle.handle));

  // Build a topological sort of the entry and any embedded computations as a
  // list. The root of the computation will be the last element in the list.
  std::set<VersionedComputationHandle> visited;
  std::list<VersionedComputationHandle> post_order;
  ComputeComputationPostOrder(entry_handle, &visited, &post_order);

  // Map from ComputationHandle value and computation version to HloComputation.
  std::map<VersionedComputationHandle, HloComputation*> hlo_computations;

  // The resolver lambda resolves VersionedHandles to embedded
  // HloComputation*. This is required by UserComputation::BuildHloComputation
  // when lowering calling operations (map, reduce etc).
  auto resolver = [&hlo_computations](
      const VersionedComputationHandle& versioned_handle) -> HloComputation* {
    CHECK_GT(hlo_computations.count(versioned_handle), 0);
    return hlo_computations.at(versioned_handle);
  };

  // Print the post-order list for this entry computation.
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Visiting UserComputations in post order:";
    for (const VersionedComputationHandle& versioned_handle : post_order) {
      VLOG(2) << "  " << versioned_handle;
    }
  }

  string module_name =
      tensorflow::strings::StrCat(entry_computation->name(), "_module");
  auto module = MakeUnique<HloModule>(module_name, entry_handle);
  for (auto versioned_handle : post_order) {
    UserComputation* computation =
        ResolveInternal(versioned_handle.handle).ValueOrDie();

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> hlo_computation,
        computation->BuildHloComputation(versioned_handle.version, resolver,
                                         include_unreachable_instructions));

    // Add the newly created computation to VersionedHandle-to-HloComputation
    // map.
    DCHECK_EQ(0, hlo_computations.count(versioned_handle));
    hlo_computations[versioned_handle] = hlo_computation.get();

    if (computation == entry_computation) {
      module->AddEntryComputation(std::move(hlo_computation));
    } else {
      module->AddEmbeddedComputation(std::move(hlo_computation));
    }
  }

  return std::move(module);
}

string ComputationTracker::ToString() const {
  tensorflow::mutex_lock lock(computation_mutex_);
  return ToStringInternal();
}

string ComputationTracker::ToStringInternal() const {
  string out;
  Appendf(&out, "ComputationTracker(%p):\n", this);
  for (const auto& handle_computation : opaque_to_computation_) {
    int64 handle = handle_computation.first;
    const std::unique_ptr<UserComputation>& computation =
        handle_computation.second;
    Appendf(&out, "  %4lld : %s \"%s\"\n", handle,
            computation->GetVersionedHandle().ToString().c_str(),
            computation->name().c_str());
  }
  return out;
}

}  // namespace xla
