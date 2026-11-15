/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/scoped_activate_context.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/stream_executor/gpu/context.h"
#include "tsl/platform/logging.h"

namespace stream_executor::gpu {

namespace {

thread_local struct ThreadLocalData {
  Context* context = nullptr;
  // -1 means no device is tracked as active, so that it never compares equal
  // to a real device ordinal (including 0) on a fresh thread.
  int device_ordinal = -1;
  int depth = 0;
} tls_data;

}  // namespace

ScopedActivateContext::ScopedActivateContext(gpu::Context* gpu_context) {
  auto* tls = &tls_data;

  // If this is an outermost scope, we must not assume that the GPU context
  // has been left in the same state we left it. Other code may have run on
  // this thread and altered the context.
  if (tls->depth == 0) {
    VLOG(3) << "ScopedActivateContext switching to "
            << gpu_context->device_ordinal();
    gpu_context->SetActive();
    tls->depth = 1;
    tls->device_ordinal = gpu_context->device_ordinal();
    tls->context = gpu_context;
    to_restore_ = nullptr;
    return;
  }

  tls->depth++;
  if (tls->device_ordinal == gpu_context->device_ordinal()) {
    DCHECK(gpu_context->IsActive());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from "
          << tls->device_ordinal << " to " << gpu_context->device_ordinal();

  to_restore_ = tls->context;
  // No tracked context means we are nested inside a ScopedDeactivateContext;
  // undo the activation on exit to preserve the deactivated state.
  deactivate_on_exit_ = (tls->context == nullptr);
  // Set the context and update thread local.
  gpu_context->SetActive();
  tls->device_ordinal = gpu_context->device_ordinal();
  tls->context = gpu_context;
}

ScopedActivateContext::~ScopedActivateContext() {
  auto* tls = &tls_data;

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (deactivate_on_exit_) {
    tls->context->SetInactive();
    tls->context = nullptr;
    tls->device_ordinal = -1;
    return;
  }
  if (to_restore_ == nullptr) {
    // Leave context, tls->device_ordinal, and tls->context set.
    return;
  }

  // Set context and update thread local.
  to_restore_->SetActive();
  tls->device_ordinal = to_restore_->device_ordinal();
  tls->context = to_restore_;
}

ScopedDeactivateContext::ScopedDeactivateContext() {
  auto* tls = &tls_data;

  tls->depth++;
  if (tls->context == nullptr) {
    // Nothing tracked as active on this thread by ScopedActivateContext, so
    // there is nothing we know how to clear. In particular, this avoids
    // needing a Context instance just to dispatch a "clear" call.
    to_restore_ = nullptr;
    return;
  }

  VLOG(3) << "ScopedDeactivateContext clearing context for device "
          << tls->device_ordinal;
  to_restore_ = tls->context;
  to_restore_->SetInactive();
  tls->context = nullptr;
  tls->device_ordinal = -1;
}

ScopedDeactivateContext::~ScopedDeactivateContext() {
  auto* tls = &tls_data;

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (to_restore_ == nullptr) {
    return;
  }

  to_restore_->SetActive();
  tls->device_ordinal = to_restore_->device_ordinal();
  tls->context = to_restore_;
}
}  // namespace stream_executor::gpu
