/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/framework/cancellation.h"

#include <forward_list>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"

namespace tsl {

const CancellationToken CancellationManager::kInvalidToken = -1;

CancellationManager::CancellationManager()
    : is_cancelling_(false),
      is_cancelled_(false),
      next_cancellation_token_(0) {}

CancellationManager::CancellationManager(CancellationManager* parent)
    : is_cancelling_(false), next_cancellation_token_(0), parent_(parent) {
  is_cancelled_ = parent->RegisterChild(this);
}

void CancellationManager::StartCancel() {
  // An "OK" status will not be logged by a callback registered by
  // RegisterCallbackWithErrorLogging.
  StartCancelWithStatus(absl::OkStatus());
}

void CancellationManager::StartCancelWithStatus(const absl::Status& status) {
  gtl::FlatMap<CancellationToken, CallbackConfiguration> callbacks_to_run;
  std::forward_list<CancellationManager*> children_to_cancel;
  absl::Notification* cancelled_notification = nullptr;
  {
    absl::MutexLock l(&mu_);
    if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
      return;
    }
    is_cancelling_ = true;
    if (state_) {
      std::swap(state_->callbacks, callbacks_to_run);

      // Remove all children from the list of children.
      CancellationManager* child = state_->first_child;
      while (child != nullptr) {
        children_to_cancel.push_front(child);
        child->is_removed_from_parent_ = true;
        child = child->next_sibling_;
      }
      state_->first_child = nullptr;

      cancelled_notification = &state_->cancelled_notification;
    }
  }
  // We call these callbacks without holding mu_, so that concurrent
  // calls to DeregisterCallback, which can happen asynchronously, do
  // not block. The callbacks remain valid because any concurrent call
  // to DeregisterCallback will block until the
  // cancelled_notification_ is notified.
  for (auto key_and_value : callbacks_to_run) {
    CallbackConfiguration& config = key_and_value.second;
    if (!status.ok() && config.log_error) {
      LOG(WARNING) << "Cancellation callback \"" << config.name
                   << "\" is triggered due to a "
                   << (StatusGroup::IsDerived(status) ? "derived" : "root")
                   << " error: " << status;
    }
    config.callback();
  }
  for (CancellationManager* child : children_to_cancel) {
    child->StartCancelWithStatus(status);
  }
  {
    absl::MutexLock l(&mu_);
    is_cancelling_ = false;
    is_cancelled_.store(true, std::memory_order_release);
  }
  if (cancelled_notification) {
    cancelled_notification->Notify();
  }
}

bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, "", false});
}

bool CancellationManager::RegisterCallbackWithErrorLogging(
    CancellationToken token, CancelCallback callback,
    absl::string_view callback_name) {
  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, std::string(callback_name), true});
}

bool CancellationManager::RegisterCallbackConfig(CancellationToken token,
                                                 CallbackConfiguration config) {
  DCHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";
  absl::MutexLock l(&mu_);
  bool should_register = !is_cancelled_ && !is_cancelling_;
  if (should_register) {
    if (!state_) {
      state_ = absl::make_unique<State>();
    }
    std::swap(state_->callbacks[token], config);
  }
  return should_register;
}

bool CancellationManager::DeregisterCallback(CancellationToken token) {
  mu_.Lock();
  if (is_cancelled_) {
    mu_.Unlock();
    return false;
  } else if (is_cancelling_) {
    absl::Notification* cancelled_notification =
        state_ ? &state_->cancelled_notification : nullptr;
    mu_.Unlock();
    // Wait for all of the cancellation callbacks to be called. This
    // wait ensures that the caller of DeregisterCallback does not
    // return immediately and free objects that may be used in the
    // execution of any currently pending callbacks in StartCancel.
    if (cancelled_notification) {
      cancelled_notification->WaitForNotification();
    }
    return false;
  } else {
    if (state_) {
      state_->callbacks.erase(token);
    }
    mu_.Unlock();
    return true;
  }
}

bool CancellationManager::RegisterChild(CancellationManager* child) {
  absl::MutexLock l(&mu_);
  if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
    child->is_removed_from_parent_ = true;
    return true;
  }

  if (!state_) {
    state_ = absl::make_unique<State>();
  }

  // Push `child` onto the front of the list of children.
  CancellationManager* current_head = state_->first_child;
  state_->first_child = child;
  child->prev_sibling_ = nullptr;
  child->next_sibling_ = current_head;
  if (current_head) {
    current_head->prev_sibling_ = child;
  }

  return false;
}

void CancellationManager::DeregisterChild(CancellationManager* child) {
  DCHECK_EQ(child->parent_, this);
  absl::Notification* cancelled_notification = nullptr;
  {
    absl::MutexLock l(&mu_);
    if (!child->is_removed_from_parent_) {
      // Remove the child from this manager's list of children.
      DCHECK(state_);

      if (child->prev_sibling_ == nullptr) {
        // The child was at the head of the list.
        DCHECK_EQ(state_->first_child, child);
        state_->first_child = child->next_sibling_;
      } else {
        child->prev_sibling_->next_sibling_ = child->next_sibling_;
      }

      if (child->next_sibling_ != nullptr) {
        child->next_sibling_->prev_sibling_ = child->prev_sibling_;
      }

      child->is_removed_from_parent_ = true;
    }
    if (is_cancelling_) {
      cancelled_notification = &state_->cancelled_notification;
    }
  }

  // Wait for an ongoing call to StartCancel() to finish. This wait ensures that
  // the caller of DeregisterChild does not return immediately and free a child
  // that may currently be being cancelled by StartCancel().
  if (cancelled_notification) {
    cancelled_notification->WaitForNotification();
  }
}

bool CancellationManager::TryDeregisterCallback(CancellationToken token) {
  absl::MutexLock lock(&mu_);
  if (is_cancelled_ || is_cancelling_) {
    return false;
  } else {
    if (state_) {
      state_->callbacks.erase(token);
    }
    return true;
  }
}

CancellationManager::~CancellationManager() {
  if (parent_) {
    parent_->DeregisterChild(this);
  }
  if (state_) {
    StartCancel();
  }
}

bool CancellationManager::IsCancelling() {
  absl::MutexLock lock(&mu_);
  return is_cancelling_;
}

absl::Status RegisterCancellationCallback(
    CancellationManager* cancellation_manager, CancelCallback callback,
    std::function<void()>* deregister_fn) {
  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token, std::move(callback))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {};
  }
  return absl::OkStatus();
}

}  // end namespace tsl
