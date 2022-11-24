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

#include "tensorflow/tsl/framework/cancellation.h"

#include <atomic>
#include <forward_list>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"

namespace tsl {

const CancellationToken CancellationManager::kInvalidToken = -1;

CancellationManager::CancellationManager(int num_shards)
    : is_cancel_requested_(false),
      is_cancelled_(false),
      next_cancellation_token_(0),
      state_(this, num_shards) {
  DCHECK_GT(num_shards, 0);
}

CancellationManager::CancellationManager(CancellationManager* parent,
                                         int num_shards)
    : next_cancellation_token_(0), parent_(parent), state_(this, num_shards) {
  DCHECK_GT(num_shards, 0);
  bool registered = parent->RegisterChild(this);
  is_cancelled_.store(registered);
  is_cancel_requested_.store(registered);
}

void CancellationManager::StartCancel() {
  // An "OK" status will not be logged by a callback registered by
  // RegisterCallbackWithErrorLogging.
  StartCancelWithStatus(OkStatus());
}

void CancellationManager::StartCancelWithStatus(const Status& status) {
  if (is_cancel_requested_.exchange(true)) {
    return;
  }

  State* state = state_.TryGet();
  std::forward_list<CancellationManager*> children_to_cancel;
  if (state != nullptr) {
    mutex_lock l(children_mu_);
    // Remove all children from the list of children.
    CancellationManager* child = state->first_child;
    while (child != nullptr) {
      children_to_cancel.push_front(child);
      child->is_removed_from_parent_ = true;
      child = child->next_sibling_;
    }
    state->first_child = nullptr;
  }

  std::vector<CallbackConfiguration> callbacks_to_run;
  if (state != nullptr) {
    for (auto& bucket : state->callback_buckets) {
      mutex_lock l(bucket.mu);
      for (auto& callback : bucket.callbacks) {
        callbacks_to_run.emplace_back(std::move(callback.second));
      }
      bucket.callbacks.clear();
    }
  }

  // We call these callbacks without holding mu_, so that concurrent
  // calls to DeregisterCallback, which can happen asynchronously, do
  // not block. The callbacks remain valid because any concurrent call
  // to DeregisterCallback will block until the
  // cancelled_notification_ is notified.
  for (auto& config : callbacks_to_run) {
    if (!status.ok() && config.log_error) {
      LOG(WARNING) << "Cancellation callback \"" << config.name
                   << "\" is triggered due to a "
                   << (StatusGroup::IsDerived(status) ? "derived" : "root")
                   << " error: " << status.ToString();
    }
    config.callback();
  }
  for (CancellationManager* child : children_to_cancel) {
    child->StartCancelWithStatus(status);
  }
  // Sets is_cancelled_ before Notify() to ensure we will at least notify once.
  is_cancelled_.store(true);
  if (state_.IsInitialized()) {
    state_.Get().cancelled_notification.Notify();
  }
}

bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, "", false});
}

bool CancellationManager::RegisterCallbackWithErrorLogging(
    CancellationToken token, CancelCallback callback,
    tsl::StringPiece callback_name) {
  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, std::string(callback_name), true});
}

bool CancellationManager::RegisterCallbackConfig(CancellationToken token,
                                                 CallbackConfiguration config) {
  DCHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";
  CallbackBucket& bucket = GetCallbackBucket(token);
  mutex_lock l(bucket.mu);
  // Check `is_cancel_requested_` while holding the lock, to make sure either:
  // a) the callback is invoked during StartCancel(), or
  // b) this method returns false.
  bool should_register = !is_cancel_requested_.load();
  if (should_register) {
    bucket.callbacks[token] = std::move(config);
  }
  return should_register;
}

bool CancellationManager::DeregisterCallback(CancellationToken token) {
  CallbackBucket& bucket = GetCallbackBucket(token);
  bucket.mu.lock();
  // Checking `is_cancel_requested_` while holding the lock, to make sure
  // either:
  // a) the callback won't be invoked during StartCancel(), or
  // b) this method block until all callbacks are invoked.
  if (is_cancel_requested_.load()) {
    bucket.mu.unlock();
    // Wait for all of the cancellation callbacks to be called. This
    // wait ensures that the caller of DeregisterCallback does not
    // return immediately and free objects that may be used in the
    // execution of any currently pending callbacks in StartCancel.
    state_.Get().cancelled_notification.WaitForNotification();
    return false;
  } else {
    bucket.callbacks.erase(token);
    bucket.mu.unlock();
    return true;
  }
}

bool CancellationManager::RegisterChild(CancellationManager* child) {
  mutex_lock l(children_mu_);
  if (is_cancel_requested_.load()) {
    child->is_removed_from_parent_ = true;
    return true;
  }

  // Push `child` onto the front of the list of children.
  State& state = state_.Get();
  CancellationManager* current_head = state.first_child;
  state.first_child = child;
  child->prev_sibling_ = nullptr;
  child->next_sibling_ = current_head;
  if (current_head) {
    current_head->prev_sibling_ = child;
  }

  return false;
}

void CancellationManager::DeregisterChild(CancellationManager* child) {
  DCHECK_EQ(child->parent_, this);
  Notification* cancelled_notification = nullptr;
  {
    mutex_lock l(children_mu_);
    if (!child->is_removed_from_parent_) {
      // Remove the child from this manager's list of children.
      State* state = state_.TryGet();
      DCHECK(state != nullptr);

      if (child->prev_sibling_ == nullptr) {
        // The child was at the head of the list.
        DCHECK_EQ(state->first_child, child);
        state->first_child = child->next_sibling_;
      } else {
        child->prev_sibling_->next_sibling_ = child->next_sibling_;
      }

      if (child->next_sibling_ != nullptr) {
        child->next_sibling_->prev_sibling_ = child->prev_sibling_;
      }

      child->is_removed_from_parent_ = true;
    }
    if (is_cancel_requested_ && !is_cancelled_) {
      // Notice that state_ may not be initialized here, if the child is
      // registered then unregistered when the parent is cancelling.
      cancelled_notification = &state_.Get().cancelled_notification;
    }
  }

  // Wait for an ongoing call to StartCancel() to finish. This wait ensures that
  // the caller of DeregisterChild does not return immediately and free a child
  // that may currently be being cancelled by StartCancel().
  if (cancelled_notification) {
    cancelled_notification->WaitForNotification();
  }
}

CancellationManager::CallbackBucket& CancellationManager::GetCallbackBucket(
    CancellationToken token) {
  auto& buckets = state_.Get().callback_buckets;
  return buckets[token % buckets.size()];
}

bool CancellationManager::TryDeregisterCallback(CancellationToken token) {
  CallbackBucket& bucket = GetCallbackBucket(token);
  mutex_lock l(bucket.mu);
  // Check `is_cancel_requested_` again while holding the lock. See the comment
  // in `DeregisterCallback` for more details.
  if (is_cancel_requested_.load()) {
    return false;
  }
  bucket.callbacks.erase(token);
  return true;
}

CancellationManager::~CancellationManager() {
  if (parent_) {
    parent_->DeregisterChild(this);
  }
  // If state_ is not initialized, there's no child or callback registered.
  if (state_.IsInitialized()) {
    StartCancel();
  }
}

Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    CancelCallback callback,
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
  return OkStatus();
}

}  // end namespace tsl
