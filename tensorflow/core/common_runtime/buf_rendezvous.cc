/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/buf_rendezvous.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"

namespace tensorflow {
namespace {
void DeregisterCancellation(BufRendezvous::Hook* h) {
  if (h->cancellation_manager != nullptr) {
    h->cancellation_manager->DeregisterCallback(h->cancellation_token);
    h->cancellation_manager = nullptr;
    h->cancellation_token = CancellationManager::kInvalidToken;
  }
}
}  // namespace

BufRendezvous::~BufRendezvous() {
  mutex_lock l(mu_);
  if (!hook_table_.empty()) {
    PurgeTable(errors::Internal("Delete called on non-empty BufRendezvous"),
               &hook_table_);
  }
}

void BufRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  HookTable dummy_table;
  {
    mutex_lock l(mu_);
    // Use a "derived" status as the status for the rendezvous. Derived
    // status messages are ignored when aggregating errors across devices: this
    // allows us to prefer our original status message over any cancellation
    // related errors.
    status_.Update(StatusGroup::MakeDerived(s));
    hook_table_.swap(dummy_table);
  }
  PurgeTable(s, &dummy_table);
}

void BufRendezvous::PurgeTable(const Status& s, HookTable* table) {
  for (auto& it : *table) {
    Hook* h = it.second;
    if (h->cancellation_manager != nullptr) {
      h->cancellation_manager->TryDeregisterCallback(h->cancellation_token);
    }
    if (h->cons_cb != nullptr) {
      h->cons_cb(s, nullptr);
    }
    if (h->prod_cb != nullptr) {
      h->prod_cb(s);
    }
    delete h;
  }
  table->clear();
}

string BufRendezvous::Hook::DebugString() const {
  return absl::StrCat("[dev:", (prod_dev ? prod_dev->name() : "none"),
                      ", ctx:", reinterpret_cast<uint64>(prod_ctx),
                      ", val:", reinterpret_cast<uint64>(prod_value),
                      ", pcb:", reinterpret_cast<uint64>(&prod_cb),
                      ", ccb:", reinterpret_cast<uint64>(&cons_cb), "]");
}

void BufRendezvous::ProvideBuf(const string& key, Device* dev,
                               DeviceContext* dev_ctx, const Tensor* v,
                               const AllocatorAttributes& attr,
                               const ProducerCallback& done,
                               CancellationManager* cancellation_manager) {
  Hook* h = nullptr;
  Status providebuf_status;
  do {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      providebuf_status = status_;
      break;
    } else {
      CancellationToken cancellation_token = CancellationManager::kInvalidToken;
      auto it = hook_table_.find(key);
      if (it == hook_table_.end()) {
        if (cancellation_manager != nullptr) {
          cancellation_token = cancellation_manager->get_cancellation_token();
        }
        h = new Hook(cancellation_manager, cancellation_token);
        it = hook_table_.insert(std::make_pair(key, h)).first;
      } else {
        if (it->second->prod_cb != nullptr) {
          providebuf_status = errors::Internal(
              "BufRendezvous::ProvideBuf already called for key ", key);
          break;
        }
        h = it->second;
      }
      // Populate Hook with all of the prod values.
      h->prod_dev = dev;
      h->prod_ctx = dev_ctx;
      h->prod_value = v;
      h->prod_attr = attr;
      h->prod_cb = done;
      if (h->cons_cb != nullptr) {
        // If consumer is waiting, kick off right away, removing Hook from
        // table.
        hook_table_.erase(it);
      } else {
        if (cancellation_manager != nullptr &&
            !cancellation_manager->RegisterCallback(
                cancellation_token, [this, key]() { CancelHook(key); })) {
          // Register cancellation callback with CancellationManager.  If it is
          // already cancelled, call done immediately with cancelled status.
          providebuf_status = errors::Cancelled(
              "Operation was cancelled for BufRendezvous key ", key);
          hook_table_.erase(it);
          delete h;
        }
        h = nullptr;
      }
    }
  } while (false);
  if (h) {
    DeregisterCancellation(h);
    h->cons_cb(OkStatus(), h);
  }
  if (!providebuf_status.ok()) {
    done(providebuf_status);
  }
}

void BufRendezvous::ConsumeBuf(const string& key, const string& device_name,
                               const uint64 device_incarnation,
                               const ConsumerCallback& done,
                               CancellationManager* cancellation_manager) {
  // Check the incarnation in the request matches the current device
  // incarnation of the producer.
  Device* device;
  Status consumebuf_status = dev_mgr_->LookupDevice(device_name, &device);
  if (consumebuf_status.ok() &&
      device->attributes().incarnation() != device_incarnation) {
    consumebuf_status = errors::FailedPrecondition(
        "RecvBuf expects a different device incarnation: ", device_incarnation,
        " vs. ", device->attributes().incarnation(),
        ". Your worker job that contains the device (\"", device_name,
        "\") was probably restarted. Check your "
        "worker job for the reason why it was restarted.");
  }
  if (!consumebuf_status.ok()) {
    done(consumebuf_status, nullptr);
    return;
  }

  Hook* existing_hook = nullptr;
  do {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      consumebuf_status = status_;
      break;
    }
    auto it = hook_table_.find(key);
    if (it != hook_table_.end()) {
      // Prepare to consume immediately.
      if (it->second->cons_cb) {
        consumebuf_status =
            errors::Internal("Second consumer arrived for key ", key);
        break;
      }
      existing_hook = it->second;
      hook_table_.erase(it);
      existing_hook->cons_cb = done;
    } else {
      // Hang consumer callback on the Hook.
      CancellationToken cancellation_token = CancellationManager::kInvalidToken;
      bool already_cancelled = false;
      if (cancellation_manager != nullptr) {
        cancellation_token = cancellation_manager->get_cancellation_token();
        already_cancelled = !cancellation_manager->RegisterCallback(
            cancellation_token, [this, key]() { CancelHook(key); });
      }
      if (already_cancelled) {
        consumebuf_status = errors::Cancelled(
            "Operation was cancelled for BufRendezvous key ", key);
      } else {
        Hook* h = new Hook(cancellation_manager, cancellation_token);
        h->cons_cb = done;
        it = hook_table_.insert(std::make_pair(key, h)).first;
        return;
      }
    }
  } while (false);
  if (existing_hook) {
    DeregisterCancellation(existing_hook);
    existing_hook->cons_cb(OkStatus(), existing_hook);
    return;
  }
  if (!consumebuf_status.ok()) {
    done(consumebuf_status, nullptr);
    return;
  }
}

void BufRendezvous::CancelHook(const string& key) {
  Hook* h = nullptr;
  {
    mutex_lock l(mu_);
    auto it = hook_table_.find(key);
    if (it == hook_table_.end()) return;
    h = it->second;
    hook_table_.erase(it);
  }
  if (h != nullptr) {
    auto s = errors::Cancelled("Operation was cancelled for BufRendezvous key ",
                               key);
    if (h->prod_cb != nullptr) {
      h->prod_cb(s);
    }
    if (h->cons_cb != nullptr) {
      h->cons_cb(s, /*Hook=*/nullptr);
    }
    delete h;
  }
}

/*static*/
void BufRendezvous::DoneWithHook(Hook* h) {
  h->prod_cb(OkStatus());
  delete h;
}

void BufRendezvous::LogContents() {
  mutex_lock l(mu_);
  LOG(INFO) << strings::StrCat("BufRendezvous ",
                               strings::Hex(reinterpret_cast<uint64>(this)),
                               " step_id=", step_id_, " current contents:");
  for (const auto& it : hook_table_) {
    LOG(INFO) << it.first << ":" << it.second->DebugString();
  }
}

}  // namespace tensorflow
