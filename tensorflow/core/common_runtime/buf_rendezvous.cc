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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"

namespace tensorflow {

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
    status_.Update(s);
    hook_table_.swap(dummy_table);
  }
  PurgeTable(s, &dummy_table);
}

void BufRendezvous::PurgeTable(const Status& s, HookTable* table) {
  for (auto& it : *table) {
    Hook* h = it.second;
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
  return strings::StrCat("[dev:", (prod_dev ? prod_dev->name() : "none"),
                         ", ctx:", reinterpret_cast<uint64>(prod_ctx),
                         ", val:", reinterpret_cast<uint64>(prod_value),
                         ", pcb:", reinterpret_cast<uint64>(&prod_cb),
                         ", ccb:", reinterpret_cast<uint64>(&cons_cb), "]");
}

void BufRendezvous::ProvideBuf(const string& key, Device* dev,
                               DeviceContext* dev_ctx, const Tensor* v,
                               const AllocatorAttributes& attr,
                               const ProducerCallback& done) {
  Hook* h = nullptr;
  Status providebuf_status;
  do {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      providebuf_status = status_;
      break;
    } else {
      auto it = hook_table_.find(key);
      if (it == hook_table_.end()) {
        h = new Hook;
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
      // If consumer is waiting, kick off right away, removing Hook from table.
      if (h->cons_cb != nullptr) {
        hook_table_.erase(it);
      } else {
        h = nullptr;
      }
    }
  } while (false);
  if (h) {
    h->cons_cb(Status::OK(), h);
  }
  if (!providebuf_status.ok()) {
    done(providebuf_status);
  }
}

void BufRendezvous::ConsumeBuf(const string& key,
                               const ConsumerCallback& done) {
  Hook* existing_hook = nullptr;
  Status consumebuf_status;
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
      Hook* h = new Hook;
      hook_table_[key] = h;
      h->cons_cb = done;
      return;
    }
  } while (false);
  if (existing_hook) {
    existing_hook->cons_cb(Status::OK(), existing_hook);
    return;
  }
  if (!consumebuf_status.ok()) {
    done(consumebuf_status, nullptr);
    return;
  }
}

/*static*/
void BufRendezvous::DoneWithHook(Hook* h) {
  h->prod_cb(Status::OK());
  delete h;
}

void BufRendezvous::LogContents() {
  mutex_lock l(mu_);
  LOG(INFO) << strings::StrCat("BufRendezvous ",
                               strings::Hex(reinterpret_cast<uint64>(this)),
                               " step_id=", step_id_, " current contents:");
  for (auto it : hook_table_) {
    LOG(INFO) << it.first << ":" << it.second->DebugString();
  }
}

}  // namespace tensorflow
