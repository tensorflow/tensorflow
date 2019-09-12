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

#include "tensorflow/core/common_runtime/eager/eager_executor.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {

EagerExecutor::EagerExecutor(bool async)
    : next_node_id_(0),
      thread_(async ? tensorflow::Env::Default()->StartThread(
                          tensorflow::ThreadOptions(), "eager_async_executor",
                          std::bind(&EagerExecutor::Run, this))
                    : nullptr) {}

EagerExecutor::~EagerExecutor() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  state_ = ExecutorState::kShutDown;
  nodes_pending_.notify_all();
}

Status EagerExecutor::ShutDown() {
  {
    std::vector<std::unique_ptr<NodeItem>> items_to_destroy;
    bool has_thread;
    Status status;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      if (state_ != ExecutorState::kShutDown) {
        // if the state is kShutDown, we don't return here because we want to
        // make sure the executor thread has ended (if there is one).
        // So, we fall through to
        // thread_exited_notification_.WaitForNotification() below.
        state_ = ExecutorState::kShuttingDown;
      }
      WaitForOrDestroyAllPendingNodes(&l, &items_to_destroy);
      state_ = ExecutorState::kShutDown;
      has_thread = thread_ != nullptr;
      status = status_;
      if (has_thread) {
        nodes_pending_.notify_all();
      }
    }
    for (auto& item : items_to_destroy) {
      item->node->Abort(status);
    }
    if (!has_thread) {
      return status;
    }
  }

  thread_exited_notification_.WaitForNotification();
  tensorflow::mutex_lock l(node_queue_mutex_);
  return status_;
}

void EagerExecutor::WaitForOrDestroyAllPendingNodes(
    mutex_lock* lock,
    std::vector<std::unique_ptr<NodeItem>>* nodes_to_destroy) {
  if (state_ == ExecutorState::kShutDown) {
    return;
  }
  if (thread_ == nullptr) {
    Status status = status_;
    if (status.ok()) {
      status = errors::FailedPrecondition(
          "Aborting eager nodes because EagerExecutor is being shut down "
          "before it got a thread to run the nodes");
      status_ = status;
    }
    while (!node_queue_.empty()) {
      nodes_to_destroy->push_back(std::move(node_queue_.front()));
      node_queue_.pop();
    }
    for (auto& it : unfinished_nodes_) {
      nodes_to_destroy->push_back(absl::WrapUnique(it.second));
    }
    unfinished_nodes_.clear();
    return;
  }

  // It is OK to ignore the returned status here because it will be saved
  // as the final status_.
  WaitForAllPendingNodesLocked(lock).IgnoreError();
}

bool EagerExecutor::Async() const {
  return thread_ != nullptr;
}

const char* EagerExecutor::StateStringLocked() {
  switch (state_) {
    case ExecutorState::kActive:
      return "Active";
    case ExecutorState::kShuttingDown:
      return "ShuttingDown";
    case ExecutorState::kShutDown:
      return "ShutDown";
  }
}

Status EagerExecutor::Add(std::unique_ptr<EagerNode> node) {
  Status status;

  // If we are unable to add the node to the queue, we must call Abort. However,
  // we want to do that outside of the scope of the lock since the Abort may
  // try to call EagerExecutor::Add()
  {
    tensorflow::mutex_lock l(node_queue_mutex_);
    VLOG(3) << "Add node [id " << next_node_id_ << "]" << node->DebugString()
            << " with status: " << status_.ToString();
    if (state_ != ExecutorState::kActive) {
      status = errors::FailedPrecondition(
          "EagerExecutor accepts new EagerNodes to run only in Active state. "
          "Current state is '",
          StateStringLocked(), "'");
    } else {
      DCHECK(thread_) << "EnableAsync should have been called before Add";
      status = status_;
      if (status.ok()) {
        auto item = absl::make_unique<NodeItem>();
        item->id = next_node_id_++;
        item->node = std::move(node);
        node_queue_.push(std::move(item));

        // If there were no previous nodes pending, wake the run thread to start
        // processing requests again.
        if (node_queue_.size() == 1) {
          nodes_pending_.notify_all();
        }

        return Status::OK();
      }
    }
  }

  // Node needs to be aborted since it was not added to the queue
  node->Abort(status);
  return status;
}

tensorflow::Status EagerExecutor::WaitForAllPendingNodes() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  return WaitForAllPendingNodesLocked(&l);
}

tensorflow::Status EagerExecutor::WaitForAllPendingNodesLocked(
    mutex_lock* lock) {
  tensorflow::condition_variable cond;
  // Don't wait if an error is already set.
  if (!status_.ok()) return status_;
  if (node_queue_.empty() && unfinished_nodes_.empty())
    return tensorflow::Status::OK();
  auto last_id = next_node_id_ - 1;
  VLOG(3) << "Wait for Node: [id " << last_id << "] ";
  node_done_notifications_.insert(std::make_pair(last_id, &cond));
  cond.wait(*lock);
  // Note that we could be woken up if an error occurs, even though the node has
  // not actually executed.
  return status_;
}

void EagerExecutor::ClearError() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  // TODO(iga): Check state_ and return an error if it is not kActive.
  if (status_.ok()) return;
  // If an error was set, node_done_notifications_ and node_queue_ should have
  // been cleared, and no new entries should have been added since.
  DCHECK(node_done_notifications_.empty());
  DCHECK(node_queue_.empty());
  status_ = tensorflow::Status::OK();
  nodes_pending_.notify_all();
}

tensorflow::Status EagerExecutor::status() const {
  tf_shared_lock l(node_queue_mutex_);
  return status_;
}

void EagerExecutor::NodeDone(NodeItem* item, const Status& status) {
  VLOG(3) << "Node Done: [id " << item->id << "] " << item->node->DebugString()
          << " with status: " << status.ToString();
  std::unique_ptr<NodeItem> current_item;
  std::vector<std::unique_ptr<NodeItem>> items_to_destroy;
  {
    mutex_lock l(node_queue_mutex_);
    if (!status_.ok()) return;
    bool need_notification = false;
    if (!node_queue_.empty() && item == node_queue_.front().get()) {
      need_notification = unfinished_nodes_.empty();
      current_item = std::move(node_queue_.front());
      node_queue_.pop();
    } else {
      DCHECK(!unfinished_nodes_.empty());
      need_notification = item->id == unfinished_nodes_.begin()->first;
      auto erase_result = unfinished_nodes_.erase(item->id);
      DCHECK_GT(erase_result, 0);
      current_item = absl::WrapUnique(item);
    }
    if (!status.ok()) {
      need_notification = true;
      status_ = status;
      // We remove any pending ops so that we don't try to execute them if
      // ClearError is called.
      errors::AppendToMessage(&status_,
                              ". Encountered when executing an operation using "
                              "EagerExecutor. This error cancels all future "
                              "operations and poisons their output tensors.");
      while (!node_queue_.empty()) {
        items_to_destroy.push_back(std::move(node_queue_.front()));
        node_queue_.pop();
      }
      for (auto& it : unfinished_nodes_) {
        items_to_destroy.push_back(absl::WrapUnique(it.second));
      }
      unfinished_nodes_.clear();
    }
    if (!node_done_notifications_.empty() && need_notification) {
      uint64 upperbound_id = 0;
      if (!unfinished_nodes_.empty()) {
        upperbound_id = unfinished_nodes_.begin()->first - 1;
      } else if (!node_queue_.empty()) {
        upperbound_id = node_queue_.front()->id - 1;
      } else {
        upperbound_id = next_node_id_ - 1;
      }
      // Note that we notify all waiting threads in case an error has
      // occurred. These calling threads are responsible for checking status_
      // before proceeding.
      const auto range =
          status_.ok()
              ? make_pair(node_done_notifications_.lower_bound(item->id),
                          node_done_notifications_.upper_bound(upperbound_id))
              : make_pair(node_done_notifications_.begin(),
                          node_done_notifications_.end());
      for (auto it = range.first; it != range.second; ++it) {
        it->second->notify_all();
      }
      node_done_notifications_.erase(range.first, range.second);
    }
  }
  for (auto& item : items_to_destroy) {
    item->node->Abort(status);
  }
  // nodes_to_destroy will be destructed here, while not holding
  // node_queue_mutex_. This is important because, unfortunately, some nodes'
  // destructors can enqueue more operations onto this executor and cause
  // a deadlock.
}

void EagerExecutor::Run() {
  auto thread_exited_notifier =
      gtl::MakeCleanup([this] { thread_exited_notification_.Notify(); });
  while (true) {
    NodeItem* curr_item_raw;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      while (node_queue_.empty() || !status_.ok()) {
        if (state_ == ExecutorState::kShutDown) return;
        nodes_pending_.wait(l);
      }
      // Obtain raw pointer since we don't want to remove from the queue until
      // the node has been run. Otherwise, WaitForAllPendingNodes can return
      // too early.
      // Note, we don't std::move from the here because the front of the queue
      // will then contain a nullptr. This can be a problem in
      // WaitForAllPendingNodes where we get the top EagerNode pointer
      // and register a notification for its completion.
      curr_item_raw = node_queue_.front().get();
    }
    VLOG(3) << "Running Node: [id " << curr_item_raw->id << "] "
            << curr_item_raw->node->DebugString();
    AsyncEagerNode* async_node_raw = curr_item_raw->node->AsAsync();
    if (async_node_raw == nullptr) {
      tensorflow::Status status = curr_item_raw->node->Run();
      NodeDone(curr_item_raw, status);
    } else {
      async_node_raw->RunAsync([this, curr_item_raw](const Status& status) {
        NodeDone(curr_item_raw, status);
      });
      {
        tensorflow::mutex_lock l(node_queue_mutex_);
        // If false, NodeDone has been called.
        if (!node_queue_.empty() &&
            curr_item_raw == node_queue_.front().get()) {
          node_queue_.front().release();
          node_queue_.pop();
          unfinished_nodes_.emplace_hint(unfinished_nodes_.end(),
                                         curr_item_raw->id, curr_item_raw);
        }
      }
    }
  }
}

}  // namespace tensorflow
