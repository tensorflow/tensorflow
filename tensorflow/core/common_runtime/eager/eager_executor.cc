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

#include <forward_list>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {

EagerExecutor::EagerExecutor(bool async)
    : next_node_id_(0),
      ok_(true),
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
    std::vector<core::RefCountPtr<NodeItem>> items_to_destroy;
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
      // It is OK to ignore the returned status here because it will be saved
      // as the final status_.
      WaitForAllPendingNodesLocked(&l).IgnoreError();
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

  return status();
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

Status EagerExecutor::AddOrExecute(std::unique_ptr<EagerNode> node) {
  Status status;
  core::RefCountPtr<NodeItem> item(new NodeItem);
  item->id = next_node_id_++;
  item->node = std::move(node);
  item->state = NodeState::kPENDING;

  status = item->node->Prepare();
  if (!status.ok()) {
    item->node->Abort(status);
    return status;
  }

  // Inline execution in sync mode.
  if (!Async()) {
    status = this->status();
    if (status.ok()) {
      status = RunItem(std::move(item));
    }
    return status;
  } else {
    tensorflow::mutex_lock l(node_queue_mutex_);
    DVLOG(3) << "Add node [id " << item->id << "]" << item->node->DebugString()
             << " with status: " << status_.ToString();
    if (state_ != ExecutorState::kActive) {
      status = errors::FailedPrecondition(
          "EagerExecutor accepts new EagerNodes to run only in Active state. "
          "Current state is '",
          StateStringLocked(), "'");
    } else {
      status = status_;
      if (status.ok()) {
        node_queue_.push(std::move(item));
        // If there were no previous nodes pending, wake the run thread to
        // start processing requests again.
        if (node_queue_.size() == 1) {
          nodes_pending_.notify_all();
        }

        return Status::OK();
      }
    }
  }

  // If we are unable to add the node to the queue, we must call Abort. However,
  // we want to do that outside of the scope of the lock since the Abort may
  // try to call EagerExecutor::AddOrExecute()
  item->node->Abort(status);

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
  // node_queue_ must be empty in sync mode.
  DCHECK(Async() || node_queue_.empty());
  auto last_id = next_node_id_ - 1;
  DVLOG(3) << "Wait for Node: [id " << last_id << "] ";
  node_done_notifications_.insert(std::make_pair(last_id, &cond));
  cond.wait(*lock);
  // Note that we could be woken up if an error occurs, even though the node has
  // not actually executed.
  return status_;
}

void EagerExecutor::ClearError() {
  // TODO(iga): Check state_ and return an error if it is not kActive.
  if (ok()) return;

  tensorflow::mutex_lock l(node_queue_mutex_);
  // If an error was set, node_done_notifications_ and node_queue_ should have
  // been cleared, and no new entries should have been added since.
  DCHECK(node_done_notifications_.empty());
  DCHECK(node_queue_.empty());
  status_ = tensorflow::Status::OK();
  ok_ = true;
  nodes_pending_.notify_all();
}

void EagerExecutor::NodeDone(const core::RefCountPtr<NodeItem>& item,
                             const Status& status) {
  DVLOG(3) << "Node Done: [id " << item->id << "] " << item->node->DebugString()
           << " with status: " << status.ToString();
  DCHECK(item->state != NodeState::kDONE);
  auto previous_state = item->state;
  item->state = NodeState::kDONE;
  if (!ok()) return;

  std::forward_list<core::RefCountPtr<NodeItem>> items_to_destroy;
  {
    mutex_lock l(node_queue_mutex_);
    bool need_notification = false;
    if (previous_state == NodeState::kPENDING) {
      if (Async()) {
        DCHECK(!node_queue_.empty() && item.get() == node_queue_.front().get());
        need_notification = unfinished_nodes_.empty();
        node_queue_.pop();
      } else {
        need_notification = unfinished_nodes_.empty();
      }
    } else {
      need_notification = item->id == unfinished_nodes_.begin()->first;
      auto result = unfinished_nodes_.erase(item->id);
      DCHECK_GT(result, 0);
    }
    if (!status.ok()) {
      need_notification = true;
      status_ = status;
      ok_ = false;
      if (Async()) {
        // We remove any pending ops so that we don't try to execute them if
        // ClearError is called.
        errors::AppendToMessage(&status_,
                                "Encountered when executing an operation using "
                                "EagerExecutor. This error cancels all future "
                                "operations and poisons their output tensors.");
      }
      while (!node_queue_.empty()) {
        items_to_destroy.push_front(std::move(node_queue_.front()));
        node_queue_.pop();
      }
      for (auto& it : unfinished_nodes_) {
        items_to_destroy.push_front(std::move(it.second));
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
      DVLOG(3) << "Notify node done: [id " << item->id << " to "
               << upperbound_id << "] ";
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
    core::RefCountPtr<NodeItem> curr_item;
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
      curr_item.reset(node_queue_.front().get());
      curr_item->Ref();
    }
    Status status = RunItem(std::move(curr_item));
    if (!status.ok()) {
      VLOG(1) << "Failed to run item: " << status;
    }
  }
}

Status EagerExecutor::RunItem(core::RefCountPtr<NodeItem> item) {
  DVLOG(3) << "Running Node: [id " << item->id << "] "
           << item->node->DebugString();
  AsyncEagerNode* async_node = item->node->AsAsync();
  if (async_node == nullptr) {
    tensorflow::Status status = item->node->Run();
    NodeDone(item, status);
    return status;
  } else {
    auto* new_ref = item.get();
    new_ref->Ref();
    async_node->RunAsync([this, new_ref](const Status& status) {
      core::RefCountPtr<NodeItem> new_item(new_ref);
      NodeDone(new_item, status);
    });

    tensorflow::mutex_lock l(node_queue_mutex_);
    if (item->state == NodeState::kPENDING) {
      item->state = NodeState::kSCHEDULED;
      if (!node_queue_.empty() && item.get() == node_queue_.front().get()) {
        node_queue_.pop();
      }
      DVLOG(3) << "Add Node: [id " << item->id << "] to unfinished map.";
      unfinished_nodes_.emplace_hint(unfinished_nodes_.end(), item->id,
                                     std::move(item));
    }
    return status_;
  }
}

}  // namespace tensorflow
