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

namespace tensorflow {

EagerExecutor::~EagerExecutor() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  thread_done_ = true;
  nodes_pending_.notify_all();
}

void EagerExecutor::EnableAsync() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  if (thread_ == nullptr) {
    thread_.reset(tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), "eager_async_executor",
        std::bind(&EagerExecutor::Run, this)));
  }
}

Status EagerExecutor::Add(std::unique_ptr<EagerNode> node) {
  tensorflow::mutex_lock l(node_queue_mutex_);
  DCHECK(thread_) << "EnableAsync should have been called before Add";
  if (!status_.ok()) {
    // node will be automatically deleted
    return status_;
  }

  node_queue_.push(std::move(node));

  // If there were no previous nodes pending, wake the run thread to start
  // processing requests again.
  if (node_queue_.size() == 1) {
    nodes_pending_.notify_all();
  }

  return Status::OK();
}

tensorflow::Status EagerExecutor::WaitForAllPendingNodes() {
  tensorflow::condition_variable cond;
  tensorflow::mutex_lock l(node_queue_mutex_);
  // Don't wait if an error is already set.
  if (!status_.ok()) return status_;
  if (node_queue_.empty()) return tensorflow::Status::OK();
  EagerNode* last_node = node_queue_.back().get();
  node_done_notifications_.insert(std::make_pair(last_node, &cond));
  cond.wait(l);
  // Note that we could be woken up if an error occurs, even though the node has
  // not actually executed.
  return status_;
}

void EagerExecutor::ClearError() {
  tensorflow::mutex_lock l(node_queue_mutex_);
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

void EagerExecutor::Run() {
  while (true) {
    EagerNode* curr_node;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      while (node_queue_.empty() || !status_.ok()) {
        if (thread_done_) return;
        nodes_pending_.wait(l);
      }
      // Obtain raw pointer since we don't want to remove from the queue until
      // the node has been run.
      curr_node = node_queue_.front().get();
    }
    tensorflow::Status status = curr_node->Run();
    const bool ok = status.ok();
    tensorflow::mutex_lock l(node_queue_mutex_);
    node_queue_.pop();
    if (!ok) {
      status_ = status;
      // We remove any pending ops so that we don't try to execute them if
      // ClearError is called.
      errors::AppendToMessage(&status,
                              ". Encountered when executing an operation using "
                              "EagerExecutor. This error cancels all future "
                              "operations and poisons their output tensors.");
      for (int i = 0; i < node_queue_.size(); ++i) {
        node_queue_.front()->Abort(status);
        // Dequeue and delete nodes
        node_queue_.pop();
      }
    }
    if (!node_done_notifications_.empty()) {
      // Note that we notify all waiting threads in case an error has occurred.
      // These calling threads are responsible for checking status_ before
      // proceeding.
      const auto range = ok ? node_done_notifications_.equal_range(curr_node)
                            : make_pair(node_done_notifications_.begin(),
                                        node_done_notifications_.end());
      for (auto it = range.first; it != range.second; ++it) {
        it->second->notify_all();
      }
      node_done_notifications_.erase(range.first, range.second);
    }
  }
}

}  // namespace tensorflow
