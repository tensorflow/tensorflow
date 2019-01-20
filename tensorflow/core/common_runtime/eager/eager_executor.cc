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

EagerNode::EagerNode(tensorflow::uint64 id) : id(id) {}

EagerExecutor::~EagerExecutor() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  thread_done_ = true;
  nodes_pending_.notify_all();
}

tensorflow::uint64 EagerExecutor::NextId() {
  tensorflow::mutex_lock l(next_id_mutex_);
  return next_id_++;
}

void EagerExecutor::EnableAsync() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  if (thread_ == nullptr) {
    thread_.reset(tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), "eager_async_executor",
        std::bind(&EagerExecutor::Run, this)));
  }
}

void EagerExecutor::Add(EagerNode* node) {
  tensorflow::mutex_lock l(node_queue_mutex_);
  DCHECK(thread_) << "EnableAsync should have been called before Add";
  if (!status_.ok()) {
    delete node;
    return;
  }
  int64 qlen = node_queue_.size();
  if (qlen > 0) {
    if (node_queue_.back()->id >= node->id) {
      status_ = tensorflow::errors::InvalidArgument(
          "Inserting EagerNode with non-increasing ids:",
          node_queue_.back()->id, " vs ", node->id);
      delete node;
      return;
    }
    node_queue_.push(node);
  } else {
    node_queue_.push(node);
    nodes_pending_.notify_all();
  }
}

tensorflow::Status EagerExecutor::WaitFor(tensorflow::uint64 node_id) {
  return WaitImpl(false, node_id);
}

tensorflow::Status EagerExecutor::WaitForAllPendingNodes() {
  return WaitImpl(true, 0);
}

tensorflow::Status EagerExecutor::WaitImpl(bool wait_all,
                                           tensorflow::uint64 node_id) {
  tensorflow::condition_variable cond;
  tensorflow::mutex_lock l(node_queue_mutex_);
  // Don't wait if an error is already set.
  if (!status_.ok()) return status_;
  if (node_queue_.empty()) return tensorflow::Status::OK();
  if (wait_all) {
    node_id = node_queue_.back()->id;
  } else if (node_id < node_queue_.front()->id) {
    // Note that we are relying on the ops being dispatched sequentially from
    // the queue.
    return tensorflow::Status::OK();
  }
  node_done_notifications_.insert(std::make_pair(node_id, &cond));
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

tensorflow::Status EagerExecutor::status() {
  tensorflow::mutex_lock l(node_queue_mutex_);
  return status_;
}

void EagerExecutor::Run() {
  while (true) {
    std::unique_ptr<EagerNode> curr_node;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      while (node_queue_.empty() || !status_.ok()) {
        if (thread_done_) return;
        nodes_pending_.wait(l);
      }
      curr_node.reset(node_queue_.front());
    }
    tensorflow::Status status = curr_node->Run();
    const bool ok = status.ok();
    tensorflow::mutex_lock l(node_queue_mutex_);
    node_queue_.pop();
    if (!ok) {
      status_ = status;
      // TODO(agarwal): mark all affected handles as corrupted before clearing
      // this queue.
      // We remove any pending ops so that we don't try to execute them if
      // ClearError is called.
      for (int i = 0; i < node_queue_.size(); ++i) {
        delete node_queue_.front();
        node_queue_.pop();
      }
    }
    if (!node_done_notifications_.empty()) {
      tensorflow::uint64 node_id = curr_node->id;
      // Note that we notify all waiting threads in case an error has occurred.
      // These calling threads are responsible for checking status_ before
      // proceeding.
      const auto range = ok ? node_done_notifications_.equal_range(node_id)
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
