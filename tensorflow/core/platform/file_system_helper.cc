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

#include "tensorflow/core/platform/file_system_helper.h"

#include <deque>
#include <string>
#include <vector>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace internal {

namespace {

const int kNumThreads = port::NumSchedulableCPUs();

// Run a function in parallel using a ThreadPool, but skip the ThreadPool
// on the iOS platform due to its problems with more than a few threads.
void ForEach(int first, int last, const std::function<void(int)>& f) {
#if TARGET_OS_IPHONE
  for (int i = first; i < last; i++) {
    f(i);
  }
#else
  int num_threads = std::min(kNumThreads, last - first);
  thread::ThreadPool threads(Env::Default(), "ForEach", num_threads);
  for (int i = first; i < last; i++) {
    threads.Schedule([f, i] { f(i); });
  }
#endif
}

}  // namespace

Status GetMatchingPaths(FileSystem* fs, Env* env, const string& pattern,
                        std::vector<string>* results) {
  results->clear();
  if (pattern.empty()) {
    return Status::OK();
  }

  string fixed_prefix = pattern.substr(0, pattern.find_first_of("*?[\\"));
  string eval_pattern = pattern;
  string dir(io::Dirname(fixed_prefix));
  // If dir is empty then we need to fix up fixed_prefix and eval_pattern to
  // include . as the top level directory.
  if (dir.empty()) {
    dir = ".";
    fixed_prefix = io::JoinPath(dir, fixed_prefix);
    eval_pattern = io::JoinPath(dir, eval_pattern);
  }
  bool is_directory = pattern[pattern.size() - 1] == '/';
#ifdef PLATFORM_WINDOWS
  is_directory = is_directory || pattern[pattern.size() - 1] == '\\';
#endif
  std::vector<string> dirs;
  if (!is_directory) {
    dirs.emplace_back(eval_pattern);
  }
  StringPiece tmp_dir(io::Dirname(eval_pattern));
  while (tmp_dir.size() > dir.size()) {
    dirs.emplace_back(string(tmp_dir));
    tmp_dir = io::Dirname(tmp_dir);
  }
  dirs.emplace_back(dir);
  std::reverse(dirs.begin(), dirs.end());
  // Setup a parallel BFS to explore everything under dir.
  std::deque<std::pair<string, int>> dir_q;
  std::deque<std::pair<string, int>> next_dir_q;
  dir_q.emplace_back(std::make_pair(dirs[0], 0));
  Status ret;  // Status to return.
  mutex results_mutex;
  condition_variable results_cond;
  mutex next_que_mutex;
  condition_variable next_que_cond;
  while (!dir_q.empty()) {
    next_dir_q.clear();
    std::vector<Status> new_rets(dir_q.size());
    auto handle_level = [fs, &results, &dir_q, &next_dir_q, &new_rets,
                         &is_directory, &dirs, &results_mutex, &results_cond,
                         &next_que_mutex, &next_que_cond](int i) {
      string current_dir = dir_q.at(i).first;
      int dir_index = dir_q.at(i).second;
      dir_index++;
      std::vector<string> children;
      Status s = fs->GetChildren(current_dir, &children);
      // In case PERMISSION_DENIED is encountered, we bail here.
      if (s.code() == tensorflow::error::PERMISSION_DENIED) {
        return;
      }
      new_rets[i] = s;
      if (children.empty()) return;

      // children_dir_status holds is_dir status for children. It can have three
      // possible values: OK for true; FAILED_PRECONDITION for false; CANCELLED
      // if we don't calculate IsDirectory (we might do that because there isn't
      // any point in exploring that child path).
      std::vector<Status> children_dir_status;

      // This IsDirectory call can be expensive for some FS. Parallelizing it.
      children_dir_status.resize(children.size());
      auto handle_children = [fs, &current_dir, &children, &dirs, dir_index,
                              is_directory, &children_dir_status](int j) {
        const string child_path = io::JoinPath(current_dir, children[j]);
        if (!fs->Match(child_path, dirs[dir_index])) {
          children_dir_status[j] =
              Status(tensorflow::error::CANCELLED, "Operation not needed");
        } else if (dir_index != dirs.size() - 1) {
          children_dir_status[j] = fs->IsDirectory(child_path);
        } else {
          children_dir_status[j] =
              is_directory ? fs->IsDirectory(child_path) : Status::OK();
        }
      };
      ForEach(0, children.size(), handle_children);

      for (size_t j = 0; j < children.size(); ++j) {
        const string child_path = io::JoinPath(current_dir, children[j]);
        // If the IsDirectory call was cancelled we bail.
        if (children_dir_status[j].code() == tensorflow::error::CANCELLED) {
          continue;
        }
        if (children_dir_status[j].ok()) {
          if (dir_index != dirs.size() - 1) {
            mutex_lock lk(next_que_mutex);
            next_dir_q.emplace_back(std::make_pair(child_path, dir_index));
            next_que_cond.notify_one();
          } else {
            mutex_lock lk(results_mutex);
            results->emplace_back(child_path);
            results_cond.notify_one();
          }
        }
      }
    };
    ForEach(0, dir_q.size(), handle_level);

    ret.Update(new_rets[dir_q.size() - 1]);
    std::swap(dir_q, next_dir_q);
  }
  return ret;
}

}  // namespace internal
}  // namespace tensorflow
