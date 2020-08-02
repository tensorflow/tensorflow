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

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace internal {

namespace {

constexpr int kNumThreads = 8;

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
    dirs.push_back(eval_pattern);
  }
  StringPiece tmp_dir(io::Dirname(eval_pattern));
  while (tmp_dir.size() > dir.size()) {
    dirs.push_back(string(tmp_dir));
    tmp_dir = io::Dirname(tmp_dir);
  }
  dirs.push_back(dir);
  std::reverse(dirs.begin(), dirs.end());
  // Setup a BFS to explore everything under dir.
  std::deque<std::pair<string, int>> dir_q;
  dir_q.push_back({dirs[0], 0});
  Status ret;  // Status to return.
  // children_dir_status holds is_dir status for children. It can have three
  // possible values: OK for true; FAILED_PRECONDITION for false; CANCELLED
  // if we don't calculate IsDirectory (we might do that because there isn't
  // any point in exploring that child path).
  std::vector<Status> children_dir_status;
  while (!dir_q.empty()) {
    string current_dir = dir_q.front().first;
    int dir_index = dir_q.front().second;
    dir_index++;
    dir_q.pop_front();
    std::vector<string> children;
    Status s = fs->GetChildren(current_dir, &children);
    // In case PERMISSION_DENIED is encountered, we bail here.
    if (s.code() == tensorflow::error::PERMISSION_DENIED) {
      continue;
    }
    ret.Update(s);
    if (children.empty()) continue;
    // This IsDirectory call can be expensive for some FS. Parallelizing it.
    children_dir_status.resize(children.size());
    ForEach(0, children.size(),
            [fs, &current_dir, &children, &dirs, dir_index, is_directory,
             &children_dir_status](int i) {
              const string child_path = io::JoinPath(current_dir, children[i]);
              if (!fs->Match(child_path, dirs[dir_index])) {
                children_dir_status[i] = Status(tensorflow::error::CANCELLED,
                                                "Operation not needed");
              } else if (dir_index != dirs.size() - 1) {
                children_dir_status[i] = fs->IsDirectory(child_path);
              } else {
                children_dir_status[i] =
                    is_directory ? fs->IsDirectory(child_path) : Status::OK();
              }
            });
    for (size_t i = 0; i < children.size(); ++i) {
      const string child_path = io::JoinPath(current_dir, children[i]);
      // If the IsDirectory call was cancelled we bail.
      if (children_dir_status[i].code() == tensorflow::error::CANCELLED) {
        continue;
      }
      if (children_dir_status[i].ok()) {
        if (dir_index != dirs.size() - 1) {
          dir_q.push_back({child_path, dir_index});
        } else {
          results->push_back(child_path);
        }
      }
    }
  }
  return ret;
}

}  // namespace internal
}  // namespace tensorflow
