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

#include "tensorflow/tsl/platform/file_system_helper.h"

#include <deque>
#include <string>
#include <vector>

#include "tensorflow/tsl/platform/cpu_info.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/platform.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/str_util.h"
#include "tensorflow/tsl/platform/threadpool.h"

namespace tsl {
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

// A globbing pattern can only start with these characters:
static const char kGlobbingChars[] = "*?[\\";

static inline bool IsGlobbingPattern(const std::string& pattern) {
  return (pattern.find_first_of(kGlobbingChars) != std::string::npos);
}

// Make sure that the first entry in `dirs` during glob expansion does not
// contain a glob pattern. This is to prevent a corner-case bug where
// `<pattern>` would be treated differently than `./<pattern>`.
static std::string PatchPattern(const std::string& pattern) {
  const std::string fixed_prefix =
      pattern.substr(0, pattern.find_first_of(kGlobbingChars));

  // Patching is needed when there is no directory part in `prefix`
  if (io::Dirname(fixed_prefix).empty()) {
    return io::JoinPath(".", pattern);
  }

  // No patching needed
  return pattern;
}

static std::vector<std::string> AllDirectoryPrefixes(const std::string& d) {
  std::vector<std::string> dirs;
  const std::string patched = PatchPattern(d);
  StringPiece dir(patched);

  // If the pattern ends with a `/` (or `\\` on Windows), we need to strip it
  // otherwise we would have one additional matching step and the result set
  // would be empty.
  bool is_directory = d[d.size() - 1] == '/';
#ifdef PLATFORM_WINDOWS
  is_directory = is_directory || (d[d.size() - 1] == '\\');
#endif
  if (is_directory) {
    dir = io::Dirname(dir);
  }

  while (!dir.empty()) {
    dirs.emplace_back(dir);
    StringPiece new_dir(io::Dirname(dir));
    // io::Dirname("/") returns "/" so we need to break the loop.
    // On Windows, io::Dirname("C:\\") would return "C:\\", so we check for
    // identity of the result instead of checking for dir[0] == `/`.
    if (dir == new_dir) break;
    dir = new_dir;
  }

  // Order the array from parent to ancestor (reverse order).
  std::reverse(dirs.begin(), dirs.end());

  return dirs;
}

static inline int GetFirstGlobbingEntry(const std::vector<std::string>& dirs) {
  int i = 0;
  for (const auto& d : dirs) {
    if (IsGlobbingPattern(d)) {
      break;
    }
    i++;
  }
  return i;
}

}  // namespace

Status GetMatchingPaths(FileSystem* fs, Env* env, const string& pattern,
                        std::vector<string>* results) {
  // Check that `fs`, `env` and `results` are non-null.
  if (fs == nullptr || env == nullptr || results == nullptr) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Filesystem calls GetMatchingPaths with nullptr arguments");
  }

  // By design, we don't match anything on empty pattern
  results->clear();
  if (pattern.empty()) {
    return OkStatus();
  }

  // The pattern can contain globbing characters at multiple levels, e.g.:
  //
  //   foo/ba?/baz/f*r
  //
  // To match the full pattern, we must match every prefix subpattern and then
  // operate on the children for each match. Thus, we separate all subpatterns
  // in the `dirs` vector below.
  std::vector<std::string> dirs = AllDirectoryPrefixes(pattern);

  // We can have patterns that have several parents where no globbing is being
  // done, for example, `foo/bar/baz/*`. We don't need to expand the directories
  // which don't contain the globbing characters.
  int matching_index = GetFirstGlobbingEntry(dirs);

  // If we don't have globbing characters in the pattern then it specifies a
  // path in the filesystem. We add it to the result set if it exists.
  if (matching_index == dirs.size()) {
    if (fs->FileExists(pattern).ok()) {
      results->emplace_back(pattern);
    }
    return OkStatus();
  }

  // To expand the globbing, we do a BFS from `dirs[matching_index-1]`.
  // At every step, we work on a pair `{dir, ix}` such that `dir` is a real
  // directory, `ix < dirs.size() - 1` and `dirs[ix+1]` is a globbing pattern.
  // To expand the pattern, we select from all the children of `dir` only those
  // that match against `dirs[ix+1]`.
  // If there are more entries in `dirs` after `dirs[ix+1]` this mean we have
  // more patterns to match. So, we add to the queue only those children that
  // are also directories, paired with `ix+1`.
  // If there are no more entries in `dirs`, we return all children as part of
  // the answer.
  // Since we can get into a combinatorial explosion issue (e.g., pattern
  // `/*/*/*`), we process the queue in parallel. Each parallel processing takes
  // elements from `expand_queue` and adds them to `next_expand_queue`, after
  // which we swap these two queues (similar to double buffering algorithms).
  // PRECONDITION: `IsGlobbingPattern(dirs[0]) == false`
  // PRECONDITION: `matching_index > 0`
  // INVARIANT: If `{d, ix}` is in queue, then `d` and `dirs[ix]` are at the
  //            same level in the filesystem tree.
  // INVARIANT: If `{d, _}` is in queue, then `IsGlobbingPattern(d) == false`.
  // INVARIANT: If `{d, _}` is in queue, then `d` is a real directory.
  // INVARIANT: If `{_, ix}` is in queue, then `ix < dirs.size() - 1`.
  // INVARIANT: If `{_, ix}` is in queue, `IsGlobbingPattern(dirs[ix + 1])`.
  std::deque<std::pair<string, int>> expand_queue;
  std::deque<std::pair<string, int>> next_expand_queue;
  expand_queue.emplace_back(dirs[matching_index - 1], matching_index - 1);

  // Adding to `result` or `new_expand_queue` need to be protected by mutexes
  // since there are multiple threads writing to these.
  mutex result_mutex;
  mutex queue_mutex;

  while (!expand_queue.empty()) {
    next_expand_queue.clear();

    // The work item for every item in `expand_queue`.
    // pattern, we process them in parallel.
    auto handle_level = [&fs, &results, &dirs, &expand_queue,
                         &next_expand_queue, &result_mutex,
                         &queue_mutex](int i) {
      // See invariants above, all of these are valid accesses.
      const auto& queue_item = expand_queue.at(i);
      const std::string& parent = queue_item.first;
      const int index = queue_item.second + 1;
      const std::string& match_pattern = dirs[index];

      // Get all children of `parent`. If this fails, return early.
      std::vector<std::string> children;
      Status s = fs->GetChildren(parent, &children);
      if (s.code() == absl::StatusCode::kPermissionDenied) {
        return;
      }

      // Also return early if we don't have any children
      if (children.empty()) {
        return;
      }

      // Since we can get extremely many children here and on some filesystems
      // `IsDirectory` is expensive, we process the children in parallel.
      // We also check that children match the pattern in parallel, for speedup.
      // We store the status of the match and `IsDirectory` in
      // `children_status` array, one element for each children.
      std::vector<Status> children_status(children.size());
      auto handle_children = [&fs, &match_pattern, &parent, &children,
                              &children_status](int j) {
        const std::string path = io::JoinPath(parent, children[j]);
        if (!fs->Match(path, match_pattern)) {
          children_status[j] =
              Status(absl::StatusCode::kCancelled, "Operation not needed");
        } else {
          children_status[j] = fs->IsDirectory(path);
        }
      };
      ForEach(0, children.size(), handle_children);

      // At this point, pairing `children` with `children_status` will tell us
      // if a children:
      //   * does not match the pattern
      //   * matches the pattern and is a directory
      //   * matches the pattern and is not a directory
      // We fully ignore the first case.
      // If we matched the last pattern (`index == dirs.size() - 1`) then all
      // remaining children get added to the result.
      // Otherwise, only the directories get added to the next queue.
      for (size_t j = 0; j < children.size(); j++) {
        if (children_status[j].code() == absl::StatusCode::kCancelled) {
          continue;
        }

        const std::string path = io::JoinPath(parent, children[j]);
        if (index == dirs.size() - 1) {
          mutex_lock l(result_mutex);
          results->emplace_back(path);
        } else if (children_status[j].ok()) {
          mutex_lock l(queue_mutex);
          next_expand_queue.emplace_back(path, index);
        }
      }
    };
    ForEach(0, expand_queue.size(), handle_level);

    // After evaluating one level, swap the "buffers"
    std::swap(expand_queue, next_expand_queue);
  }

  return OkStatus();
}

StatusOr<bool> FileExists(Env* env, const string& fname) {
  Status status = env->FileExists(fname);
  if (errors::IsNotFound(status)) {
    return false;
  }
  TF_RETURN_IF_ERROR(status);
  return true;
}

}  // namespace internal
}  // namespace tsl
