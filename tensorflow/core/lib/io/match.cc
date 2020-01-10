#include "tensorflow/core/lib/io/match.h"
#include <fnmatch.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {
namespace io {

Status GetMatchingFiles(Env* env, const string& pattern,
                             std::vector<string>* results) {
  results->clear();
  std::vector<string> all_files;
  string dir = Dirname(pattern).ToString();
  if (dir.empty()) dir = ".";
  string basename_pattern = Basename(pattern).ToString();
  Status s = env->GetChildren(dir, &all_files);
  if (!s.ok()) {
    return s;
  }
  for (const auto& f : all_files) {
    int flags = 0;
    if (fnmatch(basename_pattern.c_str(), Basename(f).ToString().c_str(),
                flags) == 0) {
      results->push_back(JoinPath(dir, f));
    }
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
