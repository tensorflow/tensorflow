#ifndef TENSORFLOW_LIB_IO_MATCH_H_
#define TENSORFLOW_LIB_IO_MATCH_H_

#include <vector>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
class Env;
namespace io {

// Given a pattern, return the set of files that match the pattern.
// Note that this routine only supports wildcard characters in the
// basename portion of the pattern, not in the directory portion.  If
// successful, return Status::OK and store the matching files in
// "*results".  Otherwise, return a non-OK status.
Status GetMatchingFiles(Env* env, const string& pattern,
                        std::vector<string>* results);

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_MATCH_H_
