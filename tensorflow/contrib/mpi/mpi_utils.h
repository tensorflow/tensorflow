/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_MPI_MPI_UTILS_H_
#define TENSORFLOW_CONTRIB_MPI_MPI_UTILS_H_

#ifdef TENSORFLOW_USE_MPI

#include <map>
#include <string>
#include <vector>

#include "tensorflow/core/lib/strings/str_util.h"

// Skip MPI C++ bindings support, this matches the usage in other places
#define OMPI_SKIP_MPICXX
#include "third_party/mpi/mpi.h"
#define MPI_CHECK(cmd)                                                \
  do {                                                                \
    int mpi_errno = cmd;                                              \
    if (MPI_SUCCESS != mpi_errno) {                                   \
      fprintf(stderr, "[%s:%d] MPI call failed with %d \n", __FILE__, \
              __LINE__, mpi_errno);                                   \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
    assert(MPI_SUCCESS == mpi_errno);                                 \
  } while (false)

namespace tensorflow {
class MPIUtils {
 public:
  explicit MPIUtils(const std::string& worker_name);

  const int GetSourceID(const std::string& task_id) const {
    auto it = name_to_id_.find(task_id);
    if (it == name_to_id_.end()) {
      LOG(FATAL) << "Failed to convert worker name to MPI index: " << task_id;
    }
    return it->second;
  }

 private:
  void InitMPI();

  std::map<std::string, int> name_to_id_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_MPI
#endif  // TENSORFLOW_CONTRIB_MPI_MPI_UTILS_H_
