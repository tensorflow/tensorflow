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

#ifdef TENSORFLOW_USE_MPI

#include "tensorflow/contrib/mpi/mpi_utils.h"
namespace tensorflow {

#define max_worker_name_length 128

MPIUtils::MPIUtils(const std::string& worker_name) {
  InitMPI();
  // Connect the MPI process IDs to the worker names that are used by TF.
  // Gather the names of all the active processes (name can't be longer than
  // 128 bytes)
  int proc_id = 0, number_of_procs = 1;
  char my_name[max_worker_name_length];
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &proc_id));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &number_of_procs));

  CHECK(worker_name.size() < max_worker_name_length)
      << "Specified worker name is too long.";
  snprintf(my_name, max_worker_name_length, worker_name.c_str());
  std::vector<char> worker_names(number_of_procs * max_worker_name_length);
  MPI_CHECK(MPI_Allgather(my_name, max_worker_name_length, MPI_CHAR,
                          &worker_names[0], max_worker_name_length, MPI_CHAR,
                          MPI_COMM_WORLD));

  if (proc_id == 0) LOG(INFO) << "MPI process-ID to gRPC server name map: \n";
  for (int i = 0; i < number_of_procs; i++) {
    name_to_id_[std::string(&worker_names[i * 128])] = i;
    if (proc_id == 0)
      LOG(INFO) << "Process: " << i
                << "\tgRPC-name: " << std::string(&worker_names[i * 128])
                << std::endl;
  }
}

void MPIUtils::InitMPI() {
  // Initialize the MPI environment if that hasn't been done
  int flag = 0;
  MPI_CHECK(MPI_Initialized(&flag));
  if (!flag) {
    int proc_id = 0, number_of_procs = 1, len = -1;
    char my_host_name[max_worker_name_length];
    // MPI_CHECK(MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &flag));
    MPI_CHECK(MPI_Init(0, 0));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &proc_id));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &number_of_procs));
    MPI_CHECK(MPI_Get_processor_name(my_host_name, &len));
    fprintf(stderr,
            "MPI Environment initialised. Process id: %d Total processes: %d "
            "|| Hostname: %s \n",
            proc_id, number_of_procs, my_host_name);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_MPI
