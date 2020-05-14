/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_DATA_SERVICE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DATA_SERVICE_H_

#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/data/service/worker.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {

// Modes for how a tf.data service job should process a dataset.
enum class ProcessingMode : int64 {
  // Each tf.data worker processes an entire epoch. If a dataset contains 2
  // elements and there are 3 workers, the job will produce 6 elements.
  PARALLEL_EPOCHS = 0,
  // Processing of a single epoch is distributed across all tf.data workers.
  ONE_EPOCH = 1,
};

// Parses a string representing a processing mode and stores the result in
// *mode. Returns an InvalidArgument status if the string is not recognized.
Status ParseProcessingMode(const std::string& s, ProcessingMode* mode);

// Converts a processing mode to its corresponding string.
std::string ProcessingModeToString(ProcessingMode mode);

// Base class for data service clients. Data service clients are
// thread-compatible, requiring external synchronization when used from multiple
// threads.
class DataServiceClientBase {
 public:
  DataServiceClientBase(const std::string& address, const std::string& protocol)
      : address_(address), protocol_(protocol) {}

  virtual ~DataServiceClientBase() = default;
  // Not copyable or movable.
  DataServiceClientBase(const DataServiceClientBase&) = delete;
  DataServiceClientBase& operator=(const DataServiceClientBase&) = delete;

  // Initializes the client. Calling `Initialize()` is not required since the
  // first RPC will perform any necessary initialization. However, it can be
  // useful to call `Initialize()` proactively so that any errors that happen
  // during initialization can be surfaced earlier.
  Status Initialize() { return EnsureInitialized(); }

 protected:
  // Initializes the client if it isn't already initialized.
  virtual Status EnsureInitialized() = 0;

  const std::string address_;
  const std::string protocol_;
};

// Client for communicating with the tf.data service master.
class DataServiceMasterClient : public DataServiceClientBase {
 public:
  DataServiceMasterClient(const std::string& address,
                          const std::string& protocol)
      : DataServiceClientBase(address, protocol) {}

  // Registers a dataset with the tf.data service, and stores the generated
  // dataset id in `*dataset_id`.
  Status RegisterDataset(GraphDef dataset, int64* dataset_id);

  // Creates a new tf.data service job for the specified dataset. The id for the
  // created job will be stored in `*job_id`.
  Status CreateJob(int64 dataset_id, ProcessingMode processing_mode,
                   int64* job_id);

  // Gets the job id for the job represented by the tuple
  // (job_name, job_name_index), and stores the id in *job_id. If the
  // job doesn't exist yet, it will be created.
  Status GetOrCreateJob(int64 dataset_id, ProcessingMode processing_mode,
                        const std::string& job_name, int job_name_index,
                        int64* job_id);

  // Queries the master for the tasks associated with the specified job.
  // The tasks will be stored in *tasks, and whether the job is finished will
  // be stored in `*job_finished`.
  Status GetTasks(int64 job_id, std::vector<TaskInfo>* tasks,
                  bool* job_finished);

 protected:
  Status EnsureInitialized() override;

 private:
  std::unique_ptr<MasterService::Stub> stub_;
};

// Client for communicating with the tf.data service worker.
class DataServiceWorkerClient : public DataServiceClientBase {
 public:
  DataServiceWorkerClient(const std::string& address,
                          const std::string& protocol)
      : DataServiceClientBase(address, protocol) {}

  // Fetches the next element for the specified task_id. The element's
  // compressed tensors will be stored in *element. If no element is available,
  // `*end_of_sequence` will be `true`, and `element` will be left unchanged.
  Status GetElement(int64 task_id, CompressedElement* element,
                    bool* end_of_sequence);

 protected:
  Status EnsureInitialized() override;

 private:
  std::unique_ptr<WorkerService::Stub> stub_;
};

// Creates and initializes a new tf.data service master client.
Status CreateDataServiceMasterClient(
    const std::string& address, const std::string& protocol,
    std::unique_ptr<DataServiceMasterClient>* out);

// Creates and initializes a new tf.data service worker client.
Status CreateDataServiceWorkerClient(
    const std::string& address, const std::string& protocol,
    std::unique_ptr<DataServiceWorkerClient>* out);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DATA_SERVICE_H_
