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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// XLA service's client object -- wraps the service with convenience and
// lifetime-oriented methods.
class Client {
 public:
  explicit Client(ServiceInterface* stub);
  virtual ~Client();

  // Executes the computation with the given arguments and returns the global
  // data that was produced from the execution.
  // * If execution_options is not nullptr, these options are passed to the
  //   service to affect how it compiles our computation.  (The pointer does not
  //   need to live beyond this call.)
  // * If execution_options.device_handles is not empty, the computation is
  //   executed on the devices associated with the handles by partitioning the
  //   computation based on the attached sharding attributes. Otherwise, a
  //   device is chosen by the service.
  // * If execution_profile is not nullptr then the pointed-to ExecutionProfile
  //   will be filled with profile data from the execution.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const ExecutionOptions* execution_options = nullptr,
      ExecutionProfile* execution_profile = nullptr);

  // A struct to represent a computation instance to be executed.
  // * If execution_options.device_handles is not empty, the computation is
  //   executed on the devices associated with the handles by partitioning the
  //   computation based on the attached sharding attributes. Otherwise, a
  //   device is chosen by the service.
  struct ComputationInstance {
    const Computation& computation;
    std::vector<GlobalData*> arguments;
    ExecutionOptions execution_options;
    ExecutionProfile* execution_profile;

    ComputationInstance(const Computation& computation,
                        std::vector<GlobalData*> arguments,
                        ExecutionOptions execution_options,
                        ExecutionProfile* execution_profile)
        : computation(computation),
          arguments(std::move(arguments)),
          execution_options(execution_options),
          execution_profile(execution_profile) {}
  };

  // Executes a list ComputationInstances and returns global data produced from
  // each computation.
  StatusOr<std::vector<std::unique_ptr<GlobalData>>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<ComputationInstance> computations);

  // Requests device_count device handles available on the target. The returned
  // device handles are used to specify the devices to execute the computations
  // (see ExecuteParallel) or to transfer data (see TransferToServer or
  // TransferToInfeed).
  StatusOr<std::vector<DeviceHandle>> GetDeviceHandles(int64 device_count);

  // Transfer the global data provided to this client process, which is
  // returned in the provided literal. Use sparingly to avoid transfer
  // overheads.
  //
  // If shape_with_layout is not nullptr, it points to a shape whose layout will
  // be the layout of the returned literal.
  StatusOr<std::unique_ptr<Literal>> Transfer(
      const GlobalData& data, const Shape* shape_with_layout = nullptr);

  // Transfer the given literal to the server. This allocates memory on the
  // device and copies the literal's contents over. Returns a global data handle
  // that can be used to refer to this value from the client.
  //
  // If device_handle is not nullptr, data is transferred to the associated
  // device (and its replicas if replication is enabled). Otherwise, data is
  // transferred to the default device (and its replicas).
  StatusOr<std::unique_ptr<GlobalData>> TransferToServer(
      const Literal& literal, const DeviceHandle* device_handle = nullptr);

  // Transfer the given literal to the Infeed interface of the device.
  //
  // device_handle and replica_id together specify a particular device; a device
  // assigned for the given replica_id among the replicas that the given device
  // handle belongs to.
  Status TransferToInfeed(const Literal& literal, int64 replica_id = 0,
                          const DeviceHandle* device_handle = nullptr);

  // Transfers from the Outfeed of the device.
  //
  // device_handle and replica_id together specify a particular device; a device
  // assigned for the given replica_id among the replicas that the given device
  // handle belongs to.
  StatusOr<std::unique_ptr<Literal>> TransferFromOutfeed(
      const Shape* shape_with_layout, int64 replica_id = 0,
      const DeviceHandle* device_handle = nullptr);

  // Resets the device, clearing all existing state on the device.
  Status ResetDevice();

  // Executes the computation with the given arguments and transfers the result
  // to the client as a literal. Parameters are defined the same as for
  // Execute() and Transfer().
  StatusOr<std::unique_ptr<Literal>> ExecuteAndTransfer(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const ExecutionOptions* execution_options = nullptr,
      ExecutionProfile* execution_profile = nullptr);

  // Unregister the memory for the given GlobalData on the device.
  Status Unregister(const GlobalData& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<std::unique_ptr<GlobalData>>> DeconstructTuple(
      const GlobalData& data);

  // Retrieves the statistics of the given computation.
  StatusOr<ComputationStats> GetComputationStats(
      const Computation& computation, const DebugOptions& debug_options) const;

  // Returns the Shape of the given array specified by 'data'. The shape
  // includes the Layout of the array as it is stored on the service.
  StatusOr<Shape> GetShape(const GlobalData& data);

  // As above, but returns the shape of the provided computation (parameter
  // types/names and return type).
  StatusOr<std::unique_ptr<ProgramShape>> GetComputationShape(
      const Computation& computation);

  // Creates a channel handle that can be used to transfer data between
  // two computations via a pair of Send and Recv instructions.
  StatusOr<ChannelHandle> CreateChannelHandle();

  StatusOr<Computation> LoadSnapshot(const SessionModule& module);

  ServiceInterface* stub() { return stub_; }

 private:
  // Returns the execution statistics (e.g., gflop/s) as a string from the
  // ExecutionProfile returned from an execution of the computation.
  StatusOr<string> ExecutionStatsAsString(const Computation& computation,
                                          const ExecutionProfile& profile);

  ServiceInterface* stub_;  // Stub that this client is connected on.

  TF_DISALLOW_COPY_AND_ASSIGN(Client);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
