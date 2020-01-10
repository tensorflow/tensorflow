// This interface provides a machine-wide resource management singleton
// interface as a convenience for users who will want to exploit all of the GPU
// resources present on the system.
//
// To use the singleton interface:
//
//  // At start of program or in your module initializer.
//  // Do not call this with different sets of arguments!
//  MachineManager::CreateSingletonOrDie(
//      MachineManager::DetectPreferredPlatform(), DeviceOptions::Default());
//
//  // At any point after that, this convenience interface avoids you having to
//  // pass those two parameters:
//  StreamExecutor *device0_executor =
//      MachineManager::singleton()->executor_for_device(0 /* = ordinal */);
//  ...

// ----------------- THIS CLASS IS DEPRECATED - DO NOT USE ------------------
// This class is not suitable for open-sourcing, as it does not support
// plugins and depends on hardcoded PlatformKind enums. MultiPlatformManager and
// Platform plugins are the replacements.
// ----------------- THIS CLASS IS DEPRECATED - DO NOT USE ------------------

#ifndef TENSORFLOW_STREAM_EXECUTOR_MACHINE_MANAGER_H_
#define TENSORFLOW_STREAM_EXECUTOR_MACHINE_MANAGER_H_

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/device_options.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace perftools {
namespace gputools {

// MachineManager is used to instantiate and manage singleton resources for
// all the GPUs present on a machine. This basically amounts to having a
// StreamExecutor-per-device pool.
//
// Thread-safe.
class MachineManager {
 public:
  // Inspects the host to determine the preferred GPU execution platform.
  // To force OpenCL from a build target on a machine that has both OpenCL and
  // CUDA capabilities, link against the :stream_executor_prefer_opencl target.
  static PlatformKind DetectPreferredPlatform();

  // Returns the machine manager singleton.
  // If the singleton has not yet been created when this is invoked, this
  // creates it with resonable default options, otherwise it returns the
  // already-created singleton. If there are errors during creation, this call
  // will terminate the program.
  static MachineManager *singleton();

  // Returns a singleton instance of the machine manager -- it's generally
  // assumed that users will have one of these for a real-world application as a
  // form of resource manager.
  //
  // This should only be called once, at the initialization of an application,
  // if at all -- MachineManager::singleton() will return a value with sensible
  // default as determined by DetectPreferredPlatform. Attempts to create the
  // singleton with options multiple times will result in an error.
  static port::StatusOr<MachineManager *> CreateSingleton(
      PlatformKind platform, DeviceOptions device_options,
      const PluginConfig &config = PluginConfig());

  // Convenience "or die" wrapper around the above call.
  static MachineManager *CreateSingletonOrDie(
      PlatformKind platform, DeviceOptions device_options,
      const PluginConfig &config = PluginConfig());

  // Creates a new instantiation of the MachineManager.
  // Warning: generally users will want to use the singleton form, see
  // MachineManager::singleton().
  //
  // The machine manager has a number of devices that it detects on creation
  // that does not change over the course of its lifetime. This does not support
  // things like hot-plugging of GPUs or the event of GPUs dropping off the bus
  // in a recoverable manner.
  static port::StatusOr<std::unique_ptr<MachineManager>> Create(
      PlatformKind kind, DeviceOptions options,
      const PluginConfig &config = PluginConfig());

  // Returns the number of devices visible to the machine manager.
  int device_count() const;

  // Returns the StreamExecutor for one of the machine-manager visible devices.
  // Checks that device_ordinal is within device_count() bound.
  StreamExecutor *executor_for_device(int device_ordinal) const;

  // Returns the bus ordinal count (as determined by the span of NUMA nodes
  // associated with the available devices).
  int bus_count() const { return limit_numa_node_ - min_numa_node_; }

  // Returns the bus ordinal associated with a given device ordinal.
  int DeviceToBus(int device_ordinal) const;

  // Returns the NUMA node associated with a given device ordinal.
  int DeviceToNumaNode(int device_ordinal) const;

  // Returns the first StreamExecutor (within device_count() ordinals that has
  // the corresponding bus ordinal, or nullptr if none is found.
  //
  // The valid bus ordinals can be enumerated by scanning through the executors
  // and seeing what bus number they are on.
  StreamExecutor *first_executor_for_bus(int bus_ordinal);

  // Returns the first StreamExecutor associated with the specified
  // numa_node, or nullptr if none is found.
  StreamExecutor *first_executor_for_numa_node(int numa_node);

  // Returns the default stream for the default executor (that returned by
  // executor_for_device()). The same stream will be returned for all calls to
  // stream_for_device() (with the same device_ordinal).
  Stream *stream_for_device(int device_ordinal);

  // Returns the platform that this machine manager was created to target.
  PlatformKind platform() const { return platform_; }

  // Enables peer access between all possible devices on this platform.
  // Only dies due to failure to enable peer access for devices in which
  // GetPeerAccessMap() is true.
  port::Status EnablePeerAccess();

  // Returns a map that says, for pairs (device ordinal i, device ordinal j),
  // whether i can access j's memory space.
  std::unique_ptr<std::map<std::pair<int, int>, bool>> GetPeerAccessMap();

 private:
  // Guts of the singleton creation mechanism that requires the exclusive
  // singleton lock to be held, in order to prevent deadlock due to method
  // composition.
  static port::StatusOr<MachineManager *> CreateSingletonInternal(
      PlatformKind platform, DeviceOptions options, const PluginConfig &config)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Private constructor used in singleton creation.
  MachineManager(PlatformKind platform, DeviceOptions options,
                 const PluginConfig &config);

  // Populates the executors_ vector with an executor per observable device
  // ordinal on the platform. Logs and returns false if any of the
  // Stream Executors cannot be created.
  port::Status Init();

  // Converts a StreamExecutor's NUMA node association into a bus ordinal for
  // this machine.
  int ExecutorToBus(const StreamExecutor *stream_exec) const;

  // Returns the NUMA node association for the StreamExecutor.
  int ExecutorToNumaNode(const StreamExecutor *stream_exec) const;

  // Mutex that guards the initialization of the machine manager static
  // variable.
  static mutex mu_;

  // Singleton MachineManager value -- assignment to this is protected by a
  // static singleton guard clause.
  static MachineManager *singleton_ GUARDED_BY(mu_);

  // Holds an executor associated with each device ordinal present in the
  // system, which are the indices. Immutable after initialization.
  std::vector<std::unique_ptr<StreamExecutor>> executors_;

  // Holds an stream associated with each device ordinal present in the
  // system, which are the indices. Immutable after initialization.
  std::vector<std::unique_ptr<Stream>> streams_;

  // The platform that this is managing for the machine.
  PlatformKind platform_;

  // Options used to create StreamExecutors on each of the respective devices.
  DeviceOptions device_options_;

  // Plugin configuration to use for all StreamExecutors created by this object.
  PluginConfig plugin_config_;

  // The smallest NUMA node value for any device managed by this machine
  // manager. Used, along with limit_numa_node_, to convert NUMA nodes into bus
  // ordinals. The NUMA node space occupied by GPUs is assumed to be dense.
  int min_numa_node_;

  // Larger than the NUMA node value for any device managed by this machine
  // manager.
  int limit_numa_node_;
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_MACHINE_MANAGER_H_
