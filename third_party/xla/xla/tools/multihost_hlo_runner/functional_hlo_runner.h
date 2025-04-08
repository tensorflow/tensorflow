/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_MULTIHOST_HLO_RUNNER_FUNCTIONAL_HLO_RUNNER_H_
#define XLA_TOOLS_MULTIHOST_HLO_RUNNER_FUNCTIONAL_HLO_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {

// Supported input formats for the input HLO module.
enum class InputFormat {
  kText,                 // Text format returned by HloModule::ToString().
  kProtoText,            // Protobuf text format of an xla::HloProto message.
  kProtoBinary,          // Protobuf binary format of an xla::HloProto message.
  kSnapshotProtoBinary,  // HloSnapshot protobuf binary format. Can be dumped by
                         // TensorFlow by setting the environment variable
                         // xla_dump_hlo_snapshots.
  kUnoptimizedSnapshotProtoBinary,  // HloUnoptimizedSnapshot protobuf binary
                                    // format. Can be dumped by
                                    // setting the flag
                                    // xla_dump_hlo_snapshots in conjunction
                                    // with xla_dump_as_text.
  kUnoptimizedSnapshotProtoText,    // HloUnoptimizedSnapshot protobuf text
                                    // format. Can be dumped by TensorFlow by
                                    // setting the flag xla_dump_hlo_snapshots
                                    // in conjunction with xla_dump_as_text.
};

enum class OutputFormat : std::uint8_t {
  kText,         // Text format returned by Literal::ToString().
  kProtoBinary,  // Protobuf binary format of an xla::LiteralProto message.
  kProtoText,    // Protobuf text format of an xla::LiteralProto message.
};

// Interface for profiler plugins. If being set in RunningOptions, profiling
// session will be created for the last run of the HLO module.
class ProfilerInterface {
 public:
  virtual ~ProfilerInterface() = default;
  // Creates profiling session while running HLO module.
  virtual void CreateSession() = 0;
  // Uploads profiling session data after finishing running HLO module.
  virtual void UploadSession() = 0;
};

// Interface that may optionally returns an XSpace proto after UploadSession()
// is called. This can be used by caller to get a programmatic handler of the
// profile data.
class XSpaceProfilerInterface : public ProfilerInterface {
 public:
  virtual const tensorflow::profiler::XSpace* GetXSpace() = 0;
};

// HLORunnerProfiler is a profiler plugin that using tsl::ProfilerSession to
// profile CPU/GPU execution and allows programmable control of
// profiling sessions for the MultihostHloRunner. It needs to be created after
// PJRT client is initialized. Example usage:
//
//   TF_ASSIGN_OR_RETURN(
//       env, xla::GetPjRtEnvironmentForGpu(...)));
//   if (env.client != nullptr) {
//     TF_ASSIGN_OR_RETURN(auto profiler, HLORunnerProfiler::Create());
//   }
//   profiler.CreateSession();
//   ...
//   profiler.UploadSession();
class HLORunnerProfiler : public XSpaceProfilerInterface {
 public:
  // Factory method to create a GPURunnerProfiler with profile result dump path.
  // If keep_xspace is true, the XSpace proto can be retrieved
  // by GetXSpace() after UploadSession() is called, which can be used by
  // caller to get a programmatic handler of the profile data and create XProf.
  static absl::StatusOr<std::unique_ptr<HLORunnerProfiler>> Create(
      absl::string_view dump_path, bool keep_xspace = false);

  // Default ctor.
  explicit HLORunnerProfiler(absl::string_view dump_path, bool keep_xspace);

  // Start a new profiling session.
  void CreateSession() override;

  // Stop the current profiling session.
  void UploadSession() override;

  // Returns the XSpace proto.
  const tensorflow::profiler::XSpace* GetXSpace() override;

 private:
  // The file path to dump the profiling result.
  std::string dump_path_;
  // Whether to keep the XSpace proto after UploadSession() is called.
  bool keep_xspace_;
  // The profiler session.
  std::unique_ptr<tsl::ProfilerSession> session_;
  // The XSpace proto to be returned by GetXSpace().
  std::unique_ptr<tensorflow::profiler::XSpace> xspace_;
};

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error);
std::string AbslUnparseFlag(InputFormat input_format);

bool AbslParseFlag(absl::string_view text, OutputFormat* output_format,
                   std::string* error);
std::string AbslUnparseFlag(OutputFormat output_format);

// FunctionalHloRunner takes an HLO module as input and runs the HLO module
// on a single or multiple hosts with various options (e.g. SPMD). The HLO
// module can be pre- or post-optimizations.
// TODO(b/306118803): replace this fully stateless class by a namespace.
class FunctionalHloRunner {
 public:
  // This class has only static methods.
  FunctionalHloRunner() = delete;

  using LiteralVec = std::vector<Literal>;
  using ShapeVec = std::vector<Shape>;
  using PerDeviceLiteralVecType = absl::btree_map<int, LiteralVec>;
  using PerDeviceShapeVecType = absl::btree_map<int, ShapeVec>;
  using PerDeviceIndexVecType = absl::btree_map<int, std::vector<int>>;

  enum class LogOutputMode { kLogOutput, kNotLogOutput };

  enum class HloPassesMode {
    // Call only XLA's RunBackend during the compilation. This is used to run a
    // post-optimization HLO module (dumped as
    // 'xxx.after_optimizations.hlo.xxx').
    kRunXLABackendOnly,
    // Calls Compile (i.e., both RunHloPasses and RunBackend) to compile the
    // module, but disables all HLO passes.
    kDisableAllHloPasses,
    // Standard XLA compilation by calling Compile (or both RunHloPasses and
    // RunBackend). This is used to run a pre-optimizations module.
    kStandardCompile
  };

  enum class SpmdMode : int8_t {
    kUseSpmdPartitioning,    // Use the GSPMD partitioner for partitioning.
    kUseShardyPartitioning,  // Use the Shardy partitioner for partitioning.
    kNotUseSpmdPartitioning  // Do not perform partitioning.
  };

  enum class SpmdPartitionedMode {
    kIsSpmdPartitionedModule,
    kIsNotSpmdPartitionedModule
  };

  enum class XlaTextDumpMode { kDumpAsText, kNotDumpAsText };

  enum class XlaProtoDumpMode { kDumpAsProto, kNotDumpAsProto };

  enum class ModuleArgumentMode {
    // Use device ID (casted to proper type) as arguments.
    kUseDeviceIdAsInput,
    // Use random values as arguments.
    kUseRandomInputs,
    // Use random values as arguments, and different local devices share the
    // same argument values.
    kUseSharedRandomInputs,
    // Use arguments which have all of their bytes set to 0 (not respecting any
    // constraints on the range).
    kUseZerosAsInput,
    // Use uninitialized device buffers as arguments (not respecting any
    // constraints on the range). This drastically reduces
    // the host memory usage and the startup time.
    kUninitialized,
  };

  enum class ModuleOutputMode {
    // Return output from all devices.
    kReturnOutputs,
    // Do not return output from any device.
    kNotReturnOutputs,
    // Return the output only from the logical device 0.
    kReturnDevice0Outputs
  };

  // The options controlling the preprocessing of the HLO before it's compiled
  // and executed.
  struct PreprocessingOptions {
    // This indicates whether the module is the partitioned result of SPMD. If
    // yes, we will add (replicated) sharding annotations to the module.
    SpmdPartitionedMode spmd_partitioned_mode =
        SpmdPartitionedMode::kIsNotSpmdPartitionedModule;
    // If set, we will flatten all while loops to the specified number of
    // iterations.
    std::optional<int> while_execution_count = std::nullopt;
    // If set, we will remove all infeed and outfeed operations.
    bool remove_infeed_outfeed = true;

    // If set, we will flatten all conditional operations by setting default
    // branch index to N-1 for indexed conditionals. Default PRED is false for
    // predicated conditionals if conditional_value is not set.
    bool flatten_conditional = false;

    // If set, used as default predicate value for predicated conditional ops.
    bool conditional_value = false;

    // If set, convert the module to StableHLO before passing to PjRt for
    // compilation.
    bool compile_as_stablehlo = false;

    // Use layouts from the HLO module.
    bool use_layouts_from_hlo_module = false;

    // Should we flatten all while loops?
    bool flatten_while_loop() const {
      return while_execution_count.has_value();
    }

    // Set / update known_trip_count in while loop backend config.
    bool annotate_while_loop_trip_count = false;

    // Is the module the partitioned result of SPMD?
    bool is_spmd_partitioned_module() const {
      return spmd_partitioned_mode ==
             SpmdPartitionedMode::kIsSpmdPartitionedModule;
    }
  };

  // The options controlling the compilation of the HLO module.
  //
  // A CompileOptions object can be created from this with CreateCompileOptions.
  struct RawCompileOptions {
    HloPassesMode hlo_passes_mode = HloPassesMode::kStandardCompile;
    SpmdMode spmd_mode = SpmdMode::kNotUseSpmdPartitioning;
    // We can set additional build options by specifying an ExecutionOptions
    // message.
    //
    // It can also specify the number of replicas and partitions - in
    // that case we don't have to set num_replicas and num_partitions.
    std::optional<ExecutionOptions> execution_options = std::nullopt;
    std::optional<int> num_replicas = 1;
    std::optional<int> num_partitions = 1;
    // See the comment on xla::MultiSliceConfig.
    std::optional<int> num_slices = std::nullopt;
    // A directory to dump xla debug data to.
    std::string xla_dump_to = "";
    XlaTextDumpMode xla_text_dump_mode = XlaTextDumpMode::kNotDumpAsText;
    XlaProtoDumpMode xla_proto_dump_mode = XlaProtoDumpMode::kNotDumpAsProto;
    // A directory to dump xspace data to (GPU profiler only).
    std::string xla_gpu_dump_xspace_to = "";
  };

  // The options controlling the execution of the HLO module.
  struct RunningOptions {
    // Option controlling the inputs of the HLO.
    ModuleArgumentMode module_argument_mode =
        ModuleArgumentMode::kUseRandomInputs;
    // Option controlling the outputs of the HLO.
    ModuleOutputMode module_output_mode = ModuleOutputMode::kReturnOutputs;
    // Repeatedly execute the HLO for this many times.
    size_t num_repeats = 1;
    // If true, we recreate the buffers between repeats to reset of effect of
    // buffer donation.
    bool recreate_buffers_between_repeats = false;
    // This indicates whether we log the inputs and outputs to stderr.
    LogOutputMode log_input_output_mode = LogOutputMode::kNotLogOutput;
    const MultiSliceConfig* multi_slice_config = nullptr;
    ProfilerInterface* profiler = nullptr;
    // Whether to untuple the result of running HLO module into a vector of
    // arrays. If unprovided, use the default in ExecuteOptions.
    std::optional<bool> untuple_result = std::nullopt;
    // If not null, profiles will be stored for this run, one per repeat.
    // Note that the first repeat is a warmup run, and uses less precise
    // profiling method.
    std::vector<ExecutionProfile>* execution_profiles = nullptr;

    // Should we log the inputs and outputs to stderr?
    bool log_input_output() const {
      return log_input_output_mode == LogOutputMode::kLogOutput;
    }
  };

  struct HloModuleAndArguments {
    std::unique_ptr<HloModule> hlo_module;

    // The outer `std::vector` represents the list of shards. The inner
    // `std::vector<Literal>` represents a list of arguments for a single shard
    // partition.
    std::vector<std::vector<Literal>> arguments;
  };

  struct ReplicasAndPartitions {
    int replicas = 1;
    int partitions = 1;
  };

  // Loads an ExecutionOptions proto (which can be used in RawCompileOptions).
  static absl::StatusOr<ExecutionOptions> LoadExecutionOptions(
      absl::string_view path);

  // Creates the compilation options.
  //
  // If RawCompileOptions::num_slices is set, the
  // CompileOptions::device_assignment has to be set manually.
  static absl::StatusOr<CompileOptions> CreateCompileOptions(
      const PjRtClient& client,
      const FunctionalHloRunner::RawCompileOptions& raw_options,
      int task_id = 0, int num_nodes = 1,
      std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr);

  // Runs on HLO module and dumps the output if needed.
  //
  // This is the highest level API in this file.
  static absl::Status LoadAndRunAndDump(
      PjRtClient& client, const DebugOptions& debug_options,
      const xla::FunctionalHloRunner::PreprocessingOptions& preproc_options,
      const xla::FunctionalHloRunner::RawCompileOptions& raw_compile_options,
      const xla::FunctionalHloRunner::RunningOptions& running_options,
      absl::string_view hlo_file, InputFormat input_format,
      std::string dump_output_to = "", int task_id = 0, int num_nodes = 1,
      std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr);

  // Loads an HLO module from hlo_file according to input_format and run it.
  // The HLO module is run with the provided arguments if the arguments map is
  // not empty. Otherwise, use arguments from the HLO file or fake arguments.
  // The hlo file might be a HLO snapshot and thus contain arguments, otherwise
  // it is run with fake arguments.
  static absl::StatusOr<PerDeviceLiteralVecType> LoadAndRun(
      PjRtClient& client, const DebugOptions& debug_options,
      const PreprocessingOptions& preproc_options,
      const CompileOptions& compile_options,
      const RunningOptions& running_options, absl::string_view hlo_file,
      InputFormat input_format, const PerDeviceLiteralVecType& arguments = {},
      std::minstd_rand0* engine = nullptr);

  // Loads and compiles an HLO for debugging purposes.
  //
  // This function allows compiling multi-device HLOs on machines with fewer
  // devices.
  static absl::Status LoadAndCompile(
      PjRtClient& client, const DebugOptions& debug_options,
      const PreprocessingOptions& preproc_options,
      const RawCompileOptions& raw_compile_options, absl::string_view hlo_file,
      InputFormat input_format, int task_id = 0, int num_nodes = 1,
      std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr,
      bool use_gpu_count_workaround = true);

  // Compiles and runs the given HLO module with the given arguments for each
  // device. The given arguments is a map from device ID to a list of arguments.
  // If the arguments map is empty, the HLO module is run with fake arguments.
  static absl::StatusOr<PerDeviceLiteralVecType> CompileAndRun(
      PjRtClient& client, const DebugOptions& debug_options,
      const PreprocessingOptions& preproc_options,
      const CompileOptions& compile_options,
      const RunningOptions& running_options, HloModule* hlo_module,
      const PerDeviceLiteralVecType& arguments = {},
      std::minstd_rand0* engine = nullptr);

  // Compiles the HLO module.
  static absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      PjRtClient& client, HloModule* hlo_module,
      const DebugOptions& debug_options,
      const PreprocessingOptions& preproc_options,
      const CompileOptions& compile_options);

  // Ahead-of-time compilation using the PjRtTopologyDescription that's passed
  // instead of using the registered topology. This enables reproduction of
  // compilation based on captured information.
  static absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      PjRtClient& client, HloModule* hlo_module,
      const DebugOptions& debug_options,
      const PreprocessingOptions& preproc_options,
      const CompileOptions& compile_options,
      const PjRtTopologyDescription& topology);

  // Runs the executable.
  static absl::StatusOr<PerDeviceLiteralVecType> Run(
      PjRtClient& client, PjRtLoadedExecutable* executable,
      const PerDeviceLiteralVecType& arguments,
      const RunningOptions& running_options,
      std::minstd_rand0* engine = nullptr);

  static absl::StatusOr<HloModuleAndArguments> LoadHloModuleAndArguments(
      absl::string_view hlo_file, InputFormat input_format);

  // This would ideally be private, but we need it for the implementation of
  // MultihostHloRunner.
  static absl::Status PrepareHloModuleForCompilation(
      HloModule* hlo_module, const DebugOptions& debug_options,
      const PreprocessingOptions& preproc_options);
  // This would ideally be private, but we need it for the implementation of
  // MultihostHloRunner.
  static CompileOptions CompleteCompileOptions(const HloModule& hlo_module,
                                               CompileOptions compile_options,
                                               const PreprocessingOptions&);

  static absl::Status DumpOutput(
      const FunctionalHloRunner::PerDeviceLiteralVecType& output,
      absl::string_view dump_output_to, int task_id,
      OutputFormat output_format = OutputFormat::kText);

 private:
  // Calculates the requested number of replicas and partitions.
  //
  // The explicit num_replicas and num_partitions options override
  // execution_options.
  //
  // Regarding the num_slices parameter, see the comment on
  // xla::MultiSliceConfig.
  static ReplicasAndPartitions GetReplicasAndPartitions(
      const std::optional<ExecutionOptions>& execution_options,
      int device_count, const std::optional<int>& num_replicas,
      const std::optional<int>& num_partitions, int num_slices = 1);

  // Creates an ExecutableBuildOptions using the specified ExecutionOptions.
  static ExecutableBuildOptions
  CreateExecutableBuildOptionsFromExecutionOptions(
      const ExecutionOptions& execution_options);

  static absl::Span<PjRtDevice* const> GetLocalDevices(
      const PjRtClient& client);

  // Creates fake arguments to run the given executable.
  static absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  CreateArgumentsOnDevice(PjRtClient& client,
                          const PjRtLoadedExecutable* executable,
                          const RunningOptions& running_options,
                          bool flatten_arguments = false,
                          std::minstd_rand0* engine = nullptr);

  // Creates uninitialized arguments to run the given executable.
  static absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  CreateUninitializedArgumentsOnDevice(PjRtClient& client,
                                       const PjRtLoadedExecutable* executable,
                                       const RunningOptions& running_options,
                                       bool flatten_arguments = false);

  // Creates argument buffers based on the given arguments map. Note that the
  // arguments might be invalid when arguments are destructed.
  static absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  CopyArgumentsToDevice(PjRtClient& client,
                        const PjRtLoadedExecutable* executable,
                        const PerDeviceLiteralVecType& arguments,
                        const RunningOptions& options, bool flattened_arguments,
                        bool clone_device0_arguments = false);

  static absl::StatusOr<PerDeviceLiteralVecType> RunInternal(
      PjRtClient& client, PjRtLoadedExecutable* executable,
      std::function<absl::StatusOr<
          std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(bool)>
          create_argument_buffers_on_device,
      const RunningOptions& running_options);

  static absl::StatusOr<PerDeviceLiteralVecType> FetchAndLogOutput(
      PjRtClient& client,
      const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>&
          output_buffers,
      ModuleOutputMode module_output_mode, bool log_output);

  static ReplicasAndPartitions GetReplicasAndPartitionsInternal(
      const std::optional<ExecutionOptions>& execution_options,
      int device_count, const std::optional<int>& num_replicas,
      const std::optional<int>& num_partitions, int num_slices = 1);
};

bool AbslParseFlag(absl::string_view text,
                   FunctionalHloRunner::ModuleArgumentMode* argument_mode,
                   std::string* error);
std::string AbslUnparseFlag(
    FunctionalHloRunner::ModuleArgumentMode argument_mode);

bool AbslParseFlag(absl::string_view text,
                   FunctionalHloRunner::ModuleOutputMode* output_mode,
                   std::string* error);
std::string AbslUnparseFlag(FunctionalHloRunner::ModuleOutputMode output_mode);

void AddShardingAnnotationsToSpmdPartitionedModule(HloModule* hlo_module);

}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_FUNCTIONAL_HLO_RUNNER_H_
