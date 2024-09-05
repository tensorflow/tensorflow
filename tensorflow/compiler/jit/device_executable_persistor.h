/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_EXECUTABLE_PERSISTOR_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_EXECUTABLE_PERSISTOR_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/jit/xla_device_compiler_client.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/util.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

// Returns the persisted compilation cache file name for the given key.
std::string XlaSerializedCacheKeyToFileName(const XlaSerializedCacheKey& key);

// Offers a way to persist and/or load compiled `ExecutableType`s along with the
// corresponding HLO (`CompilationResult`) to/from `persistent_cache_directory`
// (if one was provided during construction) on disk  using `ClientType`.
template <typename ExecutableType, typename ClientType>
class DeviceExecutablePersistor {
 public:
  // Configuration for setting up persistence (directory, filename prefix, etc).
  struct Config {
    Config() = default;
    explicit Config(absl::string_view persistent_cache_directory,
                    bool disable_strict_signature_checks,
                    absl::string_view persistence_prefix,
                    bool persistent_cache_directory_read_only)
        : persistent_cache_directory(persistent_cache_directory),
          disable_strict_signature_checks(disable_strict_signature_checks),
          persistence_prefix(persistence_prefix),
          persistent_cache_directory_read_only(
              persistent_cache_directory_read_only) {}

    explicit Config(absl::string_view persistent_cache_directory,
                    bool disable_strict_signature_checks,
                    absl::string_view persistence_prefix)
        : persistent_cache_directory(persistent_cache_directory),
          disable_strict_signature_checks(disable_strict_signature_checks),
          persistence_prefix(persistence_prefix) {}

    // If non-empty, JIT-compiled executables are saved to and loaded from the
    // specified file system directory path.
    std::string persistent_cache_directory;

    // Disable strict signature checks for entries loaded into the cache from
    // external sources.
    bool disable_strict_signature_checks = false;

    // The cache persistence prefix to use if serializing/deserialzing entries.
    std::string persistence_prefix;

    // Cache is read-only if set to true.
    bool persistent_cache_directory_read_only = false;
  };

  DeviceExecutablePersistor(const Config& config,
                            const DeviceType& device_type);
  virtual ~DeviceExecutablePersistor() = default;

  // Returns std::nullopt if persistence is not enabled (i.e.
  // `persistent_cache_directory_` is empty) or if the serialized entry is not
  // found on disk. Otherwise, loads and returns the serialized executable
  // (or returns a status).
  // TODO(b/255826209): Take in Signature instead of hash and string once cache
  // is refactored.
  std::optional<StatusOr<std::unique_ptr<ExecutableType>>> TryToLoadExecutable(
      uint64 signature_hash, const std::string& signature_str,
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& compilation_result,
      DeviceCompilerClient<ExecutableType, ClientType>* client) const;

  // Tries to serialize an already built `executable` and persist it on disk. If
  // unable to do so, tries to build a serialized executable using the AOT
  // pipeline and persists that to disk.
  // TODO(b/255826209): Take in Signature instead hash and string once cache
  // is refactored.
  virtual Status TryToPersistExecutable(
      uint64 signature_hash, const std::string& signature_str,
      const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& compilation_result,
      const ExecutableType& executable,
      DeviceCompilerClient<ExecutableType, ClientType>* client) const;

  const DeviceType& device_type() const { return device_type_; }
  const std::string& persistence_prefix() const { return persistence_prefix_; }
  const std::string& persistent_cache_directory() const {
    return persistent_cache_directory_;
  }

 private:
  // Returns a cache key proto that identifies an entry in the compilation
  // cache.
  XlaSerializedCacheKey BuildSerializedCacheKey(
      uint64 signature_hash, const xla::HloModuleProto& hlo_module) const;

  XlaSerializedCacheKey BuildSerializedCacheKey(
      uint64 signature_hash, const xla::HloModuleProto& hlo_module,
      bool compiled_using_pjrt) const;

  // Serializes the signature and its corresponding entry to a proto message.
  absl::StatusOr<XlaSerializedCacheEntry> SerializeEntry(
      uint64 signature_hash, const XlaCompiler::Options& options,
      const XlaCompiler::CompilationResult& compilation_result,
      const ExecutableType& executable,
      DeviceCompilerClient<ExecutableType, ClientType>* compiler_client) const;

  // Saves the cache entry in the file directory supplied during the
  // construction of this class. Overwrites existing entries.
  Status SaveSerializedEntry(const XlaSerializedCacheEntry& entry) const;

  // Tries to read a cache entry given a `key` by searching the file directory
  // supplied during the construction of this class. Returns std::nullopt if no
  // cache entry is found.
  absl::StatusOr<std::optional<XlaSerializedCacheEntry>>
  TryToReadSerializedEntry(const XlaSerializedCacheKey& key) const;

  // Checks if the loaded `entry` matches the expected `key` and `hlo_module`.
  Status VerifyLoadedCacheEntry(const XlaSerializedCacheKey& key,
                                const xla::HloModuleProto& hlo_module,
                                const XlaSerializedCacheEntry& entry) const;

  std::string GetFilePath(const XlaSerializedCacheKey& key) const;

  const DeviceType device_type_;
  const bool disable_strict_signature_checks_;
  const std::string persistence_prefix_;

  // If non-empty, JIT-compiled executables are saved to and loaded from the
  // specified file system directory path.
  const std::string persistent_cache_directory_;

  // Cache is read-only if set to true.
  const bool persistent_cache_directory_read_only_;

  DeviceExecutablePersistor(const DeviceExecutablePersistor&) = delete;
  void operator=(const DeviceExecutablePersistor&) = delete;
};

template <typename ExecutableType, typename ClientType>
DeviceExecutablePersistor<ExecutableType, ClientType>::
    DeviceExecutablePersistor(const Config& config,
                              const DeviceType& device_type)
    : device_type_(device_type),
      disable_strict_signature_checks_(config.disable_strict_signature_checks),
      persistence_prefix_(config.persistence_prefix),
      persistent_cache_directory_(config.persistent_cache_directory),
      persistent_cache_directory_read_only_(
          config.persistent_cache_directory_read_only) {}

template <typename ExecutableType, typename ClientType>
std::string DeviceExecutablePersistor<ExecutableType, ClientType>::GetFilePath(
    const XlaSerializedCacheKey& key) const {
  const std::string file_name = XlaSerializedCacheKeyToFileName(key);
  return io::JoinPath(persistent_cache_directory_, file_name);
}

template <typename ExecutableType, typename ClientType>
XlaSerializedCacheKey
DeviceExecutablePersistor<ExecutableType, ClientType>::BuildSerializedCacheKey(
    uint64 signature_hash, const xla::HloModuleProto& hlo_module,
    bool compiled_using_pjrt) const {
  XlaSerializedCacheKey key;
  key.set_signature_fingerprint(signature_hash);
  key.set_cluster_fingerprint(DeterministicProtoHash64(hlo_module));
  key.set_device_type(device_type().type_string());
  key.set_prefix(persistence_prefix());
  key.set_compiled_using_pjrt(compiled_using_pjrt);
  return key;
}

template <typename ExecutableType, typename ClientType>
XlaSerializedCacheKey
DeviceExecutablePersistor<ExecutableType, ClientType>::BuildSerializedCacheKey(
    uint64 signature_hash, const xla::HloModuleProto& hlo_module) const {
  return BuildSerializedCacheKey(signature_hash, hlo_module, false);
}

// This template specialization sets compiled_using_prjt to true in the cache
// key when the template arguments are PjRtLoadedExecutable and PjRtClient.
template <>
inline XlaSerializedCacheKey
DeviceExecutablePersistor<xla::PjRtLoadedExecutable, xla::PjRtClient>::
    BuildSerializedCacheKey(uint64 signature_hash,
                            const xla::HloModuleProto& hlo_module) const {
  return BuildSerializedCacheKey(signature_hash, hlo_module, true);
}

template <typename ExecutableType, typename ClientType>
absl::StatusOr<std::optional<XlaSerializedCacheEntry>>
DeviceExecutablePersistor<ExecutableType, ClientType>::TryToReadSerializedEntry(
    const XlaSerializedCacheKey& key) const {
  Env* env = Env::Default();
  const std::string file_path = GetFilePath(key);
  if (!env->FileExists(file_path).ok()) {
    return absl::StatusOr<std::optional<XlaSerializedCacheEntry>>(std::nullopt);
  }

  XlaSerializedCacheEntry entry;
  TF_RETURN_IF_ERROR(ReadTextOrBinaryProto(env, file_path, &entry));
  return std::optional<XlaSerializedCacheEntry>(entry);
}

template <typename ExecutableType, typename ClientType>
Status
DeviceExecutablePersistor<ExecutableType, ClientType>::VerifyLoadedCacheEntry(
    const XlaSerializedCacheKey& key, const xla::HloModuleProto& hlo_module,
    const XlaSerializedCacheEntry& entry) const {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat("Verifying loaded cache entry: ",
                                        hlo_module.entry_computation_name()));

  if (!AreSerializedProtosEqual(key, entry.key())) {
    VLOG(2) << "Serialized cache key does not match:\n"
            << "got:\n"
            << entry.key().DebugString() << "\nexpected:\n"
            << key.DebugString() << "\n";
    return errors::InvalidArgument("Serialized cache key does not match.");
  }

  // Perform a stricter (slower) check of the snapshot to verify that they
  // match exactly.
  if (!disable_strict_signature_checks_) {
    if (!AreSerializedProtosEqual(hlo_module, entry.hlo_module())) {
      VLOG(2) << "HLOs do not match:\n"
              << "got:\n"
              << hlo_module.DebugString() << "\nexpected:\n"
              << entry.hlo_module().DebugString() << "\n";
      return errors::InvalidArgument("Serialized HLO does not match.");
    }
  }

  if (entry.executable().empty()) {
    return errors::InvalidArgument("No binary found in serialized entry.");
  }
  return absl::OkStatus();
}

template <typename ExecutableType, typename ClientType>
Status
DeviceExecutablePersistor<ExecutableType, ClientType>::SaveSerializedEntry(
    const XlaSerializedCacheEntry& entry) const {
  Env* env = Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(persistent_cache_directory_));

  // The cache on the filesystem can be read while we're writing out the proto.
  // To prevent reads of partially-written files, we write the proto to a temp
  // file, then move it into place once we're done writing.  And we warn the
  // user if these moves are not known to be atomic.
  bool has_atomic_move = false;
  env->HasAtomicMove(persistent_cache_directory_, &has_atomic_move)
      .IgnoreError();
  if (!has_atomic_move) {
    LOG_EVERY_POW_2(WARNING)
        << "Filesystem for XLA persistent cache at "
        << persistent_cache_directory_
        << " does not support atomic moves.  Therefore the persistent cache is "
           "racy if you have multiple XLA compilations occurring "
           "simultaneously!  You have been warned. :)";
  }

  // Write to temp location, then when that completes, atomically move into the
  // final location.
  std::string temp_path =
      io::JoinPath(persistent_cache_directory_,
                   XlaSerializedCacheKeyToFileName(entry.key()));
  if (!env->CreateUniqueFileName(&temp_path, ".tmp")) {
    return absl::UnavailableError(absl::StrCat(
        "Could not create a unique file inside ", persistent_cache_directory_));
  }
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, temp_path, entry));
  return env->RenameFile(temp_path, GetFilePath(entry.key()));
}

template <typename ExecutableType, typename ClientType>
absl::StatusOr<XlaSerializedCacheEntry>
DeviceExecutablePersistor<ExecutableType, ClientType>::SerializeEntry(
    uint64 signature_hash, const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& compilation_result,
    const ExecutableType& executable,
    DeviceCompilerClient<ExecutableType, ClientType>* compiler_client) const {
  XlaSerializedCacheEntry serialized_entry;
  const xla::HloModuleProto& hlo_module =
      compilation_result.computation->proto();
  *serialized_entry.mutable_key() =
      BuildSerializedCacheKey(signature_hash, hlo_module);
  *serialized_entry.mutable_hlo_module() = hlo_module;

  // XLA compiler supports exporting executables as an AOT compilation result
  // to avoid running potentially expensive compilation pipeline twice.
  // Check if XLA compiler can export available executable.
  if (auto serialized_executable =
          compiler_client->SerializeExecutable(executable);
      serialized_executable.ok()) {
    serialized_entry.set_executable(std::move(*serialized_executable));
    return serialized_entry;
  } else if (serialized_executable.status().code() == error::UNIMPLEMENTED) {
    VLOG(1) << "Executable export is not implemented";
  } else {
    return serialized_executable.status();
  }

  TF_ASSIGN_OR_RETURN(
      auto serialized_executable,
      compiler_client->BuildSerializedExecutable(options, compilation_result));
  serialized_entry.set_executable(std::move(serialized_executable));
  return serialized_entry;
}

template <typename ExecutableType, typename ClientType>
std::optional<StatusOr<std::unique_ptr<ExecutableType>>>
DeviceExecutablePersistor<ExecutableType, ClientType>::TryToLoadExecutable(
    uint64 signature_hash, const std::string& signature_str,
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& compilation_result,
    DeviceCompilerClient<ExecutableType, ClientType>* compiler_client) const {
  if (persistent_cache_directory_.empty()) {
    return std::nullopt;
  }

  const xla::HloModuleProto& hlo_module =
      compilation_result.computation->proto();

  XlaSerializedCacheKey cache_key =
      BuildSerializedCacheKey(signature_hash, hlo_module);

  std::optional<XlaSerializedCacheEntry> serialized_entry;
  {
    XLA_SCOPED_LOGGING_TIMER(
        absl::StrCat("Try loading serialized cache entry:", signature_str));
    TF_ASSIGN_OR_RETURN(serialized_entry, TryToReadSerializedEntry(cache_key));
  }

  if (!serialized_entry.has_value()) {
    return std::nullopt;
  }

  TF_RETURN_IF_ERROR(
      VerifyLoadedCacheEntry(cache_key, hlo_module, *serialized_entry));

  VLOG(1) << "Loading cached entry for: " << signature_str;
  return compiler_client->LoadExecutable(options, compilation_result,
                                         serialized_entry->executable());
}

template <typename ExecutableType, typename ClientType>
Status
DeviceExecutablePersistor<ExecutableType, ClientType>::TryToPersistExecutable(
    uint64 signature_hash, const std::string& signature_str,
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& compilation_result,
    const ExecutableType& executable,
    DeviceCompilerClient<ExecutableType, ClientType>* client) const {
  if (persistent_cache_directory_.empty() ||
      persistent_cache_directory_read_only_) {
    VLOG(1) << "Not persisting executable. No `persistent_cache_directory` "
               "provided or cache is read-only.";
    return absl::OkStatus();
  }

  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("Serializing and saving cache entry: ", signature_str));
  TF_ASSIGN_OR_RETURN(XlaSerializedCacheEntry serialized_entry,
                      SerializeEntry(signature_hash, options,
                                     compilation_result, executable, client));
  TF_RETURN_IF_ERROR(SaveSerializedEntry(std::move(serialized_entry)));
  VLOG(2) << "XlaSerializedCacheEntry saved for signature: [" << signature_str
          << "] with signature hash: " << signature_hash;
  return absl::OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_EXECUTABLE_PERSISTOR_H_
