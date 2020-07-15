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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_filesystem.h"

#include <aws/core/config/AWSProfileConfigLoader.h>
#include <stdlib.h>
#include <string.h>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/aws_crypto.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for S3 environments.
// This filesystem will support `s3://` URI schemes.
constexpr char kS3ClientAllocationTag[] = "S3ClientAllocation";
constexpr int64_t kS3TimeoutMsec = 300000;  // 5 min

constexpr char kExecutorTag[] = "TransferManagerExecutorAllocation";
constexpr int kExecutorPoolSize = 25;

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

static void ParseS3Path(const Aws::String& fname, bool object_empty_ok,
                        Aws::String* bucket, Aws::String* object,
                        TF_Status* status) {
  size_t scheme_end = fname.find("://") + 2;
  if (fname.substr(0, scheme_end + 1) != "s3://") {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't start with 's3://'.");
    return;
  }

  size_t bucket_end = fname.find("/", scheme_end + 1);
  if (bucket_end == std::string::npos) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't contain a bucket name.");
    return;
  }

  *bucket = fname.substr(scheme_end + 1, bucket_end - scheme_end - 1);
  *object = fname.substr(bucket_end + 1);

  if (object->empty() && !object_empty_ok) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "S3 path doesn't contain an object name.");
  }
}

static Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  ABSL_CONST_INIT static absl::Mutex cfg_lock(absl::kConstInit);
  static bool init(false);
  static Aws::Client::ClientConfiguration cfg;

  absl::MutexLock l(&cfg_lock);

  if (!init) {
    const char* endpoint = getenv("S3_ENDPOINT");
    if (endpoint) cfg.endpointOverride = Aws::String(endpoint);
    const char* region = getenv("AWS_REGION");
    // TODO (yongtang): `S3_REGION` should be deprecated after 2.0.
    if (!region) region = getenv("S3_REGION");
    if (region) {
      cfg.region = Aws::String(region);
    } else {
      // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
      // is set with a truthy value.
      const char* load_config_env = getenv("AWS_SDK_LOAD_CONFIG");
      std::string load_config =
          load_config_env ? absl::AsciiStrToLower(load_config_env) : "";
      if (load_config == "true" || load_config == "1") {
        Aws::String config_file;
        // If AWS_CONFIG_FILE is set then use it, otherwise use ~/.aws/config.
        const char* config_file_env = getenv("AWS_CONFIG_FILE");
        if (config_file_env) {
          config_file = config_file_env;
        } else {
          const char* home_env = getenv("HOME");
          if (home_env) {
            config_file = home_env;
            config_file += "/.aws/config";
          }
        }
        Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
        loader.Load();
        auto profiles = loader.GetProfiles();
        if (!profiles["default"].GetRegion().empty())
          cfg.region = profiles["default"].GetRegion();
      }
    }
    const char* use_https = getenv("S3_USE_HTTPS");
    if (use_https) {
      if (use_https[0] == '0')
        cfg.scheme = Aws::Http::Scheme::HTTP;
      else
        cfg.scheme = Aws::Http::Scheme::HTTPS;
    }
    const char* verify_ssl = getenv("S3_VERIFY_SSL");
    if (verify_ssl) {
      if (verify_ssl[0] == '0')
        cfg.verifySSL = false;
      else
        cfg.verifySSL = true;
    }
    // if these timeouts are low, you may see an error when
    // uploading/downloading large files: Unable to connect to endpoint
    int64_t timeout;
    cfg.connectTimeoutMs =
        absl::SimpleAtoi(getenv("S3_CONNECT_TIMEOUT_MSEC"), &timeout)
            ? timeout
            : kS3TimeoutMsec;
    cfg.requestTimeoutMs =
        absl::SimpleAtoi(getenv("S3_REQUEST_TIMEOUT_MSEC"), &timeout)
            ? timeout
            : kS3TimeoutMsec;
    const char* ca_file = getenv("S3_CA_FILE");
    if (ca_file) cfg.caFile = Aws::String(ca_file);
    const char* ca_path = getenv("S3_CA_PATH");
    if (ca_path) cfg.caPath = Aws::String(ca_path);
    init = true;
  }
  return cfg;
};

static void GetS3Client(TF_Filesystem* filesystem) {
  auto s3_file =
      static_cast<tf_s3_filesystem::S3File*>(filesystem->plugin_filesystem);
  absl::MutexLock l(&s3_file->initialization_lock);

  if (s3_file->s3_client.get() == nullptr) {
    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<tf_s3_filesystem::AWSSHA256Factory>(
          tf_s3_filesystem::AWSCryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<tf_s3_filesystem::AWSSHA256HmacFactory>(
          tf_s3_filesystem::AWSCryptoAllocationTag);
    };
    options.cryptoOptions.secureRandomFactory_create_fn = []() {
      return Aws::MakeShared<tf_s3_filesystem::AWSSecureRandomFactory>(
          tf_s3_filesystem::AWSCryptoAllocationTag);
    };
    Aws::InitAPI(options);

    // The creation of S3Client disables virtual addressing:
    //   S3Client(clientConfiguration, signPayloads, useVirtualAddressing =
    //   true)
    // The purpose is to address the issue encountered when there is an `.`
    // in the bucket name. Due to TLS hostname validation or DNS rules,
    // the bucket may not be resolved. Disabling of virtual addressing
    // should address the issue. See GitHub issue 16397 for details.
    s3_file->s3_client = Aws::MakeShared<Aws::S3::S3Client>(
        kS3ClientAllocationTag, GetDefaultClientConfig(),
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never, false);
  }
}

static void GetExecutor(TF_Filesystem* filesystem) {
  auto s3_file =
      static_cast<tf_s3_filesystem::S3File*>(filesystem->plugin_filesystem);
  absl::MutexLock l(&s3_file->initialization_lock);

  if (s3_file->executor.get() == nullptr) {
    s3_file->executor =
        Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
            kExecutorTag, kExecutorPoolSize);
  }
}

static void ShutdownClient(Aws::S3::S3Client* s3_client) {
  if (s3_client != nullptr) {
    delete s3_client;
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
  }
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

// TODO(vnvo2409): Implement later

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

// TODO(vnvo2409): Implement later

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

// TODO(vnvo2409): Implement later

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_s3_filesystem {
S3File::S3File()
    : s3_client(nullptr, ShutdownClient),
      executor(nullptr),
      initialization_lock() {}
void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new S3File();
  TF_SetStatus(status, TF_OK, "");
}

void Cleanup(TF_Filesystem* filesystem) {
  auto s3_file = static_cast<S3File*>(filesystem->plugin_filesystem);
  delete s3_file;
}

// TODO(vnvo2409): Implement later

}  // namespace tf_s3_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "s3");
}
