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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_shared.h"

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/s3/S3Client.h>
#include <aws/transfer/TransferManager.h>
#include <aws/core/utils/threading/Executor.h>

#include <cstdlib>
#include <memory>

#include "tensorflow/c/experimental/filesystem/plugins/s3/aws_crypto.h"
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_helper.h"

namespace tf_s3_filesystem {

  namespace {
    Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
      static bool init(false);
      static Aws::Client::ClientConfiguration cfg;

      if (!init) {
        const char* endpoint = std::getenv("S3_ENDPOINT");
        if (endpoint) {
          cfg.endpointOverride = Aws::String(endpoint);
        }
        const char* region = std::getenv("AWS_REGION");
        if (!region) {
          // TODO (yongtang): `S3_REGION` should be deprecated after 2.0.
          region = std::getenv("S3_REGION");
        }
        if (region) {
          cfg.region = Aws::String(region);
        } else {
          // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
          // is set with a truthy value.

          const char* load_config_env = std::getenv("AWS_SDK_LOAD_CONFIG"); 
          if(load_config_env) {
            char* load_config_env_lower = strdup(load_config_env);
            for(int i = 0; i < strlen(load_config_env); ++i) {
              load_config_env_lower[i] = tolower(load_config_env_lower[i]);
            }
            const char* load_config = load_config_env_lower;
            if (load_config == "true" || load_config == "1") {
              Aws::String config_file;
              // If AWS_CONFIG_FILE is set then use it, otherwise use ~/.aws/config.
              const char* config_file_env = std::getenv("AWS_CONFIG_FILE");
              if (config_file_env) {
                config_file = config_file_env;
              } else {
                const char* home_env = std::getenv("HOME");
                if (home_env) {
                  config_file = home_env;
                  config_file += "/.aws/config";
                }
              }
              Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
              loader.Load();
              auto profiles = loader.GetProfiles();
              if (!profiles["default"].GetRegion().empty()) {
                cfg.region = profiles["default"].GetRegion();
              }
            }
            free(load_config_env_lower);
          }
        }
        const char* use_https = std::getenv("S3_USE_HTTPS");
        if (use_https) {
          if (use_https[0] == '0') {
            cfg.scheme = Aws::Http::Scheme::HTTP;
          } else {
            cfg.scheme = Aws::Http::Scheme::HTTPS;
          }
        }
        const char* verify_ssl = std::getenv("S3_VERIFY_SSL");
        if (verify_ssl) {
          if (verify_ssl[0] == '0') {
            cfg.verifySSL = false;
          } else {
            cfg.verifySSL = true;
          }
        }
        // if these timeouts are low, you may see an error when
        // uploading/downloading large files: Unable to connect to endpoint
        const char* connect_timeout_str = std::getenv("S3_CONNECT_TIMEOUT_MSEC");
        int64_t connect_timeout = kS3TimeoutMsec;
        if (connect_timeout_str) {
          // if conversion is unsafe, below method doesn't modify connect_timeout
          connect_timeout = std::stoll(connect_timeout_str, nullptr, 10);
        }
        cfg.connectTimeoutMs = connect_timeout;

        const char* request_timeout_str = std::getenv("S3_REQUEST_TIMEOUT_MSEC");
        int64_t request_timeout = kS3TimeoutMsec;
        if (request_timeout_str) {
          request_timeout = std::stoll(request_timeout_str, nullptr, 10);
        }
        cfg.requestTimeoutMs = request_timeout;

        const char* ca_file = std::getenv("S3_CA_FILE");
        if (ca_file) {
          cfg.caFile = Aws::String(ca_file);
        }
        const char* ca_path = std::getenv("S3_CA_PATH");
        if (ca_path) {
          cfg.caPath = Aws::String(ca_path);
        }

        init = true;
      }

      return cfg;
    }
  }  // namespace

  void GetS3Client(S3Shared* s3_shared) {
    if(s3_shared->s3_client == nullptr) {
      Aws::SDKOptions options;
      options.cryptoOptions.sha256Factory_create_fn = []() {
        return Aws::MakeShared<AWSSHA256Factory>(AWSCryptoAllocationTag);
      };
      options.cryptoOptions.sha256HMACFactory_create_fn = []() {
        return Aws::MakeShared<AWSSHA256HmacFactory>(AWSCryptoAllocationTag);
      };
      options.cryptoOptions.secureRandomFactory_create_fn = []() {
        return Aws::MakeShared<AWSSecureRandomFactory>(AWSCryptoAllocationTag);
      };
      Aws::InitAPI(options);

      // The creation of S3Client disables virtual addressing:
      //   S3Client(clientConfiguration, signPayloads, useVirtualAddressing =
      //   true)
      // The purpose is to address the issue encountered when there is an `.`
      // in the bucket name. Due to TLS hostname validation or DNS rules,
      // the bucket may not be resolved. Disabling of virtual addressing
      // should address the issue. See GitHub issue 16397 for details.
      s3_shared->s3_client = std::make_shared<Aws::S3::S3Client>(
        GetDefaultClientConfig(),
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        false
      );
    }
  }

  void GetTransferManager(S3Shared* s3_shared) {
    if(s3_shared->transfer_manager == nullptr) {
      GetS3Client(s3_shared);
      GetExecutor(s3_shared);
      Aws::Transfer::TransferManagerConfiguration config(s3_shared->thread_executor.get());
      config.s3Client = s3_shared->s3_client;
      config.bufferSize = multi_part_copy_part_size_;
      config.transferBufferMaxHeapSize = (kExecutorPoolSize + 1) * multi_part_copy_part_size_;
      s3_shared->transfer_manager = Aws::Transfer::TransferManager::Create(config);
    }
  }
  
  void GetExecutor(S3Shared* s3_shared) {
    if(s3_shared->thread_executor == nullptr) {
      s3_shared->thread_executor = Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
        kExecutorTag, kExecutorPoolSize
      );
    }
  }
}  // namespace tf_s3_filesystem
