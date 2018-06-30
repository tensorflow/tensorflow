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

#include "tensorflow/compiler/xla/client/client_library.h"

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

LocalClientOptions::LocalClientOptions(se::Platform* platform,
                                       int number_of_replicas,
                                       int intra_op_parallelism_threads)
    : platform_(platform),
      number_of_replicas_(number_of_replicas),
      intra_op_parallelism_threads_(intra_op_parallelism_threads) {}

LocalClientOptions& LocalClientOptions::set_platform(se::Platform* platform) {
  platform_ = platform;
  return *this;
}

se::Platform* LocalClientOptions::platform() const { return platform_; }

LocalClientOptions& LocalClientOptions::set_number_of_replicas(
    int number_of_replicas) {
  number_of_replicas_ = number_of_replicas;
  return *this;
}

int LocalClientOptions::number_of_replicas() const {
  return number_of_replicas_;
}

LocalClientOptions& LocalClientOptions::set_intra_op_parallelism_threads(
    int num_threads) {
  intra_op_parallelism_threads_ = num_threads;
  return *this;
}

int LocalClientOptions::intra_op_parallelism_threads() const {
  return intra_op_parallelism_threads_;
}

/* static */ ClientLibrary& ClientLibrary::Singleton() {
  static ClientLibrary* c = new ClientLibrary;
  return *c;
}

ClientLibrary::ClientLibrary() = default;
ClientLibrary::~ClientLibrary() = default;

/* static */ StatusOr<LocalClient*> ClientLibrary::GetOrCreateLocalClient(
    se::Platform* platform) {
  LocalClientOptions default_options;
  default_options.set_platform(platform);
  return GetOrCreateLocalClient(default_options);
}

/* static */ StatusOr<LocalClient*> ClientLibrary::GetOrCreateLocalClient(
    const LocalClientOptions& options) {
  se::Platform* platform = options.platform();
  int replica_count = options.number_of_replicas();
  ClientLibrary& client_library = Singleton();
  tensorflow::mutex_lock lock(client_library.service_mutex_);

  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  auto it = client_library.local_instances_.find(platform->id());
  if (it != client_library.local_instances_.end()) {
    return it->second->client.get();
  }

  ServiceOptions service_options;
  service_options.set_platform(platform);
  service_options.set_number_of_replicas(replica_count);
  service_options.set_intra_op_parallelism_threads(
      options.intra_op_parallelism_threads());

  auto instance = MakeUnique<LocalInstance>();
  TF_ASSIGN_OR_RETURN(instance->service,
                      LocalService::NewService(service_options));
  instance->client = MakeUnique<LocalClient>(instance->service.get());
  LocalClient* cl = instance->client.get();

  client_library.local_instances_.insert(
      std::make_pair(platform->id(), std::move(instance)));
  return cl;
}

/* static */ LocalClient* ClientLibrary::LocalClientOrDie() {
  auto client_status = GetOrCreateLocalClient();
  TF_CHECK_OK(client_status.status());
  return client_status.ValueOrDie();
}

/* static */ LocalService* ClientLibrary::GetXlaService(
    se::Platform* platform) {
  ClientLibrary& client_library = Singleton();
  tensorflow::mutex_lock lock(client_library.service_mutex_);
  auto it = client_library.local_instances_.find(platform->id());
  CHECK(it != client_library.local_instances_.end());
  return it->second->service.get();
}

/* static */ StatusOr<CompileOnlyClient*>
ClientLibrary::GetOrCreateCompileOnlyClient(se::Platform* platform) {
  ClientLibrary& client_library = Singleton();
  tensorflow::mutex_lock lock(client_library.service_mutex_);

  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  auto it = client_library.compile_only_instances_.find(platform->id());
  if (it != client_library.compile_only_instances_.end()) {
    return it->second->client.get();
  }

  auto instance = MakeUnique<CompileOnlyInstance>();
  TF_ASSIGN_OR_RETURN(instance->service,
                      CompileOnlyService::NewService(platform));
  instance->client = MakeUnique<CompileOnlyClient>(instance->service.get());
  CompileOnlyClient* cl = instance->client.get();

  client_library.compile_only_instances_.insert(
      std::make_pair(platform->id(), std::move(instance)));
  return cl;
}

/* static */ void ClientLibrary::DestroyLocalInstances() {
  ClientLibrary& client_library = Singleton();
  tensorflow::mutex_lock lock(client_library.service_mutex_);

  client_library.local_instances_.clear();
  client_library.compile_only_instances_.clear();
}

}  // namespace xla
