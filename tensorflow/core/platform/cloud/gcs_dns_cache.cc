/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/gcs_dns_cache.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/types.h>

namespace tensorflow {

namespace {

constexpr char kStorageHost[] = "storage.googleapis.com";
constexpr char kWwwHost[] = "www.googleapis.com";

}  // namespace

GcsDnsCache::GcsDnsCache(Env* env, int64 refresh_rate_secs)
    : env_(env), refresh_rate_secs_(refresh_rate_secs) {}

Status GcsDnsCache::AnnotateRequest(HttpRequest* request) {
  // TODO(saeta): Blacklist failing IP addresses.
  mutex_lock l(mu_);
  if (!started_) {
    DCHECK(!worker_) << "Worker thread already exists!";
    // Perform DNS resolutions to warm the cache.
    std::vector<string> www_addresses = ResolveName(kWwwHost);
    std::vector<string> storage_addresses = ResolveName(kStorageHost);
    www_addresses.swap(www_addresses_);
    storage_addresses.swap(storage_addresses_);

    // Note: we opt to use a thread instead of a delayed closure.
    worker_.reset(env_->StartThread(
        {}, "gcs_dns_worker", std::bind(&GcsDnsCache::WorkerThread, this)));
    started_ = true;
  }
  if (!storage_addresses_.empty()) {
    std::uniform_int_distribution<> storage_dist(0,
                                                 storage_addresses_.size() - 1);
    size_t index = storage_dist(random_);
    TF_RETURN_IF_ERROR(request->AddResolveOverride(kStorageHost, 443,
                                                   storage_addresses_[index]));
  } else {
    LOG(WARNING) << "No IP addresses available for " << kStorageHost;
  }
  if (!www_addresses_.empty()) {
    std::uniform_int_distribution<> www_dist(0, www_addresses_.size() - 1);
    size_t index = www_dist(random_);
    TF_RETURN_IF_ERROR(
        request->AddResolveOverride(kWwwHost, 443, www_addresses_[index]));
  } else {
    LOG(WARNING) << "No IP addresses available for " << kWwwHost;
  }
  return Status::OK();
}

/* static */ std::vector<string> GcsDnsCache::ResolveName(const string& name) {
  addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;  // Only use IPv4 for now.
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  int return_code = getaddrinfo(name.c_str(), nullptr, &hints, &result);

  std::vector<string> output;
  if (return_code == 0) {
    for (addrinfo* i = result; i != nullptr; i = i->ai_next) {
      if (i->ai_family != AF_INET || i->ai_addr->sa_family != AF_INET) {
        LOG(WARNING) << "Non-IPv4 address returned. ai_family: " << i->ai_family
                     << ". sa_family: " << i->ai_addr->sa_family << ".";
        continue;
      }
      char buf[INET_ADDRSTRLEN];
      void* address_ptr =
          &(reinterpret_cast<sockaddr_in*>(i->ai_addr)->sin_addr);
      const char* formatted = nullptr;
      if ((formatted = inet_ntop(i->ai_addr->sa_family, address_ptr, buf,
                                 INET_ADDRSTRLEN)) == nullptr) {
        LOG(ERROR) << "Error converting response to IP address for " << name
                   << ": " << strerror(errno);
      } else {
        output.emplace_back(buf);
      }
    }
  } else {
    if (return_code == EAI_SYSTEM) {
      LOG(ERROR) << "Error resolving " << name
                 << " (EAI_SYSTEM): " << strerror(errno);
    } else {
      LOG(ERROR) << "Error resolving " << name << ": "
                 << gai_strerror(return_code);
    }
  }
  if (result != nullptr) {
    freeaddrinfo(result);
  }
  return output;
}

void GcsDnsCache::WorkerThread() {
  while (true) {
    {
      // Don't immediately re-resolve the addresses.
      mutex_lock l(mu_);
      if (cancelled_) return;
      cond_var_.wait_for(l, std::chrono::seconds(refresh_rate_secs_));
      if (cancelled_) return;
    }
    // Resolve DNS values
    std::vector<string> www_addresses = ResolveName(kWwwHost);
    std::vector<string> storage_addresses = ResolveName(kStorageHost);

    {
      mutex_lock l(mu_);
      // Update instance variables.
      www_addresses.swap(www_addresses_);
      storage_addresses.swap(storage_addresses_);
    }
  }
}

}  // namespace tensorflow
