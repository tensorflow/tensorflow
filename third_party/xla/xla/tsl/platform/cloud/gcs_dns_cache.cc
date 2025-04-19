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

#include "xla/tsl/platform/cloud/gcs_dns_cache.h"

#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/retrying_utils.h"
#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#else
#include <Windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#include <sys/types.h>

namespace tsl {

namespace {

const std::vector<string>& kCachedDomainNames =
    *new std::vector<string>{"www.googleapis.com", "storage.googleapis.com"};

inline void print_getaddrinfo_error(const string& name,
                                    absl::Status return_status) {
  // Status doesn't map well to EAI type errors.
  LOG(ERROR) << "Error resolving " << name << ": " << return_status;
}

// Selects one item at random from a vector of items, using a uniform
// distribution.
template <typename T>
const T& SelectRandomItemUniform(std::default_random_engine* random,
                                 const std::vector<T>& items) {
  CHECK_GT(items.size(), 0);
  std::uniform_int_distribution<size_t> distribution(0u, items.size() - 1u);
  size_t choice_index = distribution(*random);
  return items[choice_index];
}
}  // namespace

GcsDnsCache::GcsDnsCache(Env* env, int64_t refresh_rate_secs)
    : env_(env), refresh_rate_secs_(refresh_rate_secs) {}

void GcsDnsCache::AnnotateRequest(HttpRequest* request) {
  // TODO(saeta): Denylist failing IP addresses.
  absl::MutexLock l(&mu_);
  if (!started_) {
    VLOG(1) << "Starting GCS DNS cache.";
    DCHECK(!worker_) << "Worker thread already exists!";
    // Perform DNS resolutions to warm the cache.
    addresses_ = ResolveNames(kCachedDomainNames);

    // Note: we opt to use a thread instead of a delayed closure.
    worker_.reset(env_->StartThread({}, "gcs_dns_worker",
                                    [this]() { return WorkerThread(); }));
    started_ = true;
  }

  CHECK_EQ(kCachedDomainNames.size(), addresses_.size());
  for (size_t i = 0; i < kCachedDomainNames.size(); ++i) {
    const string& name = kCachedDomainNames[i];
    const std::vector<string>& addresses = addresses_[i];
    if (!addresses.empty()) {
      const string& chosen_address =
          SelectRandomItemUniform(&random_, addresses);
      request->AddResolveOverride(name, 443, chosen_address);
      VLOG(1) << "Annotated DNS mapping: " << name << " --> " << chosen_address;
    } else {
      LOG(WARNING) << "No IP addresses available for " << name;
    }
  }
}

/* static */ std::vector<string> GcsDnsCache::ResolveName(const string& name) {
  VLOG(1) << "Resolving DNS name: " << name;

  addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;  // Only use IPv4 for now.
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  RetryConfig retryConfig(
      /* init_delay_time_us = */ 5000,
      /* max_delay_time_us = */ 50 * 1000 * 5000,
      /* max_retries = */ 5);

  const absl::Status getaddrinfo_status = RetryingUtils::CallWithRetries(
      [&name, &hints, &result]() {
        int return_code = getaddrinfo(name.c_str(), nullptr, &hints, &result);
        absl::Status return_status;
        switch (return_code) {
          case 0:
            return_status = absl::OkStatus();
            break;
#ifndef _WIN32
          case EAI_ADDRFAMILY:
          case EAI_SERVICE:
          case EAI_SOCKTYPE:
          case EAI_NONAME:
            return_status = absl::FailedPreconditionError(
                absl::StrCat("System in invalid state for getaddrinfo call: ",
                             gai_strerror(return_code)));
            break;
          case EAI_AGAIN:
          case EAI_NODATA:  // lump nodata in here - the domains being resolved
                            // should always have data
            return_status = absl::UnavailableError(absl::StrCat(
                "Resolving ", name, " is temporarily unavailable"));
            break;
          case EAI_BADFLAGS:
          case EAI_FAMILY:
            return_status = absl::InvalidArgumentError(absl::StrCat(
                "Bad arguments for getaddrinfo: ", gai_strerror(return_code)));
            break;
          case EAI_FAIL:
            return_status = absl::NotFoundError(
                absl::StrCat("Permanent failure resolving ", name, ": ",
                             gai_strerror(return_code)));
            break;
          case EAI_MEMORY:
            return_status = absl::ResourceExhaustedError("Out of memory");
            break;
          case EAI_SYSTEM:
          default:
            return_status = absl::UnknownError(strerror(return_code));
#else
          // mapping from
          // https://learn.microsoft.com/en-us/windows/win32/api/ws2tcpip/nf-ws2tcpip-getaddrinfo#return-value
          case WSATYPE_NOT_FOUND:
          case WSAESOCKTNOSUPPORT:
          case WSAHOST_NOT_FOUND:
            return_status = absl::FailedPreconditionError(
                absl::StrCat("System in invalid state for getaddrinfo call: ",
                             gai_strerror(return_code)));
            break;
          case WSATRY_AGAIN:
            return_status = absl::UnavailableError(absl::StrCat(
                "Resolving ", name, " is temporarily unavailable"));
            break;
          case WSAEINVAL:
          case WSAEAFNOSUPPORT:
            return_status = absl::InvalidArgumentError(absl::StrCat(
                "Bad arguments for getaddrinfo: ", gai_strerror(return_code)));
            break;
          case WSANO_RECOVERY:
            return_status = absl::NotFoundError(
                absl::StrCat("Permanent failure resolving ", name, ": ",
                             gai_strerror(return_code)));
            break;
          case WSA_NOT_ENOUGH_MEMORY:
            return_status = absl::ResourceExhaustedError("Out of memory");
            break;
          default:
            return_status = absl::UnknownError(strerror(return_code));
#endif
        }

        return absl::Status(return_status);
      },
      retryConfig);

  std::vector<string> output;
  if (getaddrinfo_status.ok()) {
    for (const addrinfo* i = result; i != nullptr; i = i->ai_next) {
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
        output.push_back(buf);
        VLOG(1) << "... address: " << buf;
      }
    }
  } else {
    print_getaddrinfo_error(name, getaddrinfo_status);
  }
  if (result != nullptr) {
    freeaddrinfo(result);
  }
  return output;
}

// Performs DNS resolution for a set of DNS names. The return vector contains
// one element for each element in 'names', and each element is itself a
// vector of IP addresses (in textual form).
//
// If DNS resolution fails for any name, then that slot in the return vector
// will still be present, but will be an empty vector.
//
// Ensures: names.size() == return_value.size()

std::vector<std::vector<string>> GcsDnsCache::ResolveNames(
    const std::vector<string>& names) {
  std::vector<std::vector<string>> all_addresses;
  all_addresses.reserve(names.size());
  for (const string& name : names) {
    all_addresses.push_back(ResolveName(name));
  }
  return all_addresses;
}

void GcsDnsCache::WorkerThread() {
  while (true) {
    {
      // Don't immediately re-resolve the addresses.
      absl::MutexLock l(&mu_);
      if (cancelled_) return;
      cond_var_.WaitWithTimeout(&mu_, absl::Seconds(refresh_rate_secs_));
      if (cancelled_) return;
    }

    // Resolve DNS values
    auto new_addresses = ResolveNames(kCachedDomainNames);

    {
      absl::MutexLock l(&mu_);
      // Update instance variables.
      addresses_.swap(new_addresses);
    }
  }
}

}  // namespace tsl
