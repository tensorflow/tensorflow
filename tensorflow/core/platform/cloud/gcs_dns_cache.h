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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_DNS_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_DNS_CACHE_H_

#include <random>

#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
const int64 kDefaultRefreshRateSecs = 60;

// DnsCache is a userspace DNS cache specialized for the GCS filesystem.
//
// Some environments have unreliable DNS resolvers. DnsCache ameliorates the
// situation by radically reducing the number of DNS requests by performing
// 2 DNS queries per minute (by default) on a background thread. Updated cache
// entries are used to override curl's DNS resolution processes.
class GcsDnsCache {
 public:
  // Default no-argument constructor.
  GcsDnsCache() : GcsDnsCache(kDefaultRefreshRateSecs) {}

  // Constructs a GcsDnsCache with the specified refresh rate.
  GcsDnsCache(int64 refresh_rate_secs)
      : GcsDnsCache(Env::Default(), refresh_rate_secs) {}

  GcsDnsCache(Env* env, int64 refresh_rate_secs);

  ~GcsDnsCache() {
    mutex_lock l(mu_);
    cancelled_ = true;
    cond_var_.notify_one();
  }

  // Annotate the given HttpRequest with resolve overrides from the cache.
  void AnnotateRequest(HttpRequest* request);

 private:
  static std::vector<string> ResolveName(const string& name);
  static std::vector<std::vector<string>> ResolveNames(
      const std::vector<string>& names);
  void WorkerThread();

  // Define a friend class for testing.
  friend class GcsDnsCacheTest;

  mutex mu_;
  Env* env_;
  condition_variable cond_var_;
  std::default_random_engine random_ GUARDED_BY(mu_);
  bool started_ GUARDED_BY(mu_) = false;
  bool cancelled_ GUARDED_BY(mu_) = false;
  std::unique_ptr<Thread> worker_ GUARDED_BY(mu_);  // After mutable vars.
  const int64 refresh_rate_secs_;

  // Entries in this vector correspond to entries in kCachedDomainNames.
  std::vector<std::vector<string>> addresses_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_GCS_DNS_CACHE_H_
