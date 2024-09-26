/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/weakref_lru_cache.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/lru_cache.h"

namespace nb = nanobind;

namespace jax {
namespace {

// Minimal wrapper to expose a nb::dict_iterator's value as something
// hashable with Abseil.
class HashablePyDictEntry {
 public:
  explicit HashablePyDictEntry(std::pair<nb::handle, nb::handle> entry)
      : entry_(entry) {}

  template <typename H>
  friend H AbslHashValue(H h, const HashablePyDictEntry& v) {
    return H::combine(std::move(h), nb::hash(v.entry_.first),
                      nb::hash(v.entry_.second));
  }

  std::pair<nb::handle, nb::handle> entry_;
};

// Similarly, a minimalist adaptor around the nb::detail::dict_iterator
// itself. Note that the iterator "is" also a Value. Does not meet the full
// standard iterator requirements, only enough to support H::combine_unordered.
class HashablePyDictIter {
 public:
  using iterator_category = std::input_iterator_tag;

  explicit HashablePyDictIter(nb::detail::dict_iterator& iter) : iter_(iter) {}

  // Minimal set of iterator operations.
  HashablePyDictEntry operator*() const { return HashablePyDictEntry(*iter_); }
  bool operator!=(const HashablePyDictIter& rhs) const {
    return iter_ != rhs.iter_;
  }
  void operator++() { ++iter_; }

 private:
  nb::detail::dict_iterator& iter_;
};

}  // namespace

class WeakrefLRUCache : public std::enable_shared_from_this<WeakrefLRUCache> {
 public:
  struct Key {
    nb::object context;
    nb::args args;
    nb::kwargs kwargs;

    bool operator==(const Key& other) const {
      return context.equal(other.context) && args.equal(other.args) &&
             kwargs.equal(other.kwargs);
    }

    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      // Note: Despite the fact this is an ABSL hash function, it's safe to call
      // functions that may throw exceptions such as nb::hash(), because it is
      // used by an LRUCache, which uses a std::unordered_map, which is
      // exception-safe.
      h = H::combine(std::move(h), nb::hash(key.context), nb::hash(key.args));
      nb::detail::dict_iterator begin = key.kwargs.begin();
      nb::detail::dict_iterator end = key.kwargs.end();
      h = H::combine_unordered(std::move(h), HashablePyDictIter(begin),
                               HashablePyDictIter(end));
      h = H::combine(std::move(h), key.kwargs.size());
      return h;
    }
  };

  struct CacheEntry {
    bool has_result = false;
    nb::object result;
    absl::Notification completed;
    std::thread::id thread_id = std::this_thread::get_id();
  };

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };

  struct WeakrefCacheKey {
    nb::handle object;
    size_t cached_hash;
  };

  using Cache = xla::LRUCache<Key, std::shared_ptr<CacheEntry>>;

  struct WeakrefCacheValue {
    std::optional<nb::weakref> weakref;
    std::shared_ptr<Cache> cache;
  };

  struct WeakrefKeyHash {
    size_t operator()(const WeakrefCacheKey& v) const { return v.cached_hash; }
  };

  struct WeakrefKeyEq {
    bool operator()(const WeakrefCacheKey& lhs,
                    const WeakrefCacheKey& rhs) const {
      return lhs.object.equal(rhs.object);
    }
  };

  WeakrefLRUCache(nb::callable cache_context_fn, nb::callable fn,
                  int64_t maxsize)
      : cache_context_fn_(cache_context_fn), fn_(fn), lru_list_(maxsize) {}

  std::shared_ptr<Cache> GetCache(WeakrefCacheKey key) {
    auto [it, inserted] = entries_.emplace(key, WeakrefCacheValue());
    if (!inserted) {
      return it->second.cache;
    }

    auto& value = it->second;

    value.cache = std::make_shared<Cache>(&lru_list_);
    value.weakref =
        nb::weakref(key.object, nb::cpp_function([this_weak = weak_from_this(),
                                                  key](nb::handle weakref) {
                      auto cache = this_weak.lock();
                      if (cache == nullptr) {
                        return;
                      }
                      auto it = cache->entries_.find(key);
                      if (it == cache->entries_.end()) {
                        return;
                      }
                      // Create temp-var to avoid re-entrant erase.
                      auto tmp = std::move(it->second);
                      cache->entries_.erase(it);
                    }));
    return value.cache;
  }

  nb::object Call(nb::object weakref_key, nb::args args,
                  nb::kwargs kwargs) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    nb::object context = cache_context_fn_();
    std::shared_ptr<Cache> cache_ptr = GetCache(WeakrefCacheKey{
        weakref_key, static_cast<size_t>(nb::hash(weakref_key))});
    Cache& cache = *cache_ptr;
    ++total_queries_;

    bool inserted = false;
    std::shared_ptr<CacheEntry> entry;
    {
      // Because the gil can be released during cache insertion, this forces
      // the lock order to be mu_ then gil so we must release the gil first.
      nb::gil_scoped_release release;
      // Acquire a mutex to avoid problems where the gil is released during
      // cache insertion and then a second thread invalidates the cache order.
      mu_.Lock();
    }
    {
      // GetOrCreateIfAbsent calls into Python hash and equality functions,
      // which may throw exceptions. The use of absl::Cleanup ensures mu_ is
      // released if that happens.
      absl::Cleanup unlock = [this]()
                                 ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
      Key key{context, args, kwargs};
      entry = cache.GetOrCreateIfAbsent(key, [&inserted](const Key& key) {
        inserted = true;
        return std::make_shared<CacheEntry>();
      });
    }
    if (!entry->completed.HasBeenNotified()) {
      if (inserted) {
        ++misses_;
        absl::Cleanup notify = [&] { entry->completed.Notify(); };
        entry->result = fn_(weakref_key, *args, **kwargs);
        entry->has_result = true;
      } else {
        if (entry->thread_id == std::this_thread::get_id()) {
          auto error_string =
              absl::StrCat("Recursively calling ",
                           nb::cast<std::string>(nb::repr(weakref_key)),
                           nb::cast<std::string>(nb::repr(args)));
          PyErr_SetString(PyExc_RecursionError, error_string.c_str());
          throw nb::python_error();
        }
        nb::gil_scoped_release release;
        entry->completed.WaitForNotification();
      }
    }

    if (entry->has_result) {
      return entry->result;
    } else {
      ++misses_;
      return fn_(weakref_key, *args, **kwargs);
    }
  }
  std::vector<nb::object> GetKeys() {
    std::vector<nb::object> results;
    mu_.Lock();
    for (const auto& wr_entry : entries_) {
      for (const auto& rest : *wr_entry.second.cache) {
        nb::tuple result =
            nb::make_tuple(*wr_entry.second.weakref, rest.first.context,
                           rest.first.args, rest.first.kwargs);
        results.push_back(std::move(result));
      }
    }
    mu_.Unlock();
    return results;
  }
  CacheInfo GetCacheInfo() const {
    CacheInfo result;
    result.hits = total_queries_ - misses_;
    result.misses = misses_;
    result.maxsize = lru_list_.Capacity();
    result.currsize = lru_list_.Size();
    return result;
  }
  void Clear() {
    total_queries_ = misses_ = 0;
    std::vector<std::shared_ptr<Cache>> deferred_deletes;
    deferred_deletes.reserve(entries_.size());
    for (auto& entry : entries_) {
      deferred_deletes.push_back(std::move(entry.second.cache));
    }
    entries_.clear();
    deferred_deletes.clear();
  }

  nb::callable cache_context_fn_;
  nb::callable fn_;
  Cache::LRUList lru_list_;
  std::unordered_map<WeakrefCacheKey, WeakrefCacheValue, WeakrefKeyHash,
                     WeakrefKeyEq>
      entries_;
  int64_t misses_ = 0;
  int64_t total_queries_ = 0;
  absl::Mutex mu_;
};

void BuildWeakrefLRUCacheAPI(nb::module_& m) {
  auto weakref_lru_cache =
      nb::class_<WeakrefLRUCache>(m, "WeakrefLRUCache",
                                  nb::is_weak_referenceable())
          .def("__call__", &WeakrefLRUCache::Call)
          .def("cache_keys", &WeakrefLRUCache::GetKeys)
          .def("cache_info", &WeakrefLRUCache::GetCacheInfo)
          .def("cache_clear", &WeakrefLRUCache::Clear);
  nb::class_<WeakrefLRUCache::CacheInfo>(weakref_lru_cache,
                                         "WeakrefLRUCacheInfo")
      .def_ro("hits", &WeakrefLRUCache::CacheInfo::hits)
      .def_ro("misses", &WeakrefLRUCache::CacheInfo::misses)
      .def_ro("maxsize", &WeakrefLRUCache::CacheInfo::maxsize)
      .def_ro("currsize", &WeakrefLRUCache::CacheInfo::currsize)
      .def("__repr__", [](WeakrefLRUCache::CacheInfo& info) {
        return absl::StrCat(
            "WeakrefLRUCache(hits=", info.hits, ", misses=", info.misses,
            ", maxsize=", info.maxsize, ", currsize=", info.currsize, ")");
      });
  m.def(
      "weakref_lru_cache",
      [](nb::callable cache_context_fn, nb::callable fn, int64_t maxsize) {
        return std::make_shared<WeakrefLRUCache>(cache_context_fn, fn, maxsize);
      },
      nb::arg("cache_context_fn"), nb::arg("fn"), nb::arg("maxsize") = 2048);
}

}  // namespace jax
