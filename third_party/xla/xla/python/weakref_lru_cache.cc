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

#include <Python.h>

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
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/lru_cache.h"
#include "xla/tsl/platform/logging.h"

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

struct HashableKey {
  nb::object context;
  nb::args args;
  nb::kwargs kwargs;

  template <typename H>
  friend H AbslHashValue(H h, const HashableKey& key) {
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

}  // namespace

class WeakrefLRUCache : public std::enable_shared_from_this<WeakrefLRUCache> {
 public:
  class Key {
   public:
    Key(nb::object context, nb::args args, nb::kwargs kwargs)
        : context_(std::move(context)),
          args_(std::move(args)),
          kwargs_(std::move(kwargs)),
          cached_hash_(absl::HashOf(HashableKey{context_, args_, kwargs_})) {}

    bool operator==(const Key& other) const {
      return context_.equal(other.context_) && args_.equal(other.args_) &&
             kwargs_.equal(other.kwargs_);
    }

    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      return H::combine(std::move(h), key.cached_hash_);
    }

    nb::object context() const { return context_; }
    nb::args args() const { return args_; }
    nb::kwargs kwargs() const { return kwargs_; }

    int tp_traverse(visitproc visit, void* arg) const {
      Py_VISIT(context_.ptr());
      Py_VISIT(args_.ptr());
      Py_VISIT(kwargs_.ptr());
      return 0;
    }

   private:
    nb::object context_;
    nb::args args_;
    nb::kwargs kwargs_;
    size_t cached_hash_;
  };

  struct CacheEntry {
    bool has_result = false;
    nb::object result;
    absl::Notification completed;
    std::thread::id thread_id = std::this_thread::get_id();

    int tp_traverse(visitproc visit, void* arg) const {
      Py_VISIT(result.ptr());
      return 0;
    }
  };

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };

  struct WeakrefCacheKey {
    nb::weakref ref;
    size_t cached_hash;
  };

  using Cache = xla::LRUCache<Key, std::shared_ptr<CacheEntry>>;

  struct WeakrefCacheValue {
    std::shared_ptr<Cache> cache;
  };

  struct WeakrefKeyHash {
    size_t operator()(const WeakrefCacheKey& v) const { return v.cached_hash; }
  };

  struct WeakrefKeyEq {
    bool operator()(const WeakrefCacheKey& lhs,
                    const WeakrefCacheKey& rhs) const {
      return lhs.ref.equal(rhs.ref);
    }
  };

  WeakrefLRUCache(nb::callable cache_context_fn, nb::callable fn,
                  int64_t maxsize)
      : cache_context_fn_(cache_context_fn), fn_(fn), lru_list_(maxsize) {}

  std::shared_ptr<Cache> GetCache(WeakrefCacheKey key) {
    WeakrefCacheValue& value = entries_[key];
    if (!value.cache) {
      value.cache = std::make_shared<Cache>(&lru_list_);
    }
    return value.cache;
  }

  nb::object Call(nb::object weakref_key, nb::args args,
                  nb::kwargs kwargs) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    nb::object context = cache_context_fn_();

    // We precompute all of the hash values needed by the various maps rather
    // than computing them during the std::unordered_map insertions. At the very
    // least, MSVC's std::unordered_map has undefined behavior if the hash
    // function throws an exception
    // (https://learn.microsoft.com/en-us/cpp/standard-library/unordered-map-class?view=msvc-170#emplace).
    Key key(context, args, kwargs);
    size_t wrcache_hash = static_cast<size_t>(nb::hash(weakref_key));

    // No hash computations after this point.

    auto weakref_gc_callback = nb::cpp_function(
        [this_weak = weak_from_this(), wrcache_hash](nb::handle weakref) {
          auto cache = this_weak.lock();
          if (cache == nullptr) {
            return;
          }
          // Set up PyCriticalSection for cache python associated object;
          auto py_cache = nb::find(cache);
          // This should never happen as python cache should always be found
          CHECK(py_cache.ptr() != nullptr);
          nb::ft_object_guard lock(py_cache);

          // The object the reference referred to is now in the process of being
          // destroyed, so we cannot refer to its contents. Python weakref
          // objects compare based on identity if the object they refer to is
          // gone, so the hash lookup will work fine.
          auto it = cache->entries_.find(
              WeakrefCacheKey{nb::borrow<nb::weakref>(weakref), wrcache_hash});
          if (it == cache->entries_.end()) {
            return;
          }
          // Create temp-var to avoid re-entrant erase.
          auto tmp = std::move(it->second);
          cache->entries_.erase(it);
        });
    nb::weakref weakref = nb::weakref(weakref_key, weakref_gc_callback);
    WeakrefCacheKey wrcache_key{weakref, wrcache_hash};
    std::shared_ptr<Cache> cache_ptr = GetCache(wrcache_key);
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
            nb::make_tuple(*wr_entry.first.ref, rest.first.context(),
                           rest.first.args(), rest.first.kwargs());
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

  static int tp_traverse(PyObject* self, visitproc visit, void* arg) {
    WeakrefLRUCache* cache = nb::inst_ptr<WeakrefLRUCache>(self);
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(cache->cache_context_fn_.ptr());
    Py_VISIT(cache->fn_.ptr());
    for (const auto& [wr_key, wr_value] : cache->entries_) {
      Py_VISIT(wr_key.ref.ptr());
      for (const auto& [key, cache_value] : *wr_value.cache) {
        int rval = key.tp_traverse(visit, arg);
        if (rval != 0) {
          return rval;
        }
        if (cache_value.value.has_value()) {
          cache_value.value->get()->tp_traverse(visit, arg);
        }
      }
    }
    return 0;
  }

  static int tp_clear(PyObject* self) {
    WeakrefLRUCache* cache = nb::inst_ptr<WeakrefLRUCache>(self);
    cache->Clear();
    cache->cache_context_fn_.reset();
    cache->fn_.reset();
    return 0;
  }

  static PyType_Slot slots_[];
};

/* static */ PyType_Slot WeakrefLRUCache::slots_[] = {
    {Py_tp_traverse, (void*)WeakrefLRUCache::tp_traverse},
    {Py_tp_clear, (void*)WeakrefLRUCache::tp_clear},
    {0, nullptr},
};

void BuildWeakrefLRUCacheAPI(nb::module_& m) {
  auto weakref_lru_cache =
      nb::class_<WeakrefLRUCache>(m, "WeakrefLRUCache",
                                  nb::is_weak_referenceable(),
                                  nb::type_slots(WeakrefLRUCache::slots_))
          .def("__call__", &WeakrefLRUCache::Call, nb::lock_self())
          .def("cache_keys", &WeakrefLRUCache::GetKeys, nb::lock_self())
          .def("cache_info", &WeakrefLRUCache::GetCacheInfo, nb::lock_self())
          .def("cache_clear", &WeakrefLRUCache::Clear, nb::lock_self());
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
