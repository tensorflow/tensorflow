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

#include "tensorflow/compiler/xla/python/weakref_lru_cache.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/synchronization/notification.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/lru_cache.h"

namespace jax {

class WeakrefLRUCache : public std::enable_shared_from_this<WeakrefLRUCache> {
 public:
  struct Key {
    pybind11::object context;
    pybind11::args args;
    pybind11::kwargs kwargs;

    bool operator==(const Key& other) const {
      if (!context.equal(other.context)) return false;
      if (!args.equal(other.args)) return false;
      if (!kwargs.equal(other.kwargs)) return false;
      return true;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      h = H::combine(std::move(h), pybind11::hash(key.context));
      h = H::combine(std::move(h), pybind11::hash(key.args));
      h = H::combine(std::move(h), key.kwargs.size());
      for (auto& kv : key.kwargs) {
        h = H::combine(std::move(h), pybind11::hash(kv.first));
        h = H::combine(std::move(h), pybind11::hash(kv.second));
      }
      return h;
    }
  };

  struct CacheEntry {
    bool has_result = false;
    pybind11::object result;
    absl::Notification completed;
    std::thread::id thread_id = std::this_thread::get_id();
  };

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };

  struct UnboundWeakrefCacheEntry {
    pybind11::handle object;
    WeakrefLRUCache* cache;
    size_t cached_hash;
  };

  struct WeakrefCacheEntry {
    pybind11::weakref weakref;
    size_t cached_hash;
  };

  struct WeakrefKeyHash {
    using is_transparent = void;

    size_t operator()(const UnboundWeakrefCacheEntry& v) const {
      return v.cached_hash;
    }
    size_t operator()(const WeakrefCacheEntry& v) const {
      return v.cached_hash;
    }
  };

  struct WeakrefKeyEq {
    using is_transparent = void;
    bool operator()(const WeakrefCacheEntry& lhs,
                    const WeakrefCacheEntry& rhs) const {
      return lhs.weakref.equal(rhs.weakref);
    }
    bool operator()(const WeakrefCacheEntry& lhs,
                    const UnboundWeakrefCacheEntry& rhs) const {
      PyObject* obj = PyWeakref_GET_OBJECT(lhs.weakref.ptr());
      if (obj == Py_None) {
        return false;
      }
      return pybind11::reinterpret_borrow<pybind11::object>(obj).equal(
          rhs.object);
    }
  };

  using Cache = xla::LRUCache<Key, std::shared_ptr<CacheEntry>>;
  WeakrefLRUCache(pybind11::function cache_context_fn, pybind11::function fn,
                  int64_t maxsize)
      : cache_context_fn_(cache_context_fn), fn_(fn), lru_list_(maxsize) {}

  std::shared_ptr<Cache> GetCache(const UnboundWeakrefCacheEntry& key) {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      return (it->second);
    }
    pybind11::weakref weakref(
        key.object, pybind11::cpp_function([this_weak = weak_from_this(),
                                            cached_hash = key.cached_hash](
                                               pybind11::handle weakref) {
          auto cache = this_weak.lock();
          if (cache == nullptr) {
            return;
          }
          cache->entries_.erase(WeakrefCacheEntry{
              pybind11::reinterpret_borrow<pybind11::weakref>(weakref),
              cached_hash});
        }));
    return (entries_
                .emplace(WeakrefCacheEntry{std::move(weakref), key.cached_hash},
                         std::make_shared<Cache>(&lru_list_))
                .first->second);
  }

  pybind11::object Call(pybind11::object weakref_key, pybind11::args args,
                        pybind11::kwargs kwargs) {
    pybind11::object context = cache_context_fn_();
    std::shared_ptr<Cache> cache_ptr = GetCache(UnboundWeakrefCacheEntry{
        weakref_key, this, static_cast<size_t>(pybind11::hash(weakref_key))});
    Cache& cache = *cache_ptr;
    ++total_queries_;

    bool inserted = false;
    Key key{context, args, kwargs};
    auto entry = cache.GetOrCreateIfAbsent(key, [&inserted](const Key& key) {
      inserted = true;
      return std::make_shared<CacheEntry>();
    });
    if (!entry->completed.HasBeenNotified()) {
      if (inserted) {
        ++misses_;
        absl::Cleanup notify = [&] { entry->completed.Notify(); };
        entry->result = fn_(weakref_key, *args, **kwargs);
        entry->has_result = true;
      } else {
        if (entry->thread_id == std::this_thread::get_id()) {
          auto error_string = absl::StrCat(
              "Recursively calling ",
              pybind11::cast<std::string>(pybind11::repr(weakref_key)),
              pybind11::cast<std::string>(pybind11::repr(args)));
          PyErr_SetString(PyExc_RecursionError, error_string.c_str());
          throw pybind11::error_already_set();
        }
        pybind11::gil_scoped_release release;
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
    entries_.clear();
  }

  pybind11::function cache_context_fn_;
  pybind11::function fn_;
  Cache::LRUList lru_list_;
  absl::node_hash_map<WeakrefCacheEntry, std::shared_ptr<Cache>, WeakrefKeyHash,
                      WeakrefKeyEq>
      entries_;
  int64_t misses_ = 0;
  int64_t total_queries_ = 0;
};

namespace {
namespace py = ::pybind11;
}  // namespace

void BuildWeakrefLRUCacheAPI(pybind11::module& m) {
  auto weakref_lru_cache =
      py::class_<WeakrefLRUCache, std::shared_ptr<WeakrefLRUCache>>(
          m, "WeakrefLRUCache")
          .def("__call__", &WeakrefLRUCache::Call)
          .def("cache_info", &WeakrefLRUCache::GetCacheInfo)
          .def("cache_clear", &WeakrefLRUCache::Clear);
  py::class_<WeakrefLRUCache::CacheInfo>(weakref_lru_cache,
                                         "WeakrefLRUCacheInfo")
      .def_readonly("hits", &WeakrefLRUCache::CacheInfo::hits)
      .def_readonly("misses", &WeakrefLRUCache::CacheInfo::misses)
      .def_readonly("maxsize", &WeakrefLRUCache::CacheInfo::maxsize)
      .def_readonly("currsize", &WeakrefLRUCache::CacheInfo::currsize)
      .def("__repr__", [](WeakrefLRUCache::CacheInfo& info) {
        return absl::StrCat(
            "WeakrefLRUCache(hits=", info.hits, ", misses=", info.misses,
            ", maxsize=", info.maxsize, ", currsize=", info.currsize, ")");
      });
  m.def(
      "weakref_lru_cache",
      [](pybind11::function cache_context_fn, pybind11::function fn,
         int64_t maxsize) {
        return std::make_shared<WeakrefLRUCache>(cache_context_fn, fn, maxsize);
      },
      pybind11::arg("cache_context_fn"), pybind11::arg("fn"),
      pybind11::arg("maxsize") = 2048);
}

}  // namespace jax
