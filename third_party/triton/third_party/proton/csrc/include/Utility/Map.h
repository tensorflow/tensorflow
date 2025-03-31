#ifndef PROTON_UTILITY_MAP_H_
#define PROTON_UTILITY_MAP_H_

#include <map>
#include <shared_mutex>

namespace proton {

/// A simple thread safe map with read/write lock.
template <typename Key, typename Value,
          typename Container = std::map<Key, Value>>
class ThreadSafeMap {
public:
  ThreadSafeMap() = default;

  Value &operator[](const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return map[key];
  }

  Value &operator[](Key &&key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return map[std::move(key)];
  }

  Value &at(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return map.at(key);
  }

  void insert(const Key &key, const Value &value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    map[key] = value;
  }

  bool contain(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = map.find(key);
    if (it == map.end())
      return false;
    return true;
  }

  bool erase(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return map.erase(key) > 0;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    map.clear();
  }

private:
  Container map;
  std::shared_mutex mutex;
};

} // namespace proton

#endif // PROTON_UTILITY_MAP_H_
