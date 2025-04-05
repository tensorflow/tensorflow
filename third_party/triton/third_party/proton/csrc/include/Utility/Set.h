#ifndef PROTON_UTILITY_SET_H_
#define PROTON_UTILITY_SET_H_

#include <set>
#include <shared_mutex>

namespace proton {

/// A simple thread safe set with read/write lock.
template <typename Key, typename Container = std::set<Key>>
class ThreadSafeSet {
public:
  ThreadSafeSet() = default;

  void insert(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    set.insert(key);
  }

  bool contain(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = set.find(key);
    if (it == set.end())
      return false;
    return true;
  }

  bool erase(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return set.erase(key) > 0;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    set.clear();
  }

private:
  Container set;
  std::shared_mutex mutex;
};

} // namespace proton

#endif // PROTON_UTILITY_MAP_H_
