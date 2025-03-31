#ifndef PROTON_UTILITY_ATOMIC_H_
#define PROTON_UTILITY_ATOMIC_H_

#include <atomic>
#include <mutex>

namespace proton {

template <typename T> T atomicMax(std::atomic<T> &target, T value) {
  T current = target.load();
  while (current < value && !target.compare_exchange_weak(current, value))
    ;
  return current;
}

template <typename T> T atomicMin(std::atomic<T> &target, T value) {
  T current = target.load();
  while (current > value && !target.compare_exchange_weak(current, value))
    ;
  return current;
}

template <typename Condition, typename Function>
void doubleCheckedLock(Condition enterCondition, std::mutex &lock,
                       Function function) {
  if (!enterCondition())
    return;

  std::unique_lock<std::mutex> guard(lock);

  if (!enterCondition())
    return;

  function();
}

} // namespace proton

#endif // PROTON_UTILITY_ATOMIC_H_
