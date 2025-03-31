#ifndef PROTON_UTILITY_SINGLETON_H_
#define PROTON_UTILITY_SINGLETON_H_

namespace proton {

template <typename T> class Singleton {
public:
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static T &instance() {
    static T _;
    return _;
  }

protected:
  Singleton() = default;
};

} // namespace proton

#endif // PROTON_UTILITY_SINGLETON_H_
