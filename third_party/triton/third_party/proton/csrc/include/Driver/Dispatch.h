#ifndef PROTON_DRIVER_DISPATCH_H_
#define PROTON_DRIVER_DISPATCH_H_

#include <dlfcn.h>

#include <stdexcept>
#include <string>

#define DISPATCH_ARGS_0()
#define DISPATCH_ARGS_1(t1) t1 v1
#define DISPATCH_ARGS_2(t1, t2) t1 v1, t2 v2
#define DISPATCH_ARGS_3(t1, t2, t3) t1 v1, t2 v2, t3 v3
#define DISPATCH_ARGS_4(t1, t2, t3, t4) t1 v1, t2 v2, t3 v3, t4 v4
#define DISPATCH_ARGS_N(_4, _3, _2, _1, _0, N, ...) DISPATCH_ARGS##N
#define DISPATCH_ARGS(...)                                                     \
  DISPATCH_ARGS_N(_0, ##__VA_ARGS__, _4, _3, _2, _1, _0)                       \
  (__VA_ARGS__)

#define DISPATCH_VALS_0()
#define DISPATCH_VALS_1(t1) , v1
#define DISPATCH_VALS_2(t1, t2) , v1, v2
#define DISPATCH_VALS_3(t1, t2, t3) , v1, v2, v3
#define DISPATCH_VALS_4(t1, t2, t3, t4) , v1, v2, v3, v4
#define DISPATCH_VALS_N(_4, _3, _2, _1, _0, N, ...) DISPATCH_VALS##N
#define DISPATCH_VALS(...)                                                     \
  DISPATCH_VALS_N(_0, ##__VA_ARGS__, _4, _3, _2, _1, _0)                       \
  (__VA_ARGS__)

#define DEFINE_DISPATCH_TEMPLATE(CheckSuccess, FuncName, ExternLib, FuncType,  \
                                 ...)                                          \
  template <>                                                                  \
  ExternLib::RetType FuncName<CheckSuccess>(DISPATCH_ARGS(__VA_ARGS__)) {      \
    typedef typename ExternLib::RetType (*FuncType##_t)(__VA_ARGS__);          \
    static FuncType##_t func = nullptr;                                        \
    return Dispatch<ExternLib>::exec<CheckSuccess, FuncType##_t>(              \
        func, #FuncType DISPATCH_VALS(__VA_ARGS__));                           \
  }

#define DEFINE_DISPATCH(ExternLib, FuncName, FuncType, ...)                    \
  DEFINE_DISPATCH_TEMPLATE(true, FuncName, ExternLib, FuncType, __VA_ARGS__)   \
  DEFINE_DISPATCH_TEMPLATE(false, FuncName, ExternLib, FuncType, __VA_ARGS__)

namespace proton {

struct ExternLibBase {
  using RetType = int; // Generic type, can be overridden in derived structs
  static constexpr const char *name = ""; // Placeholder
  static constexpr RetType success = 0;   // Placeholder
  ExternLibBase() = delete;
  ExternLibBase(const ExternLibBase &) = delete;
  ExternLibBase &operator=(const ExternLibBase &) = delete;
  static inline void *lib{nullptr};
  static inline std::string defaultDir{""};
};

template <typename ExternLib> class Dispatch {
public:
  Dispatch() = delete;

  static void init(const char *name, void **lib) {
    if (*lib == nullptr) {
      // If not found, try to load it from the default path
      auto dir = std::string(ExternLib::defaultDir);
      if (dir.length() > 0) {
        auto fullPath = dir + "/" + name;
        *lib = dlopen(fullPath.c_str(), RTLD_LOCAL | RTLD_LAZY);
      } else {
        // Only if the default path is not set, we try to load it from the
        // system.
        // First reuse the existing handle
        *lib = dlopen(name, RTLD_NOLOAD);
        if (*lib == nullptr) {
          // If not found, try to load it from LD_LIBRARY_PATH
          *lib = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
        }
      }
    }
    if (*lib == nullptr) {
      throw std::runtime_error("Could not load `" + std::string(name) + "`");
    }
  }

  static void check(typename ExternLib::RetType ret, const char *functionName) {
    if (ret != ExternLib::success) {
      throw std::runtime_error("Failed to execute " +
                               std::string(functionName) + " with error " +
                               std::to_string(ret));
    }
  }

  template <bool CheckSuccess, typename FnT, typename... Args>
  static typename ExternLib::RetType
  exec(FnT &handler, const char *functionName, Args... args) {
    init(ExternLib::name, &ExternLib::lib);
    if (handler == nullptr) {
      handler = reinterpret_cast<FnT>(dlsym(ExternLib::lib, functionName));
      if (handler == nullptr) {
        throw std::runtime_error("Failed to load " +
                                 std::string(ExternLib::name));
      }
    }
    auto ret = handler(args...);
    if constexpr (CheckSuccess) {
      check(ret, functionName);
    }
    return ret;
  }
};

} // namespace proton

#endif // PROTON_DRIVER_DISPATCH_H_
