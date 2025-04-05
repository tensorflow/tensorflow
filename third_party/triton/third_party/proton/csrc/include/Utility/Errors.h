#ifndef PROTON_UTILITY_ERRORS_H_
#define PROTON_UTILITY_ERRORS_H_

#include <stdexcept>

namespace proton {

class NotImplemented : public std::logic_error {
public:
  NotImplemented() : std::logic_error("Not yet implemented") {};
};

} // namespace proton

#endif // PROTON_UTILITY_ERRORS_H_
