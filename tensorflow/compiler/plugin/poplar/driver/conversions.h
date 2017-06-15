#ifndef IPU_CONVERSIONS_H_
#define IPU_CONVERSIONS_H_

#include <vector>

#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

std::vector<char> ConvertInt64ToInt32(void* src, int64 size);
std::vector<char> ConvertInt32ToInt64(void* src, int64 size);

}
}

#endif
