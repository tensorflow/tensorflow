#ifndef TENSORFLOW_PLATFORM_LOGGING_H_
#define TENSORFLOW_PLATFORM_LOGGING_H_

#include "tensorflow/core/platform/port.h"  // To pick up PLATFORM_define

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
#include "base/logging.h"
#else
#include "tensorflow/core/platform/default/logging.h"
#endif

#endif  // TENSORFLOW_PLATFORM_LOGGING_H_
