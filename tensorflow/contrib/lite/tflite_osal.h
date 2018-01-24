#ifndef TFLITE_OSAL_H
#define TFLITE_OSAL_H

/*
Tensorflow-Lite operating system abstraction layer
*/

#ifdef _WIN32
static constexpr void* MAP_FAILED = nullptr;
#define PROT_READ (0x1)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace tflite
{
    namespace osal
    {
        void *load_dynamic_library(const char* filename, int flags=0);

        void *load_dynamic_symbol(void* library_handle, const char *name);

        int unload_dynamic_library( void *library_handle );
    }
}



#endif // TFLITE_OSAL_H
