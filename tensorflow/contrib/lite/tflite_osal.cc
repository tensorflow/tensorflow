#include "tensorflow/contrib/lite/tflite_osal.h"


#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tflite
{
    namespace osal
    {
        void *load_dynamic_library(const char* filename, int flags)
        {
            #ifdef _WIN32
                return reinterpret_cast<void*>(LoadLibrary(filename));
            #else
                return dlopen(filename, flags);
            #endif
        }

        void *load_dynamic_symbol(void* library_handle, const char *name)
        {
            void *symbol = nullptr;

            if( library_handle == nullptr )
                return symbol;

            #ifdef _WIN32
                HINSTANCE dll_handle = reinterpret_cast<HINSTANCE>(library_handle);
                symbol = reinterpret_cast<void*>( GetProcAddress( dll_handle, name ) );
            #else
                symbol = dlsym( library_handle, name );
            #endif

            return symbol;
        }

        int unload_dynamic_library( void *library_handle )
        {
            #ifdef _WIN32
                return (int)FreeLibrary( reinterpret_cast<HINSTANCE>( library_handle ) );
            #else
                return dlclose(library_handle);
            #endif
        }
    }
}
