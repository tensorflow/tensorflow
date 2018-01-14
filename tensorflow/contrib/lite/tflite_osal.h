#ifndef TFLITE_OSAL_H
#define TFLITE_OSAL_H

/*
Tensorflow-Lite operating system abstraction layer
*/


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
