#pragma once
#ifdef TENSORFLOW_EXPORTS  
#define INIT_API __declspec(dllexport)   
#else  
#define INIT_API __declspec(dllimport)   
#endif  
#ifdef __cplusplus
extern "C" {
#endif
INIT_API const char* main_interface(const float* source_data);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
	INIT_API int init_main_interface();
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
	INIT_API void main_interface2(const float* source_data, char** output_label, int* a);
#ifdef __cplusplus
}
#endif
