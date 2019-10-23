/*
 * Copyright XMOS Limited - 2009
 */

#ifndef _ExamplePlugin_H_
#define _ExamplePlugin_H_

#include "xsiplugin.h"

#ifdef __cplusplus
extern "C" {
#endif

DLL_EXPORT XsiStatus plugin_create(void **instance, XsiCallbacks *xsi, const char *arguments);
DLL_EXPORT XsiStatus plugin_clock(void *instance);
DLL_EXPORT XsiStatus plugin_notify(void *instance, int type, unsigned arg1, unsigned arg2);
DLL_EXPORT XsiStatus plugin_terminate(void *instance);

#ifdef __cplusplus
}
#endif

#endif /* _ExamplePlugin_H_ */
