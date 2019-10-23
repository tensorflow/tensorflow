/*
 * @FileName XsiSlave.h
 *
 * @Version 1.0
 * @Description Slave instantiation of Xmos Simulator
 *
 * Copyright XMOS Ltd 2009
 */

#ifndef _XsiDevice_h_
#define _XsiDevice_h_

#include "xsi.h"

#ifdef __cplusplus
extern "C" {
#endif

DLL_EXPORT enum XsiStatus xsi_create(void **instance, const char *arguments);
DLL_EXPORT enum XsiStatus xsi_clock(void *instance);
DLL_EXPORT enum XsiStatus xsi_terminate(void *instance);

DLL_EXPORT enum XsiStatus xsi_read_mem(void *instance, const char *core,
                                       XsiWord32 address, unsigned num_bytes, unsigned char *data);
DLL_EXPORT enum XsiStatus xsi_write_mem(void *instance, const char *core,
                                        XsiWord32 address, unsigned num_bytes, unsigned char *data);

DLL_EXPORT enum XsiStatus xsi_read_pswitch_reg(void *instance, const char *core,
                                               unsigned reg_num, unsigned *value);
DLL_EXPORT enum XsiStatus xsi_write_pswitch_reg(void *instance, const char *core,
                                                unsigned reg_num, unsigned value);

DLL_EXPORT enum XsiStatus xsi_is_pin_driving(void *instance, const char *package,
		                                     const char *pin, unsigned int *value);
DLL_EXPORT enum XsiStatus xsi_sample_pin(void *instance, const char *package,
		                                 const char *pin, unsigned *value);
DLL_EXPORT enum XsiStatus xsi_drive_pin(void *instance, const char *package,
		                                const char *pin, unsigned value);

DLL_EXPORT enum XsiStatus xsi_is_port_pins_driving(void *instance, const char *core,
		                                           const char *port, XsiPortData mask, XsiPortData *value);
DLL_EXPORT enum XsiStatus xsi_sample_port_pins(void *instance, const char *core,
		                                       const char *port, XsiPortData mask, XsiPortData *value);
DLL_EXPORT enum XsiStatus xsi_drive_port_pins(void *instance, const char *core,
		                                      const char *port, XsiPortData mask, XsiPortData value);

DLL_EXPORT enum XsiStatus xsi_reset(void *instance, enum XsiResetType type);

DLL_EXPORT enum XsiStatus xsi_save_state(void *instance, const char *filename);
DLL_EXPORT enum XsiStatus xsi_restore_state(void *instance, const char *filename);

#ifdef __cplusplus
}
#endif

#endif
