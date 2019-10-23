/*
 * @FileName XmosSimulatorInterface.h
 * @Date 20/04/2009
 *
 * @Version 1.0
 * @Description Simulation interface header
 *
 * Copyright XMOS Ltd 2009
 */

#ifndef _XsiPlugin_h_
#define _XsiPlugin_h_

#include "xsi.h"

#define XSI_PLUGIN_INTERFACE_VERSION 1.1

#define CHECK_INTERFACE_VERSION(xsi) \
	  xsi->check_interface_version(XSI_PLUGIN_INTERFACE_VERSION)

struct XsiCallbacks 
{
	enum XsiStatus (*check_interface_version)(double version);

    enum XsiStatus (*set_mhz)(double mhz);

    enum XsiStatus (*read_mem)(const char *tile, XsiWord32 address, unsigned num_bytes, unsigned char *data);
    enum XsiStatus (*write_mem)(const char *tile, XsiWord32 address, unsigned num_bytes, unsigned char *data);
    
    enum XsiStatus (*read_pswitch_reg)(const char *tile, unsigned reg_num, unsigned *var);
    enum XsiStatus (*write_pswitch_reg)(const char *tile, unsigned reg_num, unsigned var);

    enum XsiStatus (*sample_pin)(const char *package, const char *pin, unsigned *var);
    enum XsiStatus (*drive_pin)(const char *package, const char *pin, unsigned var);
    enum XsiStatus (*is_pin_driving)(const char *package, const char *pin, unsigned *var);

    enum XsiStatus (*sample_port_pins)(const char *tile, const char *port, XsiPortData mask, XsiPortData *var);
    enum XsiStatus (*drive_port_pins)(const char *tile, const char *port, XsiPortData mask, XsiPortData var);
    enum XsiStatus (*is_port_pins_driving)(const char *tile, const char *port, XsiPortData *var);

    enum XsiStatus (*reset)(enum XsiResetType type);

    enum XsiStatus (*save_state)(const char *filename);
    enum XsiStatus (*restore_state)(const char *filename);

    enum XsiStatus (*get_xlink)(void **xlink, const char *target_node_id, unsigned target_link_num);
    enum XsiStatus (*tokens_available)(void *xlink, unsigned int *available);    
    enum XsiStatus (*receive_token)(void *xlink, unsigned char *token, unsigned char *is_ct);
    enum XsiStatus (*spaces_available)(void *xlink, unsigned int *available);    
    enum XsiStatus (*send_token)(void *xlink, unsigned char token, unsigned char is_ct);
};

#endif /* _XsiPlugin_h_ */
