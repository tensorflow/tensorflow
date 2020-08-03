/*
 * Copyright (C) EEMBC(R). All Rights Reserved
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 */

#include "tensorflow/lite/micro/benchmarks/eembc/monitor/ee_main.h"

#if EE_CFG_SELFHOSTED != 1

// Command buffer (incoming commands from host)
static char volatile g_cmd_buf[EE_CMD_SIZE + 1];
static unsigned int volatile  g_cmd_pos = 0u;
/**
 * Since the serial port ISR may be connected before the loop is ready, this
 * flag turns off the parser until the main routine is ready.
 */
static bool g_parser_enabled = false;

/**
 * This function assembles a command string from the UART. It should be called
 * from the UART ISR for each new character received. When the parser sees the
 * termination character, the user-defined th_command_ready() command is called.
 * It is up to the application to then dispatch this command outside the ISR
 * as soon as possible by calling ee_serial_command_parser_callback(), below.
 */
void
ee_serial_callback(char c)
{
    if (c == EE_CMD_TERMINATOR)
    {
        g_cmd_buf[g_cmd_pos] = (char)0;
        th_command_ready(g_cmd_buf);
        g_cmd_pos = 0;
    }
    else
    {
        g_cmd_buf[g_cmd_pos] = c;
        g_cmd_pos = g_cmd_pos >= EE_CMD_SIZE ? EE_CMD_SIZE : g_cmd_pos + 1;
    }
}

/**
 * This is the minimal parser required to test the monitor; profile-specific
 * commands are handled by whatever profile is compiled into the firmware.
 *
 * The most basic commands are:
 *
 * name             Print m-name-NAME, where NAME defines the intent of the f/w
 * timestamp        Generate a signal used for timestamping by the framework
 */
/*@-mustfreefresh*/
/*@-nullpass*/
void
ee_serial_command_parser_callback(char *p_command)
{
    char *tok;

    if (g_parser_enabled != true)
    {
        return;
    }

    tok = th_strtok(p_command, EE_CMD_DELIMITER);

    if (th_strncmp(tok, EE_CMD_NAME, EE_CMD_SIZE) == 0)
    {
        th_printf(EE_MSG_NAME, EE_DEVICE_NAME, TH_VENDOR_NAME_STRING);
    }
    else if (th_strncmp(tok, EE_CMD_TIMESTAMP, EE_CMD_SIZE) == 0)
    {
        th_timestamp();
    }
    else if (ee_profile_parse(tok) == EE_ARG_CLAIMED)
    {
    }
    else
    {
        th_printf(EE_ERR_CMD, tok);
    }

    th_printf(EE_MSG_READY);
}

/**
 * Perform the basic setup.
 */
void
ee_main(void)
{
    th_serialport_initialize();
    th_printf("\r\n");
    th_timestamp_initialize();
    th_monitor_initialize();
    th_printf(EE_MSG_INIT_DONE);
    // Enable the command parser here (the callback is connected)
    g_parser_enabled = true;
    // At this point, the serial monitor should be up and running,
    th_printf(EE_MSG_READY);
}

#endif // EE_CFG_SELFHOSTED
