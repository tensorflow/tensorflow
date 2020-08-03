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

#ifndef __EE_MAIN_H
#define __EE_MAIN_H

#include <stdbool.h>
#include "tensorflow/lite/micro/benchmarks/eembc/monitor/th_api/th_lib.h"
#include "tensorflow/lite/micro/benchmarks/eembc/monitor/th_api/th_libc.h"

#define EE_MONITOR_VERSION "2.1.0"

typedef enum { EE_ARG_CLAIMED, EE_ARG_UNCLAIMED } arg_claimed_t;
typedef enum { EE_STATUS_OK = 0, EE_STATUS_ERROR } ee_status_t;

#define EE_DEVICE_NAME    "dut"

#define EE_CMD_SIZE       80u
#define EE_CMD_DELIMITER  " "
#define EE_CMD_TERMINATOR '%'

#define EE_CMD_NAME       "name"
#define EE_CMD_TIMESTAMP  "timestamp"

#define EE_MSG_READY      "m-ready\r\n"
#define EE_MSG_INIT_DONE  "m-init-done\r\n"
#define EE_MSG_NAME       "m-name-%s-[%s]\r\n"

#define EE_ERR_CMD        "e-[Unknown command: %s]\r\n"

void ee_serial_callback(char);
void ee_serial_command_parser_callback(char *);
void ee_main(void);

// From ../profile/ee_profile.c
arg_claimed_t ee_profile_parse(char *);
void          ee_profile_initialize(void);

#endif // __EE_MAIN_H
