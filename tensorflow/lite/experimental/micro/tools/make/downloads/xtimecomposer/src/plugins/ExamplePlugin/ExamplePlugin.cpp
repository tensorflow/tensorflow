/*
 * Copyright XMOS Limited - 2009
 *
 * An example plugin which connects pairs of pins.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "ExamplePlugin.h"

#define MAX_INSTANCES 256
#define MAX_BYTES 1024
#define CHECK_STATUS if (status != XSI_STATUS_OK) return status

/*
 * Types
 */
struct LoopbackInstance
{
  XsiCallbacks *xsi;
  const char *from_package;
  const char *from_pin;
  const char *to_package;
  const char *to_pin;
};

/*
 * Static data
 */
static size_t s_num_instances = 0;
static LoopbackInstance s_instances[MAX_INSTANCES];

/*
 * Static functions
 */
static void print_usage();
static XsiStatus split_args(const char *args, char *argv[]);

/*
 * Create
 */
XsiStatus plugin_create(void **instance, XsiCallbacks *xsi, const char *arguments)
{
  if (s_num_instances >= MAX_INSTANCES) {
    fprintf(stderr, "ERROR: too many instances of plugin (max %d)\n", MAX_INSTANCES);
    return XSI_STATUS_INVALID_INSTANCE;
  }
  
  // Use the entry in the instances list to identify this instance
  *instance = (void*)s_num_instances;

  char *argv[4];
  XsiStatus status = split_args(arguments, argv);
  if (status != XSI_STATUS_OK) {
    print_usage();
    return status;
  }
  
  // Stores the from pin information
  s_instances[s_num_instances].from_package = argv[0];
  s_instances[s_num_instances].from_pin = argv[1];

  // Stores the to pin information
  s_instances[s_num_instances].to_package = argv[2];
  s_instances[s_num_instances].to_pin = argv[3];

  s_instances[s_num_instances].xsi = xsi;
  s_num_instances++;
  return XSI_STATUS_OK;
}

/*
 * Clock
 */
XsiStatus plugin_clock(void *instance)
{
  size_t instance_num = (size_t)instance;
  if (instance_num >= s_num_instances) {
    return XSI_STATUS_INVALID_INSTANCE;
  }

  XsiStatus status = XSI_STATUS_OK;

  XsiCallbacks *xsi = s_instances[instance_num].xsi;
  const char *from_package = s_instances[instance_num].from_package;
  const char *from_pin     = s_instances[instance_num].from_pin;
  const char *to_package   = s_instances[instance_num].to_package;
  const char *to_pin       = s_instances[instance_num].to_pin;

  unsigned value = 0;
  unsigned int from_driving = 0;

  unsigned int to_driving = 0;

  status = xsi->is_pin_driving(from_package, from_pin, &from_driving);
  CHECK_STATUS;
  status = xsi->is_pin_driving(to_package, to_pin, &to_driving);
  CHECK_STATUS;

  if (from_driving) {
    status = xsi->sample_pin(from_package, from_pin, &value);
    CHECK_STATUS;
    status = xsi->drive_pin(to_package, to_pin, value);
    CHECK_STATUS;

  } else if (to_driving) {
    status = xsi->sample_pin(to_package, to_pin, &value);
    CHECK_STATUS;
    status = xsi->drive_pin(from_package, from_pin, value);
    CHECK_STATUS;
    
  } else {
    // Read both in order remove the drive
    status = xsi->sample_pin(from_package, from_pin, &value);
    CHECK_STATUS;
    status = xsi->sample_pin(to_package, to_pin, &value);
    CHECK_STATUS;
  }
  return status;
}

/*
 * Notify
 */
XsiStatus plugin_notify(void *instance, int type, unsigned arg1, unsigned arg2)
{
  return XSI_STATUS_OK;
}

/*
 * Terminate
 */
XsiStatus plugin_terminate(void *instance)
{
  if ((size_t)instance >= s_num_instances) {
    return XSI_STATUS_INVALID_INSTANCE;
  }
  return XSI_STATUS_OK;
}

/*
 * Usage
 */
static void print_usage()
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  ExamplePlugin.dll/so <from package> <from pin> <to package> <to pin>\n");
}

/*
 * Split args
 */
static XsiStatus split_args(const char *args, char *argv[])
{
  char buf[MAX_BYTES];

  int arg_num = 0;
  while (arg_num < 4) {
    char *buf_ptr = buf;
    
    while (isspace(*args))
      args++;
      
    if (*args == '\0')
      return XSI_STATUS_INVALID_ARGS;

    while (*args != '\0' && !isspace(*args))
      *buf_ptr++ = *args++;

    *buf_ptr = '\0';
    argv[arg_num] = strdup(buf);
    arg_num++;
  }

  while (isspace(*args))
    args++;
  
  if (arg_num != 4 || *args != '\0')
    return XSI_STATUS_INVALID_ARGS;
  else
    return XSI_STATUS_OK;
}
