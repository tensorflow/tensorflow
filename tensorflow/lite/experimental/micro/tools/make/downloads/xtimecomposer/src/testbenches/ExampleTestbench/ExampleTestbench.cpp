/*
 * Copyright XMOS Limited - 2009
 * 
 * An example testbench which instantiates one simulator and connects pairs of pins.
 *
 */

#include <string>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "xsidevice.h"

#define MAX_INSTANCES 256

using namespace std;

struct ConnectionInstance
{
  const char *from_package;
  const char *from_pin;
  const char *to_package;
  const char *to_pin;
};

size_t  g_num_connections = 0;
ConnectionInstance g_connections[MAX_INSTANCES];

void *g_device = 0;
string g_sim_exe_name;

void print_usage()
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s <options> SIM_ARGS\n", g_sim_exe_name.c_str());
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  --help - print this message\n");
  fprintf(stderr, "  --connect <from pkg> <from pin> <to pkg> <to pin> - connect a pair of pads together\n");
  fprintf(stderr, "  SIM_ARGS - the remaining arguments will be passed to the xsim created\n");
  exit(1);
}

unsigned str_to_uint(const char *val_str, const char *description)
{
  char *end_ptr = 0;
  unsigned value = strtoul(val_str, &end_ptr, 0);

  if (strcmp(end_ptr, "") != 0) {
    fprintf(stderr, "ERROR: could not parse %s\n", description);
    print_usage();
  }
  
  return (unsigned)value;
}

int parse_connect(int argc, char **argv, int index)
{
  if ((index + 4) >= argc) {
    fprintf(stderr, "ERROR: missing arguments for --connect\n");
    print_usage();
  }

  g_connections[g_num_connections].from_package = argv[index + 1];
  g_connections[g_num_connections].from_pin     = argv[index + 2];
  g_connections[g_num_connections].to_package   = argv[index + 3];
  g_connections[g_num_connections].to_pin       = argv[index + 4];
  g_num_connections++;
  return index + 5;
}

void parse_args(int argc, char **argv)
{
  g_sim_exe_name = argv[0];
  unsigned int char_index = g_sim_exe_name.find_last_of("\\/");
  if (char_index > 0)
    g_sim_exe_name.erase(0, char_index + 1);

  bool done = false;
  int index = 1;
  while (!done && (index < argc)) {
    if (strcmp(argv[index], "--help") == 0) {
      print_usage();

    } else if (strcmp(argv[index], "--connect") == 0) {
      index = parse_connect(argc, argv, index);

    } else {
      done = true;
    }
  }

  string args;
  while (index < argc) {
    args += " ";
    args += argv[index];
    index++;
  }

  XsiStatus status = xsi_create(&g_device, args.c_str());
  if (status != XSI_STATUS_OK) {
    fprintf(stderr, "ERROR: failed to create device with args '%s'\n", args.c_str());
    print_usage();
  }
}

bool is_pin_driving(const char *package, const char *pin)
{
  unsigned int is_driving = 0;
  XsiStatus status = xsi_is_pin_driving(g_device, package, pin, &is_driving);
  if (status != XSI_STATUS_OK) {
    fprintf(stderr, "ERROR: failed to check for driving pin %s on package %s\n", pin, package);
    exit(1);
  }
  return is_driving ? true : false;
}

unsigned sample_pin(const char *package, const char *pin)
{
  unsigned value = 0;
  XsiStatus status = xsi_sample_pin(g_device, package, pin, &value);
  if (status != XSI_STATUS_OK) {
    fprintf(stderr, "ERROR: failed to sample pin %s on package %s\n", pin, package);
    exit(1);
  }
  return value;
}

void drive_pin(const char *package, const char *pin, unsigned value)
{
  XsiStatus status = xsi_drive_pin(g_device, package, pin, value);
  if (status != XSI_STATUS_OK) {
    fprintf(stderr, "ERROR: failed to drive pin %s on package %s\n", pin, package);
    exit(1);
  }
}

void manage_connections()
{
  for (size_t connection_num = 0; connection_num < g_num_connections; connection_num++) {
    const char *from_package = g_connections[connection_num].from_package;
    const char *from_pin     = g_connections[connection_num].from_pin;
    const char *to_package   = g_connections[connection_num].to_package;
    const char *to_pin       = g_connections[connection_num].to_pin;
    unsigned value = 0;
  
    int from_driving = is_pin_driving(from_package, from_pin);
    int to_driving = is_pin_driving(to_package, to_pin);
  
    if (from_driving) {
      value = sample_pin(from_package, from_pin);
      drive_pin(to_package, to_pin, value);

    } else if (to_driving) {
      value = sample_pin(to_package, to_pin);
      drive_pin(from_package, from_pin, value);
      
    } else {
      // Read both in order to stop the testbench driving
      sample_pin(from_package, from_pin);
      sample_pin(to_package, to_pin);
    }
  }
}

XsiStatus sim_clock()
{
  XsiStatus status = xsi_clock(g_device);
  if ((status != XSI_STATUS_OK) && (status != XSI_STATUS_DONE)) {
    fprintf(stderr, "ERROR: failed to clock device (status %d)\n", status);
    exit(1);
  }
  return status;
}

int main(int argc, char **argv)
{
  parse_args(argc, argv);

  bool done = false;
  while (!done) {
    manage_connections();

    XsiStatus status = sim_clock();
    if (status == XSI_STATUS_DONE)
      done = true;
  }

  XsiStatus status = xsi_terminate(g_device);
  if (status != XSI_STATUS_OK) {
    fprintf(stderr, "ERROR: failed to terminate device\n");
    exit(1);
  }
  return 0;
}
