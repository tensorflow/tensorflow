#ifndef XSCOPE_H_
#define XSCOPE_H_

/**
 * \file xscope.h
 * \brief Xscope interface
 *
 * This file contains functions to access xscope.
 * Example:
\code
#include <platform.h>
#include <xscope.h>
#include <xccompat.h>

void xscope_user_init(void) {
  xscope_register(1, XSCOPE_CONTINUOUS, "Continuous Value 1", XSCOPE_UINT, "Value");
}

int main (void) {
  par {
    on tile[0]: {
      for (int i = 0; i < 100; i++) {
        xscope_int(0, i*i);
      }
    }
  }
  return 0;
}
\endcode

 *
 * xscope_user_init()
 * Constructor for use with xscope event registration, this allows the code on the device to syncronize with the host.
 * This should be declared anywhere in the application code and if present will be called before main().
 */

/** Enum for all types of xscope events */
typedef enum {
  XSCOPE_STARTSTOP=1, /**< Start/Stop - Event gets a start and stop value representing a block of execution */
  XSCOPE_CONTINUOUS, /**<  Continuous - Only gets an event start, single timestamped "ping" */
  XSCOPE_DISCRETE, /**<  Discrete - Event generates a discrete block following on from the previous event */
  XSCOPE_STATEMACHINE, /**<  State Machine - Create a new event state for every new data value */
  XSCOPE_HISTOGRAM,
} xscope_EventType;

/** Enum for all user data types */
typedef enum {
  XSCOPE_NONE=0, /**< No user data */
  XSCOPE_UINT, /**< Unsigned int user data */
  XSCOPE_INT, /**< Signed int user data */
  XSCOPE_FLOAT, /**< Floating point user data */
} xscope_UserDataType;

/** Enum of all I/O redirection modes */
typedef enum {
  XSCOPE_IO_NONE=0, /**< I/O is not redirected */
  XSCOPE_IO_BASIC, /**< Basic I/O redirection */
  XSCOPE_IO_TIMED, /**< Timed I/O redirection */
} xscope_IORedirectionMode;

/**
 * Registers the trace probes with the host system.
 * First parameter is the number of probes that will be registered. Further parameters are in groups of four.
 * -# event type (\link #xscope_EventType \endlink)
 * -# probe name
 * -# user data type (\link #xscope_UserDataType \endlink)
 * -# user data name.
 *
 * Examples:
 * \code
 *    xscope_register(1, XSCOPE_DISCRETE, "A probe", XSCOPE_UINT, "value"); ``
 *    xscope_register(2, XSCOPE_CONTINUOUS, "Probe", XSCOPE_FLOAT, "Level",
 *                       XSCOPE_STATEMACHINE, "State machine", XSCOPE_NONE, "no name");
 * \endcode
 * \param num_probes Number of probes that will be specified.
 */
void xscope_register(int num_probes, ...);

/**
 * Enable the XSCOPE event capture on the local xCORE tile
 */
void xscope_enable();

/**
 * Disable the XSCOPE event capture on the local xCORE tile
 */
void xscope_disable();

/**
 * Configures XScope I/O redirection.
 * \param mode I/O redirection mode.
 */
void xscope_config_io(unsigned int mode);

/**
 * Generate an XSCOPE ping system timestamp event
 */
void xscope_ping();

/**
 * Send a trace event for the specified XSCOPE probe of type char.
 * \param id XSCOPE probe id. 
 * \param data User data value (char).
 * \sa xscope_short xscope_int xscope_long_long xscope_float xscope_double xscope_bytes
 */
void xscope_char(unsigned char id, unsigned char data);
#ifndef XSCOPE_IMPL
#define xscope_char(id, data) do { if ((id) != -1) xscope_char(id, data); } while (0)
#endif
/**
 * Send a trace event for the specified XSCOPE probe of type short.
 * \param id XSCOPE probe id.
 * \param data User data value (short).
 * \sa xscope_char xscope_int xscope_long_long xscope_float xscope_double xscope_bytes
 */
void xscope_short(unsigned char id, unsigned short data);
#ifndef XSCOPE_IMPL
#define xscope_short(id, data) do { if ((id) != -1) xscope_short(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type int.
 * \param id XSCOPE probe id.
 * \param data User data value (int).
 * \sa xscope_char xscope_short xscope_long_long xscope_float xscope_double xscope_bytes
 */
void xscope_int(unsigned char id, unsigned int data);
#ifndef XSCOPE_IMPL
#define xscope_int(id, data) do { if ((id) != -1) xscope_int(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type long long.
 * \param id XSCOPE probe id.
 * \param data User data value (long long).
 * \sa xscope_char xscope_short xscope_int xscope_float xscope_double xscope_bytes
 */
void xscope_longlong(unsigned char id, unsigned long long data);
#ifndef XSCOPE_IMPL
#define xscope_longlong(id, data) do { if ((id) != -1) xscope_longlong(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type float.
 * \param id XSCOPE probe id.
 * \param data User data value (float).
 * \sa xscope_char xscope_short xscope_int xscope_long_long xscope_double xscope_bytes
 */
void xscope_float(unsigned char id, float data);
#ifndef XSCOPE_IMPL
#define xscope_float(id, data) do { if ((id) != -1) xscope_float(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type double.
 * \param id XSCOPE probe id.
 * \param data User data value (double).
 * \sa xscope_char xscope_short xscope_int xscope_long_long xscope_float xscope_bytes
 */
void xscope_double(unsigned char id, double data);
#ifndef XSCOPE_IMPL
#define xscope_double(id, data) do { if ((id) != -1) xscope_double(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe with a byte array.
 * \param id XSCOPE probe id.
 * \param size User data size.
 * \param data User data bytes (char[]).
 * \sa xscope_char xscope_short xscope_int xscope_long_long xscope_float xscope_double
 */
void xscope_bytes(unsigned char id, unsigned int size, const unsigned char data[]);
#ifndef XSCOPE_IMPL
#define xscope_bytes(id, size, data) do { if ((id) != -1) xscope_bytes(id, size, data); } while (0)
#endif

/**
 * Start a trace block for the specified XSCOPE probe.
 * \param id XSCOPE probe id.
 * \sa xscope_stop xscope_start_int xscope_stop_int
 */
void xscope_start(unsigned char id);
#ifndef XSCOPE_IMPL
#define xscope_start(id) do { if ((id) != -1) xscope_start(id); } while (0)
#endif

/**
 * Stop a trace block for the specified XSCOPE probe.
 * \param id XSCOPE probe id.
 * \sa xscope_start xscope_start_int xscope_stop_int
 */
void xscope_stop(unsigned char id);
#ifndef XSCOPE_IMPL
#define xscope_stop(id) do { if ((id) != -1) xscope_stop(id); } while (0)
#endif

/**
 * Start a trace block for the specified XSCOPE probe and capture a value of type int.
 * \param id XSCOPE probe id.
 * \param data User data value (int).
 * \sa xscope_start xscope_stop xscope_stop_int
 */
void xscope_start_int(unsigned char id, unsigned int data);
#ifndef XSCOPE_IMPL
#define xscope_start_int(id, data) do { if ((id) != -1) xscope_start_int(id, data); } while (0)
#endif

/**
 * Stop a trace block for the specified XSCOPE probe and capture a value of type int.
 * \param id XSCOPE probe id.
 * \param data User data value (int).
 * \sa xscope_start xscope_stop xscope_start_int
 */
void xscope_stop_int(unsigned char id, unsigned int data);
#ifndef XSCOPE_IMPL
#define xscope_stop_int(id, data) do { if ((id) != -1) xscope_stop_int(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type char with logical core info.
 * \param id XSCOPE probe id.
 * \param data User data value (char).
 * \sa xscope_core_short xscope_core_int xscope_core_long_long xscope_core_float xscope_core_double xscope_core_bytes
 */
void xscope_core_char(unsigned char id, unsigned char data);
#ifndef XSCOPE_IMPL
#define xscope_core_char(id, data) do { if ((id) != -1) xscope_core_char(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type short with logical core info.
 * \param id XSCOPE probe id.
 * \param data User data value (short).
 * \sa xscope_core_char xscope_core_int xscope_core_long_long xscope_core_float xscope_core_double xscope_core_bytes
 */
void xscope_core_short(unsigned char id, unsigned short data);
#ifndef XSCOPE_IMPL
#define xscope_core_short(id, data) do { if ((id) != -1) xscope_core_short(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type int with logical core info.
 * \param id XSCOPE probe id.
 * \param data User data value (int).
 * \sa xscope_core_char xscope_core_short xscope_core_long_long xscope_core_float xscope_core_double xscope_core_bytes
 */
void xscope_core_int(unsigned char id, unsigned int data);
#ifndef XSCOPE_IMPL
#define xscope_core_int(id, data) do { if ((id) != -1) xscope_core_int(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type long long with logical core info.
 * \param id XSCOPE probe id.
 * \param data User data value (long long).
 * \sa xscope_core_char xscope_core_short xscope_core_int xscope_core_float xscope_core_double xscope_core_bytes
 */
void xscope_core_longlong(unsigned char id, unsigned long long data);
#ifndef XSCOPE_IMPL
#define xscope_core_longlong(id, data) do { if ((id) != -1) xscope_core_longlong(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type float with logical core info.
 * \param id XSCOPE probe id.
 * \param data User data value (float).
 * \sa xscope_core_char xscope_core_short xscope_core_int xscope_core_long_long xscope_core_double xscope_core_bytes
 */
void xscope_core_float(unsigned char id, float data);
#ifndef XSCOPE_IMPL
#define xscope_core_float(id, data) do { if ((id) != -1) xscope_core_float(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe of type double with logical core info.
 * \param id XSCOPE probe id.
 * \param data User data value (double).
 * \sa xscope_core_char xscope_core_short xscope_core_int xscope_core_long_long xscope_core_float xscope_core_bytes
 */
void xscope_core_double(unsigned char id, double data);
#ifndef XSCOPE_IMPL
#define xscope_core_double(id, data) do { if ((id) != -1) xscope_core_double(id, data); } while (0)
#endif

/**
 * Send a trace event for the specified XSCOPE probe with a byte array with logical core info.
 * \param id XSCOPE probe id.
 * \param size User data size.
 * \param data User data bytes (char[]).
 * \sa xscope_core_char xscope_core_short xscope_core_int xscope_core_long_long xscope_core_float xscope_core_double
 */
void xscope_core_bytes(unsigned char id, unsigned int size, const unsigned char data[]);
#ifndef XSCOPE_IMPL
#define xscope_core_bytes(id, size, data) do { if ((id) != -1) xscope_core_bytes(id, size, data); } while (0)
#endif

/**
 * Start a trace block for the specified XSCOPE probe with logical core info.
 * \param id XSCOPE probe id.
 * \sa xscope_core_stop xscope_core_start_int xscope_core_stop_int
 */
void xscope_core_start(unsigned char id);
#ifndef XSCOPE_IMPL
#define xscope_core_start(id) do { if ((id) != -1) xscope_core_start(id); } while (0)
#endif

/**
 * Stop a trace block for the specified XSCOPE probe with logical core info.
 * \param id XSCOPE probe id.
 * \sa xscope_core_start xscope_core_start_int xscope_core_stop_int
 */
void xscope_core_stop(unsigned char id);
#ifndef XSCOPE_IMPL
#define xscope_core_stop(id) do { if ((id) != -1) xscope_core_stop(id); } while (0)
#endif


/**
 * Start a trace block for the specified XSCOPE probe with logical core info and capture a value of type int
 * \param id XSCOPE probe id.
 * \param data User data value (int).
 * \sa xscope_core_start xscope_core_stop xscope_core_stop_int
 */
void xscope_core_start_int(unsigned char id, unsigned int data);
#ifndef XSCOPE_IMPL
#define xscope_core_start_int(id, data) do { if ((id) != -1) xscope_core_start_int(id, data); } while (0)
#endif


/**
 * Stop a trace block for the specified XSCOPE probe with logical core info and capture a value of type int
 * \param id XSCOPE probe id.
 * \param data User data value (int).
 * \sa xscope_core_start xscope_core_stop xscope_core_start_int
 */
void xscope_core_stop_int(unsigned char id, unsigned int data);
#ifndef XSCOPE_IMPL
#define xscope_core_stop_int(id, data) do { if ((id) != -1) xscope_core_stop_int(id, data); } while (0)
#endif

/**
 * Put XSCOPE into a lossless mode where timing is no longer guaranteed.
 * \sa xscope_mode_lossy
 */
void xscope_mode_lossless();

/**
 * Put XSCOPE into a lossy mode where timing is not impacted, but data is lossy.
 * This is the default XSCOPE mode.
 * \sa xscope_mode_lossless
 */
void xscope_mode_lossy();

#ifdef __XC__
#pragma select handler
void xscope_data_from_host(chanend c, char buf[256], int &n);
#else
void xscope_data_from_host(unsigned int c, char buf[256], int *n);
#endif

#ifdef __XC__
void xscope_connect_data_from_host(chanend from_host);
#else
void xscope_connect_data_from_host(unsigned int from_host);
#endif

/* Probe enabled macro */
#define XSCOPE_PROBE_ENABLED(x) ((x) != -1)

/* Backwards compatibility */

#ifdef __XC__
#define xscope_probe(id) _Pragma("warning \"xscope_probe is deprecated, use xscope_char instead\"") xscope_char(id, 0)
#else
__attribute__((deprecated)) static inline void xscope_probe(unsigned char id)
{
  xscope_char(id, 0);
}
#endif

#ifdef __XC__
#define xscope_probe_data(id, data) _Pragma("warning \"xscope_probe_data is deprecated, use xscope_int instead\"") xscope_int(id, data)
#else
__attribute__((deprecated)) static inline void xscope_probe_data(unsigned char id, unsigned int data)
{
  xscope_int(id, data);
}
#endif

#ifdef __XC__
#define xscope_probe_data_pred(id, data) _Pragma("warning \"xscope_probe_data_pred is deprecated, use xscope_int instead\"") xscope_int(id, data)
#else
__attribute__((deprecated)) static inline void xscope_probe_data_pred(unsigned char id, unsigned int data)
{
  xscope_int(id, data);
}
#endif

#ifdef __XC__
#define xscope_probe_cpu(id) _Pragma("warning \"xscope_probe_cpu is deprecated, use xscope_core_char instead\"") xscope_core_char(id, 0)
#else
__attribute__((deprecated)) static inline void xscope_probe_cpu(unsigned char id)
{
  xscope_core_char(id, 0);
}
#endif

#ifdef __XC__
#define xscope_probe_cpu_data(id, data) _Pragma("warning \"xscope_probe_cpu_data is deprecated, use xscope_core_int instead\"") xscope_core_int(id, data)
#else
__attribute__((deprecated)) static inline void xscope_probe_cpu_data(unsigned char id, unsigned int data)
{
  xscope_core_int(id, data);
}
#endif

/* This section includes the autogenerated probe definitions
   from .xscope  files */
#ifdef _XSCOPE_PROBES_INCLUDE_FILE
#include _XSCOPE_PROBES_INCLUDE_FILE
#endif

#endif /* XSCOPE_H_ */
