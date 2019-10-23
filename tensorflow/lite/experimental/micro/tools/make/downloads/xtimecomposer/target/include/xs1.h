/*
 * Copyright (C) XMOS Limited 2008-2019
 * 
 * The copyrights, all other intellectual and industrial property rights are
 * retained by XMOS and/or its licensors.
 *
 * The code is provided "AS IS" without a warranty of any kind. XMOS and its
 * licensors disclaim all other warranties, express or implied, including any
 * implied warranty of merchantability/satisfactory quality, fitness for a
 * particular purpose, or non-infringement except to the extent that these
 * disclaimers are held to be legally invalid under applicable law.
 *
 * Version: Community_15.0.0_eng
 */

#ifndef _xs1_h_
#define _xs1_h_

#if !defined(__XS3A__) && !defined(__XS2A__) && !defined(__XS1B__)
  #error "Unknown architecture"
#endif

#include "timer.h"

/** 
 * \file xs1.h
 * \brief XS1 Hardware routines
 *
 * This file contains functions to access XS1 hardware.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <xs1_g4000b-512.h>

#include <xs1_user.h>
#include <xs1_kernel.h>
#include <xs1_registers.h>
#include <xs1_clock.h>


#ifndef __ASSEMBLER__

#ifdef __XC__

/** 
 * Configures a buffered port to be a clocked input port in handshake mode.
 * If the ready-in or ready-out ports are not 1-bit ports, an exception is raised.
 * The ready-out port is asserted on the falling edge of the clock when the
 * port's buffer is not full. The port samples its pins on its sampling edge
 * when both the ready-in and ready-out ports are asserted.
 *
 * By default the port's sampling edge is the rising edge of clock. This can be
 * changed by the function set_port_sample_delay().
 * \param p The buffered port to configure.
 * \param readyin A 1-bit port to use for the ready-in signal.
 * \param readyout A 1-bit port to use for the ready-out signal.
 * \param clk The clock used to configure the port.
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 * \sa set_port_no_sample_delay
 * \sa set_port_sample_delay
 */
void configure_in_port_handshake(void port p, in port readyin,
                                 out port readyout, clock clk);

/**
 * Configures a buffered port to be a clocked output port in handshake mode. 
 * If the ready-in or ready-out ports are not 1-bit ports an exception is
 * raised. The port drives the initial value on its pins until an
 * output statement changes the value driven. The ready-in port is read on the
 * sampling edge of the buffered port. Outputs are driven on the next falling
 * edge of the clock where the previous value read from the ready-in port was
 * high.  On the falling edge of the clock the ready-out port is driven high
 * if data is output on that edge, otherwise it is driven low.
 * By default the port's sampling edge is the rising edge of clock. This can be
 * changed by the function set_port_sample_delay().
 * \param p The buffered port to configure.
 * \param readyin A 1-bit port to use for the ready-in signal.
 * \param readyout A 1-bit port to use for the ready-out signal.
 * \param clk The clock used to configure the port.
 * \param initial The initial value to output on the port.
 * \sa configure_in_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 * \sa set_port_no_sample_delay
 * \sa set_port_sample_delay
 */
void configure_out_port_handshake(void port p, in port readyin,
                                 out port readyout, clock clk,
                                 unsigned initial);

/**
 * Configures a buffered port to be a clocked input port in strobed master mode.
 * If the ready-out port is not a 1-bit port, an exception is raised.
 * The ready-out port is asserted on the falling edge of the clock when the
 * port's buffer is not full. The port samples its pins on its sampling edge
 * after the ready-out port is asserted.
 *
 * By default the port's sampling edge is the rising edge of clock. This can be
 * changed by the function set_port_sample_delay().
 * \param p The buffered port to configure.
 * \param readyout A 1-bit port to use for the ready-out signal.
 * \param clk The clock used to configure the port.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 * \sa set_port_no_sample_delay
 * \sa set_port_sample_delay
 */
void configure_in_port_strobed_master(void port p, out port readyout,
                                      const clock clk);

/**
 * Configures a buffered port to be a clocked output port in strobed master mode.
 * If the ready-out port is not a 1-bit port, an exception is raised.
 * The port drives the initial value on its pins until an
 * output statement changes the value driven. Outputs are driven on the next
 * falling edge of the clock. On the falling edge of the clock the ready-out
 * port is driven high if data is output on that edge, otherwise it is driven
 * low.
 * \param p The buffered port to configure.
 * \param readyout A 1-bit port to use for the ready-out signal.
 * \param clk The clock used to configure the port.
 * \param initial The initial value to output on the port.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 */
void configure_out_port_strobed_master(void port p, out port readyout,
                                      const clock clk, unsigned initial);

/**
 * Configures a buffered port to be a clocked input port in strobed slave mode.
 * If the ready-in port is not a 1-bit port, an exception is raised.
 * The port samples its pins on its sampling edge when the ready-in signal is
 * high. By default the port's sampling edge is the rising edge of clock. This
 * can be changed by the function set_port_sample_delay().
 * \param p The buffered port to configure.
 * \param readyin A 1-bit port to use for the ready-in signal.
 * \param clk The clock used to configure the port.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 * \sa set_port_no_sample_delay
 * \sa set_port_sample_delay
 */
void configure_in_port_strobed_slave(void port p, in port readyin, clock clk);

/** 
 * Configures a buffered port to be a clocked output port in strobed slave mode.
 * If the ready-in port is not a 1-bit port, an exception is raised.
 * The port drives the initial value on its pins until an
 * output statement changes the value driven. The ready-in port is read on the
 * buffered port's sampling edge. Outputs are driven on the next falling edge
 * of the clock where the previous value read from the ready-in port is high.
 * By default the port's sampling edge is the rising edge of clock. This
 * can be changed by the function set_port_sample_delay().
 * \param p The buffered port to configure.
 * \param readyin A 1-bit port to use for the ready-in signal.
 * \param clk The clock used to configure the port.
 * \param initial The initial value to output on the port.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 * \sa set_port_no_sample_delay
 * \sa set_port_sample_delay
 */
void configure_out_port_strobed_slave(void port p, in port readyin, clock clk,
                                      unsigned initial);

/**
 * Configures a port to be a clocked input port with no ready signals. This is the
 * default mode of a port. The port samples its pins on its sampling edge.
 * If the port is unbuffered, its direction can be changed by performing an
 * output. This change occurs on the next falling edge of the clock.
 * Afterwards, the port behaves as an output port with no ready signals.
 * 
 * By default the port's sampling edge is the rising edge of the clock. This
 * can be changed by the function set_port_sample_delay().
 * \param p The port to configure, which may be buffered or unbuffered.
 * \param clk The clock used to configure the port.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_out_port
 * \sa set_port_no_sample_delay
 * \sa set_port_sample_delay
 */
void configure_in_port(void port p, const clock clk);

/**
 * Alias for configure_in_port().
 * \sa configure_in_port
 */
void configure_in_port_no_ready(void port p, const clock clk);
#define configure_in_port_no_ready(p, clk) configure_in_port(p, clk)

/**
 * Configures a port to be a clocked output port with no ready signals. The port
 * drives the initial value on its pins until an input or output statement
 * changes the value driven. Outputs are driven on the next falling edge of the
 * clock and every port-width bits of data are held for one clock cycle. If the
 * port is unbuffered, the direction of the port can be changed by
 * performing an input. This change occurs on the falling edge of
 * the clock after any pending outputs have been held for one clock period.
 * Afterwards, the port behaves as an input port with no ready signals.
 * \param p The port to configure, which may be buffered or unbuffered.
 * \param clk The clock used to configure the port.
 * \param initial The initial value to output on the port.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 */
void configure_out_port(void port p, const clock clk, unsigned initial);

/**
 * Alias for configure_out_port().
 * \sa configure_out_port
 */
void configure_out_port_no_ready(void port p, const clock clk, unsigned initial);
#define configure_out_port_no_ready(p, clk, initial) configure_out_port(p, clk, initial)

/** Configures a 1-bit port to output a clock signal. If the port is
 *  not a 1-bit port, an exception is raised. Performing
 *  inputs or outputs on the port once it has been configured in this
 *  mode results in undefined behaviour.
 *  \param p The 1-bit port to configure.
 *  \param clk The clock to output.
 */
void configure_port_clock_output(void port p, const clock clk);

/** 
 * Activates a port. The buffer used by the port is cleared.
 * \param p The port to activate.
 * \sa clearbuf
 * \sa stop_port
 * 
 */
void start_port(void port p);

/**
 * Deactivates a port. The port is reset to being a no ready port.
 * \param p The port to deactivate.
 * \sa start_port
 */
void stop_port(void port p);

/** Configures a clock to use a 1-bit port as its source. This allows
 *  I/O operations on ports to be synchronised to an
 *  external clock signal.
 *  If the port is not a 1-bit port, an exception is raised.
 *  \param clk The clock to configure.
 *  \param p The 1-bit port to use as the clock source.
 *  \sa configure_clock_ref
 *  \sa configure_clock_xcore
 *  \sa configure_clock_rate
 *  \sa configure_clock_rate_at_least
 *  \sa configure_clock_rate_at_most
 */
void configure_clock_src(clock clk, void port p);

#if defined(__XS2A__) || defined (__XS3A__)
/**
 * Configures a clock to use a 1-bit port as its source with a divide. If divide
 * is set to zero the 1-bit port provides the clock signal for the clock block
 * directly. If divide is non zero the clock signal provided by the 1-bit port
 * is divided by 2 * \a divide. This function is only available on XS2 devices.
 *  If the port is not a 1-bit port, an exception is raised.
 *  \param clk The clock to configure.
 *  \param p The 1-bit port to use as the clock source.
 *  \sa configure_clock_ref
 *  \sa configure_clock_xcore
 *  \sa configure_clock_rate
 *  \sa configure_clock_rate_at_least
 *  \sa configure_clock_rate_at_most
 */
void configure_clock_src_divide(clock clk, void port p, unsigned char d);
#endif

/** 
 * Configures a clock to use the reference clock as it source.
 * If the divide is set to zero the reference clock frequency is used,
 * otherwise the reference clock frequency divided by 2 * \a divide is used.
 * By default the reference clock is configured to run at 100 MHz.
 * \param clk The clock to configure.
 * \param divide The clock divide.
 * \sa configure_clock_src
 * \sa configure_clock_xcore
 * \sa configure_clock_rate
 * \sa configure_clock_rate_at_least
 * \sa configure_clock_rate_at_most
 */
void configure_clock_ref(clock clk, unsigned char divide);

/** 
 * Configures a clock to use the xCORE tile clock as it source.
 * The xCORE tile clock frequency divided by 2 * \a divide is used.
 * An exception is raised if the divide is zero.
 * \param clk The clock to configure.
 * \param divide The clock divide.
 * \sa configure_clock_ref
 * \sa configure_clock_src
 * \sa configure_clock_rate
 * \sa configure_clock_rate_at_least
 * \sa configure_clock_rate_at_most
 */
void configure_clock_xcore(clock clk, unsigned char divide);

/**
 * Configures a clock to run at a rate of (\a a/\a b) MHz. If the specified rate
 * is not supported by the hardware, an exception is raised.
 * The hardware supports rates of \e ref MHz and rates of the form
 * (\e ref/\e 2n) MHz or (\e tileclk/\e 2n) MHz where \e ref is the reference
 * clock frequency, \e tileclk is the xCORE tile frequency and \e n is a number
 * in the range 1 to 255 inclusive.
 * \param clk The clock to configure.
 * \param a The dividend of the desired rate.
 * \param b The divisor of the desired rate.
 *  \sa configure_clock_src
 *  \sa configure_clock_ref
 *  \sa configure_clock_xcore
 *  \sa configure_clock_rate_at_least
 *  \sa configure_clock_rate_at_most
 */
void configure_clock_rate(clock clk, unsigned a, unsigned b);

/** 
 * Configures a clock to run the slowest rate supported by the hardware that
 * is equal to or exceeds (\a a/\a b) MHz. An exception is raised if no rate satisfies
 * this criterion.
 * \param clk The clock to configure.
 * \param a The dividend of the desired rate.
 * \param b The divisor of the desired rate.
 *  \sa configure_clock_src
 *  \sa configure_clock_ref
 *  \sa configure_clock_xcore
 *  \sa configure_clock_rate
 *  \sa configure_clock_rate_at_most
 */
void configure_clock_rate_at_least(clock clk, unsigned a, unsigned b);

/**
 * Configures a clock to run at the fastest non-zero rate supported by the
 * hardware that is less than or equal to (\a a/\a b) MHz. An exception is
 * raised if no rate satisfies this criterion.
 * \param clk The clock to configure.
 * \param a The dividend of the desired rate.
 * \param b The divisor of the desired rate.
 *  \sa configure_clock_src
 *  \sa configure_clock_ref
 *  \sa configure_clock_xcore
 *  \sa configure_clock_rate
 *  \sa configure_clock_rate_at_least
 */
void configure_clock_rate_at_most(clock clk, unsigned a, unsigned b);

/** 
 * Sets the source for a clock to a 1-bit port.
 * This corresponds with using the SETCLK instruction on a clock.
 * If the port is not a 1-bit port, an exception is raised.
 * In addition if the clock was previously configured with a non-zero divide
 * then an exception is raised. Usually the use of configure_clock_src() which
 * does not suffer from this problem is recommended.
 * \param clk The clock to configure.
 * \param p The 1-bit port to use as the clock source.
 * \sa configure_clock_src
 */
void set_clock_src(clock clk, void port p);
#define set_clock_src(clk, p)                __builtin_set_clk_src(clk, p)

/** 
 * Sets the source for a clock to the reference clock. This corresponds
 * with the using SETCLK instruction on a clock. The clock divide is left
 * unchanged.
 * \param clk The clock to configure.
 * \sa configure_clock_rate
 * \sa configure_clock_rate_at_least
 * \sa configure_clock_rate_at_most
 * \sa configure_clock_ref
 */
void set_clock_ref(clock clk);
#define set_clock_ref(clk)                   __builtin_set_clk_ref(clk)

/** 
 * Sets the source for a clock to the xCORE tile clock. This corresponds
 * with the using SETCLK instruction on a clock. The clock divide is left
 * unchanged.
 * \param clk The clock to configure.
 * \sa configure_clock_rate
 * \sa configure_clock_rate_at_least
 * \sa configure_clock_rate_at_most
 * \sa configure_clock_xcore
 */
void set_clock_xcore(clock clk);
#define set_clock_xcore(clk)                   __builtin_set_clk_xcore(clk)

/** 
 * Sets the divide for a clock. This corresponds with the SETD instruction.
 * On XS1 devices an exception is raised if the clock source is not the
 * reference clock or the xCORE tile clock. If the divide is set to zero the
 * source frequency is left unchanged, unless the clock source is the xCORE
 * tile clock, in which case the behaviour is undefined. If the divide is
 * non-zero, the source frequency is divided by 2 * \a divide.
 * \param clk The clock to configure.
 * \param div The divide to use.
 * \sa configure_clock_rate
 * \sa configure_clock_rate_at_least
 * \sa configure_clock_rate_at_most
 * \sa configure_clock_ref
 * \sa configure_clock_xcore
 */
void set_clock_div(clock clk, unsigned char div);
#define set_clock_div(clk, div)              __builtin_set_clk_div(clk, div)

/** Sets the delay for the rising edge of a clock. Each rising edge of the
 *  clock by \a n processor-clock cycles before it is
 *  seen by any port connected to the clock. The default rising edge delay
 *  is 0 and the delay must be set to values in the range 0 to 512 inclusive.
 *  If the clock edge is delayed by more than the clock period then no
 *  rising clock edges are seen by the ports connected to the clock.
 *  \param clk The clock to configure.
 *  \param n The number of processor-clock cycles by which to delay the rising
 *           edge of the clock.
 *  \sa set_clock_fall_delay
 *  \sa set_pad_delay
 */
void set_clock_rise_delay(clock clk, unsigned n);
#define set_clock_rise_delay(clk, n)       __builtin_set_clock_rise_delay (clk, n)

/** Sets the delay for the falling edge of a clock. Each falling edge of
 *  the clock is delayed by \a n processor-clock cycles before it is
 *  seen by any port connected to the clock. The default falling edge delay
 *  is 0. The delay can be set to values in the range 0 to 512 inclusive. If
 *  the clock edge is delayed by more than the clock period then no
 *  falling clock edges are seen by the ports connected to the clock.
 *  \param clk The clock to configure.
 *  \param n The number of processor-clock cycles by which to delay the falling
 *           edge of the clock.
 *  \sa set_clock_rise_delay
 *  \sa set_pad_delay
 */
void set_clock_fall_delay(clock clk, unsigned n);
#define set_clock_fall_delay(clk, n)       __builtin_set_clock_fall_delay (clk, n)

/** 
 * Attaches a clock to a port. This corresponds to using the SETCLK instruction
 * on a port. The edges of the clock are used to sample and output data.
 * Usually the use of the configure_*_port_* functions is
 * preferred since they ensure that all the port configuration changes required
 * for the desired mode are performed in the correct order.
 * \param p The port to configure.
 * \param clk The clock to attach.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 * \sa configure_in_port
 * \sa configure_out_port
 */
void set_port_clock(void port p, const clock clk);
#define set_port_clock(p, clk)               __builtin_set_port_clk(p, clk)

/** 
 * Sets a 1-bit port as the ready-out for another port.  This corresponds with
 * using the SETRDY instruction on a port. If the ready-out port is not a 1-bit
 * port then an exception is raised. The ready-out port is used to indicate that
 * the port is ready to transfer data.
 * Usually the use of the configure_*_port_* functions is
 * preferred since they ensure that all the port configuration changes required
 * for the desired mode are performed in the correct order.
 * \param p The port to configure.
 * \param ready The 1-bit port to use for the ready-out signal.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 */
void set_port_ready_src(void port ready, void port p);
#define set_port_ready_src(ready, p)         __builtin_set_ready_src(ready, p)

/** 
 * Sets a clock to use a 1-bit port for the ready-in signal. This corresponds
 * with using the SETRDY instruction on a clock. If the port is not a 1-bit
 * port then an exception is raised. The ready-in port controls when data is
 * sampled from the pins.
 * Usually the use of the configure_*_port_* functions is
 * preferred since they ensure that all the port configuration changes required
 * for the desired mode are performed in the correct order.
 * \param clk The clock to configure.
 * \param ready The 1-bit port to use for the ready-in signal.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 */
void set_clock_ready_src(clock clk, void port ready);
#define set_clock_ready_src(clk, portReady)  __builtin_set_clock_ready_src(clk, portReady)

/**
 * Turns on a clock. The clock state is initialised to the default state for a
 * clock. If the clock is already turned on then its state is reset to 
 * its default state.
 * \param clk The clock to turn on.
 * \sa set_clock_off
 */
void set_clock_on(clock clk);
#define set_clock_on(clk)                    __builtin_set_clock_on(clk)

/**
 * Turns off a clock. No action is performed if the clock is already turned
 * off. Any attempt to use the clock while it is turned off will result in an
 * exception being raised.
 * \param clk The clock to turn off.
 * \sa set_clock_on
 */
void set_clock_off(clock clk);
#define set_clock_off(clk)                   __builtin_set_clock_off(clk)

/**
 * Puts a clock into a running state. A clock generates edges only after
 * it has been put into this state. The port counters of all ports attached to 
 * the clock are reset to 0.
 * \param clk The clock to put into a running state.
 * \sa stop_clock
 */
void start_clock(clock clk);
#define start_clock(clk)                     __builtin_start_clock(clk)

/** Waits until a clock is low and then puts the clock into a stopped state.
 *  In a stopped state a clock does not generate edges.
 *  \param clk The clock to put into a stopped state.
 *  \sa start_clock
 */
void stop_clock(clock clk);
#define stop_clock(clk)                      __builtin_stop_clock(clk)

/**
 * Turns on a port. The port state is initialised to the default state for a
 * port of its type. If the port is already turned on its state is reset to
 * its default state.
 * \param p The port to turn on.
 * \sa set_port_use_off
 */
void set_port_use_on(void port p);
#define set_port_use_on(p)                __builtin_set_port_use (p, XS1_SETC_INUSE_ON)

/**
 * Turns off a port. No action is performed if the port is already turned off.
 * Any attempt to use the port while off will result in an exception
 * being raised.
 * \param p The port to turn off.
 * \sa set_port_use_on
 */
void set_port_use_off(void port p);
#define set_port_use_off(p)               __builtin_set_port_use (p, XS1_SETC_INUSE_OFF)

/**
 * Configures a port to be a data port. This is the default state of a port.
 * Output operations on the port are use to control its output signal.
 * \param p The port to configure.
 * \sa set_port_mode_clock
 * \sa set_port_mode_ready
 */
void set_port_mode_data(void port p);
#define set_port_mode_data(p)             __builtin_set_port_type(p, XS1_SETC_PORT_DATAPORT)

/**
 * Configures a 1-bit port to be a clock output port. The port will output the
 * clock connected to it. If the port is not a 1-bit port, an exception is
 * raised. The function set_port_mode_data() can be used to set the port back to
 * its default state.
 * \param p The port to configure.
 * \sa set_port_mode_data
 * \sa set_port_mode_ready
 */
void set_port_mode_clock(void port p);
#define set_port_mode_clock(p)            __builtin_set_port_type(p, XS1_SETC_PORT_CLOCKPORT)

/**
 * Configures a 1-bit port to be a ready signal output port. The port will
 * output the ready-out of a port connected with set_port_ready_src().
 * If the port is not a 1-bit port, an exception is raised. The
 * function set_port_mode_data() can be used to set the port back to
 * its default state.
 * Usually the use of the configure_*_port_* functions is
 * prefered since they ensure that all the port configuration changes required
 * for the desired mode are performed in the correct order.
 * \param p The port to configure.
 * \sa set_port_mode_data
 * \sa set_port_mode_clock
 * \sa set_port_ready_src
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 */
void set_port_mode_ready(void port p);
#define set_port_mode_ready(p)            __builtin_set_port_type(p, XS1_SETC_PORT_READYPORT)

/**
 * Configures a port in drive mode. Values output to the port are driven on the
 * pins. This is the default drive state of a port. Calling set_port_drive() has
 * the side effect disabling the port's pull up or pull down resistor.
 * \param p The port to configure.
 * \sa set_port_drive_low
 * \sa set_port_drive_high
 * \sa set_port_pull_none
 */
void set_port_drive(void port p);
#define set_port_drive(p)                 __builtin_set_port_drv (p, XS1_SETC_DRIVE_DRIVE)

/**
 * Configures a port in drive low mode. When a 0 is output to a pin it is
 * driven low and when 1 is output no value is driven. For XS1 devices, if
 * the port is not a 1-bit port, the result of an output to the port is undefined.
 * On XS2 and XS1-G devices calling set_port_drive_low() has the side effect of
 * enabling the port's internal pull-up resistor. On XS1-L devices calling
 * set_port_drive_low() has the side effect of enabling the port's internal
 * pull-down resistor.
 * \param p The port to configure.
 * \sa set_port_drive
 * \sa set_port_drive_high
 * \sa set_port_pull_up
 * \sa set_port_pull_down
 */
void set_port_drive_low(void port p);
#define set_port_drive_low(p)             __builtin_set_port_drv (p, XS1_SETC_DRIVE_PULL_UP)

/**
 * Configures a port in drive high mode. When a 1 is output to a pin it is
 * driven high and when 0 is output no value is driven. On XS2 devices
 * calling set_port_drive_high() has the side effect of enabling the port's
 * internal pull-down resistor. The function is not avaiable on XS1 devices.
 * \param p The port to configure.
 * \sa set_port_drive
 * \sa set_port_drive_low
 * \sa set_port_pull_up
 * \sa set_port_pull_down
 */
void set_port_drive_high(void port p);
#ifndef __XS1B__
# define set_port_drive_high(p)           __builtin_set_port_drv (p, XS1_SETC_DRIVE_PULL_DOWN)
#endif

/**
 * Enables a port's internal pull-up resistor. When nothing is driving a pin the
 * pull-up resistor ensures that the value sampled by the port is 1. The pull-up
 * is not strong enough to guarantee a defined external value. On XS2 and XS1-G
 * devices calling set_port_pull_up() has the side effect of configuring the
 * port in drive low mode. On XS1-L devices no pull-up resistors are available
 * and an exception will be raised if set_port_pull_up() is called.
 * \param p The port to configure.
 * \sa set_port_pull_down
 * \sa set_port_pull_none
 * \sa set_port_drive_low
 */
void set_port_pull_up(void port p);

/**
 * Enables a port's internal pull-down resistor. When nothing is driving a pin
 * the pull-down resistor ensures that the value sampled by the port is 0. The
 * pull-down is not strong enough to guarantee a defined external value. On XS2
 * devices calling set_port_pull_down() has the side effect of configuring the
 * port in drive high mode. On XS1-G devices no pull-down resistors are available
 * and an exception will be raised if set_port_pull_down() is called. On XS1-L
 * devices calling set_port_pull_down() has the side effect of configuring the
 * port in drive low mode.
 * \param p The port to configure.
 * \sa set_port_pull_up
 * \sa set_port_pull_none
 * \sa set_port_drive_high
 * \sa set_port_drive_low
 */
void set_port_pull_down(void port p);

/**
 * Disables the port's pull-up or pull-down resistor. This has the side effect of
 * configuring the port in drive mode.
 * \param p The port to configure.
 * \sa set_port_pull_up
 * \sa set_port_pull_down
 * \sa set_port_drive
 */
void set_port_pull_none(void port p);
#define set_port_pull_none(p)             __builtin_set_port_drv (p, XS1_SETC_DRIVE_DRIVE)

/**
 * Sets a port to master mode. This corresponds to using the SETC instruction
 * on the port with the value XS1_SETC_MS_MASTER.
 * Usually the use of the functions configure_in_port_strobed_master() and
 * configure_out_port_strobed_master() is preferred since they ensure that all
 * the port configuration changes required for the desired mode are performed
 * in the correct order.
 * \param p The port to configure.
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 */
void set_port_master(void port p);
#define set_port_master(p)                __builtin_set_port_ms  (p, XS1_SETC_MS_MASTER)

/**
 * Sets a port to slave mode. This corresponds to using the SETC instruction
 * on the port with the value XS1_SETC_MS_SLAVE.
 * Usually the use of the functions configure_in_port_strobed_slave() and
 * configure_out_port_strobed_slave() is preferred since they ensure that all
 * the port configuration changes required for the desired mode are performed
 * in the correct order.
 * \param p The port to configure.
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 */
void set_port_slave(void port p);
#define set_port_slave(p)                 __builtin_set_port_ms  (p, XS1_SETC_MS_SLAVE)

/**
 * Configures a port to not use ready signals. This corresponds to using the
 * SETC instruction on the port with the value XS1_SETC_RDY_NOREADY.
 * Usually the use of the functions configure_in_port() and
 * configure_out_port() is preferred since they ensure that all the port
 * configuration changes required for the desired mode are performed in the
 * correct order.
 * \param p The port to configure.
 * \sa configure_in_port
 * \sa configure_out_port
 */
void set_port_no_ready(void port p);
#define set_port_no_ready(p)              __builtin_set_port_rdy (p, XS1_SETC_RDY_NOREADY)

/**
 * Sets a port to strobed mode. This corresponds to using the SETC instruction
 * on the port with the value XS1_SETC_RDY_STROBED.
 * Usually the use of the configure_*_port_strobed_* functions is
 * preferred since they ensure that all the port configuration changes required
 * for the desired mode are performed in the correct order.
 * \param p The port to configure.
 * \sa configure_in_port_strobed_master
 * \sa configure_out_port_strobed_master
 * \sa configure_in_port_strobed_slave
 * \sa configure_out_port_strobed_slave
 */
void set_port_strobed(void port p);
#define set_port_strobed(p)               __builtin_set_port_rdy (p, XS1_SETC_RDY_STROBED)

/**
 * Sets a port to handshake mode. This corresponds to using the SETC instruction
 * on the port with the value XS1_SETC_RDY_HANDSHAKE.
 * Usually the use of the configure_*_port_handshake functions is
 * preferred since they ensure that all the port configuration changes required
 * for the desired mode are performed in the correct order.
 * \param p The port to configure.
 * \sa configure_in_port_handshake
 * \sa configure_out_port_handshake
 */
void set_port_handshake(void port p);
#define set_port_handshake(p)             __builtin_set_port_rdy (p, XS1_SETC_RDY_HANDSHAKE)

/**
 * Sets a port to no sample delay mode. This causes the port to sample input
 * data on the rising edge of its clock. This is the default state of the port.
 * \param p The port to configure.
 * \sa set_port_sample_delay
 */
void set_port_no_sample_delay(void port p);
#define set_port_no_sample_delay(p)       __builtin_set_port_sdel(p, XS1_SETC_SDELAY_NOSDELAY)

/**
 * Sets a port to sample delay mode. This causes the port to sample input data
 * on the falling edge of its clock.
 * \param p The port to configure.
 * \sa set_port_no_sample_delay
 */
void set_port_sample_delay(void port p);
#define set_port_sample_delay(p)          __builtin_set_port_sdel(p, XS1_SETC_SDELAY_SDELAY)

/** Configures a port to not invert data that is sampled and driven on its pins.
 *  This is the default state of a port.
 *  \param p The port to configure.
 *  \sa set_port_inv
 */
void set_port_no_inv(void port p);
#define set_port_no_inv(p)                __builtin_set_port_inv (p, XS1_SETC_INV_NOINVERT)

/** Configures a 1-bit port to invert data which is sampled and driven
 *  on its pin. If the port is not a 1-bit port, an
 *  exception is raised. If the port is used as the source for a
 *  clock then setting this mode has the effect of the swapping the
 *  rising and falling edges of the clock.
 *  \param p The 1-bit port to configure.
 *  \sa set_port_no_inv
 */
void set_port_inv(void port p);
#define set_port_inv(p)                   __builtin_set_port_inv (p, XS1_SETC_INV_INVERT)

/**
 * Sets the shift count for a port. This corresponds with the SETPSC
 * instruction. The new shift count must be less than the transfer width of the
 * port, greater than zero and a multiple of the port width otherwise an
 * exception is raised.
 * For a port used for input this function will cause the
 * next input to be ready when the specified amount of data has been shifted in.
 * The next input will return transfer-width bits of data with the captured data in
 * the most significant bits. For a port used for output this will cause the
 * next output to shift out this number of bits.
 * Usually the use of the functions partin() and partout() is preferred over setpsc()
 * as they perform both the required configuration and the input or output together.
 * \param p The buffered port to configure.
 * \param n The new shift count.
 * \sa partin
 * \sa partin_timestamped
 * \sa partout
 * \sa partout_timed
 * \sa partout_timestamped
 */
void set_port_shift_count(/* buffered */ void port p, unsigned n);
#define set_port_shift_count(port, n)        __builtin_set_port_shift(port, n)

/** Sets the delay on the pins connected to the port. The input signals sampled on the
 *  port's pins are delayed by this number of processor-clock cycles before they
 *  they are seen on the port. The default delay on the pins is 0.
 *  The delay must be set to values in the range 0 to 5 inclusive.
 *  If there are multiple enabled ports connected to the same pin then the delay
 *  on that pin is set by the highest priority port.
 *  \param p The port to configure.
 *  \param n The number of processor-clock cycles by which to delay the input
 *           signal.
 *  \sa set_clock_rise_delay
 *  \sa set_clock_fall_delay
 */
void set_pad_delay(void port p, unsigned n);
#define set_pad_delay(port, n)               __builtin_set_pad_delay (port, n)


/** Dynamically reconfigure the type of a port.
 *
 *  This builtin function reconfigures the type of a port. Its first argument
 *  should be a movable pointer to a port. The second argument is the new
 *  port type to reconfigure to. The return value is a movable pointer to the
 *  new type of port. You can use the function as follows:
 *
 *  \code
 *  void f(in port p) {
 *    in port * movable pp = &p;
 *    in buffered port:32 * movable q;
 *    q = reconfigure_port(move(pp), in buffered port:32);
 *
 *    // use *q here as a buffered port
 *
 *    pp = reconfigure_port(move(q), in port);
 *  }
 *  \endcode
 *
 *  When reconfiguring a port, all state (buffered input values etc.) on the
 *  port is lost.
 */
#define reconfigure_port(ptr, typ) __reconfigure_port(ptr, typ)

/**
 * Sets the current logical core to run in fast mode. The scheduler always
 * reserves a slot for a logical core in fast mode regardless of whether core
 * is waiting for an input or a select to complete. This reduces the worst case
 * latency from a change in state happening to a paused input or select
 * completing as a result of that change.
 * However, putting a core in fast mode means that other logical cores are
 * unable to use the extra slot which would otherwise be available while the
 * core is waiting. In addition setting logical cores to run in fast mode may
 * also increase the power consumption.
 * \sa set_core_fast_mode_off
 */
void set_core_fast_mode_on(void);

/**
 * Sets the current logical core to run in normal execution mode. If a core
 * has previously been put into fast mode using set_core_fast_mode_on() this
 * function resets the execution mode it to its default state.
 * \sa set_core_fast_mode_on
 */
void set_core_fast_mode_off(void);

/** \cond */
#define set_core_fast_mode_on()           __builtin_set_thread_fast()
#define set_core_fast_mode_off()          __builtin_set_thread_norm()

// For backwards compatibility
#define set_thread_fast_mode_on()         set_core_fast_mode_on()
#define set_thread_fast_mode_off()        set_core_fast_mode_off()
/** \endcond */

/**
 * Sets the current logical core to run in high-priority mode. High-priority
 * cores are run in preference to regular low-priority cores, with the
 * exception that at least one low priority cores is run on every iteration of
 * the low-priority queue.
 * \sa set_core_high_priority_on
 */
void set_core_high_priority_on(void);

/**
 * Sets the current logical core to run in low-priority mode.
 * \sa set_core_high_priority_off
 */
void set_core_high_priority_off(void);

/**
 * Streams out a value as an unsigned char on a channel end.
 * The protocol used is incompatible with the protocol used
 * by the input (:>) and output (<:) operators.
 * \param c The channel end to stream data out on.
 * \param val The value to output.
 * \sa outuint
 * \sa inuchar
 * \sa inuint
 * \sa inuchar_byref
 * \sa inuint_byref
 */
void outuchar(chanend c, unsigned char val);
#define outuchar(c, val)                   __builtin_out_uchar(c, val)

/**
 * Streams out a value as an unsigned int on a channel end.
 * The protocol used is incompatible with the protocol used
 * by the input (:>) and output (<:) operators.
 * \param c The channel end to stream data out on.
 * \param val The value to output.
 * \sa outuchar
 * \sa inuchar
 * \sa inuint
 * \sa inuchar_byref
 * \sa inuint_byref
 */
void outuint(chanend c, unsigned val);
#define outuint(c, val)                     __builtin_out_uint(c, val)

/**
 * Streams in a unsigned char from a channel end. If the next token in the
 * channel is a control token then an exception is raised.
 * The protocol used is incompatible with the protocol used
 * by the input (:>) and output (<:) operators.
 * \param c The channel end to stream data in on.
 * \return The value received.
 * \sa outuchar
 * \sa outuint
 * \sa inuint
 * \sa inuchar_byref
 * \sa inuint_byref
 */
unsigned char inuchar(chanend c);
#define inuchar(c)                           __builtin_in_uchar(c)

/**
 * Streams in a unsigned int from a channel end. If the next word of data
 * channel in the channel contains a control token then an exception is raised.
 * The protocol used is incompatible with the protocol used
 * by the input (:>) and output (<:) operators.
 * \param c The channel end to stream data in on.
 * \return The value received.
 * \sa outuchar
 * \sa outuint
 * \sa inuchar
 * \sa inuchar_byref
 * \sa inuint_byref
 */
unsigned inuint(chanend c);
#define inuint(c)                            __builtin_in_uint(c)

/**
 * Streams in a unsigned char from a channel end. The inputted value is
 * written to \a val. If the next token in channel is a control
 * token then an exception is raised.
 * The protocol used is incompatible with the protocol used
 * by the input (:>) and output (<:) operators.
 * \param c The channel end to stream data in on.
 * \param[out] val The variable to set to the received value.
 * \sa outuchar
 * \sa outuint
 * \sa inuchar
 * \sa inuint
 * \sa inuint_byref
 */
void inuchar_byref(chanend c, unsigned char &val);
#define inuchar_byref(c, val)                __builtin_in_uchar_byref(c, val)

/**
 * Streams in a unsigned int from a channel end. The inputted value is
 * written to \a val. This function may be called in a case of a
 * select, in which case it becomes ready as soon as there data available on
 * the channel.
 * The protocol used is incompatible with the protocol used
 * by the input (:>) and output (<:) operators.
 * \param c The channel end to stream data in on.
 * \param[out] val The variable to set to the received value.
 * \sa outuchar
 * \sa outuint
 * \sa inuchar
 * \sa inuint
 * \sa inuchar_byref
 */
void inuint_byref(chanend c, unsigned &val);
#define inuint_byref(c, val)                 __builtin_in_uint_byref(c, val)

/**
 * \brief Waits until a port has completed any pending outputs.
 *
 * Waits output all until a port has completed any pending outputs and the
 * last port-width bits of data has been held on the pins for one clock period.
 * \param p The port to wait on.
 */
void sync(void port p);
#define sync(p)                              __builtin_sync(p)

/**
 * Instructs the port to sample the current value on its pins.
 * The port provides the sampled port-width bits of data to the processor
 * immediately, regardless of its transfer width, clock, ready signals and
 * buffering. The input has no effect on subsequent I/O performed on the port.
 * \param p The port to peek at.
 * \return The value sampled on the pins.
 */
unsigned peek(void port p);
#define peek(p)                              __builtin_peek(p)

/**
 * Clears the buffer used by a port. Any data sampled by the port which has not
 * been input by the processor is discarded. Any data output by the processor which
 * has not been driven by the port is discarded. If the port is in the process
 * of serialising output, it is interrupted immediately.
 * If a pending output would have caused a change in direction of the port then
 * that change of direction does not take place. If the port is driving a value
 * on its pins when clearbuf() is called then it continues to drive
 * the value until an output statement changes the value driven.
 * \param p The port whose buffer is to be cleared.
 */
void clearbuf(void port p);
#define clearbuf(p)                          __builtin_clear_buff(p)

/** 
 * Ends the current input on a buffered port. The number of bits sampled by the
 * port but not yet input by the processor is returned. This count includes both
 * data in the transfer register and data in the shift register used for
 * deserialisation.
 * Subsequent inputs on the port return transfer-width bits of data
 * until there is less than one transfer-width bits of data remaining.
 * Any remaining data can be read with one further input, which
 * returns transfer-width bits of data with the remaining buffered data
 * in the most significant bits of this value.
 * \param p The port to end the current input on.
 * \return The number of bits of data remaining.
 */
unsigned endin(/* buffered */ void port p);
#define endin(p)                             __builtin_endin(p)

/**
 * Performs an input of the specified width on a buffered port.
 * The width must be less than the transfer width of the port, greater than
 * zero and a multiple of the port width, otherwise an exception is raised.
 * The value returned is undefined if the number of bits in the port's shift
 * register is greater than or equal to the specified width.
 * \param p The buffered port to input on.
 * \param n The number of bits to input.
 * \return The inputted value.
 * \sa partin_timestamped
 * \sa partout
 * \sa partout_timed
 * \sa partout_timestamped
 */
unsigned partin(/* buffered */ void port p, unsigned n);
#define partin(p, n)                             __builtin_partin(p, n)

/**
 * Performs an output of the specified width on a buffered port.
 * The width must be less than the transfer width of the port, greater than
 * zero and a multiple of the port width, otherwise an exception is raised.
 * The \a n least significant bits of \a val are output.
 * \param p The buffered port to output on.
 * \param n The number of bits to output.
 * \param val The value to output.
 * \sa partin
 * \sa partin_timestamped
 * \sa partout_timed
 * \sa partout_timestamped
 */
void partout(/* buffered */ void port p, unsigned n, unsigned val);
#define partout(p, n, val)                       __builtin_partout(p, n, val)

/**
 * Performs a output of the specified width on a buffered port when the port
 * counter equals the specified time.
 * The width must be less than the transfer width of the port, greater than
 * zero and a multiple of the port width, otherwise an exception is raised.
 * The \a n least significant bits of \a val are output.
 * \param p The buffered port to output on.
 * \param n The number of bits to output.
 * \param val The value to output.
 * \param t The port counter value to output at.
 * \sa partin
 * \sa partin_timestamped
 * \sa partout
 * \sa partout_timestamped
 */
unsigned partout_timed(/* buffered */ void port p, unsigned n, unsigned val, unsigned t);
#define partout_timed(p, n, val, t)                       __builtin_partout_timed(p, n, val, t)

/**
 * Performs an input of the specified width on a buffered port and
 * timestamps the input.
 * The width must be less than the transfer width of the port, greater than
 * zero and a multiple of the port width, otherwise an exception is raised.
 * The value returned is undefined if the number of bits in the port's shift
 * register is greater than or equal to the specified width.
 * \param p The buffered port to input on.
 * \param n The number of bits to input.
 * \return The inputted value and the timestamp.
 * \sa partin
 * \sa partout
 * \sa partout_timed
 * \sa partout_timestamped
 */
{unsigned /* value */, unsigned /* timestamp */}  partin_timestamped(/* buffered */ void port p, unsigned n);
#define partin_timestamped(p, n)                             __builtin_partin_timestamped(p, n)

/**
 * Performs an output of the specified width on a buffered port and
 * timestamps the output.
 * The width must be less than the transfer width of the port, greater than
 * zero and a multiple of the port width, otherwise an exception is raised.
 * The \a n least significant bits of \a val are output.
 * \param p The buffered port to output on.
 * \param n The number of bits to output.
 * \param val The value to output.
 * \return The timestamp of the output.
 * \sa partin
 * \sa partin_timestamped
 * \sa partout
 * \sa partout_timed
 */
unsigned partout_timestamped(/* buffered */ void port p, unsigned n, unsigned val);
#define partout_timestamped(p, n, val)                       __builtin_partout_timestamped(p, n, val)

/**
 * Streams out a control token on a channel end. Attempting to output a
 * hardware control token causes an exception to be raised.
 * \param c The channel end to stream data out on.
 * \param val The value of the control token to output.
 * \sa chkct
 * \sa inct
 * \sa inct_byref
 * \sa testct
 * \sa testwct
 */
void outct(chanend c, unsigned char val);
#define outct(c, val)                        __builtin_outct(c, val)

/**
 * Checks for a control token of a given value. If the next byte in the channel
 * is a control token which matches the expected value then it is
 * input and discarded, otherwise an exception is raised.
 * \param c The channel end.
 * \param val The expected control token value.
 * \sa outct
 * \sa inct
 * \sa inct_byref
 * \sa testct
 * \sa testwct
 */
void chkct(chanend c, unsigned char val);
#define chkct(c, val)                        __builtin_chkct(c, val)

/**
 * Streams in a control token on a channel end. If the next byte in the channel
 * is not a control token then an exception is raised, otherwise the value of
 * the control token is returned.
 * \param c The channel end to stream data in on.
 * \return The received control token.
 * \sa outct
 * \sa chkct
 * \sa inct_byref
 * \sa testct
 * \sa testwct
 */
unsigned char inct(chanend c);
#define inct(c)                              __builtin_inct(c)

/**
 * Streams in a control token on a channel end. The inputted value is written
 * to \a val. If the next byte in the channel is not a control
 * token then an exception is raised.
 * \param c The channel end to stream data in on.
 * \param[out] val The variable to set to the received value.
 * \sa outct
 * \sa chkct
 * \sa inct
 * \sa testct
 * \sa testwct
 */
void inct_byref(chanend c, unsigned char &val);
#define inct_byref(c, val)                   __builtin_inct_byref(c, val)

/**
 * Tests whether the next byte on a channel end is a control token.
 * The token is not discarded from the channel and is still available for input.
 * \param c The channel end to perform the test on.
 * \return 1 if the next byte is a control token, 0 otherwise.
 * \sa outct
 * \sa chkct
 * \sa inct
 * \sa inct_byref
 * \sa testwct
 */
int testct(chanend c);
#define testct(c)                            __builtin_testct(c)

/**
 * Tests whether the next word on a channel end contains a control token.
 * If the word does contain a control token the position in the word is
 * returned. No data is discarded from the channel.
 * \param c The channel end to perform the test on.
 * \return The position of the first control token in the word (1-4) or
           0 if the word contains no control tokens.
 * \sa chkct
 * \sa testwct
 */
int testwct(chanend c);
#define testwct(c)                           __builtin_testwct(c)

/**
 * Outputs a control token on a streaming channel end. Attempting to output a
 * hardware control token causes an exception to be raised. Attempting to
 * output a \a CT_END or \a CT_PAUSE control token is invalid.
 * \param c The channel end to stream data out on.
 * \param val The value of the control token to output.
 * \sa schkct
 * \sa sinct
 * \sa sinct_byref
 * \sa stestct
 * \sa stestwct
 */
void soutct(streaming chanend c, unsigned char val);
#define soutct(c, val)                        __builtin_soutct(c, val)

/**
 * Checks for a control token of a given value on a streaming channel end.
 * If the next byte in the channel is a control token which matches the
 * expected value then it is
 * input and discarded, otherwise an exception is raised.
 * \param c The streaming channel end.
 * \param val The expected control token value.
 * \sa soutct
 * \sa sinct
 * \sa sinct_byref
 * \sa stestct
 * \sa stestwct
 */
void schkct(streaming chanend c, unsigned char val);
#define schkct(c, val)                        __builtin_schkct(c, val)

/**
 * Inputs a control token on a streaming channel end. If the next byte in
 * the channel
 * is not a control token then an exception is raised, otherwise the value of
 * the control token is returned.
 * \param c The streaming channel end to stream data in on.
 * \return The received control token.
 * \sa outct
 * \sa chkct
 * \sa inct_byref
 * \sa testct
 * \sa testwct
 */
unsigned char sinct(streaming chanend c);
#define sinct(c)                              __builtin_sinct(c)

/**
 * Inputs a control token on a streaming channel end.
 * The inputted value is written
 * to \a val. If the next byte in the channel is not a control
 * token then an exception is raised.
 * \param c The streaming channel end to stream data in on.
 * \param[out] val The variable to set to the received value.
 * \sa soutct
 * \sa schkct
 * \sa sinct
 * \sa stestct
 * \sa stestwct
 */
void sinct_byref(streaming chanend c, unsigned char &val);
#define sinct_byref(c, val)                   __builtin_sinct_byref(c, val)

/**
 * Tests whether the next byte on a streaming channel end is a control token.
 * The token is not discarded from the channel and is still available for input.
 * \param c The channel end to perform the test on.
 * \return 1 if the next byte is a control token, 0 otherwise.
 * \sa soutct
 * \sa schkct
 * \sa sinct
 * \sa sinct_byref
 * \sa stestwct
 */
int stestct(streaming chanend c);
#define stestct(c)                            __builtin_stestct(c)

/**
 * Tests whether the next word on a streaming channel end
 * contains a control token.
 * If the word does contain a control token the position in the word is
 * returned. No data is discarded from the channel.
 * \param c The streaming channel end to perform the test on.
 * \return The position of the first control token in the word (1-4) or
           0 if the word contains no control tokens.
 * \sa schkct
 * \sa stestwct
 */
int stestwct(streaming chanend c);
#define stestwct(c)                           __builtin_stestwct(c)

/**
 * Output a block of data over a channel. A total of
 * \a size bytes of data are output on the channel end.
 * The call to out_char_array() must be matched with a call to in_char_array()
 * on the other end of the channel. The number of bytes output must match the
 * number of bytes input.
 * \param c The channel end to output on.
 * \param src The array of values to send.
 * \param size The number of bytes to output.
 * \sa in_char_array
 */
transaction out_char_array(chanend c, const char src[size], unsigned size);

/**
 * Input a block of data from a channel. A total of \a size bytes of data are
 * input on the channel end and stored in an array.
 * The call to in_char_array() must be matched with a call to out_char_array() on the
 * other end of the channel. The number of bytes input must match the
 * number of bytes output.
 * \param c The channel end to input on.
 * \param dst The array to store the values input from on the channel.
 * \param size The number of bytes to input.
 * \sa out_char_array
 */
transaction in_char_array(chanend c, char dst[size], unsigned size);

/**
 * Output a block of data over a streaming channel. A total of
 * \a size bytes of data are output on the channel end.
 * The call to sout_char_array() must be matched with a call to sin_char_array()
 * on the other end of the channel. The number of bytes output must match the
 * number of bytes input.
 * \param c The streaming channel end to output on.
 * \param src The array of values to send.
 * \param size The number of bytes to output.
 * \sa sin_char_array
 */
void sout_char_array(streaming chanend c, const char src[size], unsigned size);

/**
 * Input a block of data from a streaming channel. A total of \a size bytes of
 * data are input on the channel end and stored in an array.
 * The call to sin_char_array() must be matched with a call to sout_char_array()
 * on the other end of the channel. The number of bytes input must match the
 * number of bytes output.
 *
 * This function can be used in a select but the behavior of the function in
 * a select is undefined if the copy size is zero.
 *
 * \param c The channel end to input on.
 * \param dst The array to store the values input from on the channel.
 * \param size The number of bytes to input.
 * \sa out_char_array
 */
#pragma select handler
void sin_char_array(streaming chanend c, char dst[size], unsigned size);

/**
 * Incorporate a word into a Cyclic Redundancy Checksum. The calculation performed
 * is
 * \code
 * for (int i = 0; i < 32; i++) {
 *   int xorBit = (crc & 1);
 *
 *   checksum  = (checksum >> 1) | ((data & 1) << 31);
 *   data = data >> 1;
 *
 *   if (xorBit)
 *     checksum = checksum ^ poly;
 * }
 * \endcode
 * \param[in,out] checksum The initial value of the checksum, which is
 *                         updated with the new checksum.
 * \param data The data to compute the CRC over.
 * \param poly The polynomial to use when computing the CRC.
 * \sa chkct
 * \sa testct
 */
void crc32(unsigned &checksum, unsigned data, unsigned poly);
#define crc32(checksum, data, poly)           __builtin_crc32(checksum, data, poly)

/**
 * Incorporate 8-bits of a word into a Cyclic Redundancy Checksum.
 * The CRC is computed over the 8 least significant bits of the data and
 * the data shifted right by 8 is returned. The calculation performed is
 * \code
 * for (int i = 0; i < 8; i++) {
 *   int xorBit = (crc & 1);
 *
 *   checksum  = (checksum >> 1) | ((data & 1) << 31);
 *   data = data >> 1;
 *
 *   if (xorBit)
 *     checksum = checksum ^ poly;
 * }
 * \endcode
 * \param[in,out] checksum The initial value of the checksum which is
 *                         updated with the new checksum.
 * \param data The data.
 * \param poly The polynomial to use when computing the CRC.
 * \return The data shifted right by 8.
 */
unsigned crc8shr(unsigned &checksum, unsigned data, unsigned poly);
#define crc8shr(checksum, data, poly)         __builtin_crc8shr(checksum, data, poly)

/**
 * Multiplies two words to produce a double-word and adds two single words.
 * The high word and the low word of the result are returned. The
 * multiplication is unsigned and cannot overflow. The calculation performed
 * is
 * \code
 * (uint64_t)a * (uint64_t)b + (uint64_t)c + (uint64_t)d
 * \endcode
 * \return The high and low halves of the calculation respectively.
 * \sa mac
 * \sa macs
 */
{unsigned, unsigned} lmul(unsigned a, unsigned b, unsigned c, unsigned d);
#define lmul(a, b, c, d)                      __builtin_long_mul(a, b, c, d)

/**
 * Multiplies two unsigned words to produce a double-word and adds a double 
 * word. The high word and the low word of the result are returned. The
 * calculation performed is:
 * \code
 * (uint64_t)a * (uint64_t)b + (uint64_t)c<<32 + (uint64_t)d
 * \endcode
 * \return The high and low halves of the calculation respectively.
 * \sa lmul
 * \sa macs
 */
{unsigned, unsigned} mac(unsigned a, unsigned b, unsigned c, unsigned d);
#define mac(a, b, c, d)                       __builtin_mac(a, b, c, d)

/**
 * Multiplies two signed words and adds the double word result to a double
 * word. The high word and the low word of the result are returned. The
 * calculation performed is:
 * \code
 * (int64_t)a * (int64_t)b + (int64_t)c<<32 + (int64_t)d
 * \endcode
 * \return The high and low halves of the calculation respectively.
 * \sa lmul
 * \sa mac
 */
{signed, unsigned} macs(signed a, signed b, signed c, unsigned d);
#define macs(a, b, c, d)                       __builtin_macs(a, b, c, d)

/**
 * Sign extends an input. The first argument is the value to sign extend. 
 * The second argument contains the bit position. All bits
 * at a position higher or equal are set to the value of the 
 * bit one position lower. In effect, the lower b bits are interpreted as 
 * a signed integer. If b is less than 1 or greater than 32 then result
 * is identical to argument a.
 *
 * \return The sign extended value.
 * \sa zext
 */
signed sext(unsigned a, unsigned b);
#define sext(a,b)                       __builtin_sext(a,b)

/**
 * Incorporate a word into a Cyclic Redundancy Checksum (CRC), and
 * simultaneously increment a register by a specified amount.
 * \param[in,out] checksum The inital value of the checksum, which is updated
 *                         with the new checksum.
 * \param data The data to compute the CRC over.
 * \param poly The polynomial to use when computing the CRC.
 * \param[in,out] value A value to be incremented.
 * \param increment The increment, one of bpw, 1, 2, 3, 4, 5,
 *                  6, 7, 8, 16, 24, 32.
 * \sa crc32
 */
void crc32_inc(unsigned int &checksum, unsigned int data, unsigned int poly,
               unsigned int &value, unsigned int increment);
#if defined(__XS2A__) || defined(__XS3A__)
#define crc32_inc(d, x, p, v, b) __builtin_crc32_inc(d, x, p, v, b)
#endif

/**
 * Incorporate n-bits of a 32-bit word into a Cyclic Redundancy Checksum (CRC).
 * Executing 32/N crcn calls sequentially has the same effect as executing a
 * single crc call.
 * \param[in,out] checksum The inital value of the checksum, which is updated
 *                         with the new checksum.
 * \param data The data to compute the CRC over.
 * \param poly The polynomial to use when computing the CRC.
 * \param n The number of lower bits of the data to incorporate.
 */
void crcn(unsigned int &checksum, unsigned int data,
          unsigned int poly, unsigned int n);
#if defined(__XS2A__) || defined(__XS3A__)
#define crcn(c, d, p, n) __builtin_crcn(c, d, p, n)
#endif

/**
 * Raise an ET_ECALL exception if the specified time is less than the reference
 * time.
 * \param time The time value.
 */
void elate(unsigned int time);
#if defined(__XS2A__) || defined(__XS3A__)
#define elate(t) __builtin_elate(t)
#endif

/**
 * Extract a bitfield from a 64-bit value.
 * \param value The value to extract the bitfield from.
 * \param position The bit position of the field, which must be a value between
 *                 0 and bpw - 1, inclusive.
 * \param length The length of the field, one of bpw, 1, 2, 3, 4, 5,
 *               6, 7, 8, 16, 24, 32.
 * \return The value of the bitfield.
 */
unsigned int lextract(unsigned long long value, unsigned int position,
                      unsigned int length);
#if defined(__XS2A__) || defined(__XS3A__)
#define lextract(v, p, l) __builtin_lextract(v, p, l)
#endif

/**
 * Insert a bitfield into a 64-bit value.
 * \param value The 64-bit value to insert the bitfield in.
 * \param bitfield The value of the bitfield.
 * \param position The bit position of the field, which must be a value between
 *                 0 and bpw - 1, inclusive.
 * \param length The length of the field, one of bpw, 1, 2, 3, 4, 5,
 *               6, 7, 8, 16, 24, 32.
 * \return The 64-bit value with the inserted bitfield.
 */
unsigned long long linsert(unsigned long long value, unsigned int bitfield,
                           unsigned int position, unsigned int length);
#if defined(__XS2A__) || defined(__XS3A__)
#define linsert(v, b, p, l) __builtin_linsert(v, b, p, l)
#endif

/**
 * Perform saturation on a 64-bit value. If any arithmetic has overflowed
 * beyond a given bit index, then the value is set to MININT or MAXINT,
 * right shifted by the bit index.
 * \param value The 64-bit value to perform saturation on.
 * \param index The bit index at which overflow is checked for.
 * \result The saturated 64-bit value.
 */
signed long long lsats(signed long long value, unsigned int index);
#if defined(__XS2A__) || defined(__XS3A__)
#define lsats(v, i) __builtin_lsats(v, i)
#endif

/**
 * Unzip a 64-bit value into two 32-bit values, with a granularity of
 * bits, bit pairs, nibbles, bytes or byte pairs.
 * \param value The 64-bit zipped value.
 * \param log_granularity The logarithm of the granularity.
 * \return Two 32-bit unzipped values.
 */
{unsigned int, unsigned int} unzip(unsigned long long value,
                                   unsigned int log_granularity);
#if defined(__XS2A__) || defined(__XS3A__)
#define unzip(v, g) __builtin_unzip(v, g)
#endif

/**
 * Zip two 32-bit values into a single 64-bit value, with a granularity of
 * bits, bit pairs, nibbles, bytes or byte pairs.
 * \param value1 The first 32-bit value.
 * \param value2 The second 32-bit value.
 * \param log_granularity The logarithm of the granularity.
 * \return The 64-bit zipped value.
 */
unsigned long long zip(unsigned int value1, unsigned int value2,
                       unsigned int log_granularity);
#if defined(__XS2A__) || defined(__XS3A__)
#define zip(v1, v2, g) __builtin_zip(v1, v2, g)
#endif

/**
 * Zero extends an input. The first argument is the value to zero extend. 
 * The second argument contains the bit position. All bits
 * at a position higher or equal are set to the zero.
 * In effect, the lower b bits are interpreted as 
 * an unsigned integer. If b is less than 1 or greater than 32 then result
 * is identical to argument a.
 *
 * \return The zero extended value.
 * \sa sext
 */
unsigned zext(unsigned a, unsigned b);
#define zext(a,b)                       __builtin_zext(a,b)

/**
 * \brief Wait until the value on the port's pins equals the
 * specified value.
 *
 * This function must be called as the \p when expression of an
 * input on a port. It causes the input to become ready when the value on the
 * port's pins is equal to the least significant port-width bits of \a val.
 * \param val The value to compare against.
 * \sa pinsneq
 */
void pinseq(unsigned val);
#define pinseq(val)                           __builtin_pins_eq(val)

/**
 * \brief Wait until the value on the port's pins does not equal
 * the specified value.
 *
 * This function must be called as the \p when expression of an input on a port.
 * It causes the input to become ready when the value on the
 * port's pins is not equal to the least significant port-width bits of \a val.
 * \param val The value to compare against.
 * \sa pinseq
 */
void pinsneq(unsigned val);
#define pinsneq(val)                          __builtin_pins_ne(val)

/**
 * \brief Wait until the value on the port's pins equals the
 * specified value and the port counter equals the specified time.
 *
 * This function must be called as the \p when expression of an
 * input on a unbuffered port. It causes the input to become ready when the value on the
 * port's pins is equal to the least significant port-width bits of \a val and
 * the port counter equals \a time.
 * \param val The value to compare against.
 * \param time The time at which to make the comparison.
 * \sa pinsneq
 */
void pinseq_at(unsigned val, unsigned time);
#define pinseq_at(val, time)                  __builtin_pins_eq_at(val, time)

/**
 * \brief Wait until the value on the port's pins does not equal
 * the specified value and the port counter equals the specified time.
 *
 * This function must be called as the \p when expression of an input on a unbuffered port.
 * It causes the input to become ready when the value on the
 * port's pins is not equal to the least significant port-width bits of \a val
 * and the port counter equals \a time.
 * \param val The value to compare against.
 * \param time The time at which to make the comparison.
 * \sa pinseq
 */
void pinsneq_at(unsigned val, unsigned time);
#define pinsneq_at(val, time)                 __builtin_pins_ne_at(val, time)

/**
 * \brief Wait until the time of the timer equals the specified value.
 *
 * This function must be called as the \p when expression of an input on a timer.
 * It causes the input to become ready when timer's counter is interpreted as
 * coming after the specified value timer is after the given value. A time A is considered to be after a
 * time B if the expression <tt>((int)(B - A) < 0)</tt> is true.
 * \param val The time to compare against.
 * \sa timeafter
 */
void timerafter(unsigned val);
#define timerafter(val)                       __builtin_timer_after(val)

/** 
 * Tests whether a time input from a timer is considered to come after
 * another time input from a timer. The comparison is the same as that
 * performed by the function timerafter().
 * \param A The first time to compare.
 * \param B The second time to compare.
 * \return Whether the first time is after the second.
 */
#define timeafter(A, B) ((int)((B) - (A)) < 0)

/**
 * Tests whether the time recorded from a port using the timestamped operator
 * is after another time recorded from a port.
 * For the comparison to be meaningful, both times must be taken from ports
 * synchronised to the same clock. Two ports are synchronised
 * to the same clock by attaching both ports to the clock before starting the
 * clock. \a A is considered to come after \a B if the expression
 * <tt>((short)(\a B - \a A) < 0)</tt> is true.
 * \param A The first time to compare.
 * \param B The second time to compare.
 * \return Whether the first time is after the second.
 */
#define porttimeafter(A, B) ((short)((B) - (A)) < 0)

#endif /* __XC__*/

/**
 * Gets the value of a processor state register. This corresponds with the
 * GETPS instruction. An exception is raised if the argument is not a legal
 * processor state register.
 * \param reg The processor state register to read.
 * \return The value of the processor state register.
 */
unsigned getps(unsigned reg);

#define getps(reg) __builtin_getps(reg)

/**
 * Sets the value of a processor state register. Corresponds with the SETPS
 * instruction. An exception is raised if the argument is not a legal processor
 * state register.
 * \param reg The processor state register to write.
 * \param value The value to set the processor state register to.
 */
void setps(unsigned reg, unsigned value);

#define setps(reg, value) __builtin_setps(reg, value)
/**
 * Reads the value of a processor switch register. The read is of
 * the processor switch which is local to the specified tile id. On success
 * 1 is returned and the value of the register is assigned to \a data.
 * If an error acknowledgement is received or if the register number or tile
 * identifier is too large to fit in the read packet then 0 is returned.
 * \param tileid The tile identifier.
 * \param reg The number of the register.
 * \param[out] data The value read from the register.
 * \return Whether the read was successful.
 * \sa read_sswitch_reg
 * \sa write_pswitch_reg
 * \sa write_pswitch_reg_no_ack
 * \sa write_sswitch_reg
 * \sa write_sswitch_reg_no_ack
 * \sa get_local_tile_id
 */
#ifdef __XC__
int read_pswitch_reg(unsigned tileid, unsigned reg, unsigned &data);
#else
int read_pswitch_reg(unsigned tileid, unsigned reg, unsigned *data);
#endif

/**
 * Reads the value of a system switch register. The read is of
 * the system switch which is local to the specified tile id. On success
 * 1 is returned and the value of the register is assigned to \a data.
 * If an error acknowledgement is received or if the
 * register number or tile identifier is too large to fit in the read packet
 * then 0 is returned.
 * \param tileid The tile identifier.
 * \param reg The number of the register.
 * \param[out] data The value read from the register.
 * \return Whether the read was successful.
 * \sa read_pswitch_reg
 * \sa write_pswitch_reg
 * \sa write_pswitch_reg_no_ack
 * \sa write_sswitch_reg
 * \sa write_sswitch_reg_no_ack
 * \sa get_local_tile_id
 */
#ifdef __XC__
int read_sswitch_reg(unsigned tileid, unsigned reg, unsigned &data);
#else
int read_sswitch_reg(unsigned tileid, unsigned reg, unsigned *data);
#endif

/**
 * Writes a value to a processor switch register. The write is of
 * the processor switch which is local to the specified tile id. If a
 * successful acknowledgement is received then 1 is returned. If an error
 * acknowledgement is received or if the register number or tile
 * identifier is too large to fit in the write packet then 0 is returned.
 * \param tileid The tile identifier.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the write was successful.
 * \sa read_pswitch_reg
 * \sa read_sswitch_reg
 * \sa write_pswitch_reg_no_ack
 * \sa write_sswitch_reg
 * \sa write_sswitch_reg_no_ack
 * \sa get_local_tile_id
 */
int write_pswitch_reg(unsigned tileid, unsigned reg, unsigned data);

/**
 * Writes a value to a processor switch register without acknowledgement.
 * The write is of the processor switch which is local to the specified tile
 * id. Unlike write_pswitch_reg() this function does not wait until the write
 * has been performed. If the register number or tile identifier is too large
 * to fit in the write packet 0 is returned, otherwise 1 is returned. Because
 * no acknowledgement is requested the return value does not reflect whether
 * the write succeeded.
 * \param tileid The tile identifier.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the parameters are valid.
 * \sa read_pswitch_reg
 * \sa read_sswitch_reg
 * \sa write_pswitch_reg
 * \sa write_sswitch_reg
 * \sa write_sswitch_reg_no_ack
 */
int write_pswitch_reg_no_ack(unsigned tileid, unsigned reg, unsigned data);

/**
 * Writes a value to a system switch register. The write is of
 * the system switch which is local to the specified tile id. If a
 * successful acknowledgement is received then 1 is returned. If an error
 * acknowledgement is received or if the register number or tile
 * identifier is too large to fit in the write packet then 0 is returned.
 * \param tileid The tile identifier.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the write was successful.
 * \sa read_pswitch_reg
 * \sa read_sswitch_reg
 * \sa write_pswitch_reg
 * \sa write_pswitch_reg_no_ack
 * \sa write_sswitch_reg_no_ack
 * \sa get_local_tile_id
 */
int write_sswitch_reg(unsigned tileid, unsigned reg, unsigned data);

/**
 * Writes a value to a system switch register without acknowledgement.
 * The write is of the system switch which is local to the specified tile id.
 * Unlike write_sswitch_reg() this function does not wait until the write has
 * been performed. If the register number or tile identifier is too large to
 * fit in the write packet 0 is returned, otherwise 1 is returned. Because no
 * acknowledgement is requested the return value does not reflect whether the
 * write succeeded.
 * \param tileid The tile identifier.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the parameters are valid.
 * \sa read_pswitch_reg
 * \sa read_sswitch_reg
 * \sa write_pswitch_reg
 * \sa write_pswitch_reg_no_ack
 * \sa write_sswitch_reg
 * \sa get_local_tile_id
 */
int write_sswitch_reg_no_ack(unsigned tileid, unsigned reg, unsigned data);

#ifdef __XC__
/**
 * Reads the value of a tile configuration register. On success
 * 1 is returned and the value of the register is assigned to \a data.
 * If an error acknowledgement is received or if the register number is too
 * large to fit in the read packet then 0 is returned.
 * \param tile The tile.
 * \param reg The number of the register.
 * \param[out] data The value read from the register.
 * \return Whether the read was successful.
 * \sa write_tile_config_reg
 * \sa write_tile_config_reg_no_ack
 */
int read_tile_config_reg(tileref tile, unsigned reg, unsigned &data);

/**
 * Writes a value to a tile configuration register. If a successful
 * acknowledgement is received then 1 is returned. If an error acknowledgement
 * is received or if the register number is too large to fit in the write packet
 * then 0 is returned.
 * \param tile The tile.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the write was successful.
 * \sa read_tile_config_reg
 * \sa write_tile_config_reg_no_ack
 */
int write_tile_config_reg(tileref tile, unsigned reg, unsigned data);

/**
 * Writes a value to a tile configuration register without acknowledgement.
 * Unlike write_tile_config_reg() this function does not wait until the write
 * has been performed. If the register number is too large to fit in the write
 * packet 0 is returned, otherwise 1 is returned. Because no acknowledgement
 * is requested the return value does not reflect whether the write succeeded.
 * \param tile The tile.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the parameters are valid.
 * \sa read_tile_config_reg
 * \sa write_tile_config_reg
 */
int write_tile_config_reg_no_ack(tileref tile, unsigned reg, unsigned data);

/** \cond */
// Backwards compatibility
#define read_core_config_reg(tile, reg, data) read_tile_config_reg(tile, reg, data)
#define write_core_config_reg(tile, reg, data) write_tile_config_reg(tile, reg, data)
#define write_core_config_reg_no_ack(tile, reg, data) write_tile_config_reg_no_ack(tile, reg, data)
/** \endcond */

/**
 * Reads the value of a node configuration register. The read is of
 * the node containing the specified tile. On success
 * 1 is returned and the value of the register is assigned to \a data.
 * If an error acknowledgement is received or if the register number is too
 * large to fit in the read packet then 0 is returned.
 * \param tile The tile.
 * \param reg The number of the register.
 * \param[out] data The value read from the register.
 * \return Whether the read was successful.
 * \sa write_node_config_reg
 * \sa write_node_config_reg_no_ack
 */
int read_node_config_reg(tileref tile, unsigned reg, unsigned &data);

/**
 * Writes a value to a node configuration register. The write is of
 * the node containing the specified tile. If a successful acknowledgement is
 * received then 1 is returned. If an error acknowledgement is received or if
 * the register number is too large to fit in the write packet then 0 is
 * returned.
 * \param tile The tile.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the write was successful.
 * \sa read_node_config_reg
 * \sa write_node_config_reg_no_ack
 */
int write_node_config_reg(tileref tile, unsigned reg, unsigned data);

/**
 * Writes a value to a node configuration register without acknowledgement.
 * The write is of the node containing the specified tile. Unlike
 * write_node_config_reg() this function does not wait until the write has
 * been performed. If the register number is too large to fit in the write
 * packet 0 is returned, otherwise 1 is returned. Because no acknowledgement
 * is requested the return value does not reflect whether the write succeeded.
 * \param tile The tile.
 * \param reg The number of the register.
 * \param data The value to write to the register.
 * \return Whether the parameters are valid.
 * \sa read_node_config_reg
 * \sa write_node_config_reg
 */
int write_node_config_reg_no_ack(tileref tile, unsigned reg, unsigned data);

/**
 * Reads \a size bytes from the specified peripheral starting at the specified
 * base address. The peripheral must be a peripheral with a 8-bit interface. 
 * On success 1 is returned and \a data is filled with the values that were
 * read. Returns 0 on failure.
 * \param tile The tile.
 * \param peripheral The peripheral number.
 * \param base_address The base address.
 * \param size The number of 8-bit values to read.
 * \param[out] data The values read from the peripheral.
 * \return Whether the read was successful.
 * \sa write_periph_8
 * \sa write_periph_8_no_ack
 * \sa read_periph_32
 * \sa write_periph_32
 * \sa write_periph_32_no_ack
 */
int read_periph_8(tileref tile, unsigned peripheral, unsigned base_address,
                  unsigned size, unsigned char data[size]);
                  
/**
 * Writes \a size bytes to the specified peripheral starting at the specified
 * base address. The peripheral must be a peripheral with a 8-bit interface.
 * On success 1 is returned. Returns 0 on failure.
 * \param tile The tile.
 * \param peripheral The peripheral number.
 * \param base_address The base address.
 * \param size The number of 8-bit values to write.
 * \param data The values to write to the peripheral.
 * \return Whether the write was successful.
 * \sa read_periph_8
 * \sa write_periph_8_no_ack
 * \sa read_periph_32
 * \sa write_periph_32
 * \sa write_periph_32_no_ack
 */
int write_periph_8(tileref tile, unsigned peripheral, unsigned base_address,
                   unsigned size, const unsigned char data[size]);

/**
 * Writes \a size bytes to the specified peripheral starting at the specified
 * base address without acknowledgement. The peripheral must be a peripheral
 * with a 8-bit interface. Unlike write_periph_8() this function does not wait
 * until the write has been performed. Because no acknowledgement is requested
 * the return value does not reflect whether the write succeeded.
 * \param tile The tile.
 * \param peripheral The peripheral number.
 * \param base_address The base address.
 * \param size The number of 8-bit values to write.
 * \param data The values to write to the peripheral.
 * \return Whether the parameters are valid.
 * \sa read_periph_8
 * \sa write_periph_8
 * \sa read_periph_32
 * \sa write_periph_32
 * \sa write_periph_32_no_ack
 */
int write_periph_8_no_ack(tileref tile, unsigned peripheral,
                          unsigned base_address, unsigned size,
                          const unsigned char data[size]);

/**
 * Reads \a size 32-bit words from the specified peripheral starting at the
 * specified base address. On success 1 is returned and \a data is filled with
 * the values that were read. Returns 0 on failure. When reading a peripheral
 * with an 8-bit interface the most significant byte of each word returned is
 * the byte at the lowest address (big endian byte ordering).
 * \param tile The tile.
 * \param peripheral The peripheral number.
 * \param base_address The base address.
 * \param size The number of 32-bit words to read.
 * \param[out] data The values read from the peripheral.
 * \return Whether the read was successful.
 * \sa read_periph_8
 * \sa write_periph_8
 * \sa write_periph_8_no_ack
 * \sa write_periph_32
 * \sa write_periph_32_no_ack
 */
int read_periph_32(tileref tile, unsigned peripheral, unsigned base_address,
                   unsigned size, unsigned data[size]);

/**
 * Writes \a size 32-bit words to the specified peripheral starting at the
 * specified base address. On success 1 is returned. Returns 0 on failure.
 * When writing to a peripheral with an 8-bit interface the most significant
 * byte of each word passed to the function is written to the byte at the
 * lowest address (big endian byte ordering).
 * \param tile The tile.
 * \param peripheral The peripheral number.
 * \param base_address The base address.
 * \param size The number of 32-bit words to write.
 * \param data The values to write to the peripheral.
 * \return Whether the write was successful.
 * \sa read_periph_8
 * \sa write_periph_8
 * \sa write_periph_8_no_ack
 * \sa read_periph_32
 * \sa write_periph_32_no_ack
 */
int write_periph_32(tileref tile, unsigned peripheral, unsigned base_address,
                    unsigned size, const unsigned data[size]);

/**
 * Writes \a size 32-bit words to the specified peripheral starting at the
 * specified base address without acknowledgement. Unlike write_periph_32()
 * this function does not wait until the write has been performed. Because no
 * acknowledgement is requested the return value does not reflect whether the
 * write succeeded. When writing to a peripheral with an 8-bit interface the
 * most significant byte of each word passed to the function is written to the
 * byte at the lowest address (big endian byte ordering).
 * \param tile The tile.
 * \param peripheral The peripheral number.
 * \param base_address The base address.
 * \param size The number of 32-bit words to write.
 * \param data The values to write to the peripheral.
 * \return Whether the parameters are valid.
 * \sa read_periph_8
 * \sa write_periph_8
 * \sa write_periph_8_no_ack
 * \sa read_periph_32
 * \sa write_periph_32
 */
int write_periph_32_no_ack(tileref tile, unsigned peripheral,
                           unsigned base_address, unsigned size,
                           const unsigned data[size]);
#endif //__XC__

/**
 * Returns the routing ID of the tile on which the caller is running. The
 * routing ID uniquely identifies a tile on the network.
 * \return The tile identifier.
 * \sa get_tile_id
 * \sa get_logical_core_id
 */
unsigned get_local_tile_id(void);

#ifdef __XC__
/**
 * Returns the routing ID of the specified tile. The routing ID uniquely
 * identifies a tile on the network.
 * \param t The tile.
 * \return The tile identifier.
 * \sa get_local_tile_id
 */
unsigned get_tile_id(tileref t);
#endif //__XC__

/**
 * Returns the identifier of the logical core on which the caller is running.
 * The identifier uniquely identifies a logical core on the current tile.
 * \return The logical core identifier.
 * \sa get_local_tile_id
 */
unsigned get_logical_core_id(void);

/** \cond */
#define get_logical_core_id() __builtin_getid()
/** \endcond */

/** \cond */
// Backwards compatibility
#ifdef __XC__
#define get_core_id() _Pragma("warning \"get_core_id is deprecated, use get_local_tile_id instead\"") get_local_tile_id()
#define get_thread_id() _Pragma("warning \"get_thread_id is deprecated, use get_logical_core_id instead\"") get_logical_core_id()
#else
__attribute__((deprecated)) static inline unsigned get_core_id(void) {
  return get_local_tile_id();
}
__attribute__((deprecated)) static inline unsigned get_thread_id(void) {
  return get_logical_core_id();
}
#endif
/** \endcond */

#endif /* !defined(__ASSEMBLER__) */

#ifdef __cplusplus
} //extern "C" 
#endif

#endif /* _xs1_h_ */
