#ifndef _xs1_su_h_
#define _xs1_su_h_

#include "xs1_su_registers.h"

/** 
 * \file xs1_su.h
 * \brief XS1-SU routines
 *
 * This file contains functions to control the analogue-to-digital converter
 * (ADC) on a XS1-SU device.
 */

/**
 * Enables the ADC input specified by \a number. Samples are sent to chanend
 * \a c.
 * \param number The ADC input number.
 * \param c The channel connected to the XS1-SU ADC.
 */
void enable_xs1_su_adc_input(unsigned number, chanend c);

/**
 * Enables the ADC input specified by \a number. Samples are sent to chanend
 * \a c.
 * \param number The ADC input number.
 * \param c The channel connected to the XS1-SU ADC.
 */
void enable_xs1_su_adc_input_streaming(unsigned number, streaming chanend c);

/**
 * Disables the ADC input specified by \a number.
 * \param number The ADC input number.
 * \param c The channel connected to the XS1-SU ADC.
 */
void disable_xs1_su_adc_input(unsigned number, chanend c);

/**
 * Disables the ADC input specified by \a number.
 * \param number The ADC input number.
 * \param c The channel connected to the XS1-SU ADC.
 */
void disable_xs1_su_adc_input_streaming(unsigned number, streaming chanend c);

#endif /*_xs1_su_h_*/
