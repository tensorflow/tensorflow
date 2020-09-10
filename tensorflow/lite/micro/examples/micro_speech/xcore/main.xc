// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include "tensorflow/lite/micro/examples/micro_speech/xcore/main_support.h"

#include <platform.h>

#include "mic_array.h"
#include "fifo.h"
#include "microspeech_xcore_support.h"

/*-----------------------------------------------------------*/
/* Setup resources */
/*-----------------------------------------------------------*/
#define MIC_TILE_NO 1
#define MIC_TILE tile[MIC_TILE_NO]

/* Ports for the PDM microphones */
out port p_pdm_clk              = PORT_PDM_CLK;
in buffered port:32 p_pdm_mics  = PORT_PDM_DATA;

/* Clock port for the PDM mics */
in port p_mclk  = PORT_MCLK_IN;

/* Clock blocks for PDM mics */
clock pdmclk    = on MIC_TILE : XS1_CLKBLK_1;
clock pdmclk2   = on MIC_TILE : XS1_CLKBLK_2;

/* Setup internal clock to be mic PLL */
clock mclk_internal = on MIC_TILE : XS1_CLKBLK_3;

/*-----------------------------------------------------------*/
/* Helpers to setup a clock as a PLL */
/*-----------------------------------------------------------*/

// 24MHz in, 24.576MHz out, integer mode
// Found exact solution:   IN  24000000.0, OUT  24576000.0, VCO 2457600000.0, RD  5, FD  512                       , OD  5, FOD   10
#define APP_PLL_DISABLE 0x0201FF04
#define APP_PLL_CTL_0   0x0A01FF04
#define APP_PLL_DIV_0   0x80000004
#define APP_PLL_FRAC_0  0x00000000

void set_app_pll(void) {
	write_node_config_reg(tile[0], XS1_SSWITCH_SS_APP_PLL_CTL_NUM,            APP_PLL_DISABLE);
	delay_milliseconds(1);
	write_node_config_reg(tile[0], XS1_SSWITCH_SS_APP_PLL_CTL_NUM,            APP_PLL_CTL_0);
	write_node_config_reg(tile[0], XS1_SSWITCH_SS_APP_PLL_CTL_NUM,            APP_PLL_CTL_0);
	write_node_config_reg(tile[0], XS1_SSWITCH_SS_APP_PLL_FRAC_N_DIVIDER_NUM, APP_PLL_FRAC_0);
	write_node_config_reg(tile[0], XS1_SSWITCH_SS_APP_CLK_DIVIDER_NUM,        APP_PLL_DIV_0);
}

void mic_array_setup_ddr(
		clock pdmclk,
		clock pdmclk6,
		in port p_mclk,
		out port p_pdm_clk,
		buffered in port:32 p_pdm_mics,
		int divide)
{
	configure_clock_src_divide(pdmclk, p_mclk, divide/2);
	configure_clock_src_divide(pdmclk6, p_mclk, divide/4);
	configure_port_clock_output(p_pdm_clk, pdmclk);
	configure_in_port(p_pdm_mics, pdmclk6);

	/* start the faster capture clock */
	start_clock(pdmclk6);
	/* wait for a rising edge on the capture clock */
	partin(p_pdm_mics, 4);
	/* start the slower output clock */
	start_clock(pdmclk);
}

void tile1(chanend c_gpio) {
  streaming chan c_ds_output[1];
  streaming chan c_4x_pdm_mic_0[1];
  set_app_pll();

  mic_array_setup_ddr(pdmclk, pdmclk2, p_mclk, p_pdm_clk, p_pdm_mics, 8);

  par {
    mic_decoupler(c_ds_output[0], c_gpio);
    mic_dual_pdm_rx_decimate(p_pdm_mics, c_ds_output[0], c_4x_pdm_mic_0);
  }
}

int main() {
    chan c_gpio;

    par {
        on tile[0]: {
              tile0(c_gpio);
        }
        on tile[1]: {
            par {
              tile1(c_gpio);
              tf_main();
            }
        }
    }
    return 0;
}
