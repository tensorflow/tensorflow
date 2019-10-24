#ifndef _overlay_quadflash_h_
#define _overlay_quadflash_h_

#include <quadflash.h>

#ifdef __XC__

void overlay_quadflash_init(fl_QSPIPorts * movable ports,
                            unsigned clk_dividend,
                            unsigned clk_divisor);

fl_QSPIPorts * movable overlay_quadflash_claim_ports();

void overlay_quadflash_return_ports(fl_QSPIPorts * movable ports);

#else

void overlay_quadflash_init(fl_QSPIPorts *ports);

fl_QSPIPorts * overlay_quadflash_claim_ports();

void overlay_quadflash_return_ports(fl_QSPIPorts *ports);

#endif

#endif // _overlay_quadflash_h_
