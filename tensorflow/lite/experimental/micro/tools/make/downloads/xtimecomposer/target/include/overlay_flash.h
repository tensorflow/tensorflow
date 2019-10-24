#ifndef _overlay_flash_h_
#define _overlay_flash_h_

#include <flash.h>

#ifdef __XC__

void overlay_flash_init(fl_SPIPorts * movable ports,
                        unsigned clk_dividend,
                        unsigned clk_divisor);

fl_SPIPorts * movable overlay_flash_claim_ports();

void overlay_flash_return_ports(fl_SPIPorts * movable ports);

#else

void overlay_flash_init(fl_SPIPorts *ports);

fl_SPIPorts * overlay_flash_claim_ports();

void overlay_flash_return_ports(fl_SPIPorts *ports);

#endif

#endif // _overlay_flash_h_
