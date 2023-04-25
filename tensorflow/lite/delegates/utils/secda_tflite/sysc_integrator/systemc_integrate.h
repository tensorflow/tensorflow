
// TODO Generalise this code so it is easy for all new accelerators
#ifndef SYSTEMC_INTEGRATE
#define SYSTEMC_INTEGRATE

#include <systemc.h>
// #include "../ap_sysc/hls_bus_if.h"

int sc_main(int argc, char* argv[]) { return 0; }

void sysC_init() {
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions(SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions(SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);
}

#endif  // SYSTEMC_INTEGRATE
