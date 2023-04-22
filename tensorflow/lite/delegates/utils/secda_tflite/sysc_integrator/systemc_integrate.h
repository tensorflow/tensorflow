
// TODO Generalise this code so it is easy for all new accelerators
#ifndef SYSTEMC_INTEGRATE
#define SYSTEMC_INTEGRATE

#include <systemc.h>
#include "../ap_sysc/hls_bus_if.h"
// #include "tb_driver.h"

int sc_main(int argc, char* argv[]) { return 0; }

void sysC_init() {
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions(SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions(SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);
}

// struct systemC_sigs {
//   sc_clock clk_fast;
//   sc_signal<bool> sig_reset;

//   hls_bus_chn<unsigned long long> insn_mem;
//   hls_bus_chn<unsigned long long> inp_mem;
//   hls_bus_chn<unsigned long long> wgt_mem;
//   hls_bus_chn<unsigned int> out_mem;
//   hls_bus_chn<unsigned long long> bias_mem;

//   sc_signal<unsigned int> sig_start_acc;
//   sc_signal<unsigned int> sig_done_acc;
//   sc_signal<unsigned int> sig_reset_acc;

//   sc_signal<unsigned int> sig_insn_count;
//   sc_signal<unsigned int> sig_insn_addr;

//   sc_signal<unsigned int> sig_input_addr;
//   sc_signal<unsigned int> sig_weight_addr;
//   sc_signal<unsigned int> sig_bias_addr;
//   sc_signal<unsigned int> sig_output_addr;

//   sc_signal<int> sig_depth;
//   sc_signal<int> sig_crf;
//   sc_signal<int> sig_crx;
//   sc_signal<int> sig_ra;

//   int id;

//   systemC_sigs(int _id)
//       : insn_mem("insn_port", 0, 81920),
//         inp_mem("input_port", 0, 409600),
//         wgt_mem("weight_port", 0, 409600),
//         bias_mem("bias_port", 0, 409600),
//         out_mem("out_port", 0, 409600) {
//     sc_clock clk_fast("ClkFast", 1, SC_NS);
//     id = _id;
//   }
// };

// void systemC_binder(ACCNAME* acc, TB_Driver* tb_driver, int _insns_mem_size,
//                     int _uops_mem_size, int _data_mem_size, systemC_sigs* scs) {
//   acc->clock(scs->clk_fast);
//   acc->reset(scs->sig_reset);

//   acc->start_acc(scs->sig_start_acc);
//   acc->done_acc(scs->sig_done_acc);
//   acc->reset_acc(scs->sig_reset_acc);

//   acc->insn_count(scs->sig_insn_count);
//   acc->insn_addr(scs->sig_insn_addr);
//   acc->input_addr(scs->sig_input_addr);
//   acc->weight_addr(scs->sig_weight_addr);
//   acc->bias_addr(scs->sig_bias_addr);
//   acc->output_addr(scs->sig_output_addr);

//   acc->depth(scs->sig_depth);
//   acc->crf(scs->sig_crf);
//   acc->crx(scs->sig_crx);
//   acc->ra(scs->sig_ra);

//   acc->insn_port(scs->insn_mem);
//   acc->input_port(scs->inp_mem);
//   acc->weight_port(scs->wgt_mem);
//   acc->bias_port(scs->bias_mem);
//   acc->out_port(scs->out_mem);

//   tb_driver->clock(scs->clk_fast);
//   tb_driver->reset(scs->sig_reset);
// }

#endif  // SYSTEMC_INTEGRATE
