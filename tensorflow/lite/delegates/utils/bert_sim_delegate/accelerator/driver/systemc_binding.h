#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

// This file is specfic to FC-GEMM SystemC definition
// This contains all the correct port/signal bindings to instantiate the FC-GEMM accelerator
struct sysC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;

  hls_bus_chn<unsigned long long> insn_mem;
  hls_bus_chn<unsigned long long> inp_mem;
  hls_bus_chn<unsigned long long> wgt_mem;
  hls_bus_chn<unsigned int> out_mem;
  hls_bus_chn<unsigned long long> bias_mem;

  sc_signal<unsigned int> sig_start_acc;
  sc_signal<unsigned int> sig_done_acc;
  sc_signal<unsigned int> sig_reset_acc;

  sc_signal<unsigned int> sig_insn_count;
  sc_signal<unsigned int> sig_insn_addr;

  sc_signal<unsigned int> sig_input_addr;
  sc_signal<unsigned int> sig_weight_addr;
  sc_signal<unsigned int> sig_bias_addr;
  sc_signal<unsigned int> sig_output_addr;

  sc_signal<int> sig_depth;
  sc_signal<int> sig_crf;
  sc_signal<int> sig_crx;
  sc_signal<int> sig_ra;

  int id;

  sysC_sigs(int _id)
      : insn_mem("insn_port", 0, 81920),
        inp_mem("input_port", 0, 409600),
        wgt_mem("weight_port", 0, 409600),
        bias_mem("bias_port", 0, 409600),
        out_mem("out_port", 0, 409600) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void systemC_binder(ACCNAME* acc, sysC_sigs* scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);

  acc->start_acc(scs->sig_start_acc);
  acc->done_acc(scs->sig_done_acc);
  acc->reset_acc(scs->sig_reset_acc);

  acc->insn_count(scs->sig_insn_count);
  acc->insn_addr(scs->sig_insn_addr);
  acc->input_addr(scs->sig_input_addr);
  acc->weight_addr(scs->sig_weight_addr);
  acc->bias_addr(scs->sig_bias_addr);
  acc->output_addr(scs->sig_output_addr);

  acc->depth(scs->sig_depth);
  acc->crf(scs->sig_crf);
  acc->crx(scs->sig_crx);
  acc->ra(scs->sig_ra);

  acc->insn_port(scs->insn_mem);
  acc->input_port(scs->inp_mem);
  acc->weight_port(scs->wgt_mem);
  acc->bias_port(scs->bias_mem);
  acc->out_port(scs->out_mem);
}

#endif // SYSTEMC_BINDING
