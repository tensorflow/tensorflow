
#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

struct sysC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  hls_bus_chn<unsigned long long> insns_mem;
  hls_bus_chn<unsigned int> uops_mem;
  hls_bus_chn<unsigned long long> data_mem;
  sc_signal<unsigned int> sig_start;
  sc_signal<unsigned int> sig_vta_done;
  sc_signal<unsigned int> sig_reset_vta;
  sc_signal<unsigned int> sig_insn_count;
  sc_signal<unsigned int> sig_ins_addr;
  sc_signal<unsigned int> sig_uops_addr;
  sc_signal<unsigned int> sig_input_addr;
  sc_signal<unsigned int> sig_weight_addr;
  sc_signal<unsigned int> sig_bias_addr;
  sc_signal<unsigned int> sig_output_addr;
  sc_signal<unsigned int> sig_crf_addr;
  sc_signal<unsigned int> sig_crx_addr;
  sc_signal<unsigned int> sig_ra_sig;
  sc_signal<bool> sig_flipped;
  int id;

  sysC_sigs(int _id)
      : insns_mem("insns", 0, 40960),
        uops_mem("uops", 0, 8192),
        data_mem("data", 0, 4096000) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void systemC_binder(ACCNAME* acc, sysC_sigs* scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->start(scs->sig_start);
  acc->vta_done(scs->sig_vta_done);
  acc->reset_vta(scs->sig_reset_vta);
  acc->insn_count(scs->sig_insn_count);
  acc->ins_addr(scs->sig_ins_addr);
  acc->uops_addr(scs->sig_uops_addr);
  acc->input_addr(scs->sig_input_addr);
  acc->weight_addr(scs->sig_weight_addr);
  acc->bias_addr(scs->sig_bias_addr);
  acc->output_addr(scs->sig_output_addr);
  acc->crf_addr(scs->sig_crf_addr);
  acc->crx_addr(scs->sig_crx_addr);
  acc->ra_sig(scs->sig_ra_sig);
  acc->flipped(scs->sig_flipped);
  acc->insns(scs->insns_mem);
  acc->uops(scs->uops_mem);
  acc->data(scs->data_mem);
}

#endif  // SYSTEMC_BINDING
