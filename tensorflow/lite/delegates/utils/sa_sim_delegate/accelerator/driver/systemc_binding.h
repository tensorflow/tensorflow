#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

// This file is specfic to SA SystemC definition
// This contains all the correct port/signal bindings to instantiate the SA accelerator
struct sysC_sigs {
  int id;
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  sc_signal<int> sig_inS;
  sc_signal<int> sig_outS;
  sc_signal<int> sig_w1S;
  sc_signal<int> sig_schS;
  sc_signal<int> sig_p1S;
  sc_signal<int> sig_rmax;
  sc_signal<int> sig_lmax;

  sc_fifo<DATA> dout1;
  sc_fifo<DATA> dout2;
  sc_fifo<DATA> dout3;
  sc_fifo<DATA> dout4;

  sc_fifo<DATA> din1;
  sc_fifo<DATA> din2;
  sc_fifo<DATA> din3;
  sc_fifo<DATA> din4;

  sysC_sigs(int id_)
      : dout1("dout1_fifo", 563840),
        dout2("dout2_fifo", 563840),
        dout3("dout3_fifo", 563840),
        dout4("dout4_fifo", 563840),
        din1("din1_fifo", 554800),
        din2("din2_fifo", 554800),
        din3("din3_fifo", 554800),
        din4("din4_fifo", 554800) {
    id = id_;
    sc_clock clk_fast("ClkFast", 1, SC_NS);
  }
};

void sysC_binder(ACCNAME* acc, multi_dma* mdma, sysC_sigs* scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->inS(scs->sig_inS);
  acc->outS(scs->sig_outS);
  acc->w1SS(scs->sig_w1S);
  acc->schS(scs->sig_schS);
  acc->p1S(scs->sig_p1S);
  acc->rmax(scs->sig_rmax);
  acc->lmax(scs->sig_lmax);

  for (int i = 0; i < mdma->dma_count; i++) {
    mdma->dmas[i].dmad->clock(scs->clk_fast);
    mdma->dmas[i].dmad->reset(scs->sig_reset);
  }
  mdma->dmas[0].dmad->dout1(scs->dout1);
  mdma->dmas[1].dmad->dout1(scs->dout2);
  mdma->dmas[2].dmad->dout1(scs->dout3);
  mdma->dmas[3].dmad->dout1(scs->dout4);
  mdma->dmas[0].dmad->din1(scs->din1);
  mdma->dmas[1].dmad->din1(scs->din2);
  mdma->dmas[2].dmad->din1(scs->din3);
  mdma->dmas[3].dmad->din1(scs->din4);

  acc->dout1(scs->dout1);
  acc->dout2(scs->dout2);
  acc->dout3(scs->dout3);
  acc->dout4(scs->dout4);
  acc->din1(scs->din1);
  acc->din2(scs->din2);
  acc->din3(scs->din3);
  acc->din4(scs->din4);
}

#endif  // SYSTEMC_BINDING