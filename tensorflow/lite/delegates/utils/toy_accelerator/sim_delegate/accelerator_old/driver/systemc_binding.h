
// TODO Generalise this code so it is easy for all new accelerators
#ifndef SYSTEMC_INTEGRATION
#define SYSTEMC_INTEGRATION

#include <systemc.h>
#include "../acc.sc.h"
// #include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/axi4s_engine.sc.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

int sc_main(int argc, char* argv[]) { return 0; }

void sysC_init() {
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions(SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions(SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);
}

struct systemC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;

  sc_fifo<DATA> dout1;
  sc_fifo<DATA> din1;

  int id;

  systemC_sigs(int _id)
      : dout1("dout1_fifo", 563840), din1("din1_fifo", 554800) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void systemC_binder(ACCNAME* acc, stream_dma* sdma, systemC_sigs* scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->dout1(scs->dout1);
  acc->din1(scs->din1);

  sdma->dmad->clock(scs->clk_fast);
  sdma->dmad->reset(scs->sig_reset);
  sdma->dmad->dout1(scs->dout1);
  sdma->dmad->din1(scs->din1);


  // static int* dma_input_address =
  //     (int*)malloc(65536 * sizeof(int));
  // static int* dma_output_address =
  //     (int*)malloc(65536 * sizeof(int));

  // // Initialize with zeros
  // for (int64_t i = 0; i < 65536; i++) {
  //   *(dma_input_address + i) = 0;
  // }

  // for (int64_t i = 0; i < 65536; i++) {
  //   *(dma_output_address + i) = 0;
  // }

  // dmad->DMA_input_buffer = (int*)dma_input_address;
  // dmad->DMA_output_buffer = (int*)dma_output_address;
}

#endif  // SYSTEMC_INTEGRATION
