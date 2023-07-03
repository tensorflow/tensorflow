#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.h"
#include "pe_module.h"

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_out<bool> on;

  sc_out<int> inS;
  sc_out<int> data_inS;
  sc_out<int> scheduleS;
  sc_out<int> outS;
  sc_out<int> tempS;

  sc_fifo_in<DATA> din1;
  sc_fifo_in<DATA> din2;
  sc_fifo_in<DATA> din3;
  sc_fifo_in<DATA> din4;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

  ACC_DTYPE<8> wgt_buf[WGT_BUF_LEN][UF];
  ACC_DTYPE<8> inp_buf[INP_BUF_LEN][UF];

  ACC_DTYPE<32> wgt_sum_buf[INP_BUF_LEN];
  ACC_DTYPE<32> bias_buf[INP_BUF_LEN];
  ACC_DTYPE<32> crf_buf[INP_BUF_LEN];
  ACC_DTYPE<32> crx_buf[INP_BUF_LEN];

  ACC_DTYPE<32> outmap_buf[INP_BUF_LEN];
  ACC_DTYPE<32> out_starts[INP_BUF_LEN];
  ACC_DTYPE<32> out_size[INP_BUF_LEN];

  ACC_DTYPE<32> out_indices[INP_BUF_LEN];
  ACC_DTYPE<32> col_indices[INP_BUF_LEN];
  ACC_DTYPE<32> col_indice_starts[INP_BUF_LEN];
  ACC_DTYPE<32> col_indice_lens[INP_BUF_LEN];

  int pe_cols[PE_COUNT];

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> load_wgt;
  sc_signal<bool, SC_MANY_WRITERS> load_inp;
  sc_signal<bool, SC_MANY_WRITERS> load_map;
  sc_signal<bool, SC_MANY_WRITERS> load_col_map;
  sc_signal<bool, SC_MANY_WRITERS> load_data;
  sc_signal<bool, SC_MANY_WRITERS> data_in;
  sc_signal<bool, SC_MANY_WRITERS> schedule;

#else
  sc_signal<bool> load_wgt;
  sc_signal<bool> load_inp;
  sc_signal<bool> load_map;
  sc_signal<bool> load_col_map;
  sc_signal<bool> load_data;
  sc_signal<bool> data_in;
  sc_signal<bool> schedule;

#endif

  int row_size;
  int depth;
  int cols_per_filter;
  int inp_rows;
  int number_of_rows;
  int nfilters;

  int ra;

  struct var_array vars;

  // Modules
  void Input_Handler();

  void Output_Handler();

  void Data_In();

  void Scheduler();

  // Functions

  void init_PE_signals();

  void activate_PEs();

  void deactivate_PEs();

  void load_wgt_PEs();

  void load_inp_PEs();

  void store(int start, int length);

  bool wgt_loaded();

  bool compute_done();

  bool compute_resetted();

  bool store_done();

  void start_compute(int);

  void stop_compute();

  bool out_fifo_filled();


  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_), vars() {

    // Connect PE ports
    vars.init(clock, reset);

    SC_CTHREAD(Input_Handler, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Output_Handler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Data_In, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Scheduler, clock);
    reset_signal_is(reset, true);

#pragma HLS array_partition variable = wgt_buf dim = 2 complete
#pragma HLS array_partition variable = inp_buf dim = 2 complete
// #pragma HLS array_partition variable = ocols complete
#pragma HLS array_partition variable = pe_cols complete

#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA1" port_map = {                                    \
      {din1_0 TDATA } {                                                        \
        din1_1 TLAST } }
#pragma HLS RESOURCE variable = din2 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA2" port_map = {{din2_0 TDATA } {din2_1 TLAST } }
#pragma HLS RESOURCE variable = din3 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA3" port_map = {{din3_0 TDATA } {din3_1 TLAST } }
#pragma HLS RESOURCE variable = din4 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA4" port_map = {{din4_0 TDATA } {din4_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA1" port_map = {{dout1_0 TDATA } {dout1_1 TLAST } }
#pragma HLS RESOURCE variable = dout2 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA2" port_map = {{dout2_0 TDATA } {dout2_1 TLAST } }
#pragma HLS RESOURCE variable = dout3 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA3" port_map = {{dout3_0 TDATA } {dout3_1 TLAST } }
#pragma HLS RESOURCE variable = dout4 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA4" port_map = {{dout4_0 TDATA } {dout4_1 TLAST } }
#pragma HLS RESET variable = reset
  }
};

#endif /* ACCNAME_H */
