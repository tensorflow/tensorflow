#ifndef ACCNAME_H
#define ACCNAME_H

#include <systemc.h>

#define ACCNAME FC_ACC_v2_0

// #define __SYNTHESIS__

#ifndef __SYNTHESIS__
#include "tensorflow/lite/delegates/utils/secda_tflite/ap_sysc/AXI4_if.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/sysc_types.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"
#define DWAIT(x) wait(x)
#define DWAIT() wait(1)
#define DWAIT(x) wait(1)
#define DPROF(x) x
#define VLOG(X)
#else
#include "AXI4_if.h"
#define DWAIT(x)
#define DPROF(x)
#define VLOG(X)
#endif

typedef unsigned long long inp_bt;
typedef unsigned long long wgt_bt;
typedef int out_bt;
typedef unsigned long long acc_bt;
typedef sc_int<8> dat_t;
typedef sc_int<32> acc_t;

#define INP_ACCESS 8
#define WGT_ACCESS 8
#define ACC_ACCESS 2

#define INP_MEMS 4
#define WGT_MEMS 4
#define ACC_MEMS 1

#define INP_DEPTH 4096
#define WGT_DEPTH 8192
#define ACC_DEPTH 8192

#define INP_SIZE (INP_DEPTH * INP_ACCESS * INP_MEMS)
#define WGT_SIZE (WGT_DEPTH * WGT_ACCESS * WGT_MEMS)
#define ACC_SIZE (ACC_DEPTH * ACC_ACCESS * ACC_MEMS)

#define SC_INP_ELEM_BYTES_RATIO 4
#define MAX8 127
#define MIN8 -128
#define MAX32 2147483647
#define MIN32 -2147483648

#define DIVMAX 2147483648
#define POS 1073741824
#define NEG -1073741823

struct opcode {
  unsigned long long p1;
  unsigned long long p2;
  int dstride;
  int x_size;
  int y_size;
  int doffset;
  int op;

  opcode(sc_uint<64> _p1, sc_uint<64> _p2) {
    p1 = _p1;
    p2 = _p2;
    dstride = _p1.range(63, 32);
    x_size = _p1.range(31, 16);
    y_size = _p1.range(15, 0);
    doffset = _p2.range(63, 32);
    op = _p2.range(31, 0);
  }
};

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<unsigned int> start_acc;
  sc_out<unsigned int> done_acc;
  sc_in<unsigned int> reset_acc;

  sc_in<unsigned int> insn_count;
  sc_in<unsigned int> insn_addr;
  sc_in<unsigned int> input_addr;
  sc_in<unsigned int> weight_addr;
  sc_in<unsigned int> bias_addr;
  sc_in<unsigned int> output_addr;

  sc_in<int> crf;
  sc_in<int> crx;
  sc_in<int> ra;
  sc_in<int> depth;

  AXI4M_bus_port<unsigned long long> insn_port;
  AXI4M_bus_port<unsigned long long> input_port;
  AXI4M_bus_port<unsigned long long> weight_port;
  AXI4M_bus_port<unsigned long long> bias_port;
  AXI4M_bus_port<unsigned int> out_port;

  // Instantiate memories
#ifndef __SYNTHESIS__
  wgt_bt *wgt_mem1 = new wgt_bt[WGT_DEPTH];
  wgt_bt *wgt_mem2 = new wgt_bt[WGT_DEPTH];
  wgt_bt *wgt_mem3 = new wgt_bt[WGT_DEPTH];
  wgt_bt *wgt_mem4 = new wgt_bt[WGT_DEPTH];
  inp_bt *inp_mem1 = new inp_bt[INP_DEPTH];
  inp_bt *inp_mem2 = new inp_bt[INP_DEPTH];
  inp_bt *inp_mem3 = new inp_bt[INP_DEPTH];
  inp_bt *inp_mem4 = new inp_bt[INP_DEPTH];
  acc_bt *acc_mem = new acc_bt[ACC_DEPTH];
#else
  wgt_bt wgt_mem1[WGT_DEPTH];
  wgt_bt wgt_mem2[WGT_DEPTH];
  wgt_bt wgt_mem3[WGT_DEPTH];
  wgt_bt wgt_mem4[WGT_DEPTH];
  inp_bt inp_mem1[INP_DEPTH];
  inp_bt inp_mem2[INP_DEPTH];
  inp_bt inp_mem3[INP_DEPTH];
  inp_bt inp_mem4[INP_DEPTH];
  acc_bt acc_mem[ACC_DEPTH];
#endif
  out_bt out_mem[4][4];

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> wgt_load;
  sc_signal<bool, SC_MANY_WRITERS> inp_load;
  sc_signal<bool, SC_MANY_WRITERS> bias_load;
  sc_signal<bool, SC_MANY_WRITERS> gemm_wait;
  sc_signal<bool, SC_MANY_WRITERS> schedule;
  sc_signal<bool, SC_MANY_WRITERS> storing;
  sc_signal<bool, SC_MANY_WRITERS> loading;
#else
  sc_signal<bool> wgt_load;
  sc_signal<bool> inp_load;
  sc_signal<bool> bias_load;
  sc_signal<bool> gemm_wait;
  sc_signal<bool> schedule;
  sc_signal<bool> storing;
  sc_signal<bool> loading;
#endif

  sc_uint<64> wgt_insn1;
  sc_uint<64> wgt_insn2;
  sc_uint<64> inp_insn1;
  sc_uint<64> inp_insn2;
  sc_uint<64> bias_insn1;
  sc_uint<64> bias_insn2;

  // Scheduler
  sc_signal<unsigned int> wgt_block;
  sc_signal<unsigned int> inp_block;
  sc_signal<unsigned int> depth_val;

  // GEMM Unit signals
  sc_signal<unsigned int> wp_val;
  sc_signal<unsigned int> ip_val;

  // Store Unit signals
  sc_signal<unsigned int> store_doffset;
  sc_signal<unsigned int> store_dstride;
  sc_signal<unsigned int> m_off;
  sc_signal<unsigned int> n_off;
  sc_signal<bool> fetch_resetted;

  int start_count;
  int done_count;
  int ra_val;
  int crf_val;
  int crx_val;

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

#ifndef __SYNTHESIS__
  // Profiling variable
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  DataCount *macs = new DataCount("macs");
  DataCount *ins_count = new DataCount("ins_count");
  std::vector<Metric *> profiling_vars = {per_batch_cycles, macs, ins_count};
#endif

  sc_int<32> mul_s8(sc_int<8>, sc_int<8>);

  int Quantised_Multiplier(int, int, int);

  void fetch();

  void load_weights();

  void load_inputs();

  void load_bias();

  void scheduler();

  void compute();

  void store();

#ifndef __SYNTHESIS_
  void counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(
      sc_module_name
          name_) // @suppress("Class members should be properly initialized")
  : sc_module(name_), insn_port("insn_port"), input_port("input_port"),
    weight_port("weight_port"), bias_port("bias_port"), out_port("out_port")

  {
    SC_CTHREAD(fetch, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(load_weights, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(load_inputs, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(load_bias, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(store, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(scheduler, clock.pos());
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(counter, clock.pos());
    reset_signal_is(reset, true);
#endif

#pragma HLS RESET variable = reset
  }
};

#endif // ACCNAME_H
