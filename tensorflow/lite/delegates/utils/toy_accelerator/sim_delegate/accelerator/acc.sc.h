#ifndef ACCNAME_H
#define ACCNAME_H

#include <systemc.h>
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/sysc_types.h"

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

#define ACCNAME TOY_ADD
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define STOPPER -1

#define IN_BUF_LEN 4096
#define WE_BUF_LEN 8192
#define SUMS_BUF_LEN 1024

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648

#define MAX8 127
#define MIN8 -128

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

  // GEMM 1 Inputs
  int lshift;

  int in1_off;
  int in1_sv;
  int in1_mul;

  int in2_off;
  int in2_sv;
  int in2_mul;

  int out1_off;
  int out1_sv;
  int out1_mul;

  int qa_max;
  int qa_min;

#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> computeS;
#else
  sc_signal<int> computeS;
#endif

  // Profiling variable
  ClockCycles* per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles* active_cycles = new ClockCycles("active_cycles", true);
  std::vector<Metric*> profiling_vars = {per_batch_cycles, active_cycles};

  void Compute();

  void Counter();

  int Quantised_Multiplier(int, int, int);

  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Counter, clock);
    reset_signal_is(reset, true);

#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata = \
    "-bus_bundle S_AXIS_DATA1" port_map = {                       \
      {din1_0 TDATA } {                                           \
        din1_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata = \
    "-bus_bundle M_AXIS_DATA1" port_map = {                        \
      {dout1_0 TDATA } {                                           \
        dout1_1 TLAST } }
  }
};

#endif