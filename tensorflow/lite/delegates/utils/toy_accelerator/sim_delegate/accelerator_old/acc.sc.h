#ifndef ACCNAME_H
#define ACCNAME_H

#include <systemc.h>
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/sysc_types.h"

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

#define ACCNAME TOY_ADD
#define ACC_DTYPE sc_uint
#define ACC_C_DTYPE unsigned int
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
  ACC_DTYPE<32> A1[IN_BUF_LEN];
  ACC_DTYPE<32> A2[IN_BUF_LEN];

  ACC_DTYPE<32> B1[IN_BUF_LEN];
  ACC_DTYPE<32> B2[IN_BUF_LEN];

  ACC_DTYPE<32> C1[IN_BUF_LEN * 4];
  ACC_DTYPE<32> C2[IN_BUF_LEN * 4];

  int qm;
  sc_int<8> shift;

#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> lenX;
  sc_signal<int, SC_MANY_WRITERS> lenY;
  sc_signal<bool, SC_MANY_WRITERS> computeX;
  sc_signal<bool, SC_MANY_WRITERS> computeY;
  sc_signal<bool, SC_MANY_WRITERS> readX;
  sc_signal<bool, SC_MANY_WRITERS> readY;
  sc_signal<bool, SC_MANY_WRITERS> writeX;
  sc_signal<bool, SC_MANY_WRITERS> writeY;
#else
  sc_signal<int> lenX;
  sc_signal<int> lenY;
  sc_signal<bool> computeX;
  sc_signal<bool> computeY;
  sc_signal<bool> readX;
  sc_signal<bool> readY;
  sc_signal<bool> writeX;
  sc_signal<bool> writeY;
#endif

  void Control();

  void Data_Read();

  void PE_Add();

  void Data_Write();

  int Quantised_Multiplier(int, int, sc_int<8>);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Control, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Data_Read, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(PE_Add, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Data_Write, clock);
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