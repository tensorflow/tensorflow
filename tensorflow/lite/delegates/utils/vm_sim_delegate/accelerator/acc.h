#ifndef ACCNAME_H
#define ACCNAME_H

#include <systemc.h>
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/sysc_types.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

#define ACCNAME VM_INT8_V2_0
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int

#define IN_BUF_LEN 2048
#define WE_BUF_LEN 2048
#define GWE_BUF_LEN 8192
#define SUMS_BUF_LEN 512

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648
#define MAX8 127
#define MIN8 -128

typedef struct _ADATA {
  ACC_DTYPE<32> d2;
  ACC_DTYPE<32> d3;
  ACC_DTYPE<32> d4;
  ACC_DTYPE<32> d5;
  ACC_DTYPE<32> d6;
  ACC_DTYPE<32> d7;
  ACC_DTYPE<32> d8;
  ACC_DTYPE<32> d9;
  ACC_DTYPE<32> d10;
  ACC_DTYPE<32> d11;
  ACC_DTYPE<32> d12;
  ACC_DTYPE<32> d13;
  ACC_DTYPE<32> d14;
  ACC_DTYPE<32> d15;
  ACC_DTYPE<32> d16;
  ACC_DTYPE<32> d17;

  inline friend ostream& operator<<(ostream& os, const _ADATA& v) { return os; }
} ADATA;

SC_MODULE(ACCNAME) {
  // debug vars
  bool print_po = false;
  bool print_wo = false;

  sc_uint<14> depth;
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<DATA> din1;
  sc_fifo_in<DATA> din2;
  sc_fifo_in<DATA> din3;
  sc_fifo_in<DATA> din4;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

  sc_signal<bool> read_inputs;
  sc_signal<bool> rtake;
  sc_signal<bool> ltake;
  sc_signal<int> llen;
  sc_signal<int> rlen;
  sc_signal<int> lhs_block_max;
  sc_signal<int> rhs_block_max;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> d_in1;
  sc_signal<bool, SC_MANY_WRITERS> schedule;
  sc_signal<bool, SC_MANY_WRITERS> out_check;
  sc_signal<bool, SC_MANY_WRITERS> gemm_unit_1_ready;
  sc_signal<bool, SC_MANY_WRITERS> gemm_unit_2_ready;
  sc_signal<bool, SC_MANY_WRITERS> gemm_unit_3_ready;
  sc_signal<bool, SC_MANY_WRITERS> gemm_unit_4_ready;
  sc_signal<bool, SC_MANY_WRITERS> write1;
  sc_signal<bool, SC_MANY_WRITERS> write2;
  sc_signal<bool, SC_MANY_WRITERS> write3;
  sc_signal<bool, SC_MANY_WRITERS> write4;

  sc_signal<bool, SC_MANY_WRITERS> arrange1;
  sc_signal<bool, SC_MANY_WRITERS> arrange2;
  sc_signal<bool, SC_MANY_WRITERS> arrange3;
  sc_signal<bool, SC_MANY_WRITERS> arrange4;
  sc_signal<bool, SC_MANY_WRITERS> write1_1;
  sc_signal<bool, SC_MANY_WRITERS> write1_2;
  sc_signal<bool, SC_MANY_WRITERS> write1_3;
  sc_signal<bool, SC_MANY_WRITERS> write1_4;
  sc_signal<bool, SC_MANY_WRITERS> write2_1;
  sc_signal<bool, SC_MANY_WRITERS> write2_2;
  sc_signal<bool, SC_MANY_WRITERS> write2_3;
  sc_signal<bool, SC_MANY_WRITERS> write2_4;
  sc_signal<bool, SC_MANY_WRITERS> write3_1;
  sc_signal<bool, SC_MANY_WRITERS> write3_2;
  sc_signal<bool, SC_MANY_WRITERS> write3_3;
  sc_signal<bool, SC_MANY_WRITERS> write3_4;
  sc_signal<bool, SC_MANY_WRITERS> write4_1;
  sc_signal<bool, SC_MANY_WRITERS> write4_2;
  sc_signal<bool, SC_MANY_WRITERS> write4_3;
  sc_signal<bool, SC_MANY_WRITERS> write4_4;
#else
  sc_signal<bool> d_in1;
  sc_signal<bool> schedule;
  sc_signal<bool> out_check;
  sc_signal<bool> gemm_unit_1_ready;
  sc_signal<bool> gemm_unit_2_ready;
  sc_signal<bool> gemm_unit_3_ready;
  sc_signal<bool> gemm_unit_4_ready;
  sc_signal<bool> write1;
  sc_signal<bool> write2;
  sc_signal<bool> write3;
  sc_signal<bool> write4;

  sc_signal<bool> arrange1;
  sc_signal<bool> arrange2;
  sc_signal<bool> arrange3;
  sc_signal<bool> arrange4;
  sc_signal<bool> write1_1;
  sc_signal<bool> write1_2;
  sc_signal<bool> write1_3;
  sc_signal<bool> write1_4;
  sc_signal<bool> write2_1;
  sc_signal<bool> write2_2;
  sc_signal<bool> write2_3;
  sc_signal<bool> write2_4;
  sc_signal<bool> write3_1;
  sc_signal<bool> write3_2;
  sc_signal<bool> write3_3;
  sc_signal<bool> write3_4;
  sc_signal<bool> write4_1;
  sc_signal<bool> write4_2;
  sc_signal<bool> write4_3;
  sc_signal<bool> write4_4;
#endif

  sc_signal<int> gemm_unit_1_l_pointer;
  sc_signal<int> gemm_unit_2_l_pointer;
  sc_signal<int> gemm_unit_3_l_pointer;
  sc_signal<int> gemm_unit_4_l_pointer;

  sc_signal<bool> gemm_unit_1_iwuse;
  sc_signal<bool> gemm_unit_2_iwuse;
  sc_signal<bool> gemm_unit_3_iwuse;
  sc_signal<bool> gemm_unit_4_iwuse;

  //   ADATA g1;
  //   ADATA g2;
  //   ADATA g3;
  //   ADATA g4;

  //   ADATA r1;
  //   ADATA r2;
  //   ADATA r3;
  //   ADATA r4;

  ACC_DTYPE<32> g1[16];
  ACC_DTYPE<32> g2[16];
  ACC_DTYPE<32> g3[16];
  ACC_DTYPE<32> g4[16];

  ACC_DTYPE<8> r1[16];
  ACC_DTYPE<8> r2[16];
  ACC_DTYPE<8> r3[16];
  ACC_DTYPE<8> r4[16];

  // GEMM 1 Inputs
  ACC_DTYPE<32> lhsdata1a[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata2a[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata3a[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata4a[IN_BUF_LEN];

  // GEMM 2 Inputs
  ACC_DTYPE<32> lhsdata1b[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata2b[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata3b[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata4b[IN_BUF_LEN];

  // GEMM 3 Inputs
  ACC_DTYPE<32> lhsdata1c[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata2c[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata3c[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata4c[IN_BUF_LEN];

  // GEMM 4 Inputs
  ACC_DTYPE<32> lhsdata1d[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata2d[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata3d[IN_BUF_LEN];
  ACC_DTYPE<32> lhsdata4d[IN_BUF_LEN];

  // Global Weights
  ACC_DTYPE<32> rhsdata1[GWE_BUF_LEN];
  ACC_DTYPE<32> rhsdata2[GWE_BUF_LEN];
  ACC_DTYPE<32> rhsdata3[GWE_BUF_LEN];
  ACC_DTYPE<32> rhsdata4[GWE_BUF_LEN];

  // First Set (A)
  ACC_DTYPE<32> rhs1a_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs1b_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs1c_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs1d_1[WE_BUF_LEN];

  ACC_DTYPE<32> rhs2a_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs2b_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs2c_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs2d_1[WE_BUF_LEN];

  ACC_DTYPE<32> rhs3a_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs3b_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs3c_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs3d_1[WE_BUF_LEN];

  ACC_DTYPE<32> rhs4a_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs4b_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs4c_1[WE_BUF_LEN];
  ACC_DTYPE<32> rhs4d_1[WE_BUF_LEN];

  // new sums bram
  ACC_DTYPE<32> lhs_sum1[SUMS_BUF_LEN];
  ACC_DTYPE<32> lhs_sum2[SUMS_BUF_LEN];
  ACC_DTYPE<32> lhs_sum3[SUMS_BUF_LEN];
  ACC_DTYPE<32> lhs_sum4[SUMS_BUF_LEN];

  ACC_DTYPE<32> rhs_sum1[SUMS_BUF_LEN];
  ACC_DTYPE<32> rhs_sum2[SUMS_BUF_LEN];
  ACC_DTYPE<32> rhs_sum3[SUMS_BUF_LEN];
  ACC_DTYPE<32> rhs_sum4[SUMS_BUF_LEN];

  // crf & crx
  ACC_DTYPE<32> crf1[SUMS_BUF_LEN];
  ACC_DTYPE<32> crf2[SUMS_BUF_LEN];
  ACC_DTYPE<32> crf3[SUMS_BUF_LEN];
  ACC_DTYPE<32> crf4[SUMS_BUF_LEN];
  ACC_DTYPE<32> crx[SUMS_BUF_LEN];
  int ra = 0;

  sc_fifo<int> WRQ1;
  sc_fifo<int> WRQ2;
  sc_fifo<int> WRQ3;
  sc_fifo<int> WRQ4;

  sc_signal<int> w1S;
  sc_signal<int> w2S;
  sc_signal<int> w3S;
  sc_signal<int> w4S;

  sc_out<int> inS;
  sc_out<int> read_cycle_count;
  sc_out<int> process_cycle_count;
  sc_out<int> gemm_1_idle;
  sc_out<int> gemm_2_idle;
  sc_out<int> gemm_3_idle;
  sc_out<int> gemm_4_idle;
  sc_out<int> gemm_1_write;
  sc_out<int> gemm_2_write;
  sc_out<int> gemm_3_write;
  sc_out<int> gemm_4_write;
  sc_out<int> gemm_1;
  sc_out<int> gemm_2;
  sc_out<int> gemm_3;
  sc_out<int> gemm_4;
  sc_out<int> wstall_1;
  sc_out<int> wstall_2;
  sc_out<int> wstall_3;
  sc_out<int> wstall_4;

  sc_out<int> rmax;
  sc_out<int> lmax;

  sc_out<int> outS;
  sc_out<int> w1SS;
  sc_out<int> w2SS;
  sc_out<int> w3SS;
  sc_out<int> w4SS;
  sc_out<int> schS;
  sc_out<int> p1S;

  // Profiling variable
  ClockCycles* read_cycles = new ClockCycles("read_cycles", true);
  ClockCycles* process_cycles = new ClockCycles("process_cycles", true);
  ClockCycles* idle1 = new ClockCycles("idle1", true);
  ClockCycles* idle2 = new ClockCycles("idle2", true);
  ClockCycles* idle3 = new ClockCycles("idle3", true);
  ClockCycles* idle4 = new ClockCycles("idle4", true);
  ClockCycles* gemmw1 = new ClockCycles("gemmw1", true);
  ClockCycles* gemmw2 = new ClockCycles("gemmw2", true);
  ClockCycles* gemmw3 = new ClockCycles("gemmw3", true);
  ClockCycles* gemmw4 = new ClockCycles("gemmw4", true);
  ClockCycles* gemm1 = new ClockCycles("gemm1", true);
  ClockCycles* gemm2 = new ClockCycles("gemm2", true);
  ClockCycles* gemm3 = new ClockCycles("gemm3", true);
  ClockCycles* gemm4 = new ClockCycles("gemm4", true);
  ClockCycles* wstall1 = new ClockCycles("wstall1", true);
  ClockCycles* wstall2 = new ClockCycles("wstall2", true);
  ClockCycles* wstall3 = new ClockCycles("wstall3", true);
  ClockCycles* wstall4 = new ClockCycles("wstall4", true);
  BufferSpace* gweightbuf_p = new BufferSpace("gweightbuf_p", GWE_BUF_LEN);
  BufferSpace* inputbuf_p = new BufferSpace("inputbuf_p", IN_BUF_LEN);
  BufferSpace* weightbuf_p = new BufferSpace("weightbuf_p", WE_BUF_LEN);
  DataCountArray* gmacs = new DataCountArray("gmacs", 4);
  DataCountArray* gouts = new DataCountArray("gouts", 4);

  std::vector<Metric*> profiling_vars = {
      read_cycles,  process_cycles, idle1,       idle2,   idle4,   idle1,
      gemmw1,       gemmw2,         gemmw3,      gemmw4,  gemm1,   gemm2,
      gemm3,        gemm4,          wstall1,     wstall2, wstall3, wstall4,
      gweightbuf_p, inputbuf_p,     weightbuf_p, gmacs,   gouts,
  };

  void Input_Handler();

  void Output_Handler();

  void Worker1();

  void Worker2();

  void Worker3();

  void Worker4();

  void Data_In();

  void Tracker();

  void Scheduler();

  void Post1();

  void Post2();

  void Post3();

  void Post4();

  void Arranger1();

  void Arranger2();

  void Arranger3();

  void Arranger4();

  void WSync1();

  void WSync2();

  void WSync3();

  void WSync4();

  void load_weights(int, int);

  void schedule_gemm_unit(int, int, int, int);

  int SHR(int, int);

  void VM_PE(ACC_DTYPE<32>*, ACC_DTYPE<32>*, ACC_DTYPE<32>*, ACC_DTYPE<32>*,
             ACC_DTYPE<32>*, ACC_DTYPE<32>*, ACC_DTYPE<32>*, ACC_DTYPE<32>*,
             ACC_DTYPE<32>[][4], int, int, int);

  int Quantised_Multiplier(int, int, sc_int<8>);

  void PPU(int*, int*, int*, sc_int<8>*, ACC_DTYPE<32>*, ACC_DTYPE<8>*);

  void overwrite_weights_check();

  void Read_Cycle_Counter();

  void Process_Cycle_Counter();

  void Writer_Cycle_Counter();

  sc_int<32> mul_s8(sc_int<8>, sc_int<8>);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_)
      : sc_module(name_), WRQ1(512), WRQ2(512), WRQ3(512), WRQ4(512) {
    SC_CTHREAD(Input_Handler, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Worker1, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Worker2, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Worker3, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Worker4, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Output_Handler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Data_In, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Scheduler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Post1, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Post2, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Post3, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Post4, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WSync1, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WSync2, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WSync3, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(WSync4, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Arranger1, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Arranger2, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Arranger3, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Arranger4, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Read_Cycle_Counter, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Process_Cycle_Counter, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Writer_Cycle_Counter, clock);
    reset_signal_is(reset, true);

#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata = \
    "-bus_bundle S_AXIS_DATA1" port_map = {                       \
      {din1_0 TDATA } {                                           \
        din1_1 TLAST } }
#pragma HLS RESOURCE variable = din2 core = AXI4Stream metadata = \
    "-bus_bundle S_AXIS_DATA2" port_map = {                       \
      {din2_0 TDATA } {                                           \
        din2_1 TLAST } }
#pragma HLS RESOURCE variable = din3 core = AXI4Stream metadata = \
    "-bus_bundle S_AXIS_DATA3" port_map = {                       \
      {din3_0 TDATA } {                                           \
        din3_1 TLAST } }
#pragma HLS RESOURCE variable = din4 core = AXI4Stream metadata = \
    "-bus_bundle S_AXIS_DATA4" port_map = {                       \
      {din4_0 TDATA } {                                           \
        din4_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata = \
    "-bus_bundle M_AXIS_DATA1" port_map = {                        \
      {dout1_0 TDATA } {                                           \
        dout1_1 TLAST } }
#pragma HLS RESOURCE variable = dout2 core = AXI4Stream metadata = \
    "-bus_bundle M_AXIS_DATA2" port_map = {                        \
      {dout2_0 TDATA } {                                           \
        dout2_1 TLAST } }
#pragma HLS RESOURCE variable = dout3 core = AXI4Stream metadata = \
    "-bus_bundle M_AXIS_DATA3" port_map = {                        \
      {dout3_0 TDATA } {                                           \
        dout3_1 TLAST } }
#pragma HLS RESOURCE variable = dout4 core = AXI4Stream metadata = \
    "-bus_bundle M_AXIS_DATA4" port_map = {                        \
      {dout4_0 TDATA } {                                           \
        dout4_1 TLAST } }
#pragma HLS RESET variable = reset
  }
};
#endif /* ACCNAME_H */
