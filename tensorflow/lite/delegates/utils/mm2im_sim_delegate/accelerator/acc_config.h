
#ifndef ACC_CONFIG_H2
#define ACC_CONFIG_H2
#include <systemc.h>

#ifndef __SYNTHESIS__

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/sysc_types.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"
typedef int8_t acc_dt;
#define DWAIT(x) wait(x)

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif

#else

typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;

  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;
typedef sc_int<8> acc_dt;
#define DWAIT(x)
#define ALOG(x)

#endif

#define ATOG(x)

#define ACCNAME MM2IMv1
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int

// HERE

#define INP_BUF_LEN 2048
#define WGT_BUF_LEN 2048 * 4
#define UF 16

// // Number of PEs
// #define PE_COUNT 3

// // needs to support ks * ks * depth / UF
// #define PE_WGTCOLBUF_SIZE 128

// // wgt_col_sum needs to support ks * ks
// #define PE_WGTCOLSUMBUF_SIZE 128

// // inp_row_buf needs to support depth / UF
// #define PE_INPROWBUF_SIZE 128

// // support ir * ks * ks gemm outputs where ir is the number
// // of input rows
// #define PE_OUTBUF_SIZE 128

// // max value is ks * ks
// #define PE_POUTDEXBUF_SIZE 128

// // Max number of MM2IM outputs storable per PE, should allow OH * OW
// #define PE_ACC_BUF_SIZE 2048

// Number of PEs
#define PE_COUNT 8

// needs to support ks * ks * depth / UF
#define PE_WGTCOLBUF_SIZE 512

// wgt_col_sum needs to support ks * ks
#define PE_WGTCOLSUMBUF_SIZE 64

// inp_row_buf needs to support depth / UF
#define PE_INPROWBUF_SIZE 16

// support ir * ks * ks gemm outputs where ir is the number
// of input rows
#define PE_OUTBUF_SIZE 1024

// max value is ks * ks
#define PE_POUTDEXBUF_SIZE 64

// Max number of MM2IM outputs storable per PE, should allow OH * OW
#define PE_ACC_BUF_SIZE 256

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648
#define MAX8 127
#define MIN8 -128

// TO HERE

struct opcode {
  unsigned int packet;
  bool load_con;
  bool load_wgt;
  bool load_inp;
  bool load_map;
  bool schedule;
  bool load_col_map;
  bool store;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    load_con = _packet.range(0, 0);
    load_wgt = _packet.range(1, 1);
    load_inp = _packet.range(2, 2);
    load_map = _packet.range(3, 3);
    schedule = _packet.range(4, 4);
    load_col_map = _packet.range(5, 5);
    store = _packet.range(6, 6);
  }
};

struct wgt_packet {
  unsigned int a;
  unsigned int b;
  unsigned int wgt_rows;
  unsigned int wgt_depth;

  wgt_packet(sc_fifo_in<DATA> *din) {
    ALOG("WGT_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    wgt_rows = a;
    wgt_depth = b;
  }
};

struct inp_packet {
  unsigned int a;
  unsigned int b;
  unsigned int inp_rows;
  unsigned int inp_depth;

  inp_packet(sc_fifo_in<DATA> *din) {
    ALOG("INP_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    inp_rows = a;
    inp_depth = b;
  }
};

struct PE_vars {

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> online;
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> reset_compute;
  sc_signal<int, SC_MANY_WRITERS> col_size;
  sc_signal<int, SC_MANY_WRITERS> start_addr_p;
  sc_signal<int, SC_MANY_WRITERS> send_len_p;
  sc_signal<int, SC_MANY_WRITERS> bias_data;
  sc_signal<int, SC_MANY_WRITERS> crf_data;
  sc_signal<int, SC_MANY_WRITERS> crx_data;
  sc_signal<int, SC_MANY_WRITERS> ra_data;
  sc_signal<bool, SC_MANY_WRITERS> send;
  sc_signal<bool, SC_MANY_WRITERS> out;
  sc_signal<int, SC_MANY_WRITERS> cols_per_filter;
  sc_signal<int, SC_MANY_WRITERS> depth;
  sc_signal<bool, SC_MANY_WRITERS> compute_done;
  sc_signal<bool, SC_MANY_WRITERS> wgt_loaded;
  sc_signal<bool, SC_MANY_WRITERS> out_done;
  sc_signal<bool, SC_MANY_WRITERS> send_done;

#else
  sc_signal<bool> online;
  sc_signal<bool> compute;
  sc_signal<bool> reset_compute;
  sc_signal<int> col_size;
  sc_signal<int> start_addr_p;
  sc_signal<int> send_len_p;
  sc_signal<int> bias_data;
  sc_signal<int> crf_data;
  sc_signal<int> crx_data;
  sc_signal<int> ra_data;
  sc_signal<bool> send;
  sc_signal<bool> out;
  sc_signal<int> cols_per_filter;
  sc_signal<int> depth;
  sc_signal<bool> compute_done;
  sc_signal<bool> wgt_loaded;
  sc_signal<bool> out_done;
  sc_signal<bool> send_done;
#endif

  sc_fifo<int> col_dexs_fifo;
  sc_fifo<int> dex_fifo;
  sc_fifo<char> wgt_fifo;
  sc_fifo<char> inp_fifo;
  // sc_fifo<int> out_fifo;
  sc_fifo<DATA> out_fifo;
  sc_fifo<int> temp_fifo;

  sc_out<int> computeS;
  sc_out<int> sendS;

  // static int sid;

#ifndef __SYNTHESIS__
  PE_vars(int size, int sid)
      : online((std::string("online") + std::to_string(sid)).c_str()),
        compute((std::string("compute") + std::to_string(sid)).c_str()),
        reset_compute(
            (std::string("reset_compute") + std::to_string(sid)).c_str()),
        col_size((std::string("col_size") + std::to_string(sid)).c_str()),
        start_addr_p(
            (std::string("start_addr_p") + std::to_string(sid)).c_str()),
        send_len_p((std::string("send_len_p") + std::to_string(sid)).c_str()),
        bias_data((std::string("bias_data") + std::to_string(sid)).c_str()),
        crf_data((std::string("crf_data") + std::to_string(sid)).c_str()),
        crx_data((std::string("crx_data") + std::to_string(sid)).c_str()),
        ra_data((std::string("ra_data") + std::to_string(sid)).c_str()),
        send((std::string("send") + std::to_string(sid)).c_str()),
        out((std::string("out") + std::to_string(sid)).c_str()),
        cols_per_filter(
            (std::string("cols_per_filter") + std::to_string(sid)).c_str()),
        depth((std::string("depth") + std::to_string(sid)).c_str()),
        compute_done(
            (std::string("compute_done") + std::to_string(sid)).c_str()),
        wgt_loaded((std::string("wgt_loaded") + std::to_string(sid)).c_str()),
        out_done((std::string("out_done") + std::to_string(sid)).c_str()),
        send_done((std::string("send_done") + std::to_string(sid)).c_str()),
        col_dexs_fifo(size), dex_fifo(size), wgt_fifo(size), inp_fifo(size),
        out_fifo(size), temp_fifo(size),
        computeS((std::string("computeS") + std::to_string(sid)).c_str()),
        sendS((std::string("sendS") + std::to_string(sid)).c_str()) {
    // sid++;
  }
#else
  PE_vars(int size)
      : online("online"), compute("compute"), reset_compute("reset_compute"),
        col_size("col_size"), start_addr_p("start_addr_p"),
        send_len_p("send_len_p"), bias_data("bias_data"), crf_data("crf_data"),
        crx_data("crx_data"), ra_data("ra_data"), send("send"), out("out"),
        cols_per_filter("cols_per_filter"), depth("depth"),
        compute_done("compute_done"), wgt_loaded("wgt_loaded"),
        out_done("out_done"), send_done("send_done"), col_dexs_fifo(size),
        dex_fifo(size), wgt_fifo(size), inp_fifo(size), out_fifo(size),
        temp_fifo(size), computeS("computeS"), sendS("sendS") {
    // sid++;
  }
#endif
};

// int PE_vars::sid = 0;

// struct var_array {
//   PE_vars v[PE_COUNT];
//   var_array(int size) {}
//   PE_vars &operator[](int i) { return v[i]; }
// };

#endif // ACC_CONFIG_H2