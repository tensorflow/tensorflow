// TODO - Clean up, define this reason for this file

#ifndef ACC_HELPERS
#define ACC_HELPERS

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <typeinfo>

#include "multi_threading.h"

struct times {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> t;
  std::string name;

  times(std::string name) : name(name) {}

  void print() {
    std::cout << name << ": ";
    std::cout
        << std::chrono::duration_cast<std::chrono::milliseconds>(t).count();
  }
};

struct del_params {
  bool init;
  bool unmap;
  int delegated_nodes;
  int layer;
  int start_count;
  int* acc;

  struct MultiThreadContext mt_context;
  struct times total;

  del_params() : total("total"), mt_context() {
    init = false;
    unmap = false;
    delegated_nodes = 0;
    layer = 0;
    start_count = 0;
    // mt_context = new MultiThreadContext();
  };
};

struct DSR {
  int dID = 0;
  int sID = 0;
  int rID = 0;

  void reset() {
    dID = 0;
    sID = 0;
    rID = 0;
  };
};

struct dma_buffer {
  int offset = 0;
  int len = 0;
  bool in_use = false;
  int id = -1;
};

struct dma_buffer_set {
  struct dma_buffer* dbuf_set;
  int count;
  int buf_size;
  unsigned int base_address;

  dma_buffer_set(int _count, int _size, unsigned int _base_address) {
    count = _count;
    buf_size = _size;
    base_address = _base_address;
    dbuf_set = new dma_buffer[count];
    for (int i = 0; i < count; i++) dbuf_set[i].offset = buf_size * i;
  }
  void free() {
    delete[] dbuf_set;
  }
};

void alloc_dbuf(dma_buffer_set& dfs, int bufdex, int newID, int len) {
  dfs.dbuf_set[bufdex].len = len;
  dfs.dbuf_set[bufdex].in_use = true;
  dfs.dbuf_set[bufdex].id = newID;
}

void dealloc_dbuf(dma_buffer_set& dfs, int bufdex) {
  dfs.dbuf_set[bufdex].len = 0;
  dfs.dbuf_set[bufdex].in_use = false;
  dfs.dbuf_set[bufdex].id = -1;
}

int check_for_free_dbuf(dma_buffer_set& dfs) {
  for (int i = 0; i < dfs.count; i++) {
    if (!dfs.dbuf_set[i].in_use) return i;
  }
  return -1;
}

int find_dbuf(dma_buffer_set& dfs, int ID) {
  for (int i = 0; i < dfs.count; i++) {
    if (dfs.dbuf_set[i].id == ID) return i;
  }
  return -1;
}

int dbufs_in_use(dma_buffer_set& dfs) {
  int count = 0;
  for (int i = 0; i < dfs.count; i++) {
    if (dfs.dbuf_set[i].in_use) count++;
  }
  return count;
}

// int Check_For_Free_Buffer(conv2d_driver& drv) {
//   for (int i = 0; i < drv.bufflen; i++) {
//     if (!drv.dinb[i].in_use) return i;
//   }
//   return -1;
// }

// int Find_Buff(conv2d_driver& drv, int ID) {
//   for (int i = 0; i < drv.bufflen; i++) {
//     if (drv.dinb[i].id == ID) return i;
//   }
//   return -1;
// }

// bool Check_Done(conv2d_driver& drv) {
//   return (drv.mdma->multi_dma_check_recv() == 0);
// }

// int In_Use_Count(conv2d_driver& drv) {
//   int count = 0;
//   for (int i = 0; i < drv.bufflen; i++) {
//     if (drv.dinb[i].in_use) count++;
//   }
//   return count;
// }

// void End_Transfer(conv2d_driver& drv) { drv.mdma->multi_dma_wait_send(); }

// void Start_Transfer(conv2d_driver& drv) {
//   int s_buf = Find_Buff(drv, drv.sID);
//   drv.mdma->multi_dma_change_start_4(drv.dinb[s_buf].offset);
//   drv.mdma->dmas[0].inl = drv.dinb[s_buf].inl0;
//   drv.mdma->dmas[1].inl = drv.dinb[s_buf].inl1;
//   drv.mdma->dmas[2].inl = drv.dinb[s_buf].inl2;
//   drv.mdma->dmas[3].inl = drv.dinb[s_buf].inl3;
//   drv.mdma->multi_dma_start_send();
//   End_Transfer(drv);
//   drv.sID++;
// }

// void Set_Results(conv2d_driver& drv) {
//   int s_buf = Find_Buff(drv, drv.sID);
//   drv.mdma->multi_dma_change_end(drv.dinb[s_buf].offset);
//   drv.mdma->multi_dma_start_recv(drv.recv_len);
//   // drv.mdma->multi_dma_start_recv(100000);
// }

// void Recieve_Results(conv2d_driver& drv) { drv.mdma->multi_dma_wait_recv_4();
// }

// void saveData(conv2d_driver& drv, bool inputs, int inl0, int inl1, int inl2,
//               int inl3) {
//   ofstream wrin0;
//   ofstream wrin1;
//   ofstream wrin2;
//   ofstream wrin3;
//   string filename = "model" + std::to_string(drv.t.layer) + "_w" +
//                     std::to_string(drv.t.layer_weight_tile);
//   if (inputs)
//     filename = "model" + std::to_string(drv.t.layer) + "_w" +
//                std::to_string(drv.t.layer_weight_tile) + "_i" +
//                std::to_string(drv.t.layer_input_tile);
//   wrin0.open("aData/" + filename + "_0.txt");
//   wrin1.open("aData/" + filename + "_1.txt");
//   wrin2.open("aData/" + filename + "_2.txt");
//   wrin3.open("aData/" + filename + "_3.txt");
//   for (int i = 0; i < inl0; i++) {
//     wrin0 << drv.mdma->dmas[0].dma_get_inbuffer() << "\n";
//     if (i < inl1) wrin1 << drv.mdma->dmas[1].dma_get_inbuffer() << "\n";
//     if (i < inl2) wrin2 << drv.mdma->dmas[2].dma_get_inbuffer() << "\n";
//     if (i < inl3) wrin3 << drv.mdma->dmas[3].dma_get_inbuffer() << "\n";
//   }
//   wrin0.close();
//   wrin1.close();
//   wrin2.close();
//   wrin3.close();
// }

#endif