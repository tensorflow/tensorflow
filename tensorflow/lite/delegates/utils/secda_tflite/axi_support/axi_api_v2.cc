#ifndef SYSC

#include "axi_api_v2.h"
// ================================================================================
// AXI4Lite API
// ================================================================================
acc_regmap::acc_regmap(size_t base_addr, size_t length) {
  acc_addr = getAccBaseAddress<int>(base_addr, length);
}

void acc_regmap::writeAccReg(uint32_t offset, unsigned int val) {
  void* base_addr = (void*)acc_addr;
  *((volatile unsigned int*)(reinterpret_cast<char*>(base_addr) + offset)) =
      val;
}

unsigned int acc_regmap::readAccReg(uint32_t offset) {
  void* base_addr = (void*)acc_addr;
  return *(
      (volatile unsigned int*)(reinterpret_cast<char*>(base_addr) + offset));
}

// TODO: parse JSON file to load offset map for control and status registers
void acc_regmap::parseOffsetJSON() {}

// TODO: checks control and status register arrays to find the offsets for the
// register
uint32_t acc_regmap::findRegOffset(string reg_name) {
  uint32_t offset = 0;
  return offset;
}

void acc_regmap::writeToControlReg(string reg_name, unsigned int val) {
  writeAccReg(findRegOffset(reg_name), val);
}

unsigned int acc_regmap::readToControlReg(string reg_name) {
  return readAccReg(findRegOffset(reg_name));
}

// ================================================================================
// Memory Map API
// ================================================================================

// Make this into a struct based API

// ================================================================================
// Stream DMA API
// ================================================================================
int stream_dma::s_id = 0;

stream_dma::stream_dma(unsigned int _dma_addr, unsigned int _input,
                       unsigned int _input_size, unsigned int _output,
                       unsigned int _output_size)
    : id(s_id++) {
  dma_init(_dma_addr, _input, _input_size, _output, _output_size);
}

stream_dma::stream_dma() : id(s_id++){};

void stream_dma::dma_init(unsigned int _dma_addr, unsigned int _input,
                          unsigned int _input_size, unsigned int _output,
                          unsigned int _output_size) {
  dma_addr = mm_alloc_rw<unsigned int>(_dma_addr, PAGE_SIZE);
  input = mm_alloc_rw<int>(_input, _input_size);
  output = mm_alloc_r<int>(_output, _output_size);
  input_size = _input_size;
  output_size = _output_size;
  input_addr = _input;
  output_addr = _output;
  initDMA(_input, _output);
}

void stream_dma::writeMappedReg(uint32_t offset, unsigned int val) {
  void* base_addr = (void*)dma_addr;
  *((volatile unsigned int*)(reinterpret_cast<char*>(base_addr) + offset)) =
      val;
}

unsigned int stream_dma::readMappedReg(uint32_t offset) {
  void* base_addr = (void*)dma_addr;
  return *(
      (volatile unsigned int*)(reinterpret_cast<char*>(base_addr) + offset));
}

void stream_dma::dma_mm2s_sync() {
  msync(dma_addr, PAGE_SIZE, MS_SYNC);
  unsigned int mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  while (!(mm2s_status & 1 << 12) || !(mm2s_status & 1 << 1)) {
    msync(dma_addr, PAGE_SIZE, MS_SYNC);
    mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  }
}

void stream_dma::dma_s2mm_sync() {
  msync(dma_addr, PAGE_SIZE, MS_SYNC);
  unsigned int s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  while (!(s2mm_status & 1 << 12) || !(s2mm_status & 1 << 1)) {
    msync(dma_addr, PAGE_SIZE, MS_SYNC);
    s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  }
}

void stream_dma::dma_change_start(int offset) {
  writeMappedReg(MM2S_START_ADDRESS, input_addr + offset);
}

void stream_dma::dma_change_end(int offset) {
  writeMappedReg(S2MM_DESTINATION_ADDRESS, output_addr + offset);
}

void stream_dma::initDMA(unsigned int src, unsigned int dst) {
  writeMappedReg(S2MM_CONTROL_REGISTER, 4);
  writeMappedReg(MM2S_CONTROL_REGISTER, 4);
  writeMappedReg(S2MM_CONTROL_REGISTER, 0);
  writeMappedReg(MM2S_CONTROL_REGISTER, 0);
  writeMappedReg(S2MM_DESTINATION_ADDRESS, dst);
  writeMappedReg(MM2S_START_ADDRESS, src);
  writeMappedReg(S2MM_CONTROL_REGISTER, 0xf001);
  writeMappedReg(MM2S_CONTROL_REGISTER, 0xf001);
}

void stream_dma::dma_free() {
  munmap(input, input_size);
  munmap(output, output_size);
  munmap(dma_addr, getpagesize());
}

int* stream_dma::dma_get_inbuffer() { return input; }

int* stream_dma::dma_get_outbuffer() { return output; }

void stream_dma::dma_start_send(int length) {
  msync(input, input_size, MS_SYNC);
  writeMappedReg(MM2S_LENGTH, length * 4);
}

void stream_dma::dma_wait_send() { dma_mm2s_sync(); }

int stream_dma::dma_check_send() {
  unsigned int mm2s_status = readMappedReg(MM2S_STATUS_REGISTER);
  bool done = !((!(mm2s_status & 1 << 12)) || (!(mm2s_status & 1 << 1)));
  return done ? 0 : -1;
}

void stream_dma::dma_start_recv(int length) {
  writeMappedReg(S2MM_LENGTH, length * 4);
}

void stream_dma::dma_wait_recv() {
  dma_s2mm_sync();
  msync(output, output_size, MS_SYNC);
}

int stream_dma::dma_check_recv() {
  unsigned int s2mm_status = readMappedReg(S2MM_STATUS_REGISTER);
  bool done = !((!(s2mm_status & 1 << 12)) || (!(s2mm_status & 1 << 1)));
  return done ? 0 : -1;
}

// =========================== Multi DMAs
multi_dma::multi_dma(int _dma_count, unsigned int* _dma_addrs,
                     unsigned int* _dma_addrs_in, unsigned int* _dma_addrs_out,
                     unsigned int _buffer_size) {
  dma_count = _dma_count;
  dmas = new stream_dma[dma_count];
  dma_addrs = _dma_addrs;
  dma_addrs_in = _dma_addrs_in;
  dma_addrs_out = _dma_addrs_out;
  buffer_size = _buffer_size;

  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_init(dma_addrs[i], dma_addrs_in[i], buffer_size * 1,
                     dma_addrs_out[i], buffer_size * 1);
}

void multi_dma::multi_free_dmas() {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_free();
  }
  delete[] dmas;
}

void multi_dma::multi_dma_change_start(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_start(offset);
  }
}

void multi_dma::multi_dma_change_start_4(int offset) {
  dmas[0].dma_change_start(offset);
  dmas[1].dma_change_start(offset);
  dmas[2].dma_change_start(offset);
  dmas[3].dma_change_start(offset);
}

void multi_dma::multi_dma_change_end(int offset) {
  for (int i = 0; i < dma_count; i++) {
    dmas[i].dma_change_end(offset);
  }
}

void multi_dma::multi_dma_start_send(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_send(length);
}

void multi_dma::multi_dma_wait_send() {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_mm2s_sync();
}

int multi_dma::multi_dma_check_send() {
  bool done = true;
  for (int i = 0; i < dma_count; i++)
    done = done && (dmas[i].dma_check_send() == 0);
  return done ? 0 : -1;
}

void multi_dma::multi_dma_start_recv(int length) {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_start_recv(length);
}

void multi_dma::multi_dma_start_recv() {
  for (int i = 0; i < dma_count; i++)
    dmas[i].dma_start_recv(dmas[i].output_size);
}

void multi_dma::multi_dma_wait_recv() {
  for (int i = 0; i < dma_count; i++) dmas[i].dma_s2mm_sync();
}

void multi_dma::multi_dma_wait_recv_4() {
  dmas[0].dma_s2mm_sync();
  dmas[1].dma_s2mm_sync();
  dmas[2].dma_s2mm_sync();
  dmas[3].dma_s2mm_sync();
}

int multi_dma::multi_dma_check_recv() {
  bool done = true;
  for (int i = 0; i < dma_count; i++)
    done = done && (dmas[i].dma_check_recv() == 0);
  return done ? 0 : -1;
}

#endif