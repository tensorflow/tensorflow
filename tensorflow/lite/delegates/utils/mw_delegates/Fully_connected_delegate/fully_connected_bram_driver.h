#ifndef FULLY_CONNECTED_BRAM_DRIVER_H
#define FULLY_CONNECTED_BRAM_DRIVER_H

#include <cstdint>
#include <string>
#include <map>

class FullyConnectedBRAMDriver {
private:
    int bram_dev_mem_fd;
    void* bram_mapped_input_block;
    void* bram_mapped_weight_block;
    void* bram_mapped_bias_block;
    void* bram_mapped_output_block;
    size_t bram_size_other_than_weight;
    size_t bram_size_weight;

    uint32_t bram_input_base_address;
    uint32_t bram_weight_base_address;
    uint32_t bram_bias_base_address;
    uint32_t bram_output_base_address;

    std::map<std::string, void*> bram_address;

    void initialize_bram();


public:
    FullyConnectedBRAMDriver();
    ~FullyConnectedBRAMDriver();

    // Add methods for reading/writing to BRAMs if needed
    void write_to_bram(const std::string& bram_name, int32_t* ptr);
    int32_t* read_from_bram(const std::string& bram_name);
};

#endif // FULLY_CONNECTED_BRAM_DRIVER_H