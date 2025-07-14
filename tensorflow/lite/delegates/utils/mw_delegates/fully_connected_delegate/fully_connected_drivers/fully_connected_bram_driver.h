#ifndef FULLY_CONNECTED_BRAM_DRIVER_H
#define FULLY_CONNECTED_BRAM_DRIVER_H

#include <cstdint>
#include <string>
#include <map>

class FpgaBramDriver {
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

    
    void write_to_bram(const std::string& bram_name, float* ptr, size_t num_elements);
    float* read_from_bram(const std::string& bram_name);


public:
    FpgaBramDriver();
    ~FpgaBramDriver();
    
    void initialize_bram(bool clear_bram);
    int write_weights_to_bram(const float* weights, const int size);
    int write_bias_to_bram(const float* bias, const int size);
    int write_input_to_bram(const float* input, const int size);
    int read_output_from_bram(float* output, const int size);
    int clear_output_bram();

    int test_read_input_bram(float* output, const int size);
    int test_read_weights_bram(float* output, const int size);
    int test_read_bias_bram(float* output, const int size);
};

#endif // FULLY_CONNECTED_BRAM_DRIVER_H