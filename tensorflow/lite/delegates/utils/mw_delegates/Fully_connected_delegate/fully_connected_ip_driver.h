#ifndef FULLY_CONNECTED_IP_DRIVER_H
#define FULLY_CONNECTED_IP_DRIVER_H

#include <cstdint>
#include <string>
#include <map>

class FullyConnectedIpDriver {
private:
    int dev_mem_fd;
    void* mapped_fpga_ip;
    size_t size;

    uint32_t ip_base_address;

    uint32_t ip_input_size_offset;
    uint32_t ip_output_size_offset;
    uint32_t ip_control_register_offset;

    static constexpr uint32_t CONTROL_START    = 0x01;
    static constexpr uint32_t CONTROL_CONTINUE = 0x10; 

    std::map<std::string, uint32_t> ip_address;
    
    void initialize_fpga();

    //TODO IF and error in input and output, check type, uint32_t and int32_t
    void write_to_fpga(const std::string& reg_name, uint32_t value);
    int32_t read_from_fpga(const std::string& reg_name);

public:
    FullyConnectedIpDriver();
    ~FullyConnectedIpDriver();

    void fpga_compute(int32_t input_size, int32_t output_size);
};

#endif // FULLY_CONNECTED_IP_DRIVER_H
