#ifndef FPGA_IP_DRIVER_H
#define FPGA_IP_DRIVER_H

#include <cstdint>
#include <string>
#include <map>

class FpgaIpDriver {
private:
    int dev_mem_fd;
    void* mapped_fpga_ip;
    size_t size;

    uint32_t ip_base_address;

    uint32_t ip_input_1_offset;
    uint32_t ip_input_2_offset;
    uint32_t ip_output_offset;
    uint32_t ip_add_flag_offset;
    uint32_t ip_control_register_offset;

    static constexpr uint32_t CONTROL_START    = 0x01;
    static constexpr uint32_t CONTROL_CONTINUE = 0x10; 

    std::map<std::string, uint32_t> ip_address;

    void initialize_fpga();
    void write_to_fpga(const std::string& reg_name, uint32_t value);
    int32_t read_from_fpga(const std::string& reg_name);

public:
    FpgaIpDriver();
    ~FpgaIpDriver();

    int32_t fpga_compute(int32_t input_1, int32_t input_2, bool add_flag);
};

#endif // FPGA_IP_DRIVER_H
