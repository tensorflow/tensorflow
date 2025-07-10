#include "fully_connected_ip_driver.h"
#include <iostream>
#include <cstdint>

int main() {
    
        FpgaIpDriver driver;
        int32_t input_size;
        int32_t output_size;

        std::cout << "Enter input size: ";
        std::cin >> input_size;
        std::cout << "Enter output size: ";
        std::cin >> output_size;

        std::cout << "Testing FPGA compute with input_size = " << input_size
                  << ", output_size = " << output_size << std::endl;
        try{

            driver.fpga_compute(input_size, output_size);
        } catch (const std::exception& ex) {
            std::cerr << "Exception: " << ex.what() << std::endl;
            return 1;
        }
        
    
    return 0;
}
