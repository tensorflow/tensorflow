#include "fully_connected_bram_driver.h"
#include <iostream>
#include <vector>
#include <cstdint>

int main() {
    try {
        FpgaBramDriver driver;
        driver.initialize_bram(false); // Clear BRAM on init

        int input_size;
        int output_size;

        std::cout << "Enter input size: ";
        std::cin >> input_size;
        std::cout << "Enter output size: ";
        std::cin >> output_size;
        
        int weight_size = input_size * output_size;
        int bias_size = output_size;

        std::vector<int32_t> input(input_size, 1);
        std::vector<int32_t> weights(weight_size, 2);
        std::vector<int32_t> bias(bias_size, 3);
        std::vector<int32_t> output(input_size, 0);

        // Write to BRAMs
        if (driver.write_input_to_bram(input.data(), input_size) != 0) {
            std::cerr << "Failed to write input to BRAM" << std::endl;
        }
        if (driver.write_weights_to_bram(weights.data(), weight_size) != 0) {
            std::cerr << "Failed to write weights to BRAM" << std::endl;
        }
        if (driver.write_bias_to_bram(bias.data(), bias_size) != 0) {
            std::cerr << "Failed to write bias to BRAM" << std::endl;
        }


        //TODO: There is the error while clearing the form all the bram
        // // Clear output BRAM
        // if (driver.clear_output_bram() != 0) {
        //     std::cerr << "Failed to clear output BRAM" << std::endl;
        // }    

        // Read input,weight,bias BRAM
        if (driver.test_read_input_bram(input.data(), input_size) == 0) {
            std::cout << "Input BRAM contents:" << std::endl;
            for (int i = 0; i < input_size; ++i) {
                std::cout << input[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to read input from BRAM" << std::endl;
        }
        if (driver.test_read_weights_bram(weights.data(), weight_size) == 0) {
            std::cout << "Weight BRAM contents:" << std::endl;
            for (int i = 0; i < weight_size; ++i) {
                std::cout << weights[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to read weights from BRAM" << std::endl;
        }
        
        if (driver.test_read_bias_bram(bias.data(), bias_size) == 0) {
            std::cout << "Bias BRAM contents:" << std::endl;
            for (int i = 0; i < bias_size; ++i) {
                std::cout << bias[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to read bias from BRAM" << std::endl;
        }

        // Read output BRAM (should be all zeros)
        if (driver.read_output_from_bram(output.data(), output_size) == 0) {
            std::cout << "Output BRAM contents after clearing:" << std::endl;
            for (int i = 0; i < output_size; ++i) {
                std::cout << output[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to read output from BRAM" << std::endl;
        }
    
    return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
    
}
