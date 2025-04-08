#include "Data/Data.h"
#include "Utility/String.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <shared_mutex>

namespace proton {

void Data::dump(OutputFormat outputFormat) {
  std::shared_lock<std::shared_mutex> lock(mutex);

  std::unique_ptr<std::ostream> out;
  if (path.empty() || path == "-") {
    out.reset(new std::ostream(std::cout.rdbuf())); // Redirecting to cout
  } else {
    out.reset(new std::ofstream(
        path + "." +
        outputFormatToString(outputFormat))); // Opening a file for output
  }
  doDump(*out, outputFormat);
}

OutputFormat parseOutputFormat(const std::string &outputFormat) {
  if (toLower(outputFormat) == "hatchet") {
    return OutputFormat::Hatchet;
  }
  throw std::runtime_error("Unknown output format: " + outputFormat);
}

const std::string outputFormatToString(OutputFormat outputFormat) {
  if (outputFormat == OutputFormat::Hatchet) {
    return "hatchet";
  }
  throw std::runtime_error("Unknown output format: " +
                           std::to_string(static_cast<int>(outputFormat)));
}

} // namespace proton
