#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/env.h"
#include <sstream>
#include <chrono>

using namespace tensorflow;

int main(int argc, char* argv[]) {
  if ( argc != 7 )
  {
    std::cout << "\n Usage: ";
    std::cout << argv[0];
    std::cout << " <relativeModelFileName> <inputVectorName> <inputVectorSize> <outputNameA> <outputNameB> <nRepeats> \n";
    return 1;
  }

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), argv[1], &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  
  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  //Get the initial vector size.
  std::stringstream strValue;
  strValue << argv[3];
  unsigned int ia;
  strValue >> ia;

  //Get the initial vector size.
  std::stringstream strRepeatValue;
  strRepeatValue << argv[6];
  unsigned int nRepeats;
  strRepeatValue >> nRepeats;

  Tensor input_vector(DT_DOUBLE, TensorShape({ia}));
  
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
  { argv[2], input_vector}
  };  
  
  std::chrono::time_point<std::chrono::system_clock> start, end;
  
  start = std::chrono::system_clock::now();
  
  for( uint a = 0; a < nRepeats; a++)
  {
    std::vector<tensorflow::Tensor> outputs;

    status = session->Run(inputs, {argv[4], argv[5]}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
  }
  
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  
  
  std::cout << "Time taken per graph evaluation: " << elapsed_seconds.count()/nRepeats << "\n ";

  
}
