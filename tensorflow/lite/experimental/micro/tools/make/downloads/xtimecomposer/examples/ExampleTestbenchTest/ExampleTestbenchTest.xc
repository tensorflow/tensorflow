/*
 * Test the use of the ExampleTestbench. Test that the value 0 and 1 can be sent
 * in both directions between the ports.
 *
 * NOTE: The src/testbenches/ExampleTestbench must have been compiled for this to run without error.
 *
 */
#include <xs1.h>
#include <print.h>

port p1 = XS1_PORT_1A;
port p2 = XS1_PORT_1B;

int testAndSend(port outPort, port inPort, int value)
{
  timer t;
  unsigned time;
  int error = 0;
  
  // Transmit the desired value
  outPort <: value;
    
  // Record the start time to check for failure
  t :> time;

  select {
    case inPort when pinseq(value) :> int x:
      printstrln(" ok");
      break;
    case t when timerafter(time + 100) :> time:
      printstrln(" error");
      error = 1;
      break;
  }
  
  return error;
}

int sendP1ToP2(int value)
{
  printstr("p1 -> ");
  printint(value);
  printstr(" -> p2");
  return testAndSend(p1, p2, value);
}

int sendP2ToP1(int value)
{
  printstr("p2 -> ");
  printint(value);
  printstr(" -> p1");
  return testAndSend(p2, p1, value);
}


int main()
{
  int error = 0;
  
  error |= sendP1ToP2(0);
  error |= sendP1ToP2(1);
  
  // Stop P1 driving
  p1 :> int x;

  error |= sendP2ToP1(0);
  error |= sendP2ToP1(1);

  return error;
}
