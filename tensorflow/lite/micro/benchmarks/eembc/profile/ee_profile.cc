
#include "tensorflow/lite/micro/benchmarks/eembc/profile/ee_profile.h"

// Generic TFLM main_functions
extern "C" void setup();
extern "C" void loop();

arg_claimed_t ee_buffer_parse(char *command);

arg_claimed_t
ee_profile_parse(char *command)
{
    char *p_next; // strtok already primed from ee_main.c

    if (th_strncmp(command, "profile", EE_CMD_SIZE) == 0)
    {
        th_printf("m-profile-[%s]\r\n", EE_FW_VERSION);
    }
    else if (th_strncmp(command, "help", EE_CMD_SIZE) == 0)
    {
        th_printf("%s\r\n", EE_FW_VERSION);
        th_printf("\r\n");
        th_printf("help         : Print this information\r\n");
        th_printf("loop         : first-stab at measurement\r\n");
        th_printf("buff SUBCMD  : Input buffer functions\r\n");
        th_printf("  on         : Infer from buffer data\r\n");
        th_printf("  off        : Infer from default data\r\n");
        th_printf("  load N     : Allocate N bytes and set load counter\r\n");
        th_printf("  i16 N [N]+ : Load 16-bit signed decimal integer(s)\r\n");
        th_printf("  x8 N [N]+  : Load 8-bit hex byte(s)\r\n");
        th_printf("  print i16  : Print buffer as 16-bit signed decimal\r\n");
        th_printf("  print x8   : Print buffer as 8-bit hex\r\n");
    }
    else if (th_strncmp(command, "loop", EE_CMD_SIZE) == 0)
    {
        int loops;

        loops = 1;
        p_next = th_strtok(NULL, EE_CMD_DELIMITER);
        if (NULL != p_next)
        {
            loops = th_atoi(p_next);
        }
        if (loops < 1)
        {
            th_printf("e-[Invalid count passed to 'loop']\r\n");
        }
        else
        {
            for (; loops > 0; --loops)
            {
                th_printf("m-[Running loop #%d]\r\n", loops);
                th_timestamp();
                //th_pre();
                loop();
                //th_post();
                th_timestamp();
            }
        }
    }
    else if (ee_buffer_parse(command) == EE_ARG_CLAIMED)
    {
    }
    else 
    {
        return EE_ARG_UNCLAIMED;
    }
    return EE_ARG_CLAIMED;
}

void
ee_profile_initialize(void) {
}

/**
 * Tensorflow Lite Micro needs a main() for the micro examples.
 */

int
main(void)
{
    setup();
    ee_main();
    while (true) {
        th_check_serial();
    }
    return 0;
}