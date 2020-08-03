#include "tensorflow/lite/micro/benchmarks/eembc/profile/ee_profile.h"
/**
todo
1. buff -> buff commands
2. remove INCLUDE
*/
void   *gp_buff      = NULL;
size_t  g_buff_size  = 0u;
size_t  g_buff_pos   = 0u;
bool    g_use_buffer = false;


arg_claimed_t
ee_buffer_parse(char *p_command)
{
    char *p_next;

    if (th_strncmp(p_command, "buff", EE_CMD_SIZE) != 0)
    {
        return EE_ARG_UNCLAIMED;
    }
    
    p_next = th_strtok(NULL, EE_CMD_DELIMITER);
    
    if (th_strncmp(p_next, "on", EE_CMD_SIZE) == 0)
    {
        if (gp_buff == NULL)
        {
            th_printf("e-[Buffer not populated]\r\n");
        }
        else
        {
            g_use_buffer = true;
            th_printf("m-[Switching to buffer as input]\r\n");            
        }
    }
    else if (th_strncmp(p_next, "off", EE_CMD_SIZE) == 0)
    {
        g_use_buffer = false;
        th_printf("m-[Switching to default data as input]\r\n");
    }
    else if (th_strncmp(p_next, "load", EE_CMD_SIZE) == 0)
    {
        // person_detection is 96x96x1, or 9216 bytes
        // micro_speech is 16000 bytes
        p_next = th_strtok(NULL, EE_CMD_DELIMITER);

        if (p_next == NULL)
        {
            th_printf("e-[Command 'buff load' requires the # of bytes]\r\n");
        }
        else
        {
            g_buff_size = (size_t)atoi(p_next);
            if (g_buff_size == 0)
            {
                th_printf("e-[Command 'buff load' must be >0 bytes]\r\n");
            }
            else
            {
                g_buff_pos = 0;
                if (gp_buff != NULL)
                {
                    th_free(gp_buff);
                    gp_buff = NULL;
                }
                gp_buff = (void *)th_malloc(g_buff_size);
                if (gp_buff == NULL)
                {
                    th_printf("e-[Unable to malloc %u bytes]\r\n", g_buff_size);
                }
                else
                {
                    th_printf("m-[Begin issuing add commands']\r\n");
                }
            }
        }
    }
    else if (th_strncmp(p_next, "print-x8", EE_CMD_SIZE) == 0)
    {
        uint8_t *ptr = (uint8_t *)gp_buff;

        if (ptr == NULL)
        {
            th_printf("e-[Buffer not allocated]\r\n");
        }
        else
        {
            size_t i = 0;
            const size_t max = 8;
            for (; i < g_buff_size; ++i)
            {
                if ((i + max) % max == 0 || i == 0)
                {
                    th_printf("%04xh: ", i * 2);
                }
                th_printf("%02x", ptr[i]);
                if ((i + 1) % max == 0)
                {
                    th_printf("\r\n");
                }
                else
                {
                    th_printf(" ");
                }
            }
            if (i % max != 0)
            {
                th_printf("\r\n");
            }
        }
    }
    else if (th_strncmp(p_next, "print-i16", EE_CMD_SIZE) == 0)
    {
        int16_t *ptr = (int16_t *)gp_buff;

        if (ptr == NULL)
        {
            th_printf("e-[Buffer not allocated]\r\n");
        }
        else
        {
            size_t i = 0;
            const size_t max = 8;

            th_printf("m-[Buffer size is %d words]\r\n", g_buff_size >> 1);
            for (; i < (g_buff_size >> 1); ++i)
            {
                if ((i + max) % max == 0 || i == 0)
                {
                    th_printf("%04xh: ", i * 2);
                }
                th_printf("%+6d", ptr[i]);
                if ((i + 1) % max == 0)
                {
                    th_printf("\r\n");
                }
                else
                {
                    th_printf(" ");
                }
            }
            if (i % max != 0)
            {
                th_printf("\r\n");
            }
        }
    }
    else if (th_strncmp(p_next, "i16", EE_CMD_SIZE) == 0)
    {
        int16_t  val;
        int16_t *ptr = (int16_t *)gp_buff;
        while ((p_next = th_strtok(NULL, EE_CMD_DELIMITER)) != NULL)
        {
            // Yes, atoi is 0 on fail. Be aware.
            val = (int16_t)th_atoi(p_next);
            if (g_buff_pos >= g_buff_size)
            {
                th_printf("e-[Buffer full, use 'buff-start' to reload]\r\n");
                break;
            }
            else
            {
                ptr[g_buff_pos >> 1] = val;
                g_buff_pos += 2;
                if (g_buff_pos == g_buff_size)
                {
                    th_printf("m-load-done\r\n");
                }
            }
        }
    }
    else if (th_strncmp(p_next, "x8", EE_CMD_SIZE) == 0)
    {
        uint8_t  val;
        uint8_t *ptr = (uint8_t *)gp_buff;
        long     lval;

        while ((p_next = th_strtok(NULL, EE_CMD_DELIMITER)) != NULL)
        {
            lval = ee_hexdec(p_next);
            if (lval < 0)
            {
                th_printf("e-[Bad hex value '%s']\r\n", p_next);
                break;
            }
            else
            {
                val = (uint8_t)lval;
                if (g_buff_pos >= g_buff_size)
                {
                    th_printf("e-[Buffer full, use 'buff-start' to reload]\r\n");
                    break;
                }
                else
                {
                    ptr[g_buff_pos] = val;
                    ++g_buff_pos;
                    if (g_buff_pos == g_buff_size)
                    {
                        th_printf("m-load-done\r\n");
                    }
                }
            }
        }
    }
    else
    {
        th_printf("e-[Unknown 'buf' sub-command: %s]\r\n", p_next);
    }
    return EE_ARG_CLAIMED;
}
