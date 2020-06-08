/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_RAW8_QVGA_8BITS_LSB_5FPS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_RAW8_QVGA_8BITS_LSB_5FPS_H_

#include "HM01B0.h"

const hm_script_t sHM01B0InitScript[] = {
    // ;*************************************************************************
    // ; Sensor: HM01B0
    // ; I2C ID: 24
    // ; Resolution: 324x244
    // ; Lens:
    // ; Flicker:
    // ; Frequency:
    // ; Description: AE control enable
    // ; 8-bit mode, LSB first
    // ;
    // ;
    // ; Note:
    // ;
    // ; $Revision: 1338 $
    // ; $Date:: 2017-04-11 15:43:45 +0800#$
    // ;*************************************************************************
    //
    // // ---------------------------------------------------
    // // HUB system initial
    // // ---------------------------------------------------
    // W 20 8A04 01 2 1
    // W 20 8A00 22 2 1
    // W 20 8A01 00 2 1
    // W 20 8A02 01 2 1
    // W 20 0035 93 2 1 ; [3]&[1] hub616 20bits in, [5:4]=1 mclk=48/2=24mhz
    // W 20 0036 00 2 1
    // W 20 0011 09 2 1
    // W 20 0012 B6 2 1
    // W 20 0014 08 2 1
    // W 20 0015 98 2 1
    // ;W 20 0130 16 2 1 ; 3m soc, signal buffer control
    // ;W 20 0100 44 2 1 ; [6] hub616 20bits in
    // W 20 0100 04 2 1 ; [6] hub616 20bits in
    // W 20 0121 01 2 1 ; [0] Q1 Intf enable, [1]:4bit mode, [2] msb first, [3]
    // serial mode
    // W 20 0150 00 2 1 ;
    // W 20 0150 04 2 1 ;
    //
    //
    // //---------------------------------------------------
    // // Initial
    // //---------------------------------------------------
    // W 24 0103 00 2 1 ; software reset-> was 0x22
    {
        0x0103,
        0x00,
    },
    // W 24 0100 00 2 1; power up
    {
        0x0100,
        0x00,
    },
    //
    //
    //
    // //---------------------------------------------------
    // // Analog
    // //---------------------------------------------------
    // L HM01B0_analog_setting.txt
    {
        0x1003,
        0x08,
    },
    {
        0x1007,
        0x08,
    },
    {
        0x3044,
        0x0A,
    },
    {
        0x3045,
        0x00,
    },
    {
        0x3047,
        0x0A,
    },
    {
        0x3050,
        0xC0,
    },
    {
        0x3051,
        0x42,
    },
    {
        0x3052,
        0x50,
    },
    {
        0x3053,
        0x00,
    },
    {
        0x3054,
        0x03,
    },
    {
        0x3055,
        0xF7,
    },
    {
        0x3056,
        0xF8,
    },
    {
        0x3057,
        0x29,
    },
    {
        0x3058,
        0x1F,
    },
    {
        0x3059,
        0x1E,
    },
    {
        0x3064,
        0x00,
    },
    {
        0x3065,
        0x04,
    },
    //
    //
    // //---------------------------------------------------
    // // Digital function
    // //---------------------------------------------------
    //
    // // BLC
    // W 24 1000 43 2 1 ; BLC_on, IIR
    {
        0x1000,
        0x43,
    },
    // W 24 1001 40 2 1 ; [6] : BLC dithering en
    {
        0x1001,
        0x40,
    },
    // W 24 1002 32 2 1 ; // blc_darkpixel_thd
    {
        0x1002,
        0x32,
    },
    //
    // // Dgain
    // W 24 0350 7F 2 1 ; Dgain Control
    {
        0x0350,
        0x7F,
    },
    //
    // // BLI
    // W 24 1006 01 2 1 ; [0] : bli enable
    {
        0x1006,
        0x01,
    },
    //
    // // DPC
    // W 24 1008 00 2 1 ; [2:0] : DPC option 0: DPC off 1 : mono 3 : bayer1 5 :
    // bayer2
    {
        0x1008,
        0x00,
    },
    // W 24 1009 A0 2 1 ; cluster hot pixel th
    {
        0x1009,
        0xA0,
    },
    // W 24 100A 60 2 1 ; cluster cold pixel th
    {
        0x100A,
        0x60,
    },
    // W 24 100B 90 2 1 ; single hot pixel th
    {
        0x100B,
        0x90,
    },
    // W 24 100C 40 2 1 ; single cold pixel th
    {
        0x100C,
        0x40,
    },
    // //
    // advance VSYNC by 1 row
    {
        0x3022,
        0x01,
    },
    // W 24 1012 00 2 1 ; Sync. enable VSYNC shift
    {
        0x1012,
        0x01,
    },

    //
    // // ROI Statistic
    // W 24 2000 07 2 1 ; [0] : AE stat en [1] : MD LROI stat en [2] : MD GROI
    // stat en [3] : RGB stat ratio en [4] : IIR selection (1 -> 16, 0 -> 8)
    {
        0x2000,
        0x07,
    },
    // W 24 2003 00 2 1 ; MD GROI 0 y start HB
    {
        0x2003,
        0x00,
    },
    // W 24 2004 1C 2 1 ; MD GROI 0 y start LB
    {
        0x2004,
        0x1C,
    },
    // W 24 2007 00 2 1 ; MD GROI 1 y start HB
    {
        0x2007,
        0x00,
    },
    // W 24 2008 58 2 1 ; MD GROI 1 y start LB
    {
        0x2008,
        0x58,
    },
    // W 24 200B 00 2 1 ; MD GROI 2 y start HB
    {
        0x200B,
        0x00,
    },
    // W 24 200C 7A 2 1 ; MD GROI 2 y start LB
    {
        0x200C,
        0x7A,
    },
    // W 24 200F 00 2 1 ; MD GROI 3 y start HB
    {
        0x200F,
        0x00,
    },
    // W 24 2010 B8 2 1 ; MD GROI 3 y start LB
    {
        0x2010,
        0xB8,
    },
    //
    // W 24 2013 00 2 1 ; MD LRIO y start HB
    {
        0x2013,
        0x00,
    },
    // W 24 2014 58 2 1 ; MD LROI y start LB
    {
        0x2014,
        0x58,
    },
    // W 24 2017 00 2 1 ; MD LROI y end HB
    {
        0x2017,
        0x00,
    },
    // W 24 2018 9B 2 1 ; MD LROI y end LB
    {
        0x2018,
        0x9B,
    },
    //
    // // AE
    // W 24 2100 01 2 1 ; [0]: AE control enable
    {
        0x2100,
        0x01,
    },
    // W 24 2101 07 2 1 ; AE target mean
    {
        0x2101,
        0x5F,
    },
    // W 24 2102 0A 2 1 ; AE min mean
    {
        0x2102,
        0x0A,
    },
    // W 24 2104 03 2 1 ; AE Threshold
    {
        0x2103,
        0x03,
    },
    // W 24 2104 05 2 1 ; AE Threshold
    {
        0x2104,
        0x05,
    },
    // W 24 2105 01 2 1 ; max INTG Hb
    {
        0x2105,
        0x02,
    },
    // W 24 2106 54 2 1 ; max INTG Lb
    {
        0x2106,
        0x14,
    },
    // W 24 2108 02 2 1 ; max AGain in full
    {
        0x2107,
        0x02,
    },
    // W 24 2108 03 2 1 ; max AGain in full
    {
        0x2108,
        0x03,
    },
    // W 24 2109 04 2 1 ; max AGain in bin2
    {
        0x2109,
        0x03,
    },
    // W 24 210A 00 2 1 ; min AGAIN
    {
        0x210A,
        0x00,
    },
    // W 24 210B C0 2 1 ; max DGain
    {
        0x210B,
        0x80,
    },
    // W 24 210C 40 2 1 ; min DGain
    {
        0x210C,
        0x40,
    },
    // W 24 210D 20 2 1 ; damping factor
    {
        0x210D,
        0x20,
    },
    // W 24 210E 03 2 1 ; FS ctrl
    {
        0x210E,
        0x03,
    },
    // W 24 210F 00 2 1 ; FS 60Hz Hb
    {
        0x210F,
        0x00,
    },
    // W 24 2110 85 2 1 ; FS 60Hz Lb
    {
        0x2110,
        0x85,
    },
    // W 24 2111 00 2 1 ; Fs 50Hz Hb
    {
        0x2111,
        0x00,
    },
    // W 24 2112 A0 2 1 ; FS 50Hz Lb
    {
        0x2112,
        0xA0,
    },

    //
    //
    // // MD
    // W 24 2150 03 2 1 ; [0] : MD LROI en [1] : MD GROI en
    {
        0x2150,
        0x03,
    },
    //
    //
    // //---------------------------------------------------
    // // frame rate : 5 FPS
    // //---------------------------------------------------
    // W 24 0340 0C 2 1 ; smia frame length Hb
    {
        0x0340,
        0x0C,
    },
    // W 24 0341 7A 2 1 ; smia frame length Lb 3192
    {
        0x0341,
        0x7A,
    },
    //
    // W 24 0342 01 2 1 ; smia line length Hb
    {
        0x0342,
        0x01,
    },
    // W 24 0343 77 2 1 ; smia line length Lb 375
    {
        0x0343,
        0x77,
    },
    //
    // //---------------------------------------------------
    // // Resolution : QVGA 324x244
    // //---------------------------------------------------
    // W 24 3010 01 2 1 ; [0] : window mode 0 : full frame 324x324 1 : QVGA
    {
        0x3010,
        0x01,
    },
    //
    //
    // W 24 0383 01 2 1 ;
    {
        0x0383,
        0x01,
    },
    // W 24 0387 01 2 1 ;
    {
        0x0387,
        0x01,
    },
    // W 24 0390 00 2 1 ;
    {
        0x0390,
        0x00,
    },
    //
    // //---------------------------------------------------
    // // bit width Selection
    // //---------------------------------------------------
    // W 24 3011 70 2 1 ; [0] : 6 bit mode enable
    {
        0x3011,
        0x70,
    },
    //
    //
    // W 24 3059 02 2 1 ; [7]: Self OSC En, [6]: 4bit mode, [5]: serial mode,
    // [4:0]: keep value as 0x02
    {
        0x3059,
        0x02,
    },
    // W 24 3060 01 2 1 ; [5]: gated_clock, [4]: msb first,
    {
        0x3060,
        0x20,
    },
    // ; [3:2]: vt_reg_div -> div by 4/8/1/2
    // ; [1;0]: vt_sys_div -> div by 8/4/2/1
    //
    //
    {
        0x0101,
        0x01,
    },
    // //---------------------------------------------------
    // // CMU update
    // //---------------------------------------------------
    //
    // W 24 0104 01 2 1 ; was 0100
    {
        0x0104,
        0x01,
    },
    //
    //
    //
    // //---------------------------------------------------
    // // Turn on rolling shutter
    // //---------------------------------------------------
    // W 24 0100 01 2 1 ; was 0005 ; mode_select 00 : standby - wait fir I2C SW
    // trigger 01 : streaming 03 : output "N" frame, then enter standby 04 :
    // standby - wait for HW trigger (level), then continuous video out til HW
    // TRIG goes off 06 : standby - wait for HW trigger (edge), then output "N"
    // frames then enter standby
    {
        0x0100,
        0x01,
    },
    //
    // ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
};

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_RAW8_QVGA_8BITS_LSB_5FPS_H_
