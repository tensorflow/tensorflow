/*
 * Generated file - do not edit.
 */
#define FL_DEVICE_ALTERA_EPCS1 \
{ \
    ALTERA_EPCS1,           /* id */ \
    256,                    /* page size */ \
    512,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0xAB,                   /* SPI_RDID */ \
    3,                      /* id dummy bytes */ \
    1,                      /* id size in bytes */ \
    0x10,                   /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* Sector erase is full sector */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0,0}},          /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x03,                   /* SPI_READ */ \
    0,                      /* no read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {32768,{0,{0}}},        /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x00,                   /* no SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_AMIC_A25L016 \
{ \
    AMIC_A25L016, \
    256,                    /* page size */ \
    8192,                   /* num pages */ \
    3,                      /* address size */ \
    5,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x373015,               /* device id */ \
    0x20,                   /* SPI_SE */ \
    0,                      /* erase is full sector */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    SEC_PROT_SR,            /* no sector protection */ \
    {{0x1c,0x0},{0,0}},    /* SR values for protection */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* skip the read dummy */ \
    SECTOR_LAYOUT_REGULAR,  /* Regular sectors */ \
    {4096,{0,{0}}},         /* Regular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_AMIC_A25L40P \
{ \
    AMIC_A25L40P, \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    5,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    4,                      /* id size in bytes */ \
    0x7F372013,             /* device id */ \
    0xD8,                   /* SPI_SE */ \
    4096,                   /* pretend erase is 4KB only; hides top/bottom boot sector diffs */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    SEC_PROT_SR,            /* no sector protection */ \
    {{0x1c,0x0},{0,0}},    /* SR values for protection */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* skip the read dummy */ \
    SECTOR_LAYOUT_REGULAR,  /* regular sectors */ \
    {65536,{0,{0}}},        /* irregular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_AMIC_A25L40PT \
{ \
    AMIC_A25L40PT, \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    5,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    4,                      /* id size in bytes */ \
    0x7F372013,             /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* erase is full sector */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    SEC_PROT_SR,            /* no sector protection */ \
    {{0x1c,0x0},{0,0}},    /* SR values for protection */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* skip the read dummy */ \
    SECTOR_LAYOUT_IRREGULAR,  /* irregular sectors */ \
    {0,{12,{8,8,8,8,8,8,8,7,6,5,4,4}}},        /* irregular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_AMIC_A25L40PUM \
{ \
    AMIC_A25L40PUM, \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    5,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    4,                      /* id size in bytes */ \
    0x7F372013,             /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* erase is full sector */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    SEC_PROT_SR,            /* no sector protection */ \
    {{0x1c,0x0},{0,0}},    /* SR values for protection */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* skip the read dummy */ \
    SECTOR_LAYOUT_IRREGULAR,  /* irregular sectors */ \
    {0,{12,{4,4,5,6,7,8,8,8,8,8,8,8}}},        /* irregular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_AMIC_A25L80P \
{ \
    AMIC_A25L80P, \
    256,                    /* page size */ \
    4096,                   /* num pages */ \
    3,                      /* address size */ \
    5,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    4,                      /* id size in bytes */ \
    0x7F372014,             /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* erase is full sector */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    SEC_PROT_SR,            /* no sector protection */ \
    {{0x1c,0x0},{0,0}},    /* SR values for protection */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* skip the read dummy */ \
    SECTOR_LAYOUT_IRREGULAR,  /* irregular sectors */ \
    {0,{20,{4,4,5,6,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8}}},        /* irregular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ATMEL_AT25DF021 \
{ \
    ATMEL_AT25DF021,        /* AT25DF021 */ \
    256,                    /* page size */ \
    1024,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x1f4300,               /* device id */ \
    0x20,                   /* SPI_BE4 */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SECS,         /* no protection */ \
    {{0,0},{0x36,0x39}},    /* SPI_SP, SPI_SU */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
    {65536,{0,{0}}},        /* regular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ATMEL_AT25DF041A \
{ \
    ATMEL_AT25DF041A, \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x1f4401,               /* device id */ \
    0x20,                   /* SPI_BE4 */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SECS,         /* no protection */ \
    {{0,0},{0x36,0x39}},    /* SPI_SP, SPI_SU */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_IRREGULAR,  /* mad sectors */ \
    {0,{11,{8,8,8,8,8,8,8,7,5,5,6}}},  /* regular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ATMEL_AT25F512 \
{ \
    ATMEL_AT25F512, \
    256,                    /* page size */ \
    256,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x15,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    2,                      /* id size in bytes */ \
    0x1f65,                 /* device id */ \
    0x52,                   /* SPI_SE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* protection through status reg */ \
    {{0x0c,0x00},{0,0}},    /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {32768,{0,{0}}},         /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ATMEL_AT25FS010 \
{ \
    ATMEL_AT25FS010, \
    256,                    /* page size */ \
    512,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x1f6601,               /* device id */ \
    0xD7,                   /* SPI_SE */ \
    0,                      /* erase is full sector */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR_2X,        /* SR based protection (need double write) */ \
    {{0x0c,0x0},{0,0}},     /* SR values for protection */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* 1 read dummy byte*/ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {4096,{0,{0}}},         /* regular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ESMT_F25L004A \
{ \
    ESMT_F25L004A, \
    256,                    /* page size */  \
    2048,                   /* num pages */  \
    3,                      /* address size */  \
    8,                      /* log2 clock divider */  \
    0x9f,                   /* SPI_RDID */  \
    0,                      /* id dummy bytes */  \
    3,                      /* id size in bytes */  \
    0x8c2013,               /* device id */  \
    0x20,                   /* SPI_SSE */  \
    0,                      /* full sector erase */  \
    0x06,                   /* SPI_WREN */  \
    0x04,                   /* SPI_WRDI */  \
    PROT_TYPE_SR,           /* protection through status reg */  \
    {{0x1c,0x00},{0,0}},    /* no values */  \
    0x00|(0xad<<8)|(2<<16), /* No SPI_PP, have SPI_AAI for 2 bytes */  \
    0x0b,                   /* SPI_READFAST */  \
    1,                      /* 1 read dummy byte */  \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */  \
    {4096,{0,{0}}},         /* regular sector size */  \
    0x05,                   /* SPI_RDSR */  \
    0x01|(0x50<<8),         /* SPI_WRSR */  \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_MACRONIX_MX25L1005C \
{ \
    MACRONIX_MX25L1005C, \
    256,                   /* Page size */ \
    512,                   /* Number of pages */ \
    3,                     /* Address size */ \
    8,                     /* Clock divider */ \
    0x9F,                  /* RDID cmd */ \
    0,                     /* RDID dummy bytes */ \
    3,                     /* RDID data size in bytes */ \
    0xC22011,              /* RDID data */ \
    0x20,                  /* SE cmd */ \
    0,                     /* SE full sector erase */ \
    0x06,                  /* WREN cmd */ \
    0x04,                  /* WRDI cmd */ \
    PROT_TYPE_SR,          /* Protection type */ \
    {{0x0C, 0x00},{0,0}},  /* SR protect and unprotect values */ \
    0x02,                  /* PP cmd */ \
    0x0B,                  /* FAST_READ cmd */ \
    1,                     /* FAST_READ dummy bytes */ \
    SECTOR_LAYOUT_REGULAR, /* Sector layout */ \
    {4096,{0,{0}}},        /* Sector sizes */ \
    0x05,                  /* RDSR cmd */ \
    0x01,                  /* WRSR cmd */ \
    0x01,                  /* WIP bit mask in SR */ \
}
#define FL_DEVICE_MICRON_M25P40 \
{ \
    MICRON_M25P40, \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x202013,               /* device id */ \
    0xd8,                   /* SPI_SE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* SR protection */ \
    {{0x1c,0x0},{0,0}},     /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {65536,{0,{0}}},        /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* no SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_NUMONYX_M25P10 \
{ \
    NUMONYX_M25P10, \
    256,                    /* page size */ \
    512,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x202011,               /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* SR protection */ \
    {{0x0c,0x0},{0,0}},     /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {32768,{0,{0}}},        /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* no SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_NUMONYX_M25P16 \
{ \
    NUMONYX_M25P16, \
    256,                    /* page size */ \
    8192,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x202015,               /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* SR protection */ \
    {{0x1c,0x0},{0,0}},     /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {65536,{0,{0}}},        /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_NUMONYX_M45P10E \
{ \
    NUMONYX_M25P10, \
    256,                    /* page size */ \
    512,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x204011,               /* device id */ \
    0xD8,                   /* SPI_SE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* SR protection */ \
    {{0x0c,0x0},{0,0}},     /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {65536,{0,{0}}},        /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* no SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_SPANSION_S25FL204K \
{ \
    SPANSION_S25FL204K,     /* S25FL204K */ \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9F,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x014013,               /* device id */ \
    0x20,                   /* SPI_BE4 */ \
    4096,                   /* Sector erase is always 4KB */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SECS,         /* no protection */ \
    {{0,0},{0x36,0x39}},    /* SPI_SP, SPI_SU */ \
    0x02,                   /* SPI_PP */ \
    0x0B,                   /* SPI_READ_FAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* mad sectors */ \
    {65536,{0,{0}}},        /* regular sector sizes */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_SST_SST25VF010 \
{ \
    SST_SST25VF010, \
    256,                    /* page size */  \
    512,                    /* num pages */  \
    3,                      /* address size */  \
    8,                      /* log2 clock divider */  \
    0x90,                   /* SPI_RDID */  \
    3,                      /* id dummy bytes */  \
    2,                      /* id size in bytes */  \
    0xbf49,                 /* device id */  \
    0x20,                   /* SPI_SSE */  \
    0,                      /* full sector erase */  \
    0x06,                   /* SPI_WREN */  \
    0x04,                   /* SPI_WRDI */  \
    PROT_TYPE_SR,           /* protection through status reg */  \
    {{0x0c,0x00},{0,0}},    /* no values */  \
    0x00|(0xaf<<8)|(1<<16), /* No SPI_PP, have SPI_AAI for 1 byte */  \
    0x0b,                   /* SPI_READFAST */  \
    1,                      /* 1 read dummy byte */  \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */  \
    {4096,{0,{0}}},         /* regular sector size */  \
    0x05,                   /* SPI_RDSR */  \
    0x01|(0x50<<8),         /* SPI_WRSR */  \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_SST_SST25VF016 \
{ \
    SST_SST25VF016, \
    256,                    /* page size */  \
    8192,                   /* num pages */  \
    3,                      /* address size */  \
    8,                      /* log2 clock divider */  \
    0x9f,                   /* SPI_RDID */  \
    0,                      /* id dummy bytes */  \
    3,                      /* id size in bytes */  \
    0xbf2541,               /* device id */  \
    0x20,                   /* SPI_SSE */  \
    0,                      /* full sector erase */  \
    0x06,                   /* SPI_WREN */  \
    0x04,                   /* SPI_WRDI */  \
    PROT_TYPE_SR,           /* protection through status reg */  \
    {{0x1c,0x00},{0,0}},    /* no values */  \
    0x00|(0xad<<8)|(2<<16), /* No SPI_PP, have SPI_AAI for 2 bytes */  \
    0x0b,                   /* SPI_READFAST */  \
    1,                      /* 1 read dummy byte */  \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */  \
    {4096,{0,{0}}},         /* regular sector size */  \
    0x05,                   /* SPI_RDSR */  \
    0x01,                   /* SPI_WRSR */  \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_SST_SST25VF040 \
{ \
    SST_SST25VF040, \
    256,                    /* page size */  \
    2048,                   /* num pages */  \
    3,                      /* address size */  \
    8,                      /* log2 clock divider */  \
    0x9f,                   /* SPI_RDID */  \
    0,                      /* id dummy bytes */  \
    3,                      /* id size in bytes */  \
    0xbf258d,               /* device id */  \
    0x20,                   /* SPI_SSE */  \
    0,                      /* full sector erase */  \
    0x06,                   /* SPI_WREN */  \
    0x04,                   /* SPI_WRDI */  \
    PROT_TYPE_SR,           /* protection through status reg */  \
    {{0x1c,0x00},{0,0}},    /* no values */  \
    0x00|(0xad<<8)|(2<<16), /* No SPI_PP, have SPI_AAI for 2 bytes */  \
    0x0b,                   /* SPI_READFAST */  \
    1,                      /* 1 read dummy byte */  \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */  \
    {4096,{0,{0}}},         /* regular sector size */  \
    0x05,                   /* SPI_RDSR */  \
    0x01,                   /* SPI_WRSR */  \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ST_M25PE10 \
{ \
    ST_M25PE10, \
    256,                    /* page size */ \
    512,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x208011,               /* device id */ \
    0x20,                   /* SPI_SSE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0,0}},          /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {4096,{0,{0}}},         /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x00,                   /* no SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_ST_M25PE20 \
{ \
    ST_M25PE20, \
    256,                    /* page size */ \
    1024,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0x208012,               /* device id */ \
    0x20,                   /* SPI_SSE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_NONE,         /* no protection */ \
    {{0,0},{0,0}},          /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {4096,{0,{0}}},         /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x00,                   /* no SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_WINBOND_W25X10 \
{ \
    WINBOND_W25X10, \
    256,                    /* page size */ \
    512,                    /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xef3011,               /* device id */ \
    0x20,                   /* SPI_SSE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* protection through status reg */ \
    {{0x1c,0x00},{0,0}},    /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {4096,{0,{0}}},         /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_WINBOND_W25X20 \
{ \
    WINBOND_W25X20, \
    256,                    /* page size */ \
    1024,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xef3012,               /* device id */ \
    0x20,                   /* SPI_SSE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* protection through status reg */ \
    {{0x1c,0x00},{0,0}},    /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {4096,{0,{0}}},         /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
#define FL_DEVICE_WINBOND_W25X40 \
{ \
    WINBOND_W25X40, \
    256,                    /* page size */ \
    2048,                   /* num pages */ \
    3,                      /* address size */ \
    8,                      /* log2 clock divider */ \
    0x9f,                   /* SPI_RDID */ \
    0,                      /* id dummy bytes */ \
    3,                      /* id size in bytes */ \
    0xef3013,               /* device id */ \
    0x20,                   /* SPI_SSE */ \
    0,                      /* full sector erase */ \
    0x06,                   /* SPI_WREN */ \
    0x04,                   /* SPI_WRDI */ \
    PROT_TYPE_SR,           /* protection through status reg */ \
    {{0x1c,0x00},{0,0}},    /* no values */ \
    0x02,                   /* SPI_PP */ \
    0x0b,                   /* SPI_READFAST */ \
    1,                      /* 1 read dummy byte */ \
    SECTOR_LAYOUT_REGULAR,  /* sane sectors */ \
    {4096,{0,{0}}},         /* regular sector size */ \
    0x05,                   /* SPI_RDSR */ \
    0x01,                   /* SPI_WRSR */ \
    0x01,                   /* SPI_WIP_BIT_MASK */ \
}
