#ifndef __TWOMODE_MAIN_H_
#define __TWOMODE_MAIN_H_

// #include "lwipopts.h"
// #include "lwip/ip_addr.h"
// #include "lwip/err.h"
// #include "lwip/udp.h"
// #include "lwip/inet.h"
// #include "xil_printf.h"
// #include "platform.h"
// #include "lwip/pbuf.h"
#include <string.h>
#include <stdint.h>
//#include "udp_send.h"

//UNCOMMENT LATER**********************************
// typedef struct __attribute__((packed)) {
//     char     file_id[8];     // exactly 8 bytes, no null terminator needed
//     uint32_t file_size;      // bytes following (consider endianness!)
// } file_header_t;

//UNCOMMENT LATER**********************************
// void udp_recv_file(void *arg, struct udp_pcb *pcb, struct pbuf *p,
//                    const ip_addr_t *addr, u16_t port);
/* server port to listen on/connect to */

#define DEBUG 0

#define MAGIC_SESSION "NPY0"   // your custom session marker, NOT the .npy magic
#define MAX_NDIM 8

#define UDP_CONN_PORT 5001
#define UDP_SEND_PORT 5002


#define MAGIC_STR "NPYB"
#define VER_EXPECTED 1
#define TYPE_INIT 1
#define TYPE_DATA 2

// Expected trailing dims (400, 2, 2, 2, 32) ***(DATA SIZE)
#define T_EXP 400
#define D1_EXP 2
#define D2_EXP 2
#define D3_EXP 2
#define D4_EXP 32
#define TRAIL_ELEMS (D1_EXP*D2_EXP*D3_EXP*D4_EXP)


// ---- Tune these to your max needs ---- (MUST CHECK WITH PYTHON transmitter first)
#define T_MAX  700          // max total_first_dim you might use
#define B_MAX  100           // max block size
#define BLOCK_NPY_MAX 65536  // max bytes for ONE block .npy (safe generous)
#define CHUNK_SEEN_MAX 512   // max chunks per block we can track



#endif /* __TWOMODE_MAIN_H_ */
