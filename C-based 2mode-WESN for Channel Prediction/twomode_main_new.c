
#include "twomode_main.h"

/*--------FOR TRANSMITTING DATA TO PC ------*/
static ip_addr_t g_pc_addr;
static int g_pc_known = 0;
#define SEND_DATA_OUT 1
#define UDP_CHUNK_BYTES 1200

static double *g_last_dst = NULL;
static uint32_t g_last_payload_bytes = 0;
static volatile int g_block_ready = 0;

static int NMSE_data_cnt = 0;
static int NMSE_data_ready = 0;
/*------------------------------------------*/

static double g_tensor[T_MAX * TRAIL_ELEMS]; //Stores the data *IMPORTANT*
static int g_T = 0, g_B = 0; //Used to determine N(total data) and how many chunks per block
static uint32_t g_sid = 0;
static int g_inited = 0;
static int WESN_DONE = 0;

// Current block reassembly
static uint8_t  g_npy_buf[BLOCK_NPY_MAX];
static uint32_t g_npy_len = 0;
static uint16_t g_chunk_size = 0;
static uint32_t g_start_index = 0;

static uint8_t  g_chunk_seen[CHUNK_SEEN_MAX]; // bitmap-by-byte
static uint32_t g_chunks_expected = 0;
static uint32_t g_chunks_received = 0;



// ---- helpers ----
static inline void cp(struct pbuf *p, void *dst, uint16_t len, uint16_t off) {
    pbuf_copy_partial(p, dst, len, off);
}

static void reset_block(void) {
    g_npy_len = 0;
    g_chunk_size = 0;
    g_start_index = 0;
    g_chunks_expected = 0;
    g_chunks_received = 0;
    memset(g_chunk_seen, 0, sizeof(g_chunk_seen));
}

static int all_chunks_received(void){
    for (uint32_t i = 0; i < g_chunks_expected; i++){
        if (g_chunk_seen[i] == 0) return 0;
    }
    return 1;
}

/* ---- helper used to send header ---- */
static err_t send_header(struct udp_pcb *pcb, const ip_addr_t *addr, u16_t port,
                         const char file_id8[8], uint32_t payload_bytes)
{
    file_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.file_id, file_id8, 8);
    hdr.file_size = htonl(payload_bytes);  // keep consistent with your Python/C reader (endianness)

    struct pbuf *ph = pbuf_alloc(PBUF_TRANSPORT, sizeof(hdr), PBUF_RAM);
    if (!ph) return ERR_MEM;
    memcpy(ph->payload, &hdr, sizeof(hdr));
    xil_printf("[DEBUG]TX: payload_bytes=%lu g_B=%d TRAIL=%d\r\n",
               (unsigned long)g_last_payload_bytes, g_B, TRAIL_ELEMS);

    err_t e = udp_sendto(pcb, ph, addr, port);
    pbuf_free(ph);
    return e;
}

/* ---- helper used to send data UDP_CHUNK_BYTES length at a time---- */
static err_t send_bytes_chunked(struct udp_pcb *pcb, const ip_addr_t *addr, u16_t port,
                                const uint8_t *buf, uint32_t len)
{
    uint32_t off = 0;
    while (off < len) {
        uint16_t n = (uint16_t)((len - off) > UDP_CHUNK_BYTES ? UDP_CHUNK_BYTES : (len - off));
        struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, n, PBUF_RAM);
        if (!p) return ERR_MEM;
        memcpy(p->payload, buf + off, n);
        err_t e = udp_sendto(pcb, p, addr, port);
        pbuf_free(p);
        if (e != ERR_OK) return e;
        off += n;
    }
    return ERR_OK;
}

// xil_printf often lacks %f; print double in fixed-ish format
static void print_double_fixed(double x) {
    long long ip = (long long)x;
    double frac = x - (double)ip;
    if (frac < 0) frac = -frac;
    long long fp = (long long)(frac * 1000000.0); // 6 digits
    xil_printf("%lld.%06lld", ip, fp);
}

// Minimal .npy parse: verify magic + dtype <f8 + compute data_offset
static int npy_get_data_offset_and_check_f64(const uint8_t *npy, uint32_t npy_len, uint32_t *data_offset_out) {
    if (npy_len < 10) return -1;

    const uint8_t magic6[6] = {0x93,'N','U','M','P','Y'};
    if (memcmp(npy, magic6, 6) != 0) return -1;

    uint8_t ver_major = npy[6];
    uint32_t header_len = 0;
    uint32_t header_start = 0;

    if (ver_major == 1) {
        header_len = (uint32_t)npy[8] | ((uint32_t)npy[9] << 8); // LE u16
        header_start = 10;
    } else if (ver_major == 2 || ver_major == 3) {
        if (npy_len < 12) return -1;
        header_len = (uint32_t)npy[8] |
                     ((uint32_t)npy[9] << 8) |
                     ((uint32_t)npy[10] << 16) |
                     ((uint32_t)npy[11] << 24); // LE u32
        header_start = 12;
    } else {
        return -1;
    }

    uint32_t data_offset = header_start + header_len;
    if (data_offset > npy_len) return -1;

    // Check header text for <f8 (float64)
    // Copy only up to 512 bytes for searching
    uint32_t n = header_len;
    if (n > 512) n = 512;
    char hdr[513];
    memcpy(hdr, npy + header_start, n);
    hdr[n] = 0;

    if (strstr(hdr, "<f8") == NULL && strstr(hdr, "|f8") == NULL) {
        return -1;
    }

    // Optional: check the block first dimension matches g_B by searching "(B,"
    // This is a light sanity check; remove if too strict.
    char pat[32];
    // NOTE: if header has spaces "(10, 2, ...)" this still matches "(10,"
    sprintf(pat, "(%d,", g_B);
    if (strstr(hdr, pat) == NULL) {
        // Not fatal if you don't want shape check; but it's useful.
        return -1;
    }

    *data_offset_out = data_offset;
    return 0;
}

static int commit_block_into_tensor(void) {
    uint32_t data_offset = 0;
    if (npy_get_data_offset_and_check_f64(g_npy_buf, g_npy_len, &data_offset) != 0) {
        xil_printf("Block .npy parse failed\r\n");
        return -1;
    }

    uint32_t expected_data_bytes = (uint32_t)g_B * (uint32_t)TRAIL_ELEMS * 8;
    if (data_offset + expected_data_bytes > g_npy_len) {
        xil_printf("Block data too short\r\n");
        return -1;
    }

    if ((g_start_index + (uint32_t)g_B) > (uint32_t)g_T) {
        xil_printf("start_index out of range\r\n");
        return -1;
    }

    double *dst = &g_tensor[g_start_index * TRAIL_ELEMS]; //Check tensor declaration for clarification on how to access
    memcpy(dst, g_npy_buf + data_offset, expected_data_bytes);

    //Send to global variables so that it can be transmitted later
    g_last_dst = dst;
    g_last_payload_bytes = expected_data_bytes;   // or payload_bytes
    g_block_ready = 1;

    xil_printf("Stored block start=%lu\r\n", (unsigned long)g_start_index);
    xil_printf("First val: ");
    print_double_fixed(dst[0]);
    xil_printf("\r\n");

    return 0;
}

void udp_send_data(void){
	if (SEND_DATA_OUT){
		/*--------------------UDP TRANSMIT----------------------*/
		//For sending output back to UDP for current chunk
		if (!pcb_tx) {
			xil_printf("TX: pcb not initialized\n\r");
			return;
		}
		if (!g_pc_known) {
			xil_printf("TX: PC not learned yet; skipping send\n\r");
			return;
		}
		err_t e;

		e = send_header(pcb_tx, &g_pc_addr, UDP_SEND_PORT, "BLK64___", g_last_payload_bytes);
		if (e != ERR_OK) return;

		e = send_bytes_chunked(pcb_tx, &g_pc_addr, UDP_SEND_PORT,
		                       (const uint8_t*)g_last_dst, g_last_payload_bytes);
		if (e != ERR_OK) return;

		g_block_ready = 0;
	}
}

void udp_send_NMSE(double *nmse_data, int size)
{
    if (!SEND_DATA_OUT)
        return;
        
    if (!pcb_tx) {
        xil_printf("TX: pcb not initialized (NMSE)\n\r");
        return;
    }
    if (!g_pc_known) {
        xil_printf("TX: PC not learned yet; skipping NMSE send\n\r");
        return;
    }
    
    uint32_t payload_bytes = (uint32_t)size * sizeof(double);
    
    err_t e;
    /* Send header with file identifier "NMSE64__" (8 chars) */
    e = send_header(pcb_tx, &g_pc_addr, UDP_SEND_PORT, "NMSE64__", payload_bytes);
    if (e != ERR_OK) {
        xil_printf("TX: NMSE header send failed, err=%d\r\n", e);
        return;
    }
    
    /* Send the actual NMSE data in chunks */
    e = send_bytes_chunked(pcb_tx, &g_pc_addr, UDP_SEND_PORT,
                           (const uint8_t*)nmse_data, payload_bytes);
    if (e != ERR_OK) {
        xil_printf("TX: NMSE data send failed, err=%d\r\n", e);
        return;
    }
    
    xil_printf("TX: NMSE sent successfully, %d doubles (%lu bytes)\r\n", 
               size, (unsigned long)payload_bytes);
}

void udp_send_channel_data(double *data, int size){

	/*--------------------UDP TRANSMIT----------------------*/
	//For sending output back to UDP for current chunk
	if (!pcb_tx) {
		xil_printf("TX: pcb not initialized\n\r");
		return;
	}
	if (!g_pc_known) {
		xil_printf("TX: PC not learned yet; skipping send\n\r");
		return;
	}
	err_t e;

	e = send_header(pcb_tx, &g_pc_addr, UDP_SEND_PORT, "BLK64___", g_last_payload_bytes);
	if (e != ERR_OK) return;

	e = send_bytes_chunked(pcb_tx, &g_pc_addr, UDP_SEND_PORT,
						   (const uint8_t*)g_last_dst, g_last_payload_bytes);
	if (e != ERR_OK) return;

	g_block_ready = 0;

}

void udp_recv_file(void *arg, struct udp_pcb *pcb, struct pbuf *p,
                   const ip_addr_t *addr, u16_t port)
{
    if (!p) return;

    // Quick check: need at least magic+ver+type+reserved
    if (p->tot_len < 8) { pbuf_free(p); return; }

    uint8_t base[8];
    cp(p, base, 8, 0);

    if (memcmp(base, MAGIC_STR, 4) != 0) { pbuf_free(p); return; }
    uint8_t ver  = base[4];
    uint8_t type = base[5];
    if (ver != VER_EXPECTED) { pbuf_free(p); return; }

    // ---------------- INIT ----------------
    if (type == TYPE_INIT) {
        // Python INIT_FMT = "!4sBBH I I I I I I I"
        // offsets:
        // 0 magic(4)
        // 4 ver(1)
        // 5 type(1)
        // 6 reserved(2)
        // 8 sid(4)
        // 12 T(4)
        // 16 B(4)
        // 20 d1(4)
        // 24 d2(4)
        // 28 d3(4)
        // 32 d4(4)
        if (p->tot_len < 36) { pbuf_free(p); return; }

        uint32_t sid_be, T_be, B_be, d1_be, d2_be, d3_be, d4_be;
        cp(p, &sid_be, 4, 8);
        cp(p, &T_be,   4, 12);
        cp(p, &B_be,   4, 16);
        cp(p, &d1_be,  4, 20);
        cp(p, &d2_be,  4, 24);
        cp(p, &d3_be,  4, 28);
        cp(p, &d4_be,  4, 32);

        g_sid = ntohl(sid_be);
        g_T   = (int)ntohl(T_be);
        g_B   = (int)ntohl(B_be);

        int d1 = (int)ntohl(d1_be);
        int d2 = (int)ntohl(d2_be);
        int d3 = (int)ntohl(d3_be);
        int d4 = (int)ntohl(d4_be);

        //Check for INIT block to ensure correct data dimensions are being sent

        if (d1 != D1_EXP || d2 != D2_EXP || d3 != D3_EXP || d4 != D4_EXP) {
            xil_printf("INIT dims mismatch\r\n");
            g_inited = 0;
            reset_block();
            pbuf_free(p);
            return;
        }
        if (g_T <= 0 || g_T > T_MAX || g_B <= 0 || g_B > B_MAX) {
            xil_printf("INIT T/B out of range\r\n");
            g_inited = 0;
            reset_block();
            pbuf_free(p);
            return;
        }

        g_inited = 1;
        reset_block();

        xil_printf("INIT ok: sid=%lu T=%d B=%d\r\n",
                   (unsigned long)g_sid, g_T, g_B);

        //Used to copy pc_addr, used in sending data back to PC
		ip_addr_copy(g_pc_addr, *addr);
		g_pc_known = 1;
		WESN_DONE = 0; //New data incoming, WESN is reset

        pbuf_free(p);



        return;
    }

    // ---------------- DATA ----------------
    if (type == TYPE_DATA) {
        if (!g_inited) { pbuf_free(p); return; }
        // Python DATA_FMT = "!4sBBH I I I H I H"
        // offsets:
        // 0 magic(4)
        // 4 ver(1)
        // 5 type(1)
        // 6 reserved(2)
        // 8 sid(4)
        // 12 start(4)
        // 16 npy_len(4)
        // 20 chunk_size(u16)
        // 22 seq(u32)
        // 26 payload_len(u16)
        // 28 payload...
        if (p->tot_len < 28) { pbuf_free(p); return; }

        uint32_t sid_be, start_be, npylen_be, seq_be;
        uint16_t csize_be, paylen_be;

        cp(p, &sid_be,    4, 8);
        cp(p, &start_be,  4, 12);
        cp(p, &npylen_be, 4, 16);
        cp(p, &csize_be,  2, 20);
        cp(p, &seq_be,    4, 22);
        cp(p, &paylen_be, 2, 26);

        uint32_t sid = ntohl(sid_be);
        uint32_t start_index = ntohl(start_be);
        uint32_t npy_len = ntohl(npylen_be);
        uint16_t chunk_size = ntohs(csize_be);
        uint32_t seq = ntohl(seq_be);
        uint16_t paylen = ntohs(paylen_be);

        if (sid != g_sid) { pbuf_free(p); return; }
        if (npy_len == 0 || npy_len > BLOCK_NPY_MAX) { pbuf_free(p); return; }
        if (chunk_size == 0) { pbuf_free(p); return; }

        // Start a new block if needed
        if (g_npy_len == 0 ||
            start_index != g_start_index ||
            npy_len != g_npy_len ||
            chunk_size != g_chunk_size)
        {
            reset_block();
            g_start_index = start_index;
            g_npy_len = npy_len;
            g_chunk_size = chunk_size;

            g_chunks_expected = (g_npy_len + g_chunk_size - 1) / g_chunk_size;
            if (g_chunks_expected > CHUNK_SEEN_MAX) {
                xil_printf("Too many chunks for bitmap\r\n");
                reset_block();
                pbuf_free(p);
                return;
            }
        }
        // Validate payload and copy it into reassembly buffer
        uint32_t off = seq * (uint32_t)g_chunk_size;
        if (off + paylen > g_npy_len) { pbuf_free(p); return; }
        if (p->tot_len < (uint16_t)(28 + paylen)) { pbuf_free(p); return; }

        // Count chunk only once
        if (seq < CHUNK_SEEN_MAX && g_chunk_seen[seq] == 0) {
            g_chunk_seen[seq] = 1;
            g_chunks_received++;
        }

        cp(p, g_npy_buf + off, paylen, 28);

        // If we received all chunks for this block, commit
        uint32_t completed_start = g_start_index;
        if (all_chunks_received()) {
            (void)commit_block_into_tensor();
            reset_block();
        }

        pbuf_free(p);

        /*
         *
         * Improvements: Do realtime to reduce processing power and increase efficiency

         *	Notes:
         *		-Assume g_B is window size (0,1,...g_B-1) == (0, 1, .... t-1)
         *
         * */
        xil_printf("[DEBUG] udp_recv_file() -> g_start_index:%d\n\r", g_start_index); //TODO: check why this prints three times

        if((completed_start == g_T-g_B) && (WESN_DONE != 1)){ //All data has been recieved from PC
        	double *current_channel_window_real = (double*)malloc(g_B*TX_ANT*RX_ANT*RB_SZ*sizeof(double)); //default (10,2,2,32)
        	double *current_channel_window_imag = (double*)malloc(g_B*TX_ANT*RX_ANT*RB_SZ*sizeof(double)); //default (10,2,2,32)
        	double *NMSE_data = (double*)malloc(2*NMSE_SZ*sizeof(double)); //2 represents outdated nmse(idx 0) and predicted nmse(idx 1)

        	int range = 0;

        	for(int i = 0; i <= g_T-g_B; i++){ //double check condition //should iterate through each window size (0, 10, ... 390) if T_MAX==400
        		range = 0; //get 0...t-1 index
                double *predicted_channel_real; //default (10,2,2,32)
                double *predicted_channel_imag; //default (10,2,2,32)
                double *predicted_channel_final= (double*)malloc(TRAIL_ELEMS*sizeof(double)); //default (10,2,2,32)
                double *outdated_channel;
                double *ground_truth;
                xil_printf("[DEBUG] Processing window starting at index %d\n\r", i);
				for(int j = i; j < g_B + i; j++){
					memcpy(&current_channel_window_real[(size_t)range * TRAIL_ELEMS], //@todo: change trail elems
							&g_tensor[(size_t)(j) * TRAIL_ELEMS],
							(size_t)TX_ANT*RX_ANT*RB_SZ * sizeof(double));
					memcpy(&current_channel_window_imag[(size_t)range* TRAIL_ELEMS],
						   &g_tensor[(size_t)(j)* TRAIL_ELEMS],
						   (size_t)TX_ANT*RX_ANT*RB_SZ * sizeof(double));
					range++;

				}

        		//predicted_channel_calc
				//predicted_channel_real = channel_pred_sys(current_channel_window_real, g_B);//found in channel_pred_sys.c (must return a malloc pointer***)
                //predicted_channel_imag = channel_pred_sys(current_channel_window_imag, g_B);//found in channel_pred_sys.c (must return a malloc pointer***)
                //if (predicted_channel_real == NULL || predicted_channel_imag == NULL) return;
                //@todo: concatenate real and imag to feed into NMSE function
                //predicted_channel_final 
				/*NMSE Processing*/
                xil_printf("[DEBUG] Calculating NMSE for window starting at index %d\n\r", i);
				for(int k = 0; k < NMSE_SZ; k++){
					// NMSE_data[2*i] = base_NMSE_calc(ground_truth, predicted_channel_final, TX_ANT*RX_ANT*RB_SZ);//predicted NMSE
					// NMSE_data[2*i+1] = base_NMSE_calc(ground_truth, outdated_channel, TX_ANT*RX_ANT*RB_SZ);//outdated NMSE
                    NMSE_data[2*k] = 1.0;//outdated NMSE
                    NMSE_data[2*k+1] = 0.5;//predicted NMSE
				}


				/*Send NMSE*/
                xil_printf("[DEBUG] Sending NMSE for window starting at index %d\n\r", i);
				udp_send_NMSE(NMSE_data, NMSE_SZ*2);

                free(predicted_channel_real);
                free(predicted_channel_imag);
                free(predicted_channel_final);
                // free(outdated_channel); //USE THESE ONCE ASSIGNED
                // free(ground_truth);
        	}
        	free(NMSE_data);
        	free(current_channel_window_real);
            free(current_channel_window_imag);
        	WESN_DONE = 1;
            xil_printf("[DEBUG] All windows processed and NMSE sent\n\r");
        }
        //send data back to FPGA
        //if(g_block_ready) udp_send_data(); // will send the data recieved back
        return;
    }

    pbuf_free(p);
}




