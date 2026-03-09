#ifndef __CHANNEL_PRED_SYS_H_
#define __CHANNEL_PRED_SYS_H_

#ifdef __cplusplus
extern "C" {
#endif

// #include "lwip/udp.h"
// #include "lwip/pbuf.h"
// #include "xil_printf.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h> //remove later, used for debugging
#include <stddef.h>
#include "twomode_main.h"
#include "twomode_core.h"
#include "helper.h"

#define DEBUG 0
#define FIXED_WEIGHTS 1 // Set to 1 to use fixed weights, 0 for random weights

/*-----------------channel_prediction constants---------------------------*/
// Data dimensions (default values, can be changed)
#define N_SZ   10 //Size of Number blocks
#define TX_ANT (D1_EXP)//# of Transceiver Antenna
#define RX_ANT (D2_EXP)//# of Receiver Antenna
#define RB_SZ (D4_EXP)//Resource block size

//Init Weights parameters
// fixed window length used throughout this channel_pred_sys; adjust as needed
#define WINDOW_LENGTH 3
#define SPARSITY 0.4
#define SPECTRAL_RADIUS 0.5
#define INPUT_SCALE 0.8
#define REG 1.0
#define RNG_SEED 10

#define WINDOW_LENGTH 3 //Window Size (Default is 3, must change code if different)
#define FEATURE_DIM (RX_ANT * TX_ANT * (WINDOW_LENGTH) * 2)



/*-------------------*/
// T_ is the number of usable time steps (N-1) where N is defined
// by the channel prediction system.  We use the compile-time macro
// N_SZ from twomode_main.h so that the core functions remain consistent
// with the dimensions used by channel_pred_sys.
#define T_ (N_SZ - 1)  /* e.g. if N_SZ==10 then T_==9 */


#define A_ANT (RX_ANT)            // parameter meaning: number of Rx antennas A * IQ = 4
#define B_ANT (TX_ANT)            // parameter meaning: number of Tx antennas B = 2
#define WIN   (WINDOW_LENGTH)            // parameter meaning: window length win = 3
#define YCOL  (WIN*B_ANT)  // parameter meaning: Y columns = win*D = 3*2 = 6
#define SCOL  (YCOL)       // parameter meaning: reservoir state columns = 6
#define CCOL  (SCOL*2)     // parameter meaning: concat columns = 6*2 = 12
#define SROW  (A_ANT*CCOL) // parameter meaning: stacked rows = 4*12 = 48
#define HROW  (A_ANT*B_ANT)// parameter meaning: output rows = 4*2 = 8


#define CH_LEN 8



// Channel Prediction function, returns array of NMSE values

// Implementation currently returns `NMSE_Result*` and accepts the
// data pointer and `total_N`. Keep the prototype in sync.
double* channel_pred_sys(double *data, int total_N);

// take a 3‑D signal of shape [T, N_r, N_t] stored in row-major order
// (time runs slowest) and form a causal sliding window of length L
// along the time axis.  The output has shape [T, N_r, L * N_t]; each
// row k contains samples at times k, k-1, …, k-(L-1) with zero
// padding for negative indices.  The caller must free the returned
// buffer when finished.
//
//   Y_3D : input array, size = T * N_r * N_t
//   T     : number of time steps
//   N_r   : number of receive antennas (rows)
//   N_t   : number of transmit antennas (cols)
//   L     : window length
//
// Returns newly allocated array or NULL on failure.
double* form_window_input_signal(const double *Y_3D, int T, int N_r, int N_t, int L);

// Initialize weight matrices for the WESN predictor.
// Initializes W_res_left, W_res_right, W_in_left, W_in_right with
// appropriate random values and spectral radius scaling.
// The global arrays (W_res_left, W_res_right, W_in_left, W_in_right)
// must be allocated before calling this function.
//
//   d_left:        left reservoir dimension (= N_r)
//   d_right:       right reservoir dimension (= L * N_t)
//   N_r:           number of receive antennas
//   N_t:           number of transmit antennas
//   window_length: window length L
//
// Returns 0 on success, non-zero on failure.
int init_weights(int N_r, int N_t, int window_length);

// Build reservoir feature matrix for one RB and append to S_all.
// Implements windowing -> state transit -> concatenation -> column stacking.
//
//   S_all:        output matrix, pre-allocated to (d_left*(d_right+L*N_t))×(F*T)
//   col:          column index where to append (typically f*T for RB f)
//   Y_in:         input channel data, shape [T, N_r, N_t]
//   T:            number of time steps
//   N_r:          number of receive antennas
//   N_t:          number of transmit antennas
//   W_in_left:    input weight matrix, shape [d_left, L*N_t]
//   W_in_right:   input weight matrix, shape [d_left, L*N_t]                                       FIXME: [d_right, d_right]?
//   W_res_left:   reservoir weight matrix, shape [d_left, d_left]                                  FIXME: shape [d_left, L*N_t]?
//   W_res_right:  reservoir weight matrix, shape [d_right, d_right]
//   input_scale:  scaling factor for input
//   d_left:       left reservoir dimension
//   d_right:      right reservoir dimension
//   window_length: window length L
void append_S(double *S_all, int col,
              const double *Y_in, int T, int N_r, int N_t,
              double *W_in_left, double *W_in_right,
              double *W_res_left, double *W_res_right,
              double input_scale, int window_length);


// Build target matrix for one RB and append to Y_all.
// Implements column stacking of the ground truth channel.
//
//   Y_all:   output matrix, pre-allocated to (N_r*N_t)×(F*T)
//   col:     column index where to append (typically f*T for RB f)
//   Y_out:   output channel data, shape [T, N_r, N_t]
//   T:       number of time steps
//   N_r:     number of receive antennas
//   N_t:     number of transmit antennas
void append_Y(double *Y_all, int col,
              const double *Y_out, int T, int N_r, int N_t);

// Convert a T×N_r×N_t array into a matrix of size (N_r*N_t) × T.
// The input is stored in row-major order with time running slowest.
// Caller must free the returned pointer.
double *column_stack_from_3d(const double *Y3, int T, int N_r, int N_t);

// Append T columns from Y3 into an existing matrix dst at column index start_col.
void column_stack_append(double *dst, int total_cols, int start_col,
                         const double *Y3, int T,
                         int N_r, int N_t);

                         /**
 * Reorders (transposes) a 5D matrix by permuting dimensions.
 * 
 * Example: matrix_reorder(matrix, 400,2,2,2,6, 3,0,1,2,4)
 * - Original shape: [400][2][2][2][6] (dim0=400, dim1=2, dim2=2, dim3=2, dim4=6)
 * - Permutation: new_dim0←old_dim3, new_dim1←old_dim0, new_dim2←old_dim1, new_dim3←old_dim2, new_dim4←old_dim4
 * - Result shape: [2][400][2][2][6]
 * 
 * @param matrix_orig  Original flat array in row-major order
 * @param dim0-dim4    Original dimensions
 * @param new_dim1-5   Permutation: which old dim (0-4) goes to new position 0-4
 * @return             Newly allocated reordered array (NULL on failure)
 */
double* matrix_reorder(double *matrix_orig, 
                       int dim0, int dim1, int dim2, int dim3, int dim4,
                       int new_dim1, int new_dim2, int new_dim3, int new_dim4, int new_dim5);

// solve ridge regression readout and return newly allocated weight matrix
// caller must free the result
//   S_all: feature matrix (feature_dim × (F*T))
//   Y_all: target matrix (N_r*N_t × (F*T))
double *single_Readout_Solve(double *S_all, double *Y_all, int N);

#ifdef __cplusplus
}
#endif

#endif
