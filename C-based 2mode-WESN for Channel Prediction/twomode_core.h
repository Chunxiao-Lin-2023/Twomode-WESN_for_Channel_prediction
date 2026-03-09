#ifndef __TWOMODE_CORE_H_
#define __TWOMODE_CORE_H_

#ifdef __cplusplus
extern "C" {
#endif

// #include "lwip/udp.h"
// #include "lwip/pbuf.h"
// #include "xil_printf.h"
#include <string.h>
#include <stdlib.h>
#include "helper.h"

#define DEBUG 0



// Structure to hold NMSE results - defined here to avoid circular includes
typedef struct {
    double base_nmse;
    double pred_nmse;
} NMSE_Result;




//Helper Functions
double* concat_horizontal_3_matrix(const double *A, const double *B, const double *C,
                          	  	  	  int rows, int cols_each);


double base_NMSE_calc(const double *channels_A,
                      const double *channels_B,
                      int length);


double *matrix_mult(double *a, double *b, int dim1, int dim2, int dim3);


// Two Mode WESN key functions

void training_Wout(double *N_channels, double *N_pred_res_state, int N, int N_f, double *W_out);

// Compute regularized pseudo-inverse of matrix X (F x T).
// Returns newly allocated matrix of size T x F (row-major) or NULL on failure.
// The caller must free the result when finished.
double* reg_pseudo_inv(double *X, int F, int T, double reg);

void update_Reservoir(double *pred_res_state, double* prev_res_state, double *channels_windowed,
				double *W_in_left, double *W_in_right, double *W_res_left, double *W_res_right,
        int RX_ANT, int TX_ANT, int WIN);

// perform feed‑forward prediction on a windowed channel sequence
//
//   channels      : input data of length T, shape [T, A_ANT, B_ANT]
//   pred_channels : output buffer length A_ANT*B_ANT (one vector per RB)
//   W_in_left..   : weight matrices
//   W_out         : readout weights (HROW×SROW)
//   T             : number of time steps in `channels` (formerly macro T_)
void predict_States(double *channels, double *pred_channels, double *W_in_left, double* W_res_left,
                                            double *W_in_right, double *W_res_right, double *W_out,
                                            int T, int RX_ANT, int TX_ANT, int window_length);



#ifdef __cplusplus
}
#endif

#endif
