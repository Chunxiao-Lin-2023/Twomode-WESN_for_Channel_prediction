#include "twomode_core.h"
#include <stdlib.h>
#include <math.h>

/*
 * TWO-MODE WESN CORE (Author: Ahmed Malik, MEng | Last Updated: 2/25/26)
 *
 *  This twomode_core file is used to do core calculations for WESN training and prediction.
 *
 *
 *  NOTES: USED MALLOC AND FREE FOR VARIABLES IN THIS FILE. This allows for dynamic memory allocation and deallocation,
 *  which is important for managing memory usage in the program. However, it is important to ensure that all
 *  allocated memory is properly freed to avoid memory leaks.
 *
 *  Future Plans: Incorporate PL (AXI BRAM) into this design flow (create seperate file?).
 * */
//NOTES:
// Total time slots: channels_all = [h(0), h(1), ... h(N-1)] = [h(0), h(1), ... h(9)]
// use [h(0), h(1), ... h(N-2)] as input to predict h(N-1)
// use [h(0), h(1), ... h(8)] as input to predict h(9)

// Training: Train_channels = [h(0), h(1), ... h(N-2)] = [h(0), h(1), ... h(8)]
//           divided into 8 pairs of input-output.
//           Train_channels_in = [h(0), h(1), ... h(N-3)] = [h(0), h(1), ... h(7)]
//           Train_channels_gt = [h(1), h(2), ... h(N-2)] = [h(1), h(2), ... h(8)]


// Prediction: Test_channels = [h(1), h(2), ... h(N-1)] = [h(1), h(2), ... h(9)]
//           divided into 8 pairs of input-output.
//           Test_channels_in = [h(1), h(2), ... h(N-2)] = [h(1), h(2), ... h(8)]
//           Test_channels_gt = [h(2), h(3), ... h(N-1)] = [h(2), h(3), ... h(9)]
// Final output: predicted h(N-1) = predicted h(9)

// NMSE(predicted h(N-1), golden h(N-1)) = NMSE(predicted h(9), h(9))
// NMSE_baseline = NMSE(h(N-2), h(N-1)) = NMSE(h(8), h(9))

/****************************************HELPER FUNCTIONS ********************************************/

// ------------------------------------------------------------
// NMSE calculation
// NMSE = sum_i (A[i] - B[i])^2 / sum_i (A[i])^2
// A is treated as reference (ground truth)
// ------------------------------------------------------------
double base_NMSE_calc(const double *channels_A,
                      const double *channels_B,
                      int length)
{
    double mse = 0.0;
    double power = 0.0;

    for (int i = 0; i < length; ++i) {
        double error = channels_A[i] - channels_B[i];
        mse += error * error;
        power += channels_A[i] * channels_A[i];
    }

    return (power > 0.0) ? (mse / power) : 0.0;
}


/* transpose helper: returns new matrix cols×rows */
double* matrix_transpose(double *a, int rows, int cols){
    if (!a || rows <= 0 || cols <= 0) return NULL;
    double *t = (double*) malloc(sizeof(double) * rows * cols);
    if (!t) return NULL;
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            t[j * rows + i] = a[i * cols + j];
        }
    }
    return t;
}


double* matrix_add(double *a, double *b, int dim1, int dim2){
    // (dim1, dim2) + (dim1, dim2) = (dim1, dim2)
    if (!a || !b || dim1 <= 0 || dim2 <= 0) return NULL;
    int size = dim1 * dim2;
    double *res = (double*) malloc(sizeof(double) * size);
    if (!res) return NULL;
    for (int i = 0; i < size; ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}


double* matrix_mult(double *a, double *b, int dim1, int dim2, int dim3){
 // (dim1, dim2) x (dim2, dim3) = (dim1, dim3)
//  if(DEBUG) HELPER_LOGD("[DEBUG] matrix_mult called with dimensions: a(%d, %d), b(%d, %d)\n", dim1, dim2, dim2, dim3);
 
 if (!a || !b || dim1 <= 0 || dim2 <= 0 || dim3 <= 0) return NULL;
//  if(DEBUG) HELPER_LOGD("[DEBUG] checkpoint1");

 double *res = (double*) malloc(sizeof(double) * dim1 * dim3);
// if(DEBUG) HELPER_LOGD("[DEBUG] checkpoint2");

 if (!res) return NULL;
//  if(DEBUG) HELPER_LOGD("[DEBUG] checkpoint3");

 for (int i = 0; i < dim1; ++i) {
  for (int j = 0; j < dim3; ++j) {
//   if(DEBUG) HELPER_LOGD("[DEBUG] checkpoint4");
   double sum = 0.0;
   for (int k = 0; k < dim2; ++k) {
    // if(DEBUG) HELPER_LOGD("[DEBUG] checkpoint5");
    // if(DEBUG) HELPER_LOGD("i=%d, j=%d, k=%d\n", i, j, k);
    // if(DEBUG) HELPER_LOGD("[DEBUG] i*dim2 + k = %d, k*dim3 + j = %d\n", i*dim2 + k, k*dim3 + j);
    // if(DEBUG) HELPER_LOGD("[DEBUG] b[0][0] = %f", b[0 * dim2 + 0]);
    // if(DEBUG) HELPER_LOGD("[DEBUG] a[0][0] = %f", a[0 * dim2 + 0]);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Multiplying a[%d][%d] * b[%d][%d] = %f * %f\n", i, k, k, j, a[i * dim2 + k], b[k * dim3 + j]);
    sum += a[i * dim2 + k] * b[k * dim3 + j];
    // if(DEBUG) HELPER_LOGD("[DEBUG] DONE");

   }
   res[i * dim3 + j] = sum;
  }
 }
 return res;
}



/* scalar tanh wrapper using the C standard library */
static double tanh_scalar(double x){
 return tanh(x);
}


void tanh_matrix(double* matrix, int row, int col){
 if (!matrix || row <= 0 || col <= 0) return;
 int size = row * col;
 for (int i = 0; i < size; ++i) {
  matrix[i] = tanh_scalar(matrix[i]);
 }
}


/* Apply tanh to complex data stored interleaved as [Re,Im, Re,Im, ...]
 * Matches Python implementation: np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y)) */
void complex_tanh_interleaved(double *data, int row, int col){
 if (!data || row <= 0 || col <= 0) return;
 int elems = row * col;
 for (int i = 0; i < elems; ++i){
  double re = data[2*i];
  double im = data[2*i + 1];
  data[2*i]     = tanh_scalar(re);
  data[2*i + 1] = tanh_scalar(im);
 }
}

/* Apply tanh to complex data given as separate real and imag arrays */
void complex_tanh_separate(double *real, double *imag, int row, int col){
 if ((!real && !imag) || row <= 0 || col <= 0) return;
 int elems = row * col;
 if (real) for (int i = 0; i < elems; ++i) real[i] = tanh_scalar(real[i]);
 if (imag) for (int i = 0; i < elems; ++i) imag[i] = tanh_scalar(imag[i]);
}

/*
 * In-place Gauss-Jordan inversion of an n x n matrix.
 * The matrix is overwritten with its inverse on success.
 * Returns 0 on success, non-zero if the matrix is singular.
 */
static int mat_inverse(double *A, int n) {
    if (!A || n <= 0) return -1;
    int i, j, k;
    /* create augmented matrix [A | I] */
    double *aug = (double*) calloc(n * 2 * n, sizeof(double));
    if (!aug) return -1;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            aug[i*(2*n) + j] = A[i*n + j];
        }
        aug[i*(2*n) + (n + i)] = 1.0;
    }
    /* forward elimination */
    for (i = 0; i < n; ++i) {
        /* pivot selection: find max in column i */
        double maxval = fabs(aug[i*(2*n) + i]);
        int maxrow = i;
        for (k = i+1; k < n; ++k) {
            double val = fabs(aug[k*(2*n) + i]);
            if (val > maxval) {
                maxval = val;
                maxrow = k;
            }
        }
        if (maxrow != i) {
            /* swap rows i and maxrow */
            for (j = 0; j < 2*n; ++j) {
                double tmp = aug[i*(2*n) + j];
                aug[i*(2*n) + j] = aug[maxrow*(2*n) + j];
                aug[maxrow*(2*n) + j] = tmp;
            }
        }
        double pivot = aug[i*(2*n) + i];
        if (fabs(pivot) < 1e-12) { free(aug); return -1; }
        /* normalize pivot row */
        for (j = 0; j < 2*n; ++j) aug[i*(2*n) + j] /= pivot;
        /* eliminate other rows */
        for (k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = aug[k*(2*n) + i];
            for (j = 0; j < 2*n; ++j) {
                aug[k*(2*n) + j] -= factor * aug[i*(2*n) + j];
            }
        }
    }
    /* copy inverse back to A */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            A[i*n + j] = aug[i*(2*n) + (n + j)];
        }
    }
    free(aug);
    return 0;
}


/*
 * Compute regularized pseudo-inverse of matrix X (F x T).
 * Returns newly allocated matrix of size T x F (row-major) or NULL on failure.
 * The caller must free the result when finished.
 *
 * G = X^T * inv(X X^T + reg * I_F)
 *
 * This function is used by single_Readout_Solve in channel_pred_sys.c, so it
 * must be non-static to allow cross-file linkage.
 */
double* reg_pseudo_inv(double *X, int F, int T, double reg) {
    if (!X || F <= 0 || T <= 0) return NULL;

    /* compute X^T first */
    double *Xt = matrix_transpose(X, F, T); // dims T x F
    if (!Xt) return NULL;

    /* G = X * X^T -> dims F x F */
    double *G = matrix_mult(X, Xt, F, T, F);
    if (!G) { free(Xt); return NULL; }

    /* add regularization to diagonal */
    for (int i = 0; i < F; ++i) {
        G[i * F + i] += reg;
    }

    /* invert G */
    if (mat_inverse(G, F) != 0) {
        free(G);
        free(Xt);
        return NULL;
    }

    /* R = X^T * G_inv -> dims T x F */
    double *R = matrix_mult(Xt, G, T, F, F);
    free(G);
    free(Xt);
    return R;
}

double* concat_horizontal_3_matrix(const double *A, //does windowing
                          const double *B,
                          const double *C,
                          int rows,
                          int cols_each)
{
    int total_cols = cols_each * 3;
    double *R = malloc(sizeof(double) * rows * total_cols);
    if (!R) return NULL;

    for (int i = 0; i < rows; ++i) {
        memcpy(&R[i*total_cols],
               &A[i*cols_each],
               cols_each*sizeof(double));

        memcpy(&R[i*total_cols + cols_each],
               &B[i*cols_each],
               cols_each*sizeof(double));

        memcpy(&R[i*total_cols + 2*cols_each],
               &C[i*cols_each],
               cols_each*sizeof(double));
    }

    return R;
}


/****************************************WESN KEY FUNCTIONS ********************************************/
/*
 *  update_Reservoir():
 *  S(N) = tanh( (Win_left * [h(N) * Win_right) + (W_res_left * S(N-1) * W_res_right))
 *          				  h(N-1)
 *             			   	  h(N-2)]
 *
 *   Matrices : Size
 *  Win_left, W_res_left      : [4,4]
 *  h(N)				      : [4,2]
 *  S(N)               		  : [4,6]
 *  Win_right, W_res_right    : [6,6]
 *
 *
 *  Notes:
 *      -Determining if N-2 and N-1 are < 0 is done in twomode_main.c,
 *       so this function assumes that the correct values are passed in.
 *      -If N-2 or N-1 are < 0, they are set to 0 in twomode_main.c,
 *       so this function does not need to handle that case.
 *
 *  */
void update_Reservoir(double *pred_res_state, double* prev_res_state, double *channels_windowed, double *W_in_left,
                                                double *W_in_right, double *W_res_left, double *W_res_right, 
                                                int RX_ANT, int TX_ANT, int WIN){
    /*
     * Reservoir update for a single time step.
     * pred_res_state and prev_res_state point to pre‑allocated arrays of
     * size A_ANT×SCOL (currently 2×6 = 12).  The output is written into
     * pred_res_state in-place.
     * channels_windowed also has length A_ANT×YCOL (4×6).  Weight matrices
     * are expected to have the hardcoded shapes documented in the header.
     */

    /* first term: Win_left * channels_windowed * Win_right */
    int A_ANT = RX_ANT;
    int B_ANT = TX_ANT;
    int YCOL = WIN * B_ANT;
    int SCOL = YCOL;
    int CCOL = SCOL * 2;
    int SROW = A_ANT * CCOL;
    int HROW = A_ANT * B_ANT;
    
    
    double *temp1 = matrix_mult(W_in_left, channels_windowed, A_ANT, A_ANT, YCOL);
    double *prod1 = matrix_mult(temp1, W_in_right, A_ANT, YCOL, YCOL);
    free(temp1);

    /* second term: W_res_left * prev_res_state * W_res_right */
    double *temp2 = matrix_mult(W_res_left, prev_res_state, A_ANT, A_ANT, SCOL);
    double *prod2 = matrix_mult(temp2, W_res_right, A_ANT, SCOL, SCOL);
    free(temp2);

    /* sum = prod1 + prod2 */
    double *sum = matrix_add(prod1, prod2, A_ANT, SCOL);
    free(prod1);
    free(prod2);

    /* copy result into caller buffer and apply tanh */
    memcpy(pred_res_state, sum, A_ANT * SCOL * sizeof(double));
    tanh_matrix(pred_res_state, A_ANT, SCOL);
    free(sum);
}

/* training_States():
 *
 * 1) Collect states and groundtruth
 *
 * States:
 * S_f = [ {h(1), h(2), ... h(N-1)}, {s(1), s(2), ... s(N-1)} ] | DIM: [4*6*2, (N-1)]
 * S_all = [4*6*2, (N-1) * N_f] | S_f with (N-1) * N_f
 *
 * Ground Truth:
 * Y_f = [{h(2), h(3), ... h(N)}] | DIM: [4*2, (N-1)]
 * Y_all = [4*2, (N-1) * N_f]
 *
 * 2) Training
 *
 * G = reg_pseudo_inv(S_all)
 * DIM: [N-1*N_f, 4*6*2] -> [4*6*2, N-1*N_f]
 *
 * Wout = Yall * G
 * DIM: [4*2, 4*6*2] = [4*2, (N-1) * N_f] * [(N-1) * N_f, 4*6*2]
 *
 *  Notes:
 *      - N is defined as the number of time frames per training sequence.***** CLARIFY AND CHANGE IF NECESSARY
 *      - N should be carefully chosen to ensure memoryleak prevention.
 *
 * */
void feature_Build_Phase(double *N_channels, double *N_pred_res_state, int N, int rb_sz, double *W_out){

	//Created temp variables
    double* S_f = (double*) malloc(4*6*2 * (N-1) * sizeof(double));
    double* S_all = (double*) malloc(4*6*2 * (N-1) * rb_sz * sizeof(double));
    double* Y_f = (double*) malloc(4*2 * (N-1) * sizeof(double));
    double* Y_all = (double*) malloc(4*2 * (N-1) * rb_sz * sizeof(double));

    //(A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms)


    //collect S_all
    //S_f[i*4*6*2 + j*4*6 + k] = pred_res_state[k] for j=0, channels[k] for j=1
    for(int i = 0; i < N-1; i++){ //Double check N****
        for(int j = 0; j < 2; j++){
            if(j == 0){//collect pred_res_state
                for(int k = 0; k < 4*6; k++){
                    S_f[i*4*6*2 + j*4*6 + k] = N_pred_res_state[i*4*6 + k];
                }
            } else {//collect channels
                for(int k = 0; k < 4*6; k++){
                    S_f[i*4*6*2 + j*4*6 + k] = N_channels[i*4*6 + k];
                }
            }
        }
    }
    //S_all[i*(N-1)*4*6*2 + j*4*6*2 + k] = i for N_f, j for N-1, k for 4*6*2
    for(int i = 0; i < rb_sz; i++){
        for(int j = 0; j < N-1; j++){
            for(int k = 0; k < 4*6*2; k++){
                S_all[i*(N-1)*4*6*2 + j*4*6*2 + k] = S_f[j*4*6*2 + k];
            }
        }
    }

 //collect Y_all
    //Y_f[i*4*2 + j] = i for N-1, j for 4*2
    for(int i = 0; i < N-1; i++){ //Double check N****
        for(int j = 0; j < 4*2; j++){
            Y_f[i*4*2 + j] = N_channels[(i+1)*4*2 + j];
        }
    }

    //Y_all[i*(N-1)*4*2 + j*4*2 + k] = i for N_f, j for N-1, k for 4*2
    for(int i = 0; i < rb_sz; i++){
        for(int j = 0; j < N-1; j++){
            for(int k = 0; k < 4*2; k++){
                Y_all[i*(N-1)*4*2 + j*4*2 + k] = Y_f[j*4*2 + k];
            }
        }
    }



    free(S_f);
    free(S_all);
    free(Y_f);
    free(Y_all);
}


/* predict_States():
 *
 * Y_3D_orig = Test_input : [h(2), h(3), ... h(N)] | DIM: [(N-1), 4, 2]
 *
 * Y_3D = (windowed Y_3D_orig) | DIM: [(N-1), 4, 6]
 *
 * S_3D = update_reservoir(Y_3D) | DIM: [(N-1), 4, 6]
 *
 *  S_3D = concat(S_3D, Y_3D) | DIM: [(N-1), 4, 6*2]
 *
 *  S = (Stack of S_3D) | DIM: [4*6*2, (N-1)]
 *
 *  h_pred =        Wout     *     S
 *  [4*2*(N-1)] = [4*2, 4*6*2] * [4*6*2, (N-1)]
 *
 *  recieve and extract h_pred[N+1]:
 *
 *
 * */
 
 
void predict_States(double *channels, double *pred_channels, double *W_in_left, double* W_res_left,
										double *W_in_right, double *W_res_right, double *W_out, int T, int RX_ANT, 
                    int TX_ANT, int window_length) {
	    // T is now a parameter, passed in by the caller (e.g., total_N - 2)
    // for the channel training set or prediction set.
    // NOTE: This implementation follows the dimensions in the comment block:
    // Y_3D_orig: (T, 2, 2) where T = (N-1)
    // Y_3D     : (T, 2, 6) windowed with WIN=3 -> 3*2 = 6
    // S_3D     : (T, 2, 6)
    // concat   : (T, 2, 12)
    // S        : (24, T) where 24 = 2*6*2
    // h_pred   : (4,  T) where 4  = 2*2

    int A_ANT = RX_ANT;
    int B_ANT = TX_ANT;
    int WIN = window_length;
    int YCOL = WIN * B_ANT;
    int SCOL = YCOL;
    int CCOL = SCOL * 2;
    int SROW = A_ANT * CCOL;
    int HROW = A_ANT * B_ANT;
    // ------------------ Temp buffers ------------------
    double *Y_3D = (double*)malloc(sizeof(double) * T * A_ANT * YCOL);
    // parameter meaning: alloc Y_3D of size (T,2,6) = (T * A_ANT * YCOL) doubles

    double *S_3D = (double*)malloc(sizeof(double) * T * A_ANT * SCOL);
    // parameter meaning: alloc S_3D of size (T,2,6) = (T * A_ANT * SCOL) doubles

    double *S_3D_concat = (double*)malloc(sizeof(double) * T * A_ANT * CCOL);
    // parameter meaning: alloc concat of size (T,2,12) = (T * A_ANT * CCOL) doubles

    double *S = (double*)malloc(sizeof(double) * SROW * T);
    // parameter meaning: alloc stacked S of size (24,T) = (SROW * T) doubles, where SROW=24

    double *h_pred = (double*)malloc(sizeof(double) * HROW * T);
    // parameter meaning: alloc h_pred of size (4,T) = (HROW * T) doubles, where HROW=4
    
    
    // if(DEBUG) HELPER_LOGD("[DEBUG] Allocated Y_3D buffer for prediction with size: %d doubles\n", T * A_ANT * YCOL);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Allocated S_3D buffer for prediction with size: %d doubles\n", T * A_ANT * SCOL);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Allocated S_3D_concat buffer for prediction with size: %d doubles\n", T * A_ANT * CCOL);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Allocated S buffer for prediction with size: %d doubles\n", SROW * T);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Allocated h_pred buffer for prediction with size: %d doubles\n", HROW * T);




    // ------------------ (1) Y_3D creation: windowed Y_3D_orig ------------------
    // channels is Y_3D_orig: shape (T,2,2) with T=(N-1)
    for (int t = 0; t < T; ++t) {                    // parameter meaning: time index t in [0, T-1]
        for (int ant = 0; ant < A_ANT; ++ant) {      // parameter meaning: antenna index ant in [0,3] because A_ANT=4
            for (int j = 0; j < WIN; ++j) {          // parameter meaning: window lag j in [0,2] because WIN=3
                int src_t = t - j;                   // parameter meaning: source time src_t = t-j to pick past samples
                for (int k = 0; k < B_ANT; ++k) {    // parameter meaning: channel component k in [0,1] because B_ANT=2
                    double val = 0.0;
                    if (src_t >= 0) {
                        val = channels[(src_t * A_ANT + ant) * B_ANT + k];
                        // parameter meaning:
                        // channels pointer stores Y_3D_orig flattened as (time, ant, k)
                        // A_ANT=4, B_ANT=2 => index = ((src_t*4 + ant)*2 + k)
                    }
                    int col = j * B_ANT + k;         // parameter meaning: Y_3D column col in [0,5] because j*2+k
                    Y_3D[(t * A_ANT + ant) * YCOL + col] = val;
                    // parameter meaning:
                    // Y_3D stores windowed input as (t, ant, col) with YCOL=6
                    // index = ((t*4 + ant)*6 + col)
                }
            }
        }
    }

    // for(int i = 0; i < T * A_ANT * YCOL; i++){ printf("Y_3D[%d]=%f \n", i, Y_3D[i]); }
    
    // if(DEBUG) print_first_n_doubles("[COMPARE] First 10 doubles of Y_3D after windowing", Y_3D, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE] Last 10 doubles of Y_3D after windowing", Y_3D, T * A_ANT * SCOL, 10);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Finished creating windowed Y_3D with shape (T=%d, A_ANT=%d, YCOL=%d)\n", T, A_ANT, YCOL);

    // ------------------ (2) S_3D creation: update_reservoir(Y_3D) ------------------
    double *res_state = (double*)calloc(A_ANT*B_ANT*WIN, sizeof(double));
    double *res_state_transit = (double*)calloc(A_ANT*B_ANT*WIN, sizeof(double));  
    // parameter meaning: pred_res_state is the current reservoir state (2,6) => 2*6 = 12 doubles

    memset(res_state, 0, sizeof(double) * A_ANT * SCOL);

    for (int t = 0; t < T; ++t) {                     // parameter meaning: time index t in [0, T-1]
        double *Y_t = &Y_3D[t * A_ANT * YCOL];
        double *Y_t_scaled = (double*)malloc(sizeof(double) * A_ANT * YCOL);
        for (int i = 0; i < A_ANT * YCOL; i++){
            Y_t_scaled[i] = Y_t[i] * 0.8; }

        // parameter meaning: pointer to Y_3D at time t; size per time slice = (A_ANT*YCOL) = 2*6 = 12

        update_Reservoir(res_state_transit,              // parameter meaning: (2,6) in-place state S(t)
                         res_state, 		          // parameter meaning: (2,6) previous state S(t-1)
						 Y_t_scaled,                  // parameter meaning: (4,6) windowed input Y(t) scaled
                         W_in_left,                   // parameter meaning: Win_left matrix, size (2,2)
                         W_in_right,                  // parameter meaning: Win_right matrix, size (6,6)
                         W_res_left,                  // parameter meaning: Wres_left matrix, size (2,2)
                         W_res_right,
                         RX_ANT,
                         TX_ANT,
                         WIN
                         );                // parameter meaning: Wres_right matrix, size (6,6)

        
        // printf("For reservoir update at time t=%d:\n", t);
        // for(int i = 0; i < A_ANT * B_ANT * WIN; i++){printf("res_state[%d] = %f \n", i, res_state[i]);}
        // for(int i = 0; i < A_ANT * YCOL; i++){printf("Y_t_scaled[%d] = %f \n", i, Y_t_scaled[i]);}
        // for(int i = 0; i < A_ANT * SCOL; i++){printf("res_state_transit[%d] = %f \n", i, res_state_transit[i]);}

        memcpy(&S_3D[t * A_ANT * SCOL], res_state_transit, sizeof(double) * A_ANT * SCOL);
        memcpy(res_state, res_state_transit, sizeof(double) * A_ANT * SCOL);
        // parameter meaning:
        // store S_3D(t,:,:) = pred_res_state
        // S_3D slice size = A_ANT*SCOL = 2*6 = 12 doubles
    }

    // for(int i = 0; i < T * A_ANT * SCOL; i++){ printf("S_3D[%d]=%f \n", i, S_3D[i]); }
    

    free(res_state);
    free(res_state_transit);
    // print_first_n_doubles("[COMPARE] First 10 doubles of S_3D after reservoir updates", S_3D, 10);
    // print_last_n_doubles("[COMPARE] Last 10 doubles of S_3D after reservoir updates", S_3D, T * A_ANT * SCOL, 10);
    
    // if(DEBUG) HELPER_LOGD("[DEBUG] Finished creating S_3D by updating reservoir for T=%d time steps\n", T);

    // ------------------ (3) concat S_3D and Y_3D => S_3D_concat ------------------
    for (int t = 0; t < T; ++t) {                     // parameter meaning: time index t in [0, T-1]
        for (int ant = 0; ant < A_ANT; ++ant) {       // parameter meaning: antenna index ant in [0,3]
            for (int j = 0; j < SCOL; ++j) {          // parameter meaning: left-half column j in [0,5] because SCOL=6
                S_3D_concat[(t * A_ANT + ant) * CCOL + j] =
                    S_3D[(t * A_ANT + ant) * SCOL + j];
                // parameter meaning:
                // concat(t,ant,0..5) = S_3D(t,ant,0..5)
                // CCOL=12, SCOL=6
            }
            for (int j = 0; j < YCOL; ++j) {          // parameter meaning: right-half column j in [0,5] because YCOL=6
                S_3D_concat[(t * A_ANT + ant) * CCOL + (SCOL + j)] =
                    Y_3D[(t * A_ANT + ant) * YCOL + j];
                // parameter meaning:
                // concat(t,ant,6..11) = Y_3D(t,ant,0..5)
                // SCOL=6 so offset is +6
            }
        }
    }

    // for(int i = 0; i < T * A_ANT * CCOL; i++){ printf("S_3D_concat[%d]=%f \n", i, S_3D_concat[i]); }
    // while(1);

    // if(DEBUG) print_first_n_doubles("[COMPARE] First 10 doubles of S_3D_concat after concatenation", S_3D_concat, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE] Last 10 doubles of S_3D_concat after concatenation", S_3D_concat, T * A_ANT * CCOL, 10);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Finished concatenating S_3D and Y_3D into S_3D_concat with shape (T=%d, A_ANT=%d, CCOL=%d)\n", T, A_ANT, CCOL);
    
    

    // ------------------ (4) Stack S_3D_concat to S: S has shape (24, T) ------------------
    // Each time t becomes one column of S (length 48 vector).
    for (int t = 0; t < T; ++t) {                     // parameter meaning: time index t in [0, T-1]
        for (int ant = 0; ant < A_ANT; ++ant) {       // parameter meaning: antenna index ant in [0,3]
            for (int c12 = 0; c12 < CCOL; ++c12) {    // parameter meaning: concat column c12 in [0,11] because CCOL=12
                int idx24 = ant * CCOL + c12;         // parameter meaning: flattened index in [0,47] because 4*12=48
                double v = S_3D_concat[(t * A_ANT + ant) * CCOL + c12];
                // parameter meaning: read concat(t,ant,c12) from S_3D_concat

                S[idx24 * T + t] = v;
                // parameter meaning:
                // S is stored as (row, time) in row-major over rows but with time as fastest index:
                // S(row=idx48, col=t) is at S[idx48*T + t]
            }
        }
    }


    // if(DEBUG) print_first_n_doubles("[COMPARE] First 10 doubles of S after stacking", S, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE] Last 10 doubles of S after stacking", S, SROW * T, 10);
    // if(DEBUG) HELPER_LOGD("[DEBUG] Finished stacking S_3D_concat into S with shape (SROW=%d, T=%d)\n", SROW, T);    

    // ------------------ (5) h_pred = Wout * S ------------------
    // W_out: (4,24), S: (24,T) => h_pred: (4,T)
    // parameter meaning: compute output predictions using learned W_out matrix
    double *h_pred_temp = matrix_mult(W_out, S, HROW, SROW, T);

    // for(int i = 0; i < HROW * T; i++){ printf("h_pred_temp[%d]=%f \n", i, h_pred_temp[i]); }
    // while(1);

    // if(DEBUG) HELPER_LOGD("[DEBUG] Computed h_pred_temp = W_out * S with shape (HROW=%d, T=%d)\n", HROW, T);
    if (!h_pred_temp) {
        if(DEBUG) HELPER_LOGD("[ERROR] Matrix multiplication for h_pred failed due to memory allocation error\n");
        free(Y_3D); free(S_3D); free(S_3D_concat); free(S); free(h_pred);
        return;
    }
    memcpy(h_pred, h_pred_temp, HROW * T * sizeof(double));
    free(h_pred_temp);

    // if(DEBUG) HELPER_LOGD("[DEBUG] Finished computing h_pred with shape (HROW=%d, T=%d)\n", HROW, T);

    // ------------------ (6) Extract h_pred[N+1] ------------------
    // In this (N-1) time-slice formulation, we return the last column as the "next-step" output.
    for (int r = 0; r < HROW; ++r) {                  // parameter meaning: output index r in [0,7]
        pred_channels[r] = h_pred[r * T + (T - 1)];
        // parameter meaning:
        // pred_channels is output vector length 8 (4*2)
        // we pick the last time column (t = T-1)
    }

    for(int i = 0; i < HROW; i++){ printf("h_pred[%d][%d] = %f \n", i, T-1, h_pred[i*T + (T-1)]); }


    // if(DEBUG) HELPER_LOGD("[DEBUG] Extracted predicted h(N) from h_pred and stored in pred_channels\n");

    free(Y_3D);
    free(S_3D);
    free(S_3D_concat);
    free(S);
    free(h_pred);
}

/*
 * final_NMSE_calc():
 *
 *  1) calculate base_NMSE
 *
 *  2) calculate twomode_NMSE
 *
 *  3) return NMSE values
 *
 *
*/
