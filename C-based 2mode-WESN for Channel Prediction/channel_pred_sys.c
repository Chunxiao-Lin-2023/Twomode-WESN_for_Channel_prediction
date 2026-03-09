#include "channel_pred_sys.h"


/*
 * Inside channel pred sys:
 *
 * CHANNEL PREDICTION SYSTEM(Author: Ahmed Malik, MEng | Last Updated: 2/25/26)
 *
 *  This file is used to do WESN training and prediction by using the twomode_core.c procedures
 *  to ultimately test golden and predicted values to produce NMSE calculations to be sent
 *  back to host device via UDP (seen in twomode_main.c).
 *
 *  Order of operations:
 *   twomode_main calls channel_pred_sys once all data is sent
 *    |
 *    V
 *  (1) Reservoir Calculation
 *  (2a) Collect states and ground truths
 *  (2b) Training States
 *  (3) Channel Prediction
 *  (4) Testing (NMSE)
 *    |
 *    V
 *  Return NMSE values back to PC
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

//@todo: set variables for the dimensions

//Global Variables (for now)

//COMMENT OUT BELOW IF WEIGHTS ARE FIXED
// static double W_res_left[RX_ANT*RX_ANT];                                                                    //FIXME: [2,2]
// static double W_res_right[(TX_ANT*WINDOW_LENGTH)*(TX_ANT*WINDOW_LENGTH)];  
// static double W_in_left[RX_ANT*RX_ANT];                                                                     //FIXME: [2,2]
// static double W_in_right[(TX_ANT*WINDOW_LENGTH)*(TX_ANT*WINDOW_LENGTH)];

//COMMENT OUT BELOW IF WEIGHTS AREN'T FIXED

/* ============ SET 1 ============ */
static double *W_in_left = (double[]){
    0.21115513, 0.02693325,
    0.1956733, -0.47556868
}; // 2x2 = 4 elements

static double *W_in_right = (double[]){
    -0.39825738, -0.94920044, -0.39387488, -0.51584825, 0.11515638, 0.13101404,
    -0.04973551, -0.41440405, -0.87149788, 0.95763829, -0.32058431, -0.00990274,
    0.95416145, -0.11845235, -0.36345439, 0.03959397, 0.15627286, 0.7078675,
    -0.86380545, -0.07093838, 0.56389824, 0.43720562, 0.17204396, -0.92581117,
    -0.29868722, 0.12638137, -0.40054026, 0.02466831, 0.34693385, -0.68161253,
    -0.89904466, -0.32436823, -0.78387245, -0.64219438, 0.77165419, -0.26927006
}; // 6x6 = 36 elements

static double *W_res_left = (double[]){
    0.5, 0.0,
    0.0, 0.45850526
}; // 2x2 = 4 elements

static double *W_res_right = (double[]){
    0.0, -0.61022613, 0.0, 0.67208945, 0.0, 0.0,
    0.0, 0.0, 0.32871989, 0.0, 0.0, 0.31807728,
    0.06306582, -0.53043067, -0.18775383, 0.0, -0.08622383, -0.09781462,
    0.17457235, 0.01947553, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.26795611, 0.0, -0.29543307, 0.0, 0.48722209,
    0.0, 0.0, 0.07053953, 0.47329636, -0.44626633, 0.52897848
}; // 6x6 = 36 elements

// below assumes S_all, Y_all resets after each Time_window
static int S_all_sz = 0;
static double *S_all;
static int Y_all_sz = 0;
static double *Y_all;

// buffer for predicted outputs, allocated during channel_pred_sys
static double *pred_channels;

// Global random state (seeded once at startup)
static unsigned long rng_state = RNG_SEED;

// ------------------------------------------------------------------
// Random Number Generation
// ------------------------------------------------------------------

// Linear congruential generator for uniform [0, 1)
static double uniform_rand(void)
{
    // LCG: next = (a * state + c) mod m
    rng_state = (1103515245UL * rng_state + 12345UL) & 0x7fffffffUL;
    return (double)rng_state / (double)0x7fffffffUL;
}

// Box-Muller transform: generates standard normal N(0, 1)
static double normal_rand(void)
{
    static int has_spare = 0;
    static double spare = 0.0;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    has_spare = 1;
    double u = uniform_rand();
    double v = uniform_rand();
    
    if (u < 1e-10) u = 1e-10;
    
    double rad = sqrt(-2.0 * log(u));
    double theta = 2.0 * M_PI * v;
    spare = rad * sin(theta);
    return rad * cos(theta);
}

// Compute largest eigenvalue magnitude using power iteration
static double max_eigenvalue(const double *A, int n, int max_iter, double tol)
{
    if (!A || n <= 0) return 0.0;
    
    double *v = malloc(n * sizeof(double));
    double *Av = malloc(n * sizeof(double));
    if (!v || !Av) {
        free(v);
        free(Av);
        return 0.0;
    }
    
    for (int i = 0; i < n; ++i)
        v[i] = uniform_rand() - 0.5;
    
    double lambda = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            Av[i] = 0.0;
            for (int j = 0; j < n; ++j)
                Av[i] += A[i * n + j] * v[j];
        }
        
        double vTAv = 0.0, vTv = 0.0;
        for (int i = 0; i < n; ++i) {
            vTAv += v[i] * Av[i];
            vTv += v[i] * v[i];
        }
        
        double lambda_new = vTv > 1e-12 ? vTAv / vTv : 0.0;
        
        double norm = sqrt(vTv);
        if (norm > 1e-12) {
            for (int i = 0; i < n; ++i)
                v[i] = Av[i] / norm;
        }
        
        if (fabs(lambda_new - lambda) < tol)
            break;
        lambda = lambda_new;
    }
    
    free(v);
    free(Av);
    return lambda;
}

// ------------------------------------------------------------------
// Sparse Matrix Generation
// ------------------------------------------------------------------

// Create sparse matrix with given sparsity and spectral radius
static double* sparse_matrix(int m, double sparsity, double spectral_radius,
                             int max_tries, double eps)
{
    if (m <= 0 || sparsity < 0.0 || sparsity > 1.0 || spectral_radius <= 0.0)
        return NULL;
    
    double *W = malloc(m * m * sizeof(double));
    if (!W) return NULL;
    
    for (int attempt = 0; attempt < max_tries; ++attempt) {
        int has_nonzero = 0;
        for (int i = 0; i < m * m; ++i) {
            if (uniform_rand() < sparsity) {
                W[i] = 0.0;
            } else {
                W[i] = 2.0 * (uniform_rand() - 0.5);
                has_nonzero = 1;
            }
        }
        
        if (!has_nonzero) continue;
        
        double radius = max_eigenvalue(W, m, 50, 1e-6);
        radius = fabs(radius);
        
        if (isfinite(radius) && radius > eps) {
            double scale = spectral_radius / radius;
            for (int i = 0; i < m * m; ++i)
                W[i] *= scale;
            return W;
        }
    }
    
    free(W);
    return NULL;
}


// ------------------------------------------------------------------
// Weight Initialization
// ------------------------------------------------------------------

// Initialize weight matrices (public entry point)
int init_weights(int N_r, int N_t, int window_length)
{
    if (FIXED_WEIGHTS){
        return 1; // Weights are already set as constants
    }
    // Reset RNG state
    rng_state = RNG_SEED;
    
    int d_left = N_r;
    int d_right = N_t * window_length;
    // Generate sparse reservoir matrices
    double *temp_left = sparse_matrix(d_left, SPARSITY, SPECTRAL_RADIUS, 100, 1e-12);
    if (!temp_left) return -1;
    memcpy(W_res_left, temp_left, d_left * d_left * sizeof(double));
    free(temp_left);
    
    double *temp_right = sparse_matrix(d_right, SPARSITY, SPECTRAL_RADIUS, 100, 1e-12);
    if (!temp_right) return -1;
    memcpy(W_res_right, temp_right, d_right * d_right * sizeof(double));
    free(temp_right);
    
    // Initialize input weight matrices: uniform [-1, 1]
    for (int i = 0; i < d_left * N_r; ++i)
        W_in_left[i] = 2.0 * (uniform_rand() - 0.5);
    
    for (int i = 0; i < d_right * d_right; ++i)
        W_in_right[i] = 2.0 * (uniform_rand() - 0.5);
    
    
    // log all the weight elements for all the weights
    // for(int i = 0; i < d_left * d_left; ++i) {
    //     if(DEBUG) HELPER_LOGD("[DEBUG] W_res_left[%d] = %f\n", i, W_res_left[i]);
    // }
    // for(int i = 0; i < d_right * d_right; ++i) {
    //     if(DEBUG) HELPER_LOGD("[DEBUG] W_res_right[%d] = %f\n", i, W_res_right[i]);
    // }
    // for(int i = 0; i < d_left * N_r; ++i) {
    //     if(DEBUG) HELPER_LOGD("[DEBUG] W_in_left[%d] = %f\n", i, W_in_left[i]);
    // }
    // for(int i = 0; i < d_right * d_right; ++i) {
    //     if(DEBUG) HELPER_LOGD("[DEBUG] W_in_right[%d] = %f\n", i, W_in_right[i]);
    // }
    
    return 0;
}


// see prototype in header for documentation

double* form_window_input_signal(const double *Y_3D, int T, int N_r, int N_t, int L)
{
    if (!Y_3D || T <= 0 || N_r <= 0 || N_t <= 0 || L <= 0)
        return NULL;
    int out_cols = L * N_t;
    size_t total = (size_t)T * N_r * out_cols;
    double *out = malloc(sizeof(double) * total);
    if (!out)
        return NULL;

    // iterate over time steps
    for (int k = 0; k < T; ++k) {
        for (int ell = 0; ell < L; ++ell) {
            int t = k - ell;
            for (int i = 0; i < N_r; ++i) {
                for (int j = 0; j < N_t; ++j) {
                    double val = 0.0;
                    if (t >= 0) {
                        // input is stored as [t][i][j]
                        val = Y_3D[(t * N_r + i) * N_t + j];
                    }
                    out[(k * N_r + i) * out_cols + ell * N_t + j] = val;
                }
            }
        }
    }

    return out;
}

/*
 * Append T columns from Y3 into an existing matrix dst at column index start_col.
 */
void column_stack_append_pre(double *dst, int start_col,
                         const double *Y3, int T,
                         int N_r, int N_t)
{   
    // if(DEBUG) HELPER_LOGD("[DEBUG] start_col = %d \n", start_col);
    
    int rows = N_r * N_t;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N_r; ++i) {
            for (int j = 0; j < N_t; ++j) {
                int row = i * N_t + j;
                dst[row + (size_t)(start_col + t) * rows] =
                    Y3[(t * N_r + i) * N_t + j];
                // if(DEBUG) HELPER_LOGD("[DEBUG] row = %d\n", row);
                // if(DEBUG) HELPER_LOGD("[DEBUG] dst[row + (size_t)(start_col + t) * rows] = Y3[(t * N_r + i) * N_t + j];\n");
                // if(DEBUG) HELPER_LOGD("[DEBUG] dst[%d] = Y3[%d];\n", row + (size_t)(start_col + t) * rows, (t * N_r + i) * N_t + j);

            }
        }
    }
}

void column_stack_append(double *dst, int total_cols, int start_col,
                                  const double *Y3, int T, int N_r, int N_t)
{
    int rows = N_r * N_t;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N_r; ++i) {
            for (int j = 0; j < N_t; ++j) {
                int row = i * N_t + j;
                int col = start_col + t;
                dst[row * (size_t)total_cols + col] =
                    Y3[(t * N_r + i) * N_t + j];
            }
        }
    }
}

/*
 * append_Y: Build target matrix for one RB and append to Y_all.
 * Mirrors Python: Y_f = build_S_Y(..., Y_out)[1]
 */
void append_Y(double *Y_all, int col,
              const double *Y_out, int T, int N_r, int N_t)
{
    int total_cols = T * RB_SZ;   
    column_stack_append(Y_all, total_cols,col, Y_out, T, N_r, N_t);
}

/*
 * Convert a T×N_r×N_t array into a matrix of size (N_r*N_t) × T.
 * The input is assumed stored as contiguous double data with
 * time index running slowest:
 *
 *     idx = (t * N_r + i) * N_t + j
 *
 * which matches NumPy’s default layout for a shape (T,N_r,N_t) array.
 *
 * The returned pointer must be freed by the caller.
 */
double *column_stack_from_3d(const double *Y3, int T, int N_r, int N_t)                      // unused
{
    int rows = N_r * N_t;      /* number of rows in output */
    int cols = T;              /* number of columns */
    double *Y = malloc((size_t)rows * cols * sizeof(double));
    if (!Y) return NULL;

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N_r; ++i) {
            for (int j = 0; j < N_t; ++j) {
                int row = i * N_t + j;                /* flatten [i,j] */
                /* store in column t */
                Y[row + (size_t)t * rows] =
                    Y3[(t * N_r + i) * N_t + j];
            }
        }
    }
    return Y;
}
/*
 * append_S: Build reservoir feature matrix for one RB and append to S_all.
 * Mirrors Python: S_f = build_S_Y(Y_in, ...)[0]
 */
 
 //append_S(S_all, col, Y_in, total_N-2, RX_ANT, TX_ANT,
 //		         W_in_left, W_in_right, W_res_left, W_res_right,
 //		         INPUT_SCALE, WINDOW_LENGTH);
void append_S(double *S_all, int col,
              const double *Y_in, int T, int N_r, int N_t,
              double *W_in_left, double *W_in_right,
              double *W_res_left, double *W_res_right,
              double input_scale, int window_length)
{
  int L = window_length;
  int d_left = N_r;
  int d_right = N_t * L;
	// if(DEBUG) HELPER_LOGD("\t[DEBUG] Forming windowed input signal...\n");
	// allocate windowed input buffer via helper (it returns a new pointer)
	// the initial malloc below is unnecessary so drop it:
	// double *Y_win = malloc(...);
	// instead just call the builder directly and check result:
	double *Y_win = form_window_input_signal(Y_in, T, N_r, N_t, L);
   
	if (!Y_win) return;
	if(DEBUG) HELPER_LOGD("\t[DEBUG] Windowed input signal formed.\n");

	double *S_state = (double*)calloc(d_left*d_right,  sizeof(double));
	if(!S_state) {
		printf("[ERROR] Failed to allocate memory for S_state.\n");
		return;
	}
  double *S_3D_transit = (double*)malloc((size_t)T * d_left * d_right * sizeof(double));
  if (!S_3D_transit) {
	printf("[ERROR] Failed to allocate memory for S_3D_transit.\n");
      return;
  }
	// if(DEBUG) HELPER_LOGD("\t[DEBUG] Starting reservoir state transit...\n");
	double *Y_t_scaled = (double*)malloc(N_r * L * N_t * sizeof(double));
	// if(DEBUG) HELPER_LOGD("\t[DEBUG] Allocated Y_t_scaled for input scaling.\n");
  for (int t = 0; t < T; ++t) {
		// if(DEBUG) HELPER_LOGD("\t[DEBUG] Processing time step %d/%d...\n", t+1, T);
    for (int i = 0; i < N_r * L * N_t; ++i) {
        Y_t_scaled[i] = Y_win[t * N_r * L * N_t + i] * input_scale;
    }
		// if(DEBUG) HELPER_LOGD("\t[DEBUG] Scaled input for time step %d.\n", t+1);
		// if(DEBUG) HELPER_LOGD("\t[DEBUG] Updating reservoir state for time step %d...\n", t+1);
    update_Reservoir(&S_3D_transit[t * d_left * d_right],
                       S_state, Y_t_scaled,
                       W_in_left, W_in_right, W_res_left, W_res_right, N_r, N_t, L);

    // printf("[CHECK] Update_reservoir at %d:\n", t);
    // for(int i = 0; i < d_left*d_right; ++i) {
    //     printf("S_state[%d]=%f \n", i, S_state[i]);
    //     printf("Y_t_scaled[%d]=%f \n", i, Y_t_scaled[i]);
    //     printf("S_3D_transit[%d]=%f \n", i, S_3D_transit[t * d_left * d_right + i]);
    // }

    

	// if(DEBUG) HELPER_LOGD("\t[DEBUG] Reservoir state updated for time step %d.\n", t+1);

    // log both the res_state and res_state_transit after reservoir update for debugging
        // if(DEBUG) {
        //     HELPER_LOGD("[DEBUG] Reservoir state before update at time t=%d:\n", t);
        //     for (int i = 0; i < A_ANT; ++i) {
        //         for (int j = 0; j < SCOL; ++j) {
        //             HELPER_LOGD("%f ", S_state[i * SCOL + j]);
        //         }
        //         HELPER_LOGD("\n");
        //     }
        //     HELPER_LOGD("[DEBUG] Y_in before update at time t=%d:\n", t);
        //     for (int i = 0; i < A_ANT; ++i) {
        //         for (int j = 0; j < YCOL; ++j) {
        //             HELPER_LOGD("%f ", Y_t_scaled[i * YCOL + j]);
        //         }
        //         HELPER_LOGD("\n");
        //     }

        //     HELPER_LOGD("[DEBUG] Reservoir state after update at time t=%d:\n", t);
        //     for (int i = 0; i < A_ANT; ++i) {
        //         for (int j = 0; j < SCOL; ++j) {
        //             HELPER_LOGD("%f ", S_3D_transit[i * SCOL + j]);
        //         }
        //         HELPER_LOGD("\n");
        //     }
        // } 


    memcpy(S_state, &S_3D_transit[t * d_left * d_right],
               d_left * d_right * sizeof(double));
    
    }

	
//   if(DEBUG) HELPER_LOGD("\t[DEBUG] Reservoir state transit completed.\n");
	
// 	if(DEBUG) HELPER_LOGD("\t[DEBUG] Concatenating reservoir states and windowed inputs...\n");
	int concat_cols = d_right + L * N_t;
  double *S_3D_concat = (double*)malloc((size_t)T * d_left * concat_cols * sizeof(double));
  // if(DEBUG) HELPER_LOGD("\t[DEBUG] S_3D_concat = %d * %d * %d\n", T, d_left, concat_cols);
  if (!S_3D_concat) {
        printf("[ERROR] Failed to allocate memory for S_3D_concat.\n");
        return;
    }

    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < d_left; ++i) {
            for (int j = 0; j < d_right; ++j) {
                S_3D_concat[(t * d_left + i) * concat_cols + j] =
                    S_3D_transit[(t * d_left + i) * d_right + j];
                S_3D_concat[(t * d_left + i) * concat_cols + d_right + j] =
                    Y_win[(t * d_left + i) * d_right + j];
            }
        }
    }

    // printf("[CHECK] --------------start concatenation:\n");
    // for(int i = 0; i < T * d_left * d_right; ++i) {
    //     printf("S_3D_transit[%d] = %f\n", i, S_3D_transit[i]);
    //     printf("Y_win[%d] = %f\n", i, Y_win[i]);
    // }
    // printf("[CHECK] --------------end concatenation:\n");

    // for(int i = 0; i < T * d_left * concat_cols; ++i) {
    //     printf("S_3D_concat[%d] = %f\n", i, S_3D_concat[i]);
    // }

    int total_cols = T * RB_SZ;
    column_stack_append(S_all, total_cols, col, S_3D_concat, T, d_left, concat_cols);
    // if(DEBUG) HELPER_LOGD("\t[DEBUG] d_left = %d, concat_cols = %d.\n", d_left, concat_cols);
    

    free(Y_win);
    free(S_3D_transit);
    free(S_3D_concat);
    free(S_state);
  	free(Y_t_scaled);
}


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
                       int new_dim1, int new_dim2, int new_dim3, int new_dim4, int new_dim5)
{
    int old_dims[5] = {dim0, dim1, dim2, dim3, dim4};
    int perm[5] = {new_dim1, new_dim2, new_dim3, new_dim4, new_dim5}; // perm[i] = old dim index for new position i
    
    /* Validate: must be permutation of {0,1,2,3,4} */
    char seen[5] = {0,0,0,0,0};
    for (int i = 0; i < 5; i++) {
        if (perm[i] < 0 || perm[i] >= 5) return NULL;
        if (seen[perm[i]]) return NULL;  // Duplicate dimension index
        seen[perm[i]] = 1;
    }
    
    /* Build new dimensions: new_shape[i] = old_shape[perm[i]] */
    int new_dims[5];
    for (int i = 0; i < 5; i++) {
        new_dims[i] = old_dims[perm[i]];
    }
    
    size_t total_elements = (size_t)new_dims[0] * new_dims[1] * new_dims[2] * new_dims[3] * new_dims[4];
    
    double *result = (double*)malloc(total_elements * sizeof(double));
    if (!result) return NULL;
    
    /* Build inverse permutation: inv[old_idx] = new_position */
    int inv[5];
    for (int i = 0; i < 5; i++) {
        inv[perm[i]] = i;
    }
    
    /* Pre-calculate strides for original matrix (row-major) */
    size_t old_stride[5];
    old_stride[4] = 1;
    old_stride[3] = dim4;
    old_stride[2] = (size_t)dim3 * dim4;
    old_stride[1] = (size_t)dim2 * dim3 * dim4;
    old_stride[0] = (size_t)dim1 * dim2 * dim3 * dim4;
    
    /* Iterate through destination array and gather from source */
    size_t n0 = new_dims[0], n1 = new_dims[1], n2 = new_dims[2], n3 = new_dims[3], n4 = new_dims[4];
    
    for (size_t i0 = 0; i0 < n0; i0++) {
        for (size_t i1 = 0; i1 < n1; i1++) {
            for (size_t i2 = 0; i2 < n2; i2++) {
                for (size_t i3 = 0; i3 < n3; i3++) {
                    for (size_t i4 = 0; i4 < n4; i4++) {
                        /* Coordinates in new array */
                        size_t new_coord[5] = {i0, i1, i2, i3, i4};
                        
                        /* Map to coordinates in old array using inverse permutation */
                        size_t old_coord[5];
                        for (int d = 0; d < 5; d++) {
                            old_coord[d] = new_coord[inv[d]];
                        }
                        
                        /* Calculate linear indices */
                        size_t src_idx = old_coord[0]*old_stride[0] + old_coord[1]*old_stride[1] 
                                       + old_coord[2]*old_stride[2] + old_coord[3]*old_stride[3] 
                                       + old_coord[4]*old_stride[4];
                        
                        size_t dst_idx = (((i0*n1 + i1)*n2 + i2)*n3 + i3)*n4 + i4;
                        
                        result[dst_idx] = matrix_orig[src_idx];
                    }
                }
            }
        }
    }
    
    return result;
}


/*
 * single_Readout_Solve: Solve for readout weights W_out using ridge regression.
 *
 * Parameters:
 *   S_all: feature matrix of size (F x T), where F is reservoir feature dimension
 *   Y_all: target matrix of size (H x T), where H is output dimension (TX_ANT * RX_ANT)
 *
 * Returns a newly allocated weight matrix (caller must free).
 */
double *single_Readout_Solve(double *S_all, double *Y_all, int N)
{
	/* Compute dimensions from the macros.
	 * S_all has shape (F, T) where:
	 *   F = FEATURE_DIM (twomode_core.h defines this)
	 *   T = (N_SZ - 1) * RB_SZ
	 * Y_all has shape (H, T) where:
	 *   H = TX_ANT * RX_ANT
	 */
	int F = FEATURE_DIM;
	int H = TX_ANT * RX_ANT; //4
	int T = N * RB_SZ;  // 

	/* compute pseudo‑inverse of S_all: G has shape (T x F) */
	double *G = reg_pseudo_inv(S_all, F, T, REG);
	double *W_out = NULL;

	if (G) {
		/* Y_all is (H x T); result W_out should be (H x F) */
        if(DEBUG) print_first_n_doubles("[COMPARE] G (pseudo-inverse of S_all)", G, 10);
        if(DEBUG) print_last_n_doubles("[COMPARE] G (pseudo-inverse of S_all)", G, F*T, 10);
		W_out = matrix_mult(Y_all, G, H, T, F);
        if(DEBUG) HELPER_LOGD("[DEBUG] Computed W_out with shape (%d, %d)\n", H, F);
		free(G);
	}

    // printf("[CHECK] W_out[0]=%.6e\n", W_out[0]);
    // printf("[CHECK] W_out[last]=%.6e\n", W_out[(size_t)TX_ANT*RX_ANT*FEATURE_DIM - 1]);


	return W_out;
}

// Main Channel Prediction System function, returns array of NMSE values

double* channel_pred_sys(double *data, int total_N){
	if(DEBUG) HELPER_LOGD("[DEBUG] Starting channel prediction system...\n");
	// data (total_t, tx_ant, rx_ant, rb_sz) consists of below:
	// total_t(total_N) = time slots being used to pred new value (default: 10) (ALSO FOUND IN twomode_main.h)
	// tx_ant (default: 2)
	// rx_ant (default: 2)
	// rb_sz = resource block value, must iterate through each one during matrix operations (default: 32)

	//declared training input and ground truth channels for training phase
	if(DEBUG) HELPER_LOGD("[DEBUG] Allocating training input and ground truth channels...");
	double *channel_train_input = (double*)malloc((total_N-1)*TX_ANT*RX_ANT*RB_SZ*sizeof(double)); //Training input channels               FIXME: changed from N-2 to N-1
	double *channel_train_gt = (double*)malloc((total_N-1)*TX_ANT*RX_ANT*RB_SZ*sizeof(double)); //Training ground truth channels           FIXME: changed from N-2 to N-1
  if(DEBUG) HELPER_LOGD("[DEBUG] channel_train_input shape: (%d, %d, %d, %d)\n", total_N-1, TX_ANT, RX_ANT, RB_SZ);
	if(DEBUG) HELPER_LOGD("[DEBUG] Allocated training input and ground truth channels.\n");
	// fill channel_train_input and channel_train_gt with data                                                                                         //    FIXME: changed from N-2 to N-1

	for(int N = 0; N < total_N-1; N++){                                                                                          //    FIXME: changed from N-2 to N-1
		for(int j = 0; j < TX_ANT*RX_ANT*RB_SZ; j++){
			channel_train_input[N*TX_ANT*RX_ANT*RB_SZ + j] = data[N*TX_ANT*RX_ANT*RB_SZ + j];
			channel_train_gt[N*TX_ANT*RX_ANT*RB_SZ + j] = data[(N+1)*TX_ANT*RX_ANT*RB_SZ + j];
		}   
	}

    // for (int N = 0; N < (total_N-1)*TX_ANT*RX_ANT*RB_SZ ; N++){ printf("[DEBUG] channel_train_gt[%d]=%.6e\n", N, channel_train_gt[N]); }

    
    //print channel train input and gt for validation
    // if(DEBUG){
    //     print_first_n_doubles("[COMPARE]channel_train_input", channel_train_input, 10);
    //     print_last_n_doubles("[COMPARE]channel_train_input", channel_train_input, (total_N)*TX_ANT*RX_ANT*RB_SZ, 10);
    //     print_first_n_doubles("[COMPARE]channel_train_gt", channel_train_gt, 10);
    //     print_last_n_doubles("[COMPARE]channel_train_gt", channel_train_gt, (total_N)*TX_ANT*RX_ANT*RB_SZ, 10);
    // }

    

	
// window length constant used repeatedly
	int L = WINDOW_LENGTH;
	if(DEBUG) HELPER_LOGD("[DEBUG] Initializing weights for WESN predictor...\n");
	//Initialize weights with correct reservoir sizes
    if(!FIXED_WEIGHTS) init_weights(RX_ANT, TX_ANT, WINDOW_LENGTH); //FIXED_WEIGHTS is in .h file, set to 1 for fixed weights and 0 for random weights

    // if(DEBUG){
    //     print_first_n_doubles("[COMPARE]W_res_left", W_res_left, 10);
    //     print_last_n_doubles("[COMPARE]W_res_left", W_res_left, RX_ANT*RX_ANT, 10);
    //     print_first_n_doubles("[COMPARE]W_res_right", W_res_right, 10);
    //     print_last_n_doubles("[COMPARE]W_res_right", W_res_right, (TX_ANT*WINDOW_LENGTH)*(TX_ANT*WINDOW_LENGTH), 10);
    //     print_first_n_doubles("[COMPARE]W_in_left", W_in_left, 10);
    //     print_last_n_doubles("[COMPARE]W_in_left", W_in_left, RX_ANT*RX_ANT, 10);
    //     print_first_n_doubles("[COMPARE]W_in_right", W_in_right, 10);
    //     print_last_n_doubles("[COMPARE]W_in_right", W_in_right, (TX_ANT*WINDOW_LENGTH)*(TX_ANT*WINDOW_LENGTH), 10);
    // }
	if(DEBUG) HELPER_LOGD("[DEBUG] Weights initialized.\n");

    


	// allocate buffers for training features and targets
	// only (total_N-2) columns are added per RB, so size accordingly
	if(DEBUG) HELPER_LOGD("[DEBUG] Allocating S_all and Y_all buffers...\n");
	double *S_all = (double*)malloc(FEATURE_DIM*(total_N-1)*RB_SZ*sizeof(double)); //(F, Sum_T)                                         FIXME: changed from N-2 to N-1
	double *Y_all = (double*)malloc(TX_ANT*RX_ANT*(total_N-1)*RB_SZ*sizeof(double)); //(N_r*N_t, Sum_T)
	if(DEBUG) HELPER_LOGD("[DEBUG] Allocated S_all and Y_all buffers.\n");
  if(DEBUG) HELPER_LOGD("[DEBUG] S_all.shape = (%d, %d)\n", FEATURE_DIM, (total_N-1) * RB_SZ);
  if(DEBUG) HELPER_LOGD("[DEBUG] Y_all.shape = (%d, %d)\n", TX_ANT*RX_ANT, (total_N-1) * RB_SZ); 

	if(DEBUG) HELPER_LOGD("[DEBUG] Allocating prediction buffer for all RBs...\n");
	/* prediction buffer: allocate storage for outputs of all RBs */
	pred_channels = (double*)malloc(RB_SZ * TX_ANT * RX_ANT * sizeof(double));
	if(DEBUG) HELPER_LOGD("[DEBUG] Allocated prediction buffer for all RBs.\n");
	if (!pred_channels) {
		// handle allocation failure
	}

	//(A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms)

	/* we will compute a full windowed version of the channel sequence
	 * once and then index into it inside the loop.  the helper above
	 * produces an array of size [total_N, 4, WINDOW_LENGTH*2]. */
	/* note: N_t is hard‑coded to 2 in this prototype, so out_cols = L*2 */


	if(DEBUG) HELPER_LOGD("[DEBUG] Building features and targets for all RBs...\n");
	int col = 0;  // when f is incremented, col is incremented by N-1.
	double *Y_in; // (8, 2, 2, f)
	double *Y_out; // (8, 2, 2, f)

    double *channel_train_input_reorder_f = matrix_reorder(channel_train_input, total_N-1, TX_ANT, RX_ANT, 32, 1, 3,0,1,2,4);
    double *channel_train_gt_reorder_f = matrix_reorder(channel_train_gt, total_N-1, TX_ANT, RX_ANT, 32, 1, 3,0,1,2,4);

	for(int f = 0; f < RB_SZ; f++){
		if(DEBUG) HELPER_LOGD("\t[DEBUG] Allocated Y_in and Y_out buffers for RB %d.\n", f);
		Y_in = &channel_train_input_reorder_f[(total_N-1)*TX_ANT*RX_ANT*f]; // (8, 2, 2, f)
		Y_out = &channel_train_gt_reorder_f[(total_N-1)*TX_ANT*RX_ANT*f]; // (8, 2, 2, f)
        // if(DEBUG && (f == 0 || f == RB_SZ-1)){
        // for(int N = 0; N < (total_N-1)*TX_ANT*RX_ANT; N++){ printf("[DEBUG] Y_in[%d]=%.6e\n, at freq %d\n", N, Y_in[N], f); }
        // for(int N = 0; N < (total_N-1)*TX_ANT*RX_ANT; N++){ printf("[DEBUG] Y_out[%d]=%.6e\n, at freq %d\n", N, Y_out[N], f); }
        // }
        

        // if(DEBUG && (f == 0 || f == RB_SZ-1)){ // only print for first and last RB for sanity check
        //     print_first_n_doubles("[COMPARE]Y_in for RB 0", Y_in, 10);
        //     print_last_n_doubles("[COMPARE]Y_in for RB 0", Y_in, (total_N-1)*TX_ANT*RX_ANT, 10);
        //     print_first_n_doubles("[COMPARE]Y_out for RB 0", Y_out, 10);
        //     print_last_n_doubles("[COMPARE]Y_out for RB 0", Y_out, (total_N-1)*TX_ANT*RX_ANT, 10);
        // }
		if(DEBUG) HELPER_LOGD("\t[DEBUG] Aliased Y_in and Y_out buffers for RB %d.\n", f);
		if(DEBUG) HELPER_LOGD("\t[DEBUG] Appending S and Y for RB %d...\n", f);

        // printf(" [CHECK] The frequency index is %d\n", f);
		append_S(S_all, col, Y_in, total_N-1, RX_ANT, TX_ANT,
		         W_in_left, W_in_right, W_res_left, W_res_right,
		         INPUT_SCALE, WINDOW_LENGTH);

        
		if(DEBUG) HELPER_LOGD("\t[DEBUG] Appended S for RB %d.\n", f);
		append_Y(Y_all, col, Y_out, total_N-1, RX_ANT, TX_ANT);
		if(DEBUG) HELPER_LOGD("\t[DEBUG] Appended Y for RB %d.\n", f);
		col += total_N-1; //-1
	}

    

    // for(int N = 0; N < (total_N-1)*TX_ANT*RX_ANT*RB_SZ; N++){ printf("[DEBUG] Y_all[%d]=%.6e\n", N, Y_all[N]); }
    // for(int N = 0; N < FEATURE_DIM*(total_N-1)*RB_SZ; N++){ printf("[DEBUG] S_all[%d]=%.6e\n", N, S_all[N]); }



    // if(DEBUG) print_first_n_doubles("[COMPARE]S_all", S_all, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE]S_all", S_all, FEATURE_DIM*(total_N-1)*RB_SZ, 10);
    // if(DEBUG) print_first_n_doubles("[COMPARE]Y_all", Y_all, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE]Y_all", Y_all, TX_ANT*RX_ANT*(total_N-1)*RB_SZ, 10);
	free(channel_train_input);
	if(DEBUG) HELPER_LOGD("[DEBUG] Built features and targets for all RBs.\n");
	
	//(B) SINGLE READOUT SOLVE (shared accross all RBs)
	/* compute readout weights */
	if(DEBUG) HELPER_LOGD("[DEBUG] Computing readout weights W_out...\n");
	double *W_out = single_Readout_Solve(S_all, Y_all, (total_N-1));

    // for(int i = 0; i < TX_ANT*RX_ANT*FEATURE_DIM; i++){ printf("[DEBUG] W_out[%d]=%.6e\n", i, W_out[i]);}
    

    // if(DEBUG) print_first_n_doubles("[COMPARE]W_out", W_out, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE]W_out", W_out, TX_ANT*RX_ANT*FEATURE_DIM, 10);
	// if(DEBUG) HELPER_LOGD("[DEBUG] Computed readout weights W_out.\n");
    //if(DEBUG) HELPER_LOGD("[CHECK] W_out[0]=%.6e\n", *W_out);
    if (!W_out) {
        if(DEBUG) HELPER_LOGD("[ERROR] W_out is NULL (readout solve failed)\n");
    // print diagnostics and abort / return
    return NULL;
    }


	//(C) PREDICTION PHASE with shared W_out
	/* no allocation; alias into channel_train_gt below */
	if(DEBUG) HELPER_LOGD("[DEBUG] Starting prediction phase for all RBs...\n");
	double *channel_test_input;
	for(int f = 0; f < RB_SZ; f++){
        // if (DEBUG) HELPER_LOGD("\t[DEBUG] Predicting for RB %d...\n", f);
        channel_test_input = &channel_train_gt_reorder_f[(total_N-1)*TX_ANT*RX_ANT*f]; // (9, 2, 2, f)
        printf("-----------------------------for frequency %d-----------------------------\n", f);
        // for(int N = 0; N < (total_N-1)*TX_ANT*RX_ANT; N++){ printf("[DEBUG] channel_test_input[%d]=%.6e\n, at freq %d\n", N, channel_test_input[N], f); }
        // if (DEBUG) HELPER_LOGD("\t[DEBUG] Aliased channel_test_input for RB %d.\n", f);
        // if(DEBUG && f==0){ 
        //     print_first_n_doubles("[COMPARE]channel_test_input for RB 0", channel_test_input, 10);
        //     print_last_n_doubles("[COMPARE]channel_test_input for RB 0", channel_test_input, (total_N-1)*TX_ANT*RX_ANT, 10);
        // }
        // if(DEBUG && f==RB_SZ-1){ 
        //     print_first_n_doubles("[COMPARE]channel_test_input for RB last", channel_test_input, 10);
        //     print_last_n_doubles("[COMPARE]channel_test_input for RB last", channel_test_input, (total_N-1)*TX_ANT*RX_ANT, 10);
        // }   
        /* length = total_N-2 (training length) */
        predict_States(channel_test_input, &pred_channels[f*TX_ANT*RX_ANT], W_in_left, W_res_left, W_in_right, 
                    W_res_right, W_out, total_N - 1, RX_ANT, TX_ANT, WINDOW_LENGTH); // write prediction into pred_channels buffer
        }
    
    
	// if(DEBUG) HELPER_LOGD("[DEBUG] Completed prediction phase for all RBs.\n");
    // if(DEBUG) print_first_n_doubles("[COMPARE]pred_channels", pred_channels, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE]pred_channels", pred_channels, RB_SZ*TX_ANT*RX_ANT, 10);
    free(channel_train_gt);
	free(S_all);
	free(Y_all);
	free(W_out);
 
  return pred_channels;
	//(D) NMSE CALCULATION PHASE
	/*if(DEBUG) printf("[DEBUG] Starting NMSE calculation phase for all RBs...\n");
	NMSE_Result *NMSE_results = (NMSE_Result*)malloc(RB_SZ*sizeof(NMSE_Result)); //NMSE results for each RB
	for(int f = 0; f < RB_SZ; f++){
		double *predicted_channels = &pred_channels[f*TX_ANT*RX_ANT]; // slice into pred_channels
		double *base_channels = &channel_train_gt[(total_N-2)*TX_ANT*RX_ANT*f];
		double *ground_truth_channels = &channel_train_gt[(total_N-1)*TX_ANT*RX_ANT*f];

		NMSE_Result current_NMSE_result = final_NMSE_calc(predicted_channels, base_channels, ground_truth_channels);
		NMSE_results[f] = current_NMSE_result;
	}
	if(DEBUG) printf("[DEBUG] Completed NMSE calculation phase for all RBs.\n");
  */
	/* free temporary buffers */
	/*free(channel_train_gt);
	free(S_all);
	free(Y_all);
	free(W_out);
	free(pred_channels);*/
	
	/* do not free pointers inside the NMSE loop; they referenced existing buffers */
	/*if(DEBUG) printf("[DEBUG] Channel prediction system completed successfully.\n");
	return NMSE_results;*/


}
