/*
 * simple test harness for channel_pred_sys
 *
 * Compiling: you must supply the same antenna/RB macros that match
 * the .npy file you plan to load.  The code below assumes a file of
 * shape (400,2,2,6) and therefore overrides the default macros from
 * twomode_main.h accordingly.  If you change the file you must also
 * recompile with matching values (see comments below).
 */

// ------------------------------------------------------------------
// Override dimensions to match the test file
// ------------------------------------------------------------------
#define D1_EXP 2    /* tx antennas */
#define D2_EXP 2    /* rx antennas */
//#define D4_EXP 6    /* resource block count in the .npy file */
#define N_SZ    10  /* prediction window (total_N) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper.h"

#include "channel_pred_sys.h"  /* pulls in twomode_main.h which uses the
                                     macros above */

/*
 * Minimal .npy loader limited to 4‑D arrays of doubles (little‑endian)
 * with C order.  If the header deviates from this assumption the
 * loader will return NULL.
 *
 * On success the returned pointer must be freed by the caller.
 */

 //Stddef for .npy used in matrix_reorder, which is used in channel_pred_sys.c
// #include <stddef.h>
// /**
//  * Reorders (transposes) a 5D matrix by permuting dimensions.
//  * 
//  * Example: matrix_reorder(matrix, 400,2,2,2,6, 3,0,1,2,4)
//  * - Original shape: [400][2][2][2][6] (dim0=400, dim1=2, dim2=2, dim3=2, dim4=6)
//  * - Permutation: new_dim0←old_dim3, new_dim1←old_dim0, new_dim2←old_dim1, new_dim3←old_dim2, new_dim4←old_dim4
//  * - Result shape: [2][400][2][2][6]
//  * 
//  * @param matrix_orig  Original flat array in row-major order
//  * @param dim0-dim4    Original dimensions
//  * @param new_dim1-5   Permutation: which old dim (0-4) goes to new position 0-4
//  * @return             Newly allocated reordered array (NULL on failure)
//  */
// double* matrix_reorder(double *matrix_orig, 
//                        int dim0, int dim1, int dim2, int dim3, int dim4,
//                        int new_dim1, int new_dim2, int new_dim3, int new_dim4, int new_dim5)
// {
//     int old_dims[5] = {dim0, dim1, dim2, dim3, dim4};
//     int perm[5] = {new_dim1, new_dim2, new_dim3, new_dim4, new_dim5}; // perm[i] = old dim index for new position i
    
//     /* Validate: must be permutation of {0,1,2,3,4} */
//     char seen[5] = {0,0,0,0,0};
//     for (int i = 0; i < 5; i++) {
//         if (perm[i] < 0 || perm[i] >= 5) return NULL;
//         if (seen[perm[i]]) return NULL;  // Duplicate dimension index
//         seen[perm[i]] = 1;
//     }
    
//     /* Build new dimensions: new_shape[i] = old_shape[perm[i]] */
//     int new_dims[5];
//     for (int i = 0; i < 5; i++) {
//         new_dims[i] = old_dims[perm[i]];
//     }
    
//     size_t total_elements = (size_t)new_dims[0] * new_dims[1] * new_dims[2] * new_dims[3] * new_dims[4];
    
//     double *result = (double*)malloc(total_elements * sizeof(double));
//     if (!result) return NULL;
    
//     /* Build inverse permutation: inv[old_idx] = new_position */
//     int inv[5];
//     for (int i = 0; i < 5; i++) {
//         inv[perm[i]] = i;
//     }
    
//     /* Pre-calculate strides for original matrix (row-major) */
//     size_t old_stride[5];
//     old_stride[4] = 1;
//     old_stride[3] = dim4;
//     old_stride[2] = (size_t)dim3 * dim4;
//     old_stride[1] = (size_t)dim2 * dim3 * dim4;
//     old_stride[0] = (size_t)dim1 * dim2 * dim3 * dim4;
    
//     /* Iterate through destination array and gather from source */
//     size_t n0 = new_dims[0], n1 = new_dims[1], n2 = new_dims[2], n3 = new_dims[3], n4 = new_dims[4];
    
//     for (size_t i0 = 0; i0 < n0; i0++) {
//         for (size_t i1 = 0; i1 < n1; i1++) {
//             for (size_t i2 = 0; i2 < n2; i2++) {
//                 for (size_t i3 = 0; i3 < n3; i3++) {
//                     for (size_t i4 = 0; i4 < n4; i4++) {
//                         /* Coordinates in new array */
//                         size_t new_coord[5] = {i0, i1, i2, i3, i4};
                        
//                         /* Map to coordinates in old array using inverse permutation */
//                         size_t old_coord[5];
//                         for (int d = 0; d < 5; d++) {
//                             old_coord[d] = new_coord[inv[d]];
//                         }
                        
//                         /* Calculate linear indices */
//                         size_t src_idx = old_coord[0]*old_stride[0] + old_coord[1]*old_stride[1] 
//                                        + old_coord[2]*old_stride[2] + old_coord[3]*old_stride[3] 
//                                        + old_coord[4]*old_stride[4];
                        
//                         size_t dst_idx = (((i0*n1 + i1)*n2 + i2)*n3 + i3)*n4 + i4;
                        
//                         result[dst_idx] = matrix_orig[src_idx];
//                     }
//                 }
//             }
//         }
//     }
    
//     return result;
// }

/**
 * Extracts submatrix [M:N+1][0:N2][0:N3][0:N4][0:N5] from 5D matrix
 * 
 * @param src    Source matrix (N1 x N2 x N3 x N4 x N5)
 * @param N1     Original size of dim 0
 * @param N2-N5  Other dimensions  
 * @param M      Start index (inclusive, 0-based)
 * @param N      End index (inclusive, 0-based)
 * @return       Newly allocated submatrix (NULL on failure/invalid range)
 */
double* extract_dim0_range(double *src, int N1, int N2, int N3, int N4, int N5, int M, int N)
{
    // Bounds validation
    if (M < 0 || N >= N1 || M > N) return NULL;
    
    size_t slice_size = (size_t)N2 * N3 * N4 * N5;        // Elements per dim0 index
    size_t num_slices = (size_t)(N - M + 1);               // How many slices to copy
    size_t total_bytes = num_slices * slice_size * sizeof(double);
    size_t offset_bytes = (size_t)M * slice_size * sizeof(double);  // Skip M slices
    
    double *dst = (double*)malloc(total_bytes);
    if (!dst) return NULL;
    
    // Copy contiguous block starting from M
    memcpy(dst, src + (offset_bytes / sizeof(double)), total_bytes);
    // Or equivalently: memcpy(dst, &src[M * slice_size], total_bytes);
    
    return dst;
}


static double *load_npy(const char *path,
                        int *dim0, int *dim1, int *dim2, int *dim3, int *dim4)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("opening npy");
        return NULL;
    }

    char magic[7] = {0};
    if (fread(magic, 1, 6, f) != 6 || strncmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "not a .npy file\n");
        fclose(f);
        return NULL;
    }

    unsigned char ver[2];
    fread(ver, 1, 2, f);

    unsigned short header_len;
    fread(&header_len, 2, 1, f);

    char *header = malloc(header_len + 1);
    if (!header) { fclose(f); return NULL; }
    fread(header, 1, header_len, f);
    header[header_len] = '\0';

    /* look for shape tuple */
    char *p = strstr(header, "shape");
    if (!p) {
        fprintf(stderr, "header has no shape\n");
        free(header); fclose(f);
        return NULL;
    }
    p = strchr(p, '(');
    if (!p) {
        fprintf(stderr, "malformed shape\n");
        free(header); fclose(f);
        return NULL;
    }

    /* try to parse five dimensions (e.g. 400,2,2,2,32) */
    int n = sscanf(p, "(%d,%d,%d,%d,%d", dim0, dim1, dim2, dim3, dim4);
    free(header);
    if (n != 5) {
        fprintf(stderr, "unable to parse five dims, got %d\n", n);
        fclose(f);
        return NULL;
    }

    /* assume dtype is <f8 (double little-endian) and fortran_order False */
    /* skip to data; file pointer is already after the header */

    size_t nelems = (size_t)(*dim0) * (*dim1) * (*dim2) * (*dim3) * (*dim4);
    double *buf = malloc(nelems * sizeof(double));
    if (!buf) { fclose(f); return NULL; }

    if (fread(buf, sizeof(double), nelems, f) != nelems) {
        fprintf(stderr, "failed to read %zu elements\n", nelems);
        free(buf);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{

  helper_log_init("channel_pred.log");
  helper_log_set_level(HELPER_LOG_DEBUG);



    if (argc != 2) {
    fprintf(stderr, "usage: %s channel_data.npy\n", argv[0]);
    return 1;
    }

    int t, tx, rx, d3, rb;
    double *data = load_npy(argv[1], &t, &tx, &rx, &d3, &rb);                                         // FIXME: order of tx and rx
    if (!data) return 1;

    printf("loaded npy dims : %d x %d x %d x %d x %d\n", t, tx, rx, d3, rb);

    /* sanity checks; d3 is a separate axis with size <= 2.  we'll run the
       predictor for each slice independently. */
    if (tx != TX_ANT || rx != RX_ANT) {
        fprintf(stderr, "antenna mismatch: file has %d x %d, compiled for %d x %d\n",
                tx, rx, TX_ANT, RX_ANT);
        free(data);
        return 1;
    }
    if (rb != RB_SZ) {
        fprintf(stderr, "resource‑block mismatch: file has %d, compiled for %d (RB_SZ)\n",
                rb, RB_SZ);
        fprintf(stderr, "please recompile with D4_EXP=%d or adjust the file\n", rb);
        free(data);
        return 1;
    }
    if (t < N_SZ) {
        fprintf(stderr, "time dimension %d smaller than window %d\n", t, N_SZ);
        free(data);
        return 1;
    }
    if (d3 <= 0) {
        fprintf(stderr, "unexpected zero length for d3 dimension\n");
        free(data);
        return 1;
    }

    /* call the predictor using the first N_SZ time steps */                              // FIXME: send the entire time slots instead of 10 slots
    /* run predictor on each d3 slice separately */
    /*
    size_t time_slot_size = tx * rx * rb;
    size_t slice_size_all = (N_SZ) * time_slot_size;
    size_t slice_size = (N_SZ-1) * time_slot_size;  // elements per d3 index
    
    for (int k = 0; k < d3; ++k) {
    
        double *slice = data + (size_t)k * slice_size;
        printf("[DEBUG] running channel_pred_sys on slice %d (d3 index)\n", k);
        double *pred_channel = channel_pred_sys(slice, N_SZ-1);
        double *outdated_channel = data + (size_t)k * (slice_size_all - 2*time_slot_size);
        double *groundtruth_channel = data + (size_t)k * (slice_size_all - *time_slot_size); 
        for(
        if (!res) {
            fprintf(stderr, "channel_pred_sys failed on slice %d\n", k);
            free(data);
            return 1;
        }
        printf("slice %d (d3 index) baseline NMSE=%f pred NMSE=%f\n",
               k, res[0].base_nmse, res[0].pred_nmse);
        free(res);
    }
    */
    
    // chatGPT-version
    size_t time_slot_size = (size_t)tx * rx * rb;                                                     // 2x2x32 = 128
    size_t time_stride      = (size_t)N_SZ  * time_slot_size;                                         // 10x128 = 1280
    
    if(DEBUG) HELPER_LOGD(" [DEBUG] ----------------- Initial Start ---------------------\n");
    if(DEBUG) HELPER_LOGD(" [DEBUG] time_slot_size = %d * %d * %d = %zu\n", tx, rx, rb, time_slot_size);
    if(DEBUG) HELPER_LOGD(" [DEBUG] time_stride = N_SZ * time_slot_size = %d * %zu = %zu\n", N_SZ, time_slot_size, time_stride);    
    // double *test_slice = extract_dim0_range(data, t, tx, rx, d3, rb, 0, N_SZ-1); // get first N_SZ time steps across all d3 slices
    // test_slice = matrix_reorder(test_slice, N_SZ, tx, rx, d3, rb, 3,0,1,2,4); // reorder to (d3, N_SZ, tx, rx, rb)

    // double *channel_pred_real = channel_pred_sys(test_slice, N_SZ-1); // imaginary (0, 10, 2, 2, 32)
    // double *channel_pred_imaginary = channel_pred_sys(&test_slice[N_SZ * tx * rx * rb], N_SZ-1); // real (1, 10, 2, 2, 32)
    double *pred_channel_complex = (double*)malloc(d3 * time_slot_size * sizeof(double));             // 2x128 = 256
    if(DEBUG) HELPER_LOGD("[DEBUG] predchannel_complex = %d * %zu = %zu\n", d3, time_slot_size, (size_t)d3 * time_slot_size);


    //if(DEBUG) print_first_n_doubles("[COMPARE] First 10 doubles of data_0-10", data, 10);
    //if(DEBUG) print_last_n_doubles("[COMPARE] Last 10 doubles of data", data, (size_t)9 * tx * rx * d3 * rb, 10);


    double *test_slice = extract_dim0_range(data, t, tx, rx, d3, rb, 0, N_SZ-2); // get first N_SZ time steps across all d3 slices
    if(DEBUG) print_first_n_doubles("[COMPARE] First 10 doubles of test_slice", test_slice, 10);
    if(DEBUG) print_last_n_doubles("[COMPARE] Last 10 doubles of test_slice", test_slice, (size_t)(N_SZ-1) * tx * rx * d3 * rb, 10);

    test_slice = matrix_reorder(test_slice, N_SZ-1, tx, rx, d3, rb, 3,0,1,2,4); // reorder to (d3, N_SZ-1, tx, rx, rb)
    double *test_slice_real = test_slice; // real part (0, N_SZ-1, tx, rx, rb)
    double *test_slice_imaginary = test_slice + (size_t)(N_SZ-1) * tx * rx * rb; // imaginary part (1, N_SZ-1, tx, rx, rb)
    // print_first_n_doubles("[COMPARE] First 10 doubles of test_slice_real", test_slice_real, 10);
    // print_last_n_doubles("[COMPARE] Last 10 doubles of test_slice_real", test_slice_real, (size_t)(N_SZ-1) * tx * rx * rb, 10);
    // print_first_n_doubles("[COMPARE] First 10 doubles of test_slice_imaginary", test_slice_imaginary, 10);
    // print_last_n_doubles("[COMPARE] Last 10 doubles of test_slice_imaginary", test_slice_imaginary, (size_t)(N_SZ-1) * tx * rx * rb, 10);

    double *outdated_channel_complex = (double*)malloc(d3 * time_slot_size * sizeof(double));
    double *gt_channel_complex = (double*)malloc(d3 * time_slot_size * sizeof(double));
    double nmse_pred_total[t - N_SZ + 1];
    double nmse_outdated_total[t - N_SZ + 1];
    double nmse_pred = 0.0;
    double nmse_outdated = 0.0;
    for(int i = 0; i < t - N_SZ + 1; ++i) {
    // for(int i = 0; i < 2; ++i) {
        double *test_slice = extract_dim0_range(data, t, tx, rx, d3, rb, i, i+(N_SZ-2)); // get first N_SZ time steps across all d3 slices
        test_slice = matrix_reorder(test_slice, N_SZ-1, tx, rx, d3, rb, 3,0,1,2,4); // reorder to (d3, N_SZ-1, tx, rx, rb)
        double *test_slice_real = test_slice; // real part (0, N_SZ-1, tx, rx, rb)
        double *test_slice_imaginary = test_slice + (size_t)(N_SZ-1) * tx * rx * rb; // imaginary part (1, N_SZ-1, tx, rx, rb)

        double *pred_channels_real_raw = channel_pred_sys(test_slice_real, N_SZ-1); // real part prediction (d3, tx, rx, rb)
        //for(int j = 0; j < TX_ANT*RX_ANT*RB_SZ; ++j){ printf("pred_channels_real[%d] = %f\n", j, pred_channels_real_raw[j]);}
        double *pred_channels_real = matrix_reorder(pred_channels_real_raw, rb, tx, rx, 1, 1, 1, 2, 0, 3, 4); // reorder to (tx, d3, rx, rb)

        double *pred_channels_imaginary_raw = channel_pred_sys(test_slice_imaginary, N_SZ-1); // imaginary part prediction (d3, tx, rx, rb)
        double *pred_channels_imaginary = matrix_reorder(pred_channels_imaginary_raw, rb, tx, rx, 1, 1, 1, 2, 0, 3, 4); // reorder to (tx, d3, rx, rb)
        double *predicted_channel_final = (double*)malloc(TRAIL_ELEMS*sizeof(double)); // final predicted channel (d3, 2, 2, 32)
        

        memcpy(predicted_channel_final, //([0], 2, 2, 32)
                 pred_channels_real,
                 TX_ANT*RX_ANT*RB_SZ * sizeof(double));

        memcpy(predicted_channel_final + (TX_ANT*RX_ANT*RB_SZ), //([1], 2, 2, 32)
                 pred_channels_imaginary,
				 TX_ANT*RX_ANT*RB_SZ * sizeof(double));

        double *outdated_channel_raw = extract_dim0_range(data, T_MAX, TX_ANT, RX_ANT, D3_EXP, RB_SZ, i + (N_SZ-2), i + (N_SZ-2)); //(2, 2, 2, 32) (double check last parameters)
        double *ground_truth_raw = extract_dim0_range(data, T_MAX, TX_ANT, RX_ANT, D3_EXP, RB_SZ, i + (N_SZ-1), i + (N_SZ-1)); //( 2, 2, 2, 32)
          
        double *outdated_channel = matrix_reorder(outdated_channel_raw, 1, TX_ANT, RX_ANT, D3_EXP, RB_SZ, 3,0,1,2,4);//Switches 3rd dimension and 0th dimension
        free(outdated_channel_raw);
        double *ground_truth = matrix_reorder(ground_truth_raw, 1, TX_ANT, RX_ANT, D3_EXP, RB_SZ, 3,0,1,2,4);//Switches 3rd dimension and 0th dimension
        free(ground_truth_raw);

        
        for(int j = 0; j < TRAIL_ELEMS; ++j){
            printf("predicted_channel_final[%d] = %f\n", j, predicted_channel_final[j]);
            printf("outdated_channel[%d] = %f\n", j, outdated_channel[j]);
            printf("ground_truth[%d] = %f\n", j, ground_truth[j]);
        }


        nmse_pred = base_NMSE_calc(ground_truth,
                       predicted_channel_final,
                       TX_ANT*RX_ANT*RB_SZ);
    
        nmse_outdated = base_NMSE_calc(ground_truth,
                       outdated_channel,
                       TX_ANT*RX_ANT*RB_SZ);

        // for(int j = 0; j < TRAIL_ELEMS; ++j){
        //     printf("ground_truth[%d] = %f\n", j, ground_truth[j]);
        // }
        // for(int j = 0; j < TRAIL_ELEMS; ++j){
        //     printf("predicted_channel_final[%d] = %f\n", j, predicted_channel_final[j]);
        // }
        // for(int j = 0; j < TRAIL_ELEMS; ++j){
        //     printf("outdated_channel[%d] = %f\n", j, outdated_channel[j]);
        // }
        
        
        printf("time step %d baseline NMSE = %f pred NMSE = %f\n", i, nmse_outdated, nmse_pred);
        nmse_pred_total[i] = nmse_pred;
        nmse_outdated_total[i] = nmse_outdated;
        free(pred_channels_real);
        free(pred_channels_imaginary);
        free(test_slice);
        free(predicted_channel_final);
        free(outdated_channel);
        free(ground_truth);
    }
    double nmse_pred_avg = 0.0;
    double nmse_outdated_avg = 0.0;
    for(int i = 0; i < (t - N_SZ + 1); ++i){
            nmse_pred_avg += nmse_pred_total[i];
            nmse_outdated_avg += nmse_outdated_total[i];

        if(i%N_SZ == 0 && i > 0){
            printf("time step %d to %d average NMSE: baseline = %f, pred = %f\n", 
                i-N_SZ, i-1, nmse_outdated_avg/N_SZ, nmse_pred_avg/N_SZ);
            nmse_pred_avg = 0.0;
            nmse_outdated_avg = 0.0;
        }
    }
    // for (int k = 0; k < d3; ++k) {
    //     double *pred_channel = channel_pred_sys(test_slice + (size_t)k * (N_SZ-1) * tx * rx * rb, N_SZ-1); // get prediction for this slice
    //     if (!pred_channel) {
    //         fprintf(stderr, "channel_pred_sys failed on slice %d\n", k);
    //         free(data);
    //         return 1;
    //     }
    //     memcpy(pred_channel_complex + (size_t)k * time_slot_size, pred_channel, time_slot_size * sizeof(double));
    //     free(pred_channel);

    //     // copy outdated and gt channels for this slice
    //     memcpy(outdated_channel_complex + (size_t)k * time_slot_size, data + (size_t)k * time_stride + (N_SZ-2)*time_slot_size, time_slot_size * sizeof(double));
    //     memcpy(gt_channel_complex + (size_t)k * time_slot_size, data + (size_t)k * time_stride + (N_SZ-1)*time_slot_size, time_slot_size * sizeof(double));
    // }
//     for (int k = 0; k < d3; ++k) {
//       if(DEBUG) HELPER_LOGD("[DEBUG] Processing slice %d (d3 index)\n", k);
//       double *channel_address = data + (size_t)k * time_stride;
  
//       double *slice = channel_address;

//       print_first_n_doubles("[COMPARE] First 5 doubles of input slice:", slice, 5);
//       print_last_n_doubles("[COMPARE] Last 5 doubles of input slice:", slice, N_SZ * time_slot_size, 5);
//       double *pred_channel = channel_pred_sys(slice, N_SZ - 1);
//       double *outdated_channel = channel_address + (size_t)(N_SZ - 2) * time_slot_size;
//       double *groundtruth_channel = channel_address + (size_t)(N_SZ - 1) * time_slot_size;
  
//       // destination offsets
//       double *dst_pred = pred_channel_complex + (size_t)k * time_slot_size;
//       double *dst_out  = outdated_channel_complex + (size_t)k * time_slot_size;
//       double *dst_gt   = gt_channel_complex + (size_t)k * time_slot_size;
  
//       memcpy(dst_pred, pred_channel,       time_slot_size * sizeof(double));
//       memcpy(dst_out,  outdated_channel,   time_slot_size * sizeof(double));
//       memcpy(dst_gt,   groundtruth_channel,time_slot_size * sizeof(double));
  
//       free(pred_channel);
//   }
  
    // size_t total_len = (size_t)d3 * time_slot_size;
    // if(DEBUG) print_first_n_doubles("[COMPARE] First 10 doubles of pred_channel_complex", pred_channel_complex, 10);
    // if(DEBUG) print_last_n_doubles("[COMPARE] Last 10 doubles of pred_channel_complex", pred_channel_complex, total_len, 10);
    // double nmse_pred =
    //     base_NMSE_calc(gt_channel_complex,
    //                    pred_channel_complex,
    //                    total_len);
    
    // double nmse_outdated =
    //     base_NMSE_calc(gt_channel_complex,
    //                    outdated_channel_complex,
    //                    total_len);
    
    // printf("baseline NMSE = %f\n", nmse_outdated);
    // printf("pred NMSE = %f\n", nmse_pred);
    
    
    helper_log_close();
    free(data);
    free(pred_channel_complex);
    free(outdated_channel_complex);
    free(gt_channel_complex);
    return 0;
}