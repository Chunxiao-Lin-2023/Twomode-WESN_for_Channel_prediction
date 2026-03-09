
Updated build comments
gcc -I. -DHELPER_ENABLE_DEBUG=1 -o main main.c twomode_core.c channel_pred_sys.c helper.c -lm -lpthread

./main H_real_imag_filtered_float64.npy
