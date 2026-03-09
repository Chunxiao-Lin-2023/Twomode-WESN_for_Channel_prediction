run this command if gcc is present:
gcc -I. -o main main.c twomode_core.c channel_pred_sys.c -lm

then to execute the code:
./main H_real_imag_filtered.npy



Current Problem/ Issue:
-Error once building features (line 494 of channel_pred_sys.c) is being iterated through (around RB 14)
-Error message is: "malloc(): unaligned tcache chunk detected \n aborted (core dumped)"



Updated build comments
gcc -I. -DHELPER_ENABLE_DEBUG=1 -o main main.c twomode_core.c channel_pred_sys.c helper.c -lm -lpthread

./main H_real_imag_filtered_float64.npy