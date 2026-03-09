// helper.h
#ifndef HELPER_H
#define HELPER_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------
// Log level constants
// ------------------------------
enum {
    HELPER_LOG_DEBUG = 0,
    HELPER_LOG_INFO  = 1,
    HELPER_LOG_WARN  = 2,
    HELPER_LOG_ERROR = 3
};

// ------------------------------
// Public API
// ------------------------------

// Initialize logging.
// - path != NULL: append to that file
// - path == NULL: log file stream becomes stderr
// Returns 0 on success. If file open fails, returns non-zero but still falls back to stderr.
int  helper_log_init(const char *path);

// Close log file (if it is not stderr). Safe to call multiple times.
void helper_log_close(void);

// Flush both log file stream and console stream (safe no-op if not initialized).
void helper_log_flush(void);

// Enable/disable timestamps (default: enabled).
void helper_log_set_timestamp(int enabled);

// Set minimum log level (default: INFO).
// 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
void helper_log_set_level(int level);

// Enable/disable console output, and select stdout/stderr.
// - enabled: 0/1 (default: 1)
// - use_stderr: 0 => stdout, 1 => stderr (default: 1)
void helper_log_set_console(int enabled, int use_stderr);

// Low-level logging function (usually use macros below).
void helper_log_printf(int level,
                       const char *file,
                       const char *func,
                       int line,
                       const char *fmt, ...);

// Helper function to print first N doubles in fixed-point format (useful for checking start of tensor).
void print_first_n_doubles(const char *label, const double *data, size_t n);

// Helper function to print the last n doubles of a large array (useful for checking end of tensor).
void print_last_n_doubles(const char *label, const double *data, size_t total_len, size_t n);
// ------------------------------
// Convenience macros
// ------------------------------
#define HELPER_LOGD(fmt, ...) helper_log_printf(HELPER_LOG_DEBUG, __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define HELPER_LOGI(fmt, ...) helper_log_printf(HELPER_LOG_INFO,  __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define HELPER_LOGW(fmt, ...) helper_log_printf(HELPER_LOG_WARN,  __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define HELPER_LOGE(fmt, ...) helper_log_printf(HELPER_LOG_ERROR, __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // HELPER_H