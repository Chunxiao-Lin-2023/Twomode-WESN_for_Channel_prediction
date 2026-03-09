// helper.c
#include "helper.h"

#include <stdarg.h>
#include <string.h>
#include <time.h>

// If you don't want thread-safety or you compile on embedded/no-pthread,
// set this to 0 and remove -lpthread from compile flags.
#define HELPER_LOG_THREAD_SAFE 1

#if HELPER_LOG_THREAD_SAFE
#include <pthread.h>
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;
#define LOCK()   pthread_mutex_lock(&g_log_mtx)
#define UNLOCK() pthread_mutex_unlock(&g_log_mtx)
#else
#define LOCK()   ((void)0)
#define UNLOCK() ((void)0)
#endif

// ------------------------------
// Globals (module-private)
// ------------------------------
static FILE *g_log_fp = NULL;          // file stream (or stderr fallback)
static int   g_log_level = HELPER_LOG_INFO;
static int   g_log_ts_enabled = 1;

// Console ��double write��
static int   g_console_enabled = 1;    // default: enabled
static FILE *g_console_fp = NULL;      // default: stderr

// Message buffer size for one log line (format body)
#ifndef HELPER_LOG_MSG_BUFSZ
#define HELPER_LOG_MSG_BUFSZ 4096
#endif

static const char *level_to_string(int level) {
    switch (level) {
        case HELPER_LOG_DEBUG: return "DEBUG";
        case HELPER_LOG_INFO:  return "INFO";
        case HELPER_LOG_WARN:  return "WARN";
        case HELPER_LOG_ERROR: return "ERROR";
        default:               return "INFO";
    }
}

static void set_line_buffered(FILE *fp) {
    if (!fp) return;
    // Best-effort: if setvbuf fails, ignore
    setvbuf(fp, NULL, _IOLBF, 0);
}

// Ensure streams are not NULL
static FILE *ensure_log_stream(void) {
    if (g_log_fp) return g_log_fp;
    g_log_fp = stderr;
    set_line_buffered(g_log_fp);
    return g_log_fp;
}

static FILE *ensure_console_stream(void) {
    if (g_console_fp) return g_console_fp;
    g_console_fp = stderr;
    set_line_buffered(g_console_fp);
    return g_console_fp;
}

// Build prefix into 'out' and return length written
static int build_prefix(char *out, size_t out_sz,
                        int level, const char *file, const char *func, int line)
{
    size_t off = 0;

    if (g_log_ts_enabled) {
        time_t now = time(NULL);
        struct tm tm_now;
        char ts[32] = {0};

#if defined(_POSIX_C_SOURCE) || defined(__linux__) || defined(__APPLE__)
        localtime_r(&now, &tm_now);
        strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_now);
#else
        struct tm *ptm = localtime(&now);
        if (ptm) {
            tm_now = *ptm;
            strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_now);
        } else {
            strncpy(ts, "0000-00-00 00:00:00", sizeof(ts) - 1);
        }
#endif
        off += (size_t)snprintf(out + off, (off < out_sz ? out_sz - off : 0),
                                "%s ", ts);
    }

    off += (size_t)snprintf(out + off, (off < out_sz ? out_sz - off : 0),
                            "[%s] %s:%d (%s): ",
                            level_to_string(level),
                            file ? file : "?", line,
                            func ? func : "?");

    if (off >= out_sz) return (int)(out_sz - 1);
    return (int)off;
}

int helper_log_init(const char *path) {
    LOCK();

    // Close old file stream if it was a real file
    if (g_log_fp && g_log_fp != stderr) {
        fclose(g_log_fp);
    }
    g_log_fp = NULL;

    // Default console stream
    if (!g_console_fp) g_console_fp = stderr;
    set_line_buffered(g_console_fp);

    int ret = 0;
    if (path == NULL) {
        g_log_fp = stderr;
        set_line_buffered(g_log_fp);
        UNLOCK();
        return 0;
    }

    FILE *fp = fopen(path, "a");
    if (!fp) {
        // fallback to stderr
        g_log_fp = stderr;
        set_line_buffered(g_log_fp);
        ret = 1;
    } else {
        g_log_fp = fp;
        set_line_buffered(g_log_fp);
    }

    UNLOCK();
    return ret;
}

void helper_log_close(void) {
    LOCK();

    if (g_log_fp && g_log_fp != stderr) {
        fclose(g_log_fp);
    }
    g_log_fp = NULL;

    UNLOCK();
}

void helper_log_flush(void) {
    LOCK();
    fflush(ensure_log_stream());
    if (g_console_enabled) fflush(ensure_console_stream());
    UNLOCK();
}

void helper_log_set_timestamp(int enabled) {
    LOCK();
    g_log_ts_enabled = (enabled != 0);
    UNLOCK();
}

void helper_log_set_level(int level) {
    LOCK();
    if (level < HELPER_LOG_DEBUG) level = HELPER_LOG_DEBUG;
    if (level > HELPER_LOG_ERROR) level = HELPER_LOG_ERROR;
    g_log_level = level;
    UNLOCK();
}

void helper_log_set_console(int enabled, int use_stderr) {
    LOCK();
    g_console_enabled = (enabled != 0);
    g_console_fp = use_stderr ? stderr : stdout;
    set_line_buffered(g_console_fp);
    UNLOCK();
}

void helper_log_printf(int level,
                       const char *file,
                       const char *func,
                       int line,
                       const char *fmt, ...)
{
    if (level < g_log_level) return;

    LOCK();

    FILE *lfp = ensure_log_stream();
    FILE *cfp = (g_console_enabled ? ensure_console_stream() : NULL);

    // Build prefix once
    char prefix[256];
    (void)build_prefix(prefix, sizeof(prefix), level, file, func, line);

    // Format body once
    char body[HELPER_LOG_MSG_BUFSZ];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);

    // Write to log stream
    fputs(prefix, lfp);
    fputs(body, lfp);
    fputc('\n', lfp);

    // Double-write to console stream
    if (cfp) {
        fputs(prefix, cfp);
        fputs(body, cfp);
        fputc('\n', cfp);
    }

    UNLOCK();
}

void print_first_n_doubles(const char *label, const double *data, size_t n) {
    HELPER_LOGD("%s (first %zu doubles):", label, n);
    for (size_t i = 0; i < n; i++) {
        HELPER_LOGD("%s [%d]: %f ", label, i, data[i]);
    }
}

void print_last_n_doubles(const char *label, const double *data, size_t total_len, size_t n) {
    HELPER_LOGD("%s (last %zu of %zu doubles):", label, n, total_len);
    for (size_t i = (total_len > n ? total_len - n : 0); i < total_len; i++) {
        HELPER_LOGD("%s [%d]: %f ", label, i, data[i]);
    }
}