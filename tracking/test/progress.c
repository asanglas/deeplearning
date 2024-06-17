#include "progress.h"
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static struct winsize w;
static int initialized = 0;

static void progress_set_window_height(int height) {
  fprintf(stdout, "\n\0337\033[0;%dr\0338\033[1A\033[J", height);
  fflush(stdout);
}

static void progress_init() {
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  progress_set_window_height(w.ws_row - 1);
}

static void progress_abort() {
  progress_stop();
  exit(0);
}

void progress_start(const char *title) {
  progress_init();
  initialized = 1;
  progress_step(0, title);
}

void progress_step(float progress, const char *title) {
  if (!initialized) {
    fprintf(stderr, "error: progress_step() called before progress_start().\n");
    exit(1);
  }

  if (progress < 0)
    progress = 0;
  if (progress > 100)
    progress = 100;

  int width = w.ws_col - 20;
  char bar[width + 1];

  memset(bar, '.', width);
  memset(bar, '#', (int)(width * (progress / 100)));
  bar[width] = '\0';

  fprintf(stdout, "\e[s\e[%d;0H\e[42;30m%s: [%3d%%]\e[0m [%s]\e[u",
          w.ws_row + 1, title, (int)progress, bar);
  fflush(stdout);
}

void progress_stop() { progress_set_window_height(w.ws_row); }
