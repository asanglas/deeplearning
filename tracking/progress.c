#include "progress.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

// Define constants
#define ROUNDS 10000
#define STRINGS_COUNT 10

// Structure to hold the window size
static struct winsize w;
static int initialized = 0;

// Function to set the terminal window height
static void progress_set_window_height(int height) {
  fprintf(stdout,
          "\n\0337"        // Save cursor
          "\033[0;%dr"     // Set scroll region
          "\0338"          // Restore cursor
          "\033[1A\033[J", // Move cursor inside the scroll area
          height);
  fflush(stdout);
}

// Initialize the progress bar
static void progress_init() {
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
  progress_set_window_height(w.ws_row - 1);
}

// Abort the progress bar and exit
static void progress_abort() {
  progress_stop();
  exit(0);
}

// Start the progress bar with a given title
void progress_start(const char *title) {
  signal(SIGWINCH, progress_init); // Handle window resize
  signal(SIGINT, progress_abort);  // Handle interrupt signal
  progress_init();
  initialized = 1;
  progress_step(0, title);
}

// Update the progress bar
void progress_step(float progress, const char *title) {
  if (!initialized) {
    fprintf(stderr, "error: progress_step() called before progress_start().\n");
    exit(1);
  }

  if (progress < 0)
    progress = 0;
  if (progress > 100)
    progress = 100;

  int width = w.ws_col - 20; // Bar width
  char bar[width + 1];

  memset(bar, '.', width);
  memset(bar, '#', (int)(width * (progress / 100)));
  bar[width] = '\0';

  fprintf(stdout, "\e[s\e[%d;0H\e[42;30m%s: [%3d%%]\e[0m [%s]\e[u",
          w.ws_row + 1, title, (int)progress, bar);
  fflush(stdout);
}

// Stop the progress bar
void progress_stop() { progress_set_window_height(w.ws_row); }

#if 1
int main(int argc, char **argv) {

  static const char *strings[] = {
      "Lorem ipsum dolor sit amet",
      "Consectetur adipiscing elit",
      "Vivamus faucibus sagittis dui, tincidunt rhoncus mi",
      "Fringilla sollicitudin. Donec eget sagittis",
      "Quam, vitae fringilla nisl",
      "Donec dolor justo, hendrerit sed accumsan id, sodales",
      "Eu odio",
      "Nunc vehicula hendrerit risus, vel condimentum dui rutrum sed.",
      "Quisque metus enim, pellentesque nec nibh sit amet.",
      "Commodo molestie diam."};
  // Example usage
  progress_start("Hola");

  // Just echoes random strings and updates the progress bar
  srand(time(NULL));
  int r;
  printf("This is the first line\n");

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  for (int i = 0; i <= ROUNDS; i++) {
    r = rand() % STRINGS_COUNT;
    printf("%s...\n", strings[r]);
    // usleep(0.1 * 1000);
    progress_step(((float)i / ROUNDS) * 100, "Hola");
  }
  progress_stop();

  cpu_time_used = ((double)(clock() - start)) / CLOCKS_PER_SEC * 1000;
  printf("It took %f ms to execute \n", cpu_time_used);

  return 0;
}
#endif
