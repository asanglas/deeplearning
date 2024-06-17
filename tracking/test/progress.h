#ifndef PROGRESS_H
#define PROGRESS_H

void progress_start(const char *title);
void progress_step(float progress, const char *title);
void progress_stop();

#endif // PROGRESS_H
