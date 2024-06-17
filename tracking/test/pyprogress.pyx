cdef extern from "progress.h":
    void progress_start(const char *title)
    void progress_step(float progress, const char *title)
    void progress_stop()

def py_progress_start(str title):
    progress_start(title.encode('utf-8'))

def py_progress_step(float progress, str title):
    progress_step(progress, title.encode('utf-8'))

def py_progress_stop():
    progress_stop()