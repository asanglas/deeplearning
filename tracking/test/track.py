import pyprogress
import time

pyprogress.py_progress_start("Processing")
for i in range(100):
    pyprogress.py_progress_step(i, "Processing")
    time.sleep(0.1)
pyprogress.py_progress_stop()