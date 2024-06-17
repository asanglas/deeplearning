import ctypes
import time
import random
from tqdm import tqdm
import pyprogress


class ProgressBar:
    def __init__(self, title, total):
        self.title = title
        self.libprogress = ctypes.CDLL("./libprogress.so")  # Adjust path as necessary
        self.total = total

    def __enter__(self):
        self.libprogress.progress_start(self.title.encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.libprogress.progress_stop()

    def update_progress(self, progress):
        self.libprogress.progress_step(
            ctypes.c_float(progress / self.total * 100), self.title.encode("utf-8")
        )

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < self.total:
            self.update_progress(self.current)
            self.current += 1
            return self.current
        # else:
        #     self.complete_progress()
        #     raise StopIteration


# Generator function to create progress bar
def progress_bar(iterable, title):
    total = len(iterable)
    with ProgressBar(title, total) as pbar:
        for i, item in enumerate(iterable):
            yield item
            pbar.update_progress(i + 1)


strings = [
    "Lorem ipsum dolor sit amet",
    "Consectetur adipiscing elit",
    "Vivamus faucibus sagittis dui, tincidunt rhoncus mi",
    "Fringilla sollicitudin. Donec eget sagittis",
    "Quam, vitae fringilla nisl",
    "Donec dolor justo, hendrerit sed accumsan id, sodales",
    "Eu odio",
    "Nunc vehicula hendrerit risus, vel condimentum dui rutrum sed.",
    "Quisque metus enim, pellentesque nec nibh sit amet.",
    "Commodo molestie diam.",
]


# Example usage
if __name__ == "__main__":
    print("hey")
    total_iterations = 100000

    t0_tqdm = time.time_ns()
    for i in tqdm(range(total_iterations), desc="Hola"):
        # Simulate work inside the loop
        print(f"{strings[random.randint(0, len(strings)-1)]}")

    t1_tqdm = time.time_ns()

    t0_progress = time.time_ns()
    for i in progress_bar(range(total_iterations), "Hola"):
        print(f"{strings[random.randint(0, len(strings)-1)]}")
    t1_progress = time.time_ns()
    print(pyprogress.__dict__)

    t0_progress_cython = time.time_ns()
    pyprogress.py_progress_start("Hola")
    for i in range(total_iterations):
        print(f"{strings[random.randint(0, len(strings)-1)]}")
        pyprogress.py_progress_step(i / 100.0, "Hola")
    pyprogress.py_progress_stop()
    t1_progress_cthon = time.time_ns()

    # print(f"time progressbar wrapper : {(t1-t0)/1e6} ms")
    print(f"time progressbar: {(t1_progress-t0_progress)/1e6} ms")
    print(f"time progressbar cython: {(t1_progress_cthon-t0_progress_cython)/1e6} ms")
    print(f"time tqdm: {(t1_tqdm-t0_tqdm)/1e6} ms")
# time.sleep(0.01)
