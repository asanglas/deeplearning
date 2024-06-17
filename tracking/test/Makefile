# Define variables
CYTHON = cython
PYTHON = python
GCC = gcc
AR = ar
CFLAGS = -std=c99 -fPIC
LDFLAGS = 
INCLUDES = 
LIBRARY = libprogress.a

# Default target
all: build_ext

# Compile the C code into a static library
$(LIBRARY): progress.o
	$(AR) rcs $@ $^

progress.o: progress.c progress.h
	$(GCC) $(CFLAGS) -c $< -o $@

# Build the Cython extension
build_ext: $(LIBRARY)
	$(PYTHON) setup.py build_ext --inplace

# Clean up build artifacts
clean:
	rm -f *.o $(LIBRARY) pyprogress.c
	rm -rf build
	rm -f pyprogress.cpython-*.so

.PHONY: all clean
