# Define variables
CYTHON = cython
PYTHON = python
GCC = gcc
AR = ar
CFLAGS = -std=c99 -fPIC -O3 -flto
LDFLAGS = -shared -flto
INCLUDES = 
STATIC_LIB = libprogress.a
SHARED_LIB = libprogress.so

# Default target
all: build_ext

# Compile the C code into a static library
$(STATIC_LIB): progress.o
	$(AR) rcs $@ $^

# Compile the C code into a shared library
$(SHARED_LIB): progress.o
	$(GCC) $(LDFLAGS) $^ -o $@

progress.o: progress.c progress.h
	$(GCC) $(CFLAGS) -c $< -o $@

# Build the Cython extension
build_ext: $(STATIC_LIB) $(SHARED_LIB)
	$(PYTHON) setup.py build_ext --inplace

# Clean up build artifacts
clean:
	rm -f *.o $(STATIC_LIB) $(SHARED_LIB) pyprogress.c
	rm -rf build
	rm -f pyprogress.cpython-*.so

.PHONY: all clean

