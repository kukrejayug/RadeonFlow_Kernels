# Makefile for HIP hook library

CC = gcc
CFLAGS = -Wall -Wextra -fPIC -shared -O2
LDFLAGS = -ldl
TARGET = libhip_hook.so
SOURCE = hip_hook.c

# ROCm include path (adjust if needed)
ROCM_PATH = /opt/rocm
INCLUDES = -I$(ROCM_PATH)/include

.PHONY: all clean install test

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig

# Test target - you can replace 'your_hip_program' with your actual HIP program
test: $(TARGET)
	@echo "To test the hook, run:"
	@echo "LD_PRELOAD=./$(TARGET) your_hip_program"

# Debug version with more verbose output
debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all     - Build the hook library"
	@echo "  clean   - Remove built files"
	@echo "  install - Install library to system (requires sudo)"
	@echo "  test    - Show how to use the library"
	@echo "  debug   - Build debug version"
	@echo "  help    - Show this help message"