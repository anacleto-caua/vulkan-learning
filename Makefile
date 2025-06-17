# Project config
PROJECT_NAME = learning
BUILD_DIR = build
SOURCE_DIR = /src
# Defined source directory
VPATH = src

CFLAGS = -std=c++17
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
REALEASE_FLAGS = -O2

# Define BUILD_TYPE, defaulting to 'debug'
BUILD_TYPE ?= debug

# Conditionally add release flags to CFLAGS based on BUILD_TYPE
ifeq ($(BUILD_TYPE),release)
	CFLAGS += $(REALEASE_FLAGS)
    # Change the output name for release builds
	TARGET_NAME = $(PROJECT_NAME)_release
else
	TARGET_NAME = $(PROJECT_NAME)
endif

# Ensure the build directory exists silently
$(shell mkdir -p $(BUILD_DIR))

# Main build target
build: main.cpp
	g++ $(CFLAGS) -o $(BUILD_DIR)/$(TARGET_NAME) $< $(LDFLAGS)

# Phony target for release builds
release:
	$(MAKE) BUILD_TYPE=release build

.PHONY: build release test clean

run: build
	./$(BUILD_DIR)/$(TARGET_NAME)

clean:
	rm -rf $(BUILD_DIR)/*