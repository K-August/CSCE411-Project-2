# Compiler and flags
CXX = c++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC

# Dynamic Python and Pybind11 bindings
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
SUFFIX = $(shell python3-config --extension-suffix)

# External library paths 
# NOTE: Check if your Eigen is here. Sometimes it's in /usr/local/include/eigen3
EIGEN_INCLUDE = -I /usr/include/eigen3
HIGHS_INCLUDE = -I /usr/local/include/highs
HIGHS_LIB = -L /usr/local/lib -lhighs

# Source and Target files
SRC = fast_evaluator.cpp
TARGET = fast_evaluator$(SUFFIX)

# Default rule to build the module
all: $(TARGET)

# The compilation command
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND_INCLUDES) $(EIGEN_INCLUDE) $(HIGHS_INCLUDE) $< -o $@ $(HIGHS_LIB)

# Clean up compiled binaries
clean:
	rm -f fast_evaluator*.so fast_evaluator*.pyd