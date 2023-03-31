CC = g++


SRC_DIR = src
OBJ_DIR = obj
INC_DIR = inc
CSRC = $(wildcard $(SRC_DIR)/*.cpp)
COBJ = $(CSRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
EXE = GridGenerator


CFLAGS = -c -g -O3 -Wall -std=c++17 -I$(INC_DIR) -I/usr/local/include/hdf5/ -I/usr/include/hdf5/serial  


all: $(EXE)

$(EXE): $(COBJ)
	$(CC) $^ -o $@  -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -lhdf5_cpp -g

# $(EXE): $(FOBJ) $(COBJ)
# $(CC) $(FLINK) $^ -o $@  $(FLINK) $(CLINK) 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $< $ -o  $@ -g

clean:
	rm -r $(COBJ) GridGenerator

# GridGenerator: Element.o Node.o Vertex.o Face.o Grid.o GridGenerator.o 
# 	$(CC) $^ -o $@  

# Element.o: Element.cpp
# 	$(CC) $(CFLAGS) Element.cpp

# Node.o: Node.cpp
# 	$(CC) $(CFLAGS) Node.cpp

# Face.o: Face.cpp
# 	$(CC) $(CFLAGS) Face.cpp

# Vertex.o: Vertex.cpp
# 	$(CC) $(CFLAGS) Vertex.cpp

# Grid.o: Grid.cpp
# 	$(CC) $(CFLAGS) Grid.cpp

# GridGenerator.o: GridGenerator.cpp
# 	$(CC) $(CFLAGS) GridGenerator.cpp


