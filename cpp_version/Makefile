CC = g++-6

CFLAGS = -c -Wall

all: GridGenerator

GridGenerator: Node.o Grid.o GridGenerator.o

Node.o: Node.cpp
	$(CC) $(CFLAGS) Node.cpp

Grid.o: Grid.cpp
	$(CC) $(CFLAGS) Grid.cpp

GridGenerator.o: GridGenerator.cpp
	$(CC) $(CFLAGS) GridGenerator.cpp
