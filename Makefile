CC=g++
CFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`
DEBUG = -g

DEPS = machine_learning.hpp

PROG=machinel
#OBJS=$(PROG).o

OBJS = main.o rgb_to_hsi.o kmeans.o feature_extraction.o svm_classifier.o

.PHONY: all clean
$(PROG): $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<

all: $(PROG)

clean:
	rm -f $(OBJS) $(PROG)
