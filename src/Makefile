#
# Simple makefile for xocr
#

# REQUIRED: libcv, libcvaux

CC      = g++
AR      = ar

OBJS    = kocr.o subr.o cropnums.o main.o
TARGET  = kocr
CPPFLAGS = `pkg-config --cflags opencv`
OPT_OPENCV = `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`
OPT	= -O3
#OPT	= -g
# OPT	= -O3 -static

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) `pkg-config --libs opencv`

# doesn't work...orz
static:
	$(CC) $(OPT) $(OPT_OPENCV) -o $@ $(OBJS)

library:
	rm *.o
	$(CC) $(OPT) $(OPT_OPENCV) -DLIBRARY -c kocr.cpp
	$(CC) $(OPT) $(OPT_OPENCV) -DLIBRARY -c subr.cpp
	$(CC) $(OPT) $(OPT_OPENCV) -DLIBRARY -c cropnums.cpp
	$(AR) rv libkocr.a kocr.o subr.o cropnums.o
	ranlib libkocr.a
	rm *.o

kocr.o: kocr.cpp kocr.h Labeling.h subr.h

subr.o: subr.cpp kocr.h Labeling.h subr.h

cropnums.o: cropnums.cpp cropnums.h

main.o: main.cpp kocr.h

.cpp.o:
	$(CC) -c $(OPT) $(OPT_OPENCV) $<

clean:
	-rm -f a.out *.o $(TARGET) libkocr.a

