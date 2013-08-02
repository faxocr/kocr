#
# Simple makefile for kocr
#

# makefile format: GNU make

# REQUIRED: libcv, libcvaux

PREFIX         = /usr/local

OBJS           = main.o
LIB_OBJS       = kocr.o subr.o cropnums.o
TARGET         = kocr
CFLAGS_OPENCV  = `pkg-config --cflags opencv`
LDFLAGS_OPENCV = `pkg-config --libs opencv`

CFLAGS		= -O3 -DUSE_SVM
# CFLAGS	= -g -DUSE_SVM
# CFLAGS	= -O3
# CFLAGS	= -g
# CFLAGS	= -O3 -static

# all: $(TARGET) library static

all: $(TARGET)

install: all
	-(for dir in bin include lib share/kocr/databases; do mkdir -p $(PREFIX)/$$dir; done)
	libtool --mode=install install -c kocr $(PREFIX)/bin
	libtool --mode=install install -c libkocr.la $(PREFIX)/lib
	install -c -m 444 -o root -g root kocr.h $(PREFIX)/include
	install -c -m 444 -o root -g root ../databases/* $(PREFIX)/share/kocr/databases

# dynamic link version of kocr
$(TARGET): $(LIB_OBJS) $(OBJS) libkocr.la
	libtool --mode=link $(CXX) -o $@ $(OBJS) libkocr.la $(LDFLAGS_OPENCV) $(LDFLAGS)

# static link version of kocr
static: $(LIB_OBJS) $(OBJS)
	libtool --mode=link $(CXX) -o kocr-static -static $(OBJS) libkocr.la $(LDFLAGS_OPENCV) $(LDFLAGS)

# library: libkocr.la

library:
	rm -f *.o
	$(CXX) $(CFLAGS) $(OPT_OPENCV) -DLIBRARY -c kocr.cpp
	$(CXX) $(CFLAGS) $(OPT_OPENCV) -DLIBRARY -c subr.cpp
	$(CXX) $(CFLAGS) $(OPT_OPENCV) -DLIBRARY -c cropnums.cpp
	$(AR) rv libkocr.a kocr.o subr.o cropnums.o
	ranlib libkocr.a
	rm -f *.o

libkocr.la: $(LIB_OBJS:.o=.lo)
	libtool --mode=link $(CXX) -o $@ $^ $(LDFLAGS_OPENCV) $(LDFLAGS) -rpath $(PREFIX)/lib

kocr.o: kocr.cpp kocr.h Labeling.h subr.h

subr.o: subr.cpp kocr.h Labeling.h subr.h

cropnums.o: cropnums.cpp cropnums.h

main.o: main.cpp kocr.h
	$(CXX) -c main.cpp -o main.o $(CFLAGS) $(CFLAGS_OPENCV)

%.o: %.cpp
	libtool --mode=compile $(CXX) -c $(CFLAGS) $(CFLAGS_OPENCV) $<

clean:
	libtool --mode=clean rm -f $(TARGET) kocr-static libkocr.la libkocr.a $(OBJS) $(LIB_OBJS:.o=.lo)
