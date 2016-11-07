# makefile format: GNU make

# REQUIRED: libcv, libcvaux

PREFIX         = /usr/local

OBJS           = main.o
TARGET         = kocr
CFLAGS_OPENCV  = `pkg-config --cflags opencv`
LDFLAGS_OPENCV = `pkg-config --libs opencv`
CLEAN_TARGET   = main.o kocr_cnn.o cropnums.o thinning.o kocr.o subr.o preprocess.o preprocess
CFLAGS         = -O3

SOLVER = CNN


ifeq ($(SOLVER), CNN)
	LIB_OBJS      = kocr_cnn.o cropnums.o thinning.o
	CFLAGS_SOLVER = -DUSE_CNN -DTHINNING
else ifeq ($(SOLVER), SVM)
	LIB_OBJS      = kocr.o subr.o cropnums.o thinning.o
	CFLAGS_SOLVER = -DUSE_SVM -DTHINNING
else
	LIB_OBJS      = kocr.o subr.o cropnums.o thinning.o
	CFLAGS_SOLVER =
endif


all: $(TARGET)

thin:
	libtool --mode=compile $(CXX) -DTHINNING_MAIN -c $(CFLAGS) $(CFLAGS_SOLVER) $(CFLAGS_OPENCV) thinning.cpp
	libtool --mode=link $(CXX) -o thin thinning.o $(LDFLAGS_OPENCV) $(LDFLAGS)

preprocess: thinning.o preprocess.o
	libtool --mode=link $(CXX) -o preprocess thinning.o preprocess.o $(LDFLAGS_OPENCV)

install: all
	-(for dir in bin include lib; do mkdir -p $(PREFIX)/$$dir; done)
	libtool --mode=install install -c kocr $(PREFIX)/bin
	libtool --mode=install install -c libkocr.la $(PREFIX)/lib
	install -c -m 444 -o root -g root kocr.h $(PREFIX)/include
	(cd ..; $(MAKE) install-db)

# dynamic link version of kocr
$(TARGET): $(LIB_OBJS) $(OBJS) libkocr.la
	libtool --mode=link $(CXX) -o $@ $(OBJS) libkocr.la $(LDFLAGS_OPENCV) $(LDFLAGS)

# static link version of kocr
static: $(LIB_OBJS) $(OBJS) libkocr.la
	libtool --mode=link $(CXX) -o kocr-static -static $(OBJS) libkocr.la $(LDFLAGS_OPENCV) $(LDFLAGS)

build-library: $(LIB_OBJS) libkocr.la

library:
	$(MAKE) build-library CFLAGS_LIBRARY_MODE=-DLIBRARY

install-library: library
	libtool --mode=install install -c libkocr.la $(PREFIX)/lib

libkocr.la: $(LIB_OBJS:.o=.lo)
	libtool --mode=link $(CXX) -o $@ $^ $(LDFLAGS_OPENCV) $(LDFLAGS) -rpath $(PREFIX)/lib

%.o: %.cpp
	libtool --mode=compile $(CXX) -c $(CFLAGS) $(CFLAGS_SOLVER) $(CFLAGS_LIBRARY_MODE) $(CFLAGS_OPENCV) $<

clean:
	libtool --mode=clean rm -f $(TARGET) kocr-static libkocr.la thin $(CLEAN_TARGET:.o=.lo)
