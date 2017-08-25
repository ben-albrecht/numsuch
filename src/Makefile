CC=chpl
ZMQ_BASE=/usr/local/Cellar/zeromq/4.2.2
INCLUDES=-I${ZMQ_BASE}/include
LIBS=-L${ZMQ_BASE}/lib -lzmq

default: all

all: numsuch.chpl
	$(CC) $(INCLUDES) $(LIBS) -o numsuch $<
