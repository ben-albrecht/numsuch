CC=chpl
ZMQ_BASE=/usr/local/Cellar/zeromq/4.2.2
BLAS_HOME= /usr/local/Cellar/openblas/0.2.20
INCLUDES=-I${ZMQ_BASE}/include -I$(BLAS_HOME)/include
LIBS=-L${ZMQ_BASE}/lib -lzmq -L${BLAS_HOME}/lib -lblas

default: all

all: numsuch.chpl
	$(CC) $(INCLUDES) $(LIBS) -o numsuch $<

mlp: mlp.chpl
	$(CC) $(INCLUDES) $(LIBS) -o mlp $<
