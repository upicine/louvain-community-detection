CC          := nvcc
CFLAGS      := -O3 -dc
LFLAGS      := -O3
ALL         := \
	louvian

louvian : $(ALL)

louvian : main.o utils.o aggregation.o modularity.o
	$(CC) $(LFLAGS) -o $@ $^

main.o : main.cu utils.cuh aggregation.cuh modularity.cuh
	$(CC) $(CFLAGS) $<

aggregation.o : aggregation.cu aggregation.cuh
	$(CC) $(CFLAGS) $<

modularity.o : modularity.cu modularity.cuh
	$(CC) $(CFLAGS) $<

utils.o : utils.cu utils.cuh
	$(CC) $(CFLAGS) $<

clean :
	rm -f *.o $(ALL)