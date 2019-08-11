SRCFILES := $(shell find . -type f -name "*.c")
OBJFILES := $(patsubst %.c,%.o,$(SRCFILES))
CFLAGS = -O3
CC = clang
INCLUDES = -I/usr/local/opt/llvm/include/c++/v1 -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

wordnet: $(OBJFILES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^

info: wordnet
	./wordnet info $(load)

train: wordnet
	./wordnet train $(load) $(save)

forward: wordnet
	./wordnet forward $(load)

.PHONY: clean

clean:
	rm -rf *.o main wordnet
