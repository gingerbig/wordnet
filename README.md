# WordNet: A Toy Example for Deep Learning

This project is based on the second programming assignment of Hinton's Coursera
course [*Neural Networks for Machine
Learning*](https://www.coursera.org/learn/neural-networks/home/welcome). The
original code is written in MATLAB/Octave. Since I find this example elegent for
showing how deep learning works, I rewrite the code with C from scratch and call
it WordNet (without Hinton's permission). For such a teaching purpose, the code doesn't have a GPU mode or a
good performance in terms of memory or CPU usage, and currently I have no plan for
optimization.

Basically, WordNet reads three consecutive words and predict the fourth word.
The layout of WordNet is defined as

![Layout of
WordNet](https://github.com/gingerbig/wordnet/blob/master/pics/layout.png
"Layout of WordNet")

For more technical details, please read `Slides.pdf`.

Building the project is quite simple since it only relies on the standard C
libs. If your building tool chain is properly configured, you may simply modify
the following lines in `src/makefile` with your compiler and header include path:

```makefile
CC = clang
INCLUDES = -I/usr/local/opt/llvm/include/c++/v1 -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
```

and type 

``` shell
make forward load=model9-3000.bin
```

to interact with a console UI doing inference with the pre-trained model
`model9-3000.bin`. If everything goes well, you'll see

```
> make forward load=model9-3000.bin 
./wordnet forward model9-3000.bin
# Load all data
# Load model: model9-3000.bin
## Model Info
   Mini-batch size          =      100
   Layer 1 Neurons          =       50
   Layer 2 Neurons          =      200
   Training epochs          =        9
   Early stop @ iteration   =     3000
   Momentum                 = 0.900000
   Learning rate            = 0.100000
   Verify per iteration     = 2147483647
   Raw training data rows   =   372550
   Raw validation data rows =    46568
   Raw test data rows       =    46568
   Raw data columns         =        4
   Input dimension          =        3
   Vocabulary size          =      250
##------Interactive UI------##
[... here lists the vocabulary.]
|Input first 3 words > have a good 
have a good 
*Top 5 = 1.time(0.332076) 2.day(0.102091) 3.game(0.059815) 4.team(0.057552) 5.year(0.041762) 
|Choose a number (default = 1)>
```

Of course, this project also covers codes for training:

``` shell
Usage:
  ./wordnet info model.bin               | Show info of pretrained model.
  ./wordnet train model.bin              | Train from scratch and save model.
  ./wordnet train pretrain.bin model.bin | Read pretrained data, finetune it, & save model.
  ./wordnet forward pretrain.bin         | Read pretrained data and do inferences.
Or
  make info load=model.bin
  make train save=model.bin
  make train load=pretrain.bin save=model.bin
  make forward load=pretrain.bin
```

Enjoy!
