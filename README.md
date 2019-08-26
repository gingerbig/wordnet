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

to interact with a console UI doing inference with the pre-trained model `model9-3000.bin`:

```
make forward load=model9-3000.bin 
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
- -- , ; : ? . 's ) $ a about after against ago all also american among an and another any are around as at back be because been before being best between big both business but by called can case center children city come companies company could country court day days department did director do does down dr. during each end even every family federal few first five for former found four from game general get go going good government group had has have he her here high him his home house how i if in including into is it its john just know last law left less life like little long made make man many market may me members might million money more most mr. ms. much music my national never new next night no not now nt of off office officials old on one only or other our out over own part people percent place play police political president program public put right said same say says school season second see set several she should show since so some state states still street such take team than that the their them then there these they think this those though three through time times to today too two under united university until up us use used very want war was way we week well were west what when where which while white who will with without women work world would year years yesterday york you your
|Input first 3 words > have a good 
have a good 
*Top 5 = 1.time(0.332076) 2.day(0.102091) 3.game(0.059815) 4.team(0.057552) 5.year(0.041762) 
|Choose a number (default = 1)>
```

Enjoy!
