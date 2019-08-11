#ifndef _MODEL_H
#define _MODEL_H

typedef struct _HyperParameter {
  // Tunable
  int miniBatchSize;
  int layer1Neurons;
  int layer2Neurons;
  int epoch;
  int earlyStopIteration;
  double momentum;
  double learningRate;
  int verifyPerIterBatch;

  // Fixed
  int rawDataRow4Training;
  int rawDataRow4Validation;
  int rawDataRow4Test;
  int rawDataColumn;
  int inputDimension;
  int vocabSize;
}HyperParameter;

typedef struct _WordNet {
  HyperParameter hp;

  // Some shortcuts
  int D;         // Input dimension
  int N;         // Mini-batch size
  int vocabSize; // Vocabulary size
  int h1; // Neuron number of hidden layer 1 (one word embedding length)
  int h2; // Neuron number of hidden layer 2

  /*
   Word embedding weights
   * h1 x vocabSize
   * The D input words will share the same W1
   * Respectively, convert input word vectors into word embedding vectors,
   so that each word is now represented by a feature vector with h1 entries
  */
  double *W1;
  double *dW1; //* h1 x vocabSize, for update delta

  /*
   * As each input batch has a size of N x D, convert it to word vector
   * representations: vocabSize x (D x N) matrix.
   ! Each column of the matrix stands for a word, e.g. [0 0 ... 1 ... 0]'
   ! So-called "One-hot vector encoding"
  */
  double *inputWordVectorBatch;
  double *xit; //* Xi', N x vocabSize, for backprop

  /*
   * Now calculate
   * bufferW1xInputWordVectorBatch = W1 * inputWordVectorBatch
   * We get an h1 x (D x N) matrix
   */
  double *bufferW1xInputWordVectorBatch;

  /*
   * Now merge every D columns into a new column,
   * which is the word embedding layer,
   * and bufferW1xInputWordVectorBatch becomes a (h1 x D) x N matrix
   */
  double *layer1StateBatch; // Embedding layer state, y1
  double *y1t;              //* y1', for backprop

  /*
   * Now we use a fully connected layer 2, with a weighting matrix W2
   * h2 x (h1 x D)
   */
  double *W2;
  double *dW2; //* h2 x (h1 x D), for update delta
  double *W2t; //* W2', for backprop

  /*
   * layer2StateBatch = sigmoid(W2 * layer1StateBatch + bias2)
   * layer2StateBatch: h2 x N
   * bias2: h2 x 1, adding to each column of (W2 * Layer1StateBatch)
   */
  double *layer2StateBatch; //! Used as buffer as well, y2
  double *y2t;              //* y2', for backprop
  double *bias2;
  double *db2; //* h2 x 1, for update delta

  /*
   * Now we use a fully connected layer 3, with a weighting matrix W3
   * vocabSize x h2
   */
  double *W3;
  double *dW3; //* vocabSize x h2, for update delta
  double *W3t; //* W3', for backprop

  /*
   * layer3StateBatch = softmax(W3 * layer2StateBatch + bias3)
   * layer3StateBatch: vocabSize x N
   * bias3: vocabSize x 1, adding to each column of (W3 * layer2StateBatch)
   */
  double *layer3StateBatch; //! Used as buffer & output as well, y3
  double *bias3;
  double *db3; //* vocabSize x 1, for update delta

  const double *outputStateBatch; // = layer3StateBatch, y3

  double *targetVectorBatch; // vocabSize x N, Compared with outputStateBatch

  //! ================================
  //! Following: for back-propagation
  //! ================================

  /*
   * y1 = [ W1X1 ]
   *      [ W1X2 ]
   *      [ W1X3 ]
   * y2_ = W2y1 + b2
   * y2  = sigmoid(y2_)
   * y3_ = W3y2 + b3
   * y3  = softmax(y3_)
   ! output: y3
   ! target: yt
   ! Loss = - sum( yt .* log(y3) )
   */

  const double *yt; // = targetVectorBatch, vocabSize x N

  // dL/dy3_ batch: vocabSize x N
  // dL/dy3_ = y3 - yt
  // dLdy3_t: transpose of dLdy3_
  double *dLdy3_;
  double *dLdy3_t;

  // dL/dW3: vocabSize x h2
  // dL/dW3 = dL/dy3_ * (dy3_/dW3)'
  //        = dLdy3_ * y2'
  // (vocabSize x N) x (N x h2)
  double *dLdW3;

  /*
   * dL/db3 = dy3_/db3 * dL/dy3_
   *        = dL/dy3_
   * dL/db3 = sumColumn2Vector(dL/dy3_)
   * vocabSize x 1
   */
  double *dLdb3;

  // dL/dy2: h2 x N
  // dL/dy2 = dy3_/dy2 * dL/dy3_
  //        = W3' * dLdy3_
  // (h2 x vocabSize) x (vocabSize x N)
  double *dLdy2;

  // dLdy2_: h2 x N
  // dL/dy2_ = dy2/dy2_ * dL/dy2
  //         = y2 .* (1 - y2) .* dLdy2
  // (h2 x N) .* (h2 x N) .* (h2 x N)
  double *dLdy2_;

  /*
   * dL/db2 = dy2_/db2 * dL/dy2_
   *        = dL/dy2_
   * dL/db2 = sumColumn2Vector(dL/dy2_)
   * h2 x 1
   */
  double *dLdb2;

  // dL/dW2: h2 x (D x h1)
  // dL/dW2 = dL/dy2_ * (dy2_/dW2)'
  //        = dLdy2_ * y1'
  // (h2 x N) x (N x (h1 x D))
  double *dLdW2;

  // dL/dy1: (h1 x D) x N
  // dL/dy1 = dy2_/dy1 * dL/dy2_
  //        = W2' * dLdy2_
  // ((h1 x D) x h2) x (h2 x N)
  double *dLdy1;
  double *dLdy1i; //* h1 x N

  // dL/dW1: h1 x vocabSize
  // dL/dW1 = sum_{i=1}^D dL/dy1[i] * Xi'
  // (h1 x N) x (N x vocabSize)
  double *dLdW1;
} WordNet;

/*
 Create and initialize a WordNet
 */
void createWordNet(WordNet *model);

/*
 Destroy a WordNet
 */
void destroyWordNet(WordNet *model);

/*
 Forward propagation for batch training
 * @param model
 * @param miniBatchInput: n x inputDimension, where
 !        n <= model->N
 *        miniBatchSize == model->N,
 *        inputDimension == model->D
 * @return model->outputStateBatch: vocabSize x N
 *         outputStateBatch
 *         = softmax(W3 * (sigmoid(W2*layer1StateBatch + bias2)) + bias3)
 */
void forwardPropagate(WordNet *model, const int *miniBatchInput, int n);

/*
 Load index target batch and convert to one-hot batch
 * @param targetBatch: N x 1
 * @return model->targetVectorBatch, vocabSize x N
 */
void loadTarget2TargetVectorBatch(WordNet *model, const int *targetBatch);

/*
 Back propagation for batch training
 * @param model
 * @return
 */
void backPropagate(WordNet *model);

/*
 Update network parameters: weights and biases
 * @param model, momentum, learningRate
 * @return W3, bias3, W2, bias2, W1
 */
void updateNetworkParameters(WordNet *model, double momentum,
                             double learningRate);

#endif
