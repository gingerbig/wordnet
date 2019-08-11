#ifndef _LOAD_DATA
#define _LOAD_DATA

/*
 Global variables: declaration
*/
extern int g_train_data[372550][4];
extern int g_validation_data[46568][4];
extern int g_test_data[46568][4];
extern char g_vocabulary[251][20]; // start from 1
static const int g_input_dimension = 3;

/*
 Load all data to global variables
 * data/train_data.csv => g_train_data
 * data/valid_data.csv => g_validation_data
 * data/test_data.csv  => g_test_data
 * data/vocab.txt      => g_vocabulary
*/
void loadAllData(void);

/*
 Create mini-batches for training
 * @param rawData, row, column, inputDimension: rawData is a row x column data
 matrix for training. Each row stands for one training datum: the first
 inputDimension columns are fed to the network, and the last (column -
 inputDimension) columns are the ground truth output.
 * @param miniBatchSize: how many items of training data should be contained in
 a mini-batch.
 * @return batchInput: a pointer to the input of mini-batches, batchNum x
 miniBatchSize x inputDimension
 * @return batchOutput: a pointer to the target output (i.e. ground truth) of
 mini-batches, batchNum x N x (column-inputDimension)
 * @return batchNum: after dividing the rawData with miniBatchSize, batchNum
 shows how many batches there are
*/
void createMiniBatch(const int *rawData, int row, int column,
                     int inputDimension, int miniBatchSize, int **batchInput,
                     int **batchOutput, int *batchNum);

/*
 Destroy the resourses used by mini-batches
 * @param batchInput
 * @param batchOutput
*/
void destroyMiniBatch(int *batchInput, int *batchOutput);

#endif