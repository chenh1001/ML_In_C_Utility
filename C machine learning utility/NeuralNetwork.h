#ifndef nnHeader
#define nnHeader
#include "Matrix.h"

typedef struct NeuralNets {
	int inputsLength;
	int hiddenLength;
	int outputsLength;
	Matrix weigthsIH;
	Matrix weigthsHO;
	Matrix biasH;
	Matrix biasO;

} NeuralNetwork;

void freeMatrix(Matrix m);
void train(double* inputsArr, int inputSize, double* targetsArr, int targetsSize, NeuralNetwork nn);
NeuralNetwork newNeuralNetwork(int inputsLength, int hiddenLength, int outputsLength);
NeuralNetwork copyNeuralNetwork(NeuralNetwork nn);
Matrix NNfeedingForward(double* inputsArr, int inputSize, NeuralNetwork nn);
void NNMutate(double mutateRate, NeuralNetwork nn);

#endif
