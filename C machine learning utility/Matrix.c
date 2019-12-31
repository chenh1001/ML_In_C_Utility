#include "Matrix.h"

Matrix newMatrix(int rows, int cols)//returns a new empty matrix by given size
{
	double **data = malloc(rows * sizeof(double));
	for (int i = 0; i<rows; i++)
	{
		data[i] = malloc(cols * sizeof(double));
	}
	for (int i = 0; i<rows; i++)
	{
		for (int j = 0; j<cols; j++)
		{
			data[i][j] = 0;
		}
	}
	Matrix temp = { rows,cols,data };
	return temp;
}

double randBetweenZeroAndOne()//returnes a random bum between 0-1
{
	return ((double)rand() / (RAND_MAX));
}

void matrixAddtoMatrix(Matrix dest, Matrix m)//add two given matixes
{
	for (int i = 0; i<dest.rows; i++)
	{
		for (int j = 0; j<dest.cols; j++)
		{
			dest.data[i][j] += m.data[i][j];
		}
	}
}
void matrixMutate(double Mrate, Matrix m)//mutate a matrix by a given value
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			if (randBetweenZeroAndOne() < Mrate)
				m.data[i][j] += (randBetweenZeroAndOne()) * 2 - 1;
		}
	}
}

void matrixSigmoid(Matrix m)//sigmoid on all values
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			m.data[i][j] = (1.0 / (1 + exp(-m.data[i][j])));
		}
	}
}

void matrixDSigmoid(Matrix m)//d sigmoid on all values
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			m.data[i][j] = m.data[i][j] * (1 - m.data[i][j]);
		}
	}
}

Matrix copyMatrix(Matrix a)//copy matrix
{
	Matrix retM = newMatrix(a.rows, a.cols);
	for (int i = 0; i < retM.rows; i++)
	{
		for (int j = 0; j < retM.cols; j++)
		{
			retM.data[i][j] = a.data[i][j];
		}
	}
	return retM;
}

Matrix dSigmoid(Matrix a)//dSigmoid that returnes a matrix
{
	Matrix retM = copyMatrix(a);
	for (int i = 0; i < retM.rows; i++)
	{
		for (int j = 0; j < retM.cols; j++)
		{
			retM.data[i][j] = retM.data[i][j] * (1 - retM.data[i][j]);
		}
	}
	return retM;
}

Matrix transpose(Matrix m)//transposes a matrix (x,y)->(y,x)
{
	Matrix temp = newMatrix(m.cols, m.rows);
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			temp.data[j][i] = m.data[i][j];
		}
	}
	return temp;
}

void matrixAddMatrix(Matrix dest, Matrix m)//add two matrixes
{
	for (int i = 0; i < dest.rows; i++)
	{
		for (int j = 0; j < dest.cols; j++)
		{
			dest.data[i][j] += m.data[i][j];
		}
	}
}

Matrix substuct(Matrix a, Matrix b)//subtruct two matrixes
{
	Matrix temp = newMatrix(a.rows, a.cols);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			temp.data[i][j] = a.data[i][j] - b.data[i][j];
		}
	}
	return temp;
}

void matrixRandomize(Matrix m)//randomize a matrix values
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			double r = randBetweenZeroAndOne();
			m.data[i][j] = (r * 2 - 1);
		}
	}
}


void matrixMultiplyNum(double n, Matrix m)//multiply a matrix by a num
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			m.data[i][j] *= n;
		}
	}
}

void matrixMultiplySimple(Matrix dest, Matrix m)//matrix simple multiplication
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			dest.data[i][j] = dest.data[i][j] * m.data[i][j];
		}
	}
}
Matrix matrixMultiply(Matrix a, Matrix b)//matrix multiply
{
	if (a.cols != b.rows)
		puts("cant multiply matrixes");
	Matrix temp = newMatrix(a.rows, b.cols);
	for (int i = 0; i < temp.rows; i++)
	{
		for (int j = 0; j < temp.cols; j++)
		{
			double sum = 0;
			for (int k = 0; k < a.cols; k++)
			{
				sum += a.data[i][k] * b.data[k][j];
			}
			temp.data[i][j] = sum;
		}
	}
	return temp;
}

void matrixPrint(Matrix m)//print matrix
{
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			printf("%0.4f ", m.data[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

