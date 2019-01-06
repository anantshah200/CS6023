/* Program : To find the matrix multiplication of rectangular matrices without tiling
 * Author : Anant Shah
 * Date : 11-9-2018
 * Roll Number : EE16B105
 **/

#include<stdio.h>

#define ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)
#define ROWS_M 4096
#define COLS_M 8192
#define ROWS_N 8192
#define COLS_N 16384
#define NUM_THREADS_X 16
#define NUM_THREADS_Y 16

void error_handler(cudaError_t error_msg,int line){
	/* Will terminate the program if an error caused due to a CUDA statement */
	if(error_msg!=cudaSuccess){
		printf("%s in %s at line %d",cudaGetErrorString(error_msg),__FILE__,line);
		exit(EXIT_FAILURE);
	}
}

void fill_matrix(double *mat,unsigned numRows,unsigned numCols){
	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			mat[i*numCols+j] = i*2.1f+j*3.2f;
		}
	}
}

void print_matrix_to_file(double *mat,unsigned numRows,unsigned numCols){
	const char 	*fname = "assignment2_out";
	FILE		*f = fopen(fname,"a");
	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			fprintf(f,"%4.4f ", mat[i*numCols+j]);
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void matrixMul(double *M,double *N,double *P,unsigned numRows_M,unsigned numCols_N,unsigned dim){
	/* Program : To find the rectangular matrix matrix multiplication 
	 * Shared Memory and Tiling has not been used in this kernel
	 * Parameters : M - Matrix M of size (numRows_M,dim)
	 * 		N - Matrix N of size (dim,numCols_N)
	 *		P - Output matrix (M*N)
	 *		numRows_M - Number of rows in the M matrix
	 *		numCols_N - Number of columns in the N matrix
	 *		dim - Numbor of columns in M = Number of rows in N
 	 */

	int	Row = blockIdx.y*blockDim.y+threadIdx.y;
	int 	Col = blockIdx.x*blockDim.x+threadIdx.x;

	if(Row<numRows_M && Col<numCols_N){
		double	pSum = 0.0;
		for(int i=0;i<dim;i++){
			pSum += M[Row*dim+i]*N[i*numCols_N+Col];
		}
		P[Row*numCols_N+Col] = pSum;
	}
}

int main(int argc,char **argv){
	if(argc!=1){
		printf("error : Invalid number of arguments\n");
		exit(EXIT_FAILURE);
	}

	if(COLS_M!=ROWS_N){
		printf("Error : Invalid matrix dimensions");
		exit(EXIT_FAILURE);
	}

	/************************************* Variable Initialization **************************************/
	double		*h_M; /*Rectangular matrix M on the host */	
	double		*d_M; /*Rectangular matrix M on the device */
	size_t		size_M; /* Size of the rectangular matrix M  in bytes */
	double		*h_N; /* Rectangular matrix N on the host */
	double		*d_N; /* Rectangular matrix N on the device */
	size_t		size_N; /* Size of the matrix N in bytes */
	double		*h_P; /* Product M*N on the host */
	double		*d_P; /* Product M*N on the device */
	size_t		size_P; /* Size of the matrix P in bytes */
	cudaEvent_t	start,stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	size_M = sizeof(double)*ROWS_M*COLS_M;
	size_N = sizeof(double)*ROWS_N*COLS_N;
	size_P = sizeof(double)*ROWS_M*COLS_N;

	/************************************** Memory Allocation on the Host ********************************/
	h_M = (double *)malloc(size_M);
	h_N = (double *)malloc(size_N);
	h_P = (double *)malloc(size_P);
	
	/************************************** Initialize the matrices ***************************************/
	fill_matrix(h_M,ROWS_M,COLS_M);
	fill_matrix(h_N,ROWS_N,COLS_N);

	/************************************** Allocate memory on the device *********************************/
	ERROR_HANDLER(cudaMalloc((void **)&d_M,size_M),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_N,size_N),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_P,size_P),__LINE__);

	/************************************** Copy Matrices to the device ***********************************/

	ERROR_HANDLER(cudaMemcpy(d_M,h_M,size_M,cudaMemcpyHostToDevice),__LINE__);
	ERROR_HANDLER(cudaMemcpy(d_N,h_N,size_N,cudaMemcpyHostToDevice),__LINE__);

	/************************************** Kernel invocation *********************************************/

	dim3	threads(NUM_THREADS_X,NUM_THREADS_Y); /*2-D layout of the threads in a block */
	dim3	blocks((COLS_N+NUM_THREADS_X-1)/NUM_THREADS_X,(ROWS_M+NUM_THREADS_Y-1)/NUM_THREADS_Y); /*2-D layout of blocks in a grid */

	cudaEventRecord(start);	
	matrixMul<<<blocks,threads>>>(d_M,d_N,d_P,ROWS_M,COLS_N,COLS_M); /* The last parameter could have been <ROWS_N> */
	cudaEventRecord(stop);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size_P,cudaMemcpyDeviceToHost),__LINE__);

	cudaEventSynchronize(stop);

	float run_time = 0.0;
	cudaEventElapsedTime(&run_time,start,stop);
	printf("Run-Time(seconds) : %.4f",run_time/1000);

	print_matrix_to_file(h_P,ROWS_M,COLS_N);
	
	/********************************** Free Allocated Memory ********************************************/

	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	free(h_M);
	free(h_N);
	free(h_P);
}	
