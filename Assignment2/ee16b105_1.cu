/* Program : Performing a matrix multiplication with a 2-D block and 2-D thread layout
 * Author : Anant Shah
 * Date : 5-9-2018
 * Roll Number : EE16B105
 */

#include<stdio.h>
#define SIZE 8192
#define NUM_THREADS 16
#define ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)

void error_handler(cudaError_t error_msg,int line){
	if(error_msg != cudaSuccess){
		printf("%s in %s at %d",cudaGetErrorString(error_msg),__FILE__,line);
		exit(EXIT_FAILURE);
	}
}

void fill_matrix(double *mat, unsigned numRows, unsigned numCols){
	/* Function to populate a 2-D matrix with the specified number of rows and columns */
	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			mat[i*numCols+j] = i*2.1f + j*3.2f;
		}
	}
}

void print_matrix_to_file(double *mat, unsigned numRows, unsigned numCols){
	/* Function to print the matrix elements into a file */
	const char *fname = "assignment2_1_out";
	FILE	*f = fopen(fname,"a");

	for(unsigned i=0; i<numRows; i++){
		for(unsigned j=0;j<numCols;j++){
			fprintf(f,"%4.4f ",mat[i*numCols+j]);
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void matrixMulX(double *M,double *N,double *P,int width){
	/* Kernel to perform the matrix multiplication between two N*N matrices where the fastest changing index is ".y"
	 * Each thread calculates one cell in the output matrix 
	 * Parameters : M - matrix M stored as 1-D layout
	 * 	      : N - matrix N stored as 1-D layout
	 *	      : P - matrix P which stores the multiplication of the matrices A and B
	 *	      : width - Number of rows/columns in the matrix as they are square matrices
	 */
	 int		col = blockIdx.x*blockDim.x+threadIdx.x; /* Column of the cell to be calculated */
	 int 		row = blockIdx.y*blockDim.y+threadIdx.y; /* Row of the cell to be calculated */
	 double		pSum = 0.0 ; /* Variable to store the partial sum of each multiplication */
	 for(unsigned i=0;i<width;i++){
	 	pSum += M[row*width+i]*N[i*width+col];
	 }
	 P[row*width+col] = pSum;
}

__global__ void matrixMulY(double *M,double *N,double *P,int width){
	/* Kernel to perform the matrix multiplication between two N*N matrices where the fastest changing index is ".x"
	 * Each thread calculates one cell in the output matrix
	 * Parameters : M - matrix M stored as 1-D layout
	 *	      : N - matrix N stored as 1-D layout
	 *	      : P - matrix P stored as 1-D layout
	 *	      : width - Number of rows/columns in the matrix as they are square matrices
	 */
	 int		col = blockIdx.x*blockDim.x+threadIdx.y; /* Column of the cell to be calculated */
	 int		row = blockIdx.y*blockDim.y+threadIdx.x; /* Row of the cell to be calculated */
	 double		pSum = 0.0;
	 for(unsigned i=0;i<width;i++){
	 	//pSum += M[i*width+row]*N[i*width+col];
		pSum += M[row*width+i]*N[i*width+col];
	 }
	 P[row*width+col] = pSum;
}

int main(int argc,char **argv){
	/************************************* Variable Initialization ***************************************/

	double		*h_M; /* Matrix multiplicand on the host*/
	double		*h_N; /* Matrix multiplicand on the host*/
	double		*h_P; /* Result of the matrix multiplication on the host*/
	double		*h_P_t; /* Result of the transpose matrix multiplication on the host*/
	double		*d_M; /* Matrix multilpicand on the device */
	double		*d_N; /* Martix multiplicand on the device */
	double		*d_P; /* Matrix multiplicand on the device */
	double		*d_P_t; /* Result of the transpose matrix multiplication on the device */
	size_t		size; /* Number of bytes required to store the matrices in the memory */
	cudaEvent_t	start, stop; /* Cuda Events to be used to measure the run-time of the multiplication kernel where the fastest moving index is ".x" */
	cudaEvent_t	start_t,stop_t; /* Cuda Events to be used to measure the run-time of the multiplication kernel where the fastest moving index is "".y*/
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_t);
	cudaEventCreate(&stop_t);

	/*****************************************************************************************************/

	/*************************** Allocating memory to the matrices on the device *************************/
	
	size = sizeof(double)*SIZE*SIZE; /* Each array is a 2-D matrix with size N*N */
	
	h_M = (double *)malloc(size);
	h_N = (double *)malloc(size);
	h_P = (double *)malloc(size);
	h_P_t = (double *)malloc(size);

	/*************************** Initialize the matrices with values ************************************/
	fill_matrix(h_M,SIZE,SIZE);
	fill_matrix(h_N,SIZE,SIZE);

	/************************** Allocate memory on the GPU to store the matrices ************************/
	ERROR_HANDLER(cudaMalloc((void **)&d_M,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_N,size),__LINE__);	
	ERROR_HANDLER(cudaMalloc((void **)&d_P,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_P_t,size),__LINE__);

	/************************** Copy the matrices from the host to device ******************************/
	ERROR_HANDLER(cudaMemcpy(d_M,h_M,size,cudaMemcpyHostToDevice),__LINE__);
	ERROR_HANDLER(cudaMemcpy(d_N,h_N,size,cudaMemcpyHostToDevice),__LINE__);

	/************************* Define the block dimensions and grid dimensions **************************/
	
	dim3	threads(NUM_THREADS,NUM_THREADS); /* Define a (16,16) block of threads */
	dim3	blocks((SIZE+NUM_THREADS-1)/NUM_THREADS,(SIZE+NUM_THREADS-1)/NUM_THREADS); /* The ceiling function has been used to define the blocks, generating a 2-D grid structure of blocks */

	cudaEventRecord(start);
	matrixMulX<<<blocks,threads>>>(d_M,d_N,d_P,SIZE); /* Execute the kernel */
	cudaEventRecord(stop);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);

	cudaEventRecord(start_t);
	matrixMulY<<<blocks,threads>>>(d_M,d_N,d_P_t,SIZE); /*Execute the kernel */
	cudaEventRecord(stop_t);

	ERROR_HANDLER(cudaMemcpy(h_P_t,d_P_t,size,cudaMemcpyDeviceToHost),__LINE__);
	cudaEventSynchronize(stop_t);

	float milliseconds = 0;
	float milliseconds_t = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventElapsedTime(&milliseconds_t, start_t, stop_t);	

	print_matrix_to_file(h_P,SIZE,SIZE);
	print_matrix_to_file(h_P_t,SIZE,SIZE);
	printf("Run-Time(seconds) for index change '.x' : %.10f \n",milliseconds_t/1000);
	printf("Run-Time(seconds) for index chnage '.y' : %.10f \n",milliseconds/1000);

	/************************ Free the Memory that was allocated ****************************************/
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	cudaFree(d_P_t);
	free(h_M);
	free(h_N);
	free(h_P);
	free(h_P_t);
}
