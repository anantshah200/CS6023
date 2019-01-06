/* Program : To perform matrix multiplication using shared memory
 * Author : Anant Shah
 * Roll Number : EE16B105
 * Date : 7-9-2018
 **/

#include<stdio.h>

#define ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)
#define SIZE 32
#define NUM_THREADS_X 16
#define NUM_THREADS_Y 16

void error_handler(cudaError_t error_msg,int line){
	/* Error handler function */
	if( error_msg!=cudaSuccess ){
		printf("%s in %s at %d",cudaGetErrorString(error_msg),__FILE__,line);
		exit(EXIT_FAILURE);
	}
}

void fill_matrix(double *mat, unsigned numRows, unsigned numCols){
	/*Program to fill the elements of a matrix with the given number of rows and columns */
	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			mat[i*numCols+j] = i*2.1f+j*3.2f;	
		}
	}
}

void print_matrix_to_file(double *mat, unsigned numRows, unsigned numCols){
        /* Function to print the matrix elements into a file */
        const char *fname = "assignment2_out";
        FILE    *f = fopen(fname,"a");

        for(unsigned i=0; i<numRows; i++){
                for(unsigned j=0;j<numCols;j++){
                        fprintf(f,"%4.4f ",mat[i*numCols+j]);
                }
                fprintf(f,"\n");
        }
        fclose(f);
}

__global__ void matrixMul(double *M,double *N,double *P,int width){
	/* Kernel to perform the matrix multiplication output for each cell 
	 * Shared Memory on the SM will be utilized for each thread
	 * Parameters : M - Matrix multiplicand
	 * 		N - matrix Multiplicand
	 *		P - Output of the product M*N
	 *		width - Dimensions of the square matrices
	 */
	int		tx = threadIdx.x; /* x-ID of a thread in a block */
	int		ty = threadIdx.y; /* y-ID of a thread in a block */
	
	int 		Row = blockIdx.y*blockDim.y+ty; /* Row of the cell in the output matrix */
	int 		Col = blockIdx.x*blockDim.x+tx; /* Row of the cell in the input matrix */

	/* Declaring array variables in the shared memory to be used by the threads to perform the matrix multiplication  */
	__shared__ double	Ms[NUM_THREADS_Y][SIZE]; /* A 2-D array to store the variables required by a block from matrix M */
	__shared__ double	Ns[SIZE][NUM_THREADS_X]; /* A 2-D array to store the variables required by a block from matrix N */

	/* Loading the matrices from the global memory to the shared memory by collaborative loading */
	int		num_col_cells = (width+blockDim.x-1)/blockDim.x; /* Number of cells to be stored from matrix M in  global memory for a given cell in the outpt matrix*/
	int		num_row_cells = (width+blockDim.y-1)/blockDim.y; /* Number of cells to be stored from matrix N in global memory for a given cell in the output matrix */

	/*Since for this case, we are taking the square matrix to be a block, num_col_cells = num_row_cells and hence both the loading operations can be completed in the same probelm  */

	for(int i=0;i<num_col_cells;i++){
		Ms[ty][tx*num_col_cells+i] = M[Row*width+tx*num_col_cells+i];
		Ns[ty*num_row_cells+i][tx] = N[(ty*num_row_cells+i)*width+Col];
	}
	__syncthreads(); /*All threads in the block need to finish loading before we can proceed with the multiplication */
	
	/*The partial matrices have been stored in the shared memory, we now need to perform matrix multiplication */
	double		pSum=0.0; /*Partial sum to store the multiplication */
	for(int i=0;i<width;i++){
		pSum += Ms[ty][i]*Ns[i][tx];
	}
	P[Row*width+Col] = pSum;
}

int main(int argc,char **argv){
	
	if(argc!=1){
		printf("Error : Invalid number of arguments");
		exit(EXIT_FAILURE);
	}

	/******************************** Variable Declarations  ***************************************************/

	double		*h_M; /* Matrix multiplicand M on the host */
	double		*h_N; /* Matrix multiplicand N on the host */
	double		*h_P; /* Matrix product(M*N) on the host */
	double		*d_M; /* Matrix multiplicand M on the device */
	double		*d_N; /* Matrix multiplicand N on the device */
	double		*d_P; /* Matrix product(M*N) on the device */
	size_t		size; /* The size in bytes of each matrix */
	cudaEvent_t	start,stop; /* Cuda Events to measure the run-time of the kernel  */

	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	size = sizeof(double)*SIZE*SIZE; /* These are square matrices of dimensions SIZE*SIZE  */

	/******************************** Allocate Memory on the host *********************************************/
	h_M = (double *)malloc(size);
	h_N = (double *)malloc(size);
	h_P = (double *)malloc(size);
	
	/******************************** Initialize the matrices ************************************************/
	fill_matrix(h_M,SIZE,SIZE);
	fill_matrix(h_N,SIZE,SIZE);
	
	/******************************** Allocate Memory on the device ******************************************/
	ERROR_HANDLER(cudaMalloc((void **)&d_M,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_N,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_P,size),__LINE__);

	/******************************* Copy matrices from host to device **************************************/

	ERROR_HANDLER(cudaMemcpy(d_M,h_M,size,cudaMemcpyHostToDevice),__LINE__);
	ERROR_HANDLER(cudaMemcpy(d_N,h_N,size,cudaMemcpyHostToDevice),__LINE__);

	/****************************** Kernel Invocation *******************************************************/
	dim3	threads(NUM_THREADS_X,NUM_THREADS_Y);	
	dim3 	blocks((SIZE+NUM_THREADS_X-1)/NUM_THREADS_X,(SIZE+NUM_THREADS_Y-1)/NUM_THREADS_Y);

	cudaEventRecord(start);
	matrixMul<<<blocks,threads>>>(d_M,d_N,d_P,SIZE);
	cudaEventRecord(stop);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);

	cudaEventSynchronize(stop);

	float	milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);	
	printf("Run-Time(milli-seconds) : %.10f \n",milliseconds);

	print_matrix_to_file(h_P,SIZE,SIZE);
	/***************************** Free the memory that was allocated **************************************/
	
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	free(h_M);
	free(h_N);
	free(h_P);
}
