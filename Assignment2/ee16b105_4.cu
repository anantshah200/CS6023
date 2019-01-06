/* Program : Perform matrix multiplication using the tiling algorithm
 * Author : Anant Shah
 * Roll Number : EE16B105
 * Date : 8-9-2018
 **/

#include<stdio.h>
#include<cuda.h>

#define ERROR_HANDLER(error_msg,line)	error_handler(error_msg,line)
#define SIZE 8192
#define NUM_THREADS_X 16
#define NUM_THREADS_Y 16
#define TILE_WIDTH 16

void error_handler(cudaError_t error_msg, int line){
	/* Function to flag an error if found in the CUDA program */
	if(error_msg!=cudaSuccess){
		printf("%s in %s at %d",cudaGetErrorString(error_msg),__FILE__,__LINE__);
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
	/* Kernel to find the matrix multiplication for each cell using the tiling algorithm
	 * TILE_WIDTH is taken to be the block size which is 16*16
	 * Parameters : M - Multiplicand matrix M
	 *		N - Multiplicand matrix N
	 *		P - Multiplicand matrix P
	 *		width - Dimension of the square block
	 */
	
	 int		bx = blockDim.x; /* Number of threads in the block in the x-direction */
	 int		by = blockDim.y; /* Number of threads in the block in the y-direction */
	 int		tx = threadIdx.x; /* x-Thread-ID in the block */
	 int		ty = threadIdx.y; /* y-Thread-ID in the block */
	 double		pSum = 0.0; /* Sum to store the partial matrix multiplication */

	 int 	Row = blockIdx.y*by+ty; /* Specific row of the cell in the output matrix */
	 int	Col = blockIdx.x*bx+tx; /* Column of the cell in the output matrix */

	 /* Decalring the shared memory variables */
	 __shared__ double	Ms[TILE_WIDTH][TILE_WIDTH]; /* Shared memory allocation for a tile of matrix M */
	 __shared__ double	Ns[TILE_WIDTH][TILE_WIDTH]; /* Shared memory allocation for a tile on matrix N */

	 /* We will use a for loop to go through each phase */
	 for(unsigned m=0;m<(width+TILE_WIDTH-1)/TILE_WIDTH;m++){
	 	/* Loading Operations */
		Ms[ty][tx] = M[Row*width+m*TILE_WIDTH+tx];
		Ns[ty][tx] = N[(m*TILE_WIDTH+ty)*width+Col];

		/* We now need to synchronize as all threads in a block have to finidh loading for the multiplication to continue */
		__syncthreads();
		
		/* Iterations to perform the partial matrix multiplication from the tile */
		for(unsigned i=0;i<TILE_WIDTH;i++){
			pSum += Ms[ty][i]*Ns[i][tx];
		}
		/* Another synchronization need to be performed so that all threads finish one phase and then start loading the next phase together */
		__syncthreads();
	 }
	 P[Row*width+Col] = pSum;
}

int main(int argc,char **argv){
	if(argc!=1){
		printf("error : Invalid number of arguments");
		exit(EXIT_FAILURE);
	}

	/********************************************** Variable Declaration *********************************/
	double		*h_M; /* Matrix multiplicand M on the host */
	double		*h_N; /* Matrix multiplicand N on the host */
	double		*h_P; /* Matrix product M*N on the host */
	double		*d_M; /* Matrix multiplicand M on the device */
	double		*d_N; /* Matrix multiplicand N on the device */
	double		*d_P; /* Matrix product M*N on the device */
	size_t		size; /* Size in bytes of the square matrices */
	cudaEvent_t	start,stop; /* CUDA events to measure the time of the matrix multiplication execution */

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	size = sizeof(double)*SIZE*SIZE;
	
	/************************************** Memory Allocation on the host ********************************/
	h_M = (double *)malloc(size);
	h_N = (double *)malloc(size);
	h_P = (double *)malloc(size);

	/************************************* Initialize the matrices **************************************/

	fill_matrix(h_M,SIZE,SIZE);
	fill_matrix(h_N,SIZE,SIZE);

	/************************************ Allocate memory on the device *********************************/
	ERROR_HANDLER(cudaMalloc((void **)&d_M,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_N,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_P,size),__LINE__);

	/**************************** Copy the matrices from the host to device ****************************/
	ERROR_HANDLER(cudaMemcpy(d_M,h_M,size,cudaMemcpyHostToDevice),__LINE__);
	ERROR_HANDLER(cudaMemcpy(d_N,h_N,size,cudaMemcpyHostToDevice),__LINE__);

	/************************************ Kernel Invocation *******************************************/
	dim3	threads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3	blocks((SIZE+NUM_THREADS_X-1)/NUM_THREADS_X,(SIZE+NUM_THREADS_Y-1)/NUM_THREADS_Y);

	cudaEventRecord(start);
	matrixMul<<<blocks,threads>>>(d_M,d_N,d_P,SIZE);
	cudaEventRecord(stop);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
	cudaEventSynchronize(stop);

	float	run_time = 0.0;
	cudaEventElapsedTime(&run_time,start,stop);
	printf("Run-Time(milli-seconds) : %.10f\n",run_time);

	print_matrix_to_file(h_P,SIZE,SIZE);

	/*********************************** Free the memory *********************************************/
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	free(h_M);
	free(h_N);
	free(h_P);
}
