/* Program : To find the matrix multiplication of rectangular matrices using tiling
 * Author : Anant Shah
 * Date : 11-9-2018
 * Roll Number : EE16B105
 **/

#include<stdio.h>

#define ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)
#define ROWS_M 8192
#define COLS_M 16384
#define ROWS_N 16384
#define COLS_N 32768
#define NUM_THREADS_X 32
#define NUM_THREADS_Y 32
#define TILE_WIDTH_X 32
#define TILE_WIDTH_Y 32

void error_handler(cudaError_t error_msg,int line){
	if(error_msg!=cudaSuccess){
		printf("%s in %s at %d",cudaGetErrorString(error_msg),__FILE__,line);
		exit(EXIT_FAILURE);
	}
}

void fill_matrix(double *mat,unsigned numRows,unsigned numCols){
	/* function to fill a mtrix with values */
	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			mat[i*numCols+j] = i*2.1f + j*3.2f;
		}
	}
}

void print_matrix_to_file(double *mat,unsigned numRows,unsigned numCols){
	/* function to print a matrix to a file */

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

__global__ void matrixMul(double *M,double *N,double *P,int numRows_M,int numCols_N,int width){
	/* Kernel function to calculate the matrix multiplication of rectangular matrices */
	int tx = threadIdx.x; /* Thread-ID in the x-direction */
	int ty = threadIdx.y; /* Thread-ID in the y-direction */

	__shared__ double Ms[TILE_WIDTH_Y][TILE_WIDTH_X]; /* Shared memory to be used by threads in a block */
	__shared__ double Ns[TILE_WIDTH_X][TILE_WIDTH_Y]; /* Shared memory to be used by therads in a block */

	int row = blockIdx.y*blockDim.y + ty; /* row in the output matrix */
	int col = blockIdx.x*blockDim.x + tx; /* column in the output matrix */
	double pSum = 0.0;

	for(int m=0;m<(width+NUM_THREADS_X-1)/NUM_THREADS_X;m++){
		/* Load Pahse : Load elements cooperatively into the shared memeory */
		Ms[ty][tx] = M[row*width+m*TILE_WIDTH_X+tx];
		Ns[ty][tx] = N[(ty+m*TILE_WIDTH_Y)*numCols_N+col]; /* This is assuming that the tile is a sqaure */

		__syncthreads();

		for(int i=0;i<TILE_WIDTH_X;i++){
			pSum += Ms[ty][i]*Ns[i][tx];
		}
		__syncthreads();
	}
	P[row*numCols_N+col] = pSum;
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
        double          *h_M; /*Rectangular matrix M on the host */
        double          *d_M; /*Rectangular matrix M on the device */
        size_t          size_M; /* Size of the rectangular matrix M  in bytes */
        double          *h_N; /* Rectangular matrix N on the host */
        double          *d_N; /* Rectangular matrix N on the device */
        size_t          size_N; /* Size of the matrix N in bytes */
        double          *h_P; /* Product M*N on the host */
        double          *d_P; /* Product M*N on the device */
        size_t          size_P; /* Size of the matrix P in bytes */
        cudaEvent_t     start,stop;

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

        dim3    threads(NUM_THREADS_X,NUM_THREADS_Y); /*2-D layout of the threads in a block */
        dim3    blocks((COLS_N+NUM_THREADS_X-1)/NUM_THREADS_X,(ROWS_M+NUM_THREADS_Y-1)/NUM_THREADS_Y); /*2-D layout of blocks in a grid */

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

