/* Program : To find the run-time for the matrix multiplication kernel without tiling for various block sizes
 * Author : Anant Shah
 * Date : 13-9-2018
 * Roll Number : EE16B105
 **/

#include<stdio.h>

#define ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)
#define NUM_THREADS_X 16
#define NUM_THREADS_Y 16
#define X_1 4
#define Y_1 4
#define X_2 4
#define Y_2 8
#define X_3 8
#define Y_3 4
#define X_4 8
#define Y_4 8
#define X_5 8
#define Y_5 16
#define X_6 16
#define Y_6 8
#define X_7 16
#define Y_7 32
#define SIZE 8192

void error_handler(cudaError_t error_msg,int line){
	/* Function to check if a CUDA statement resulted in a function */
	if(error_msg!=cudaSuccess){
		printf("%s in %s at %d",cudaGetErrorString(error_msg),__FILE__,line);
		exit(EXIT_FAILURE);
	}
}

void fill_matrix(double *mat,unsigned numRows,unsigned numCols){
	/* Function to fill a mtrix with values */
	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			mat[i*numCols+j] = i*2.1f + j*3.2f;
		}
	}
}

void print_matrix_to_file(double *mat,unsigned numRows,unsigned numCols){
	/* Function to wite a matrix into a file */
	const char 	*fname = "assignment2_out";
	FILE		*f = fopen(fname,"a");

	for(unsigned i=0;i<numRows;i++){
		for(unsigned j=0;j<numCols;j++){
			fprintf(f,"%4.4f ",mat[i*numCols+j]);
		}
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void matrixMul(double *M,double *N,double *P,int width){
	/* Kernel to perform the matrix multiplication between two N*N matrices where the fastest changing index is ".x"
         * Each thread calculates one cell in the output matrix 
         * Parameters : M - matrix M stored as 1-D layout
         *            : N - matrix N stored as 1-D layout
         *            : P - matrix P which stores the multiplication of the matrices A and B
         *            : width - Number of rows/columns in the matrix as they are square matrices
         */
         int            col = blockIdx.x*blockDim.x+threadIdx.x; /* Column of the cell to be calculated */
         int            row = blockIdx.y*blockDim.y+threadIdx.y; /* Row of the cell to be calculated */
         double         pSum = 0.0 ; /* Variable to store the partial sum of each multiplication */

	 if( (row<width) && (col<width)){ /* Condition to check if the element to be calculated is within bounds */
         	for(unsigned i=0;i<width;i++){
                	pSum += M[row*width+i]*N[i*width+col];
         	}
	 }
         P[row*width+col] = pSum;	
}

int main(int argc,char **argv){
	if(argc!=1){
		printf("error: Invalid number of arguments\n");
		exit(EXIT_FAILURE);
	}

	/******************************* Variable Initialization  ********************************/

	double		*h_M; /* Matrix multiplicand on the host */
	double		*h_N; /* Matrix multiplicand on the host */
	double		*h_P; /* Matrix multiplication result  on the host */
	double		*d_M; /* Matrix multiplicand on the device */
	double		*d_N; /* Matrix multiplicand on the device */
	double		*d_P; /* Matrix multiplication result on the device */
	size_t		size; /* Size of each matrix in bytes */
	cudaEvent_t	start,stop; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_1,stop_1; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_2,stop_2; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_3,stop_3; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_4,stop_4; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_5,stop_5; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_6,stop_6; /* CUDA events to be used to calculate the kernel-run time */
	cudaEvent_t     start_7,stop_7; /* CUDA events to be used to calculate the kernel-run time */

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_1);
        cudaEventCreate(&stop_1);
	cudaEventCreate(&start_2);
        cudaEventCreate(&stop_2);
	cudaEventCreate(&start_3);
        cudaEventCreate(&stop_3);
	cudaEventCreate(&start_4);
        cudaEventCreate(&stop_4);
	cudaEventCreate(&start_5);
        cudaEventCreate(&stop_5);
	cudaEventCreate(&start_6);
        cudaEventCreate(&stop_6);
	cudaEventCreate(&start_7);
        cudaEventCreate(&stop_7);
	
	/*********************************** Allocate Memory on the Host *************************/

	size = sizeof(double)*SIZE*SIZE;

	h_M = (double *)malloc(size);
	h_N = (double *)malloc(size);
	h_P = (double *)malloc(size);

	/********************************* Initialize matrix with values ***********************/
	fill_matrix(h_M,SIZE,SIZE);
	fill_matrix(h_N,SIZE,SIZE);

	/******************************* Allocate Memory on the GPU ****************************/
	ERROR_HANDLER(cudaMalloc((void **)&d_M,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_N,size),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_P,size),__LINE__);

	/****************************** Copy multiplicands to the device ***********************/
	ERROR_HANDLER(cudaMemcpy(d_M,h_M,size,cudaMemcpyHostToDevice),__LINE__);
	ERROR_HANDLER(cudaMemcpy(d_N,h_N,size,cudaMemcpyHostToDevice),__LINE__);
	
	/****************************** Kernel Invocation **************************************/

	dim3	threads(NUM_THREADS_X,NUM_THREADS_Y);
	dim3	blocks((SIZE+NUM_THREADS_X-1)/NUM_THREADS_X,(SIZE+NUM_THREADS_Y-1)/NUM_THREADS_Y);
	float   milliseconds = 0.0;

	cudaEventRecord(start);	
	matrixMul<<<blocks,threads>>>(d_M,d_N,d_P,SIZE);
	cudaEventRecord(stop);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
	print_matrix_to_file(h_P,SIZE,SIZE);

	dim3 threads_1(X_1,Y_1);
	dim3 blocks_1((SIZE+X_1-1)/X_1,(SIZE+Y_1-1)/Y_1);
	cudaEventRecord(start_1);
        matrixMul<<<blocks_1,threads_1>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_1);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);


	dim3 threads_2(X_2,Y_2);
        dim3 blocks_2((SIZE+X_2-1)/X_2,(SIZE+Y_2-1)/Y_2);
        cudaEventRecord(start_2);
        matrixMul<<<blocks_2,threads_2>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_2);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

 	dim3 threads_3(X_3,Y_3);
        dim3 blocks_3((SIZE+X_3-1)/X_3,(SIZE+Y_3-1)/Y_3);
        cudaEventRecord(start_3);
        matrixMul<<<blocks_3,threads_3>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_3);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

 	dim3 threads_4(X_4,Y_4);
        dim3 blocks_4((SIZE+X_4-1)/X_4,(SIZE+Y_4-1)/Y_4);
        cudaEventRecord(start_4);
        matrixMul<<<blocks_4,threads_4>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_4);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

 	dim3 threads_5(X_5,Y_5);
        dim3 blocks_5((SIZE+X_5-1)/X_5,(SIZE+Y_5-1)/Y_5);
        cudaEventRecord(start_5);
        matrixMul<<<blocks_5,threads_5>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_5);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

	dim3 threads_6(X_6,Y_6);
        dim3 blocks_6((SIZE+X_6-1)/X_6,(SIZE+Y_6-1)/Y_6);
        cudaEventRecord(start_6);
        matrixMul<<<blocks_6,threads_6>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_6);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

	dim3 threads_7(X_7,Y_7);
        dim3 blocks_7((SIZE+X_7-1)/X_7,(SIZE+Y_7-1)/Y_7);
        cudaEventRecord(start_7);
        matrixMul<<<blocks_7,threads_7>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_7);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds,start_1,stop_1);
	printf("Run-Time(seconds) (4*4): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start_2,stop_2);
	printf("Run-Time(seconds) (4*8): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start_3,stop_3);
	printf("Run-Time(seconds) (8*4): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start_4,stop_4);
	printf("Run-Time(seconds) (8*8): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start_5,stop_5);
	printf("Run-Time(seconds) (8*16): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start_6,stop_6);
	printf("Run-Time(seconds) (16*8): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start,stop);
	printf("Run-Time(seconds) (16*16): %.10f\n",milliseconds/1000);
	cudaEventElapsedTime(&milliseconds,start_7,stop_7);
	printf("Run-Time(seconds) (16*32): %.10f\n",milliseconds/1000);

	/**************************** Free Allocated Memory **********************************/
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	free(h_M);
	free(h_N);
	free(h_P);
}
