/* Program : To find the run-time of matrix multiplication using tiling for different tile sizes 
 * Author : Anant Shah
 * Roll Number : EE16B105
 * Date : 10-9-2018
 */

#include<stdio.h>

#define	ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)
#define T_1 4
#define T_2 8
#define T_3 16
#define T_4 32
#define SIZE 8192

void error_handler(cudaError_t error_msg,int line){
	/* Function to check if a CUDA C command causes an error */
	if(error_msg!=cudaSuccess){
		printf("%s in %s at line %d",cudaGetErrorString(error_msg),__FILE__,line);
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

__global__ void matrixMula(double *M,double *N,double *P,int width){
	/* Function to find the matrix multiplication using the tiling algorithm 
	 * Shared memory has been used to avoid high latency global memory accesses
	 * Parameters : M - matrix multiplicand on the device
	 *		N - matrix multiplicand on the device
	 *		P - output of the matrix multiplication M*N
	 *		width - dimensions of the matrix(it is a square matrix)
	 */
	 int            bx = blockDim.x; /* Number of threads in the block in the x-direction */
         int            by = blockDim.y; /* Number of threads in the block in the y-direction */
         int            tx = threadIdx.x; /* x-Thread-ID in the block */
         int            ty = threadIdx.y; /* y-Thread-ID in the block */
         double          pSum = 0.0; /* Sum to store the partial matrix multiplication */

         int    Row = blockIdx.y*by+ty; /* Specific row of the cell in the output matrix */
         int    Col = blockIdx.x*bx+tx; /* Column of the cell in the output matrix */

         /* Decalring the shared memory variables */
         __shared__ double       Ms[T_1][T_1]; /* Shared memory allocation for a tile of matrix M */
         __shared__ double       Ns[T_1][T_1]; /* Shared memory allocation for a tile on matrix N */

         /* We will use a for loop to go through each phase */
         for(int m=0;m<(width+T_1-1)/T_1;m++){
                /* Loading Operations */
                Ms[ty][tx] = M[Row*width+m*T_1+tx];
                Ns[ty][tx] = N[(m*T_1+ty)*width+Col];

                /* We now need to synchronize as all threads in a block have to finidh loading for the multiplication to continue */
                __syncthreads();

                /* Iterations to perform the partial matrix multiplication from the tile */
                for(int i=0;i<T_1;i++){
                        pSum += Ms[ty][i]*Ns[i][tx];
                }
                /* Another synchronization need to be performed so that all threads finish one phase and then start loading the next phase together */
                __syncthreads();
         }
         P[Row*width+Col] = pSum;	 
}

__global__ void matrixMulb(double *M,double *N,double *P,int width){
        /* Function to find the matrix multiplication using the tiling algorithm 
         * Shared memory has been used to avoid high latency global memory accesses
         * Parameters : M - matrix multiplicand on the device
         *              N - matrix multiplicand on the device
         *              P - output of the matrix multiplication M*N
         *              width - dimensions of the matrix(it is a square matrix)
         */
         int            bx = blockDim.x; /* Number of threads in the block in the x-direction */
         int            by = blockDim.y; /* Number of threads in the block in the y-direction */
         int            tx = threadIdx.x; /* x-Thread-ID in the block */
         int            ty = threadIdx.y; /* y-Thread-ID in the block */
         double          pSum = 0.0; /* Sum to store the partial matrix multiplication */

         int    Row = blockIdx.y*by+ty; /* Specific row of the cell in the output matrix */
         int    Col = blockIdx.x*bx+tx; /* Column of the cell in the output matrix */

         /* Decalring the shared memory variables */
         __shared__ double       Ms[T_2][T_2]; /* Shared memory allocation for a tile of matrix M */
         __shared__ double       Ns[T_2][T_2]; /* Shared memory allocation for a tile on matrix N */

         /* We will use a for loop to go through each phase */
         for(int m=0;m<(width+T_2-1)/T_2;m++){
                /* Loading Operations */
                Ms[ty][tx] = M[Row*width+m*T_2+tx];
                Ns[ty][tx] = N[(m*T_2+ty)*width+Col];

                /* We now need to synchronize as all threads in a block have to finidh loading for the multiplication to continue */
                __syncthreads();

                /* Iterations to perform the partial matrix multiplication from the tile */
                for(int i=0;i<T_2;i++){
                        pSum += Ms[ty][i]*Ns[i][tx];
                }
                /* Another synchronization need to be performed so that all threads finish one phase and then start loading the next phase together */
                __syncthreads();
         }
         P[Row*width+Col] = pSum;
}

__global__ void matrixMulc(double *M,double *N,double *P,int width){
        /* Function to find the matrix multiplication using the tiling algorithm 
         * Shared memory has been used to avoid high latency global memory accesses
         * Parameters : M - matrix multiplicand on the device
         *              N - matrix multiplicand on the device
         *              P - output of the matrix multiplication M*N
         *              width - dimensions of the matrix(it is a square matrix)
         */
         int            bx = blockDim.x; /* Number of threads in the block in the x-direction */
         int            by = blockDim.y; /* Number of threads in the block in the y-direction */
         int            tx = threadIdx.x; /* x-Thread-ID in the block */
         int            ty = threadIdx.y; /* y-Thread-ID in the block */
         double          pSum = 0.0; /* Sum to store the partial matrix multiplication */

         int    Row = blockIdx.y*by+ty; /* Specific row of the cell in the output matrix */
         int    Col = blockIdx.x*bx+tx; /* Column of the cell in the output matrix */

         /* Decalring the shared memory variables */
         __shared__ double       Ms[T_3][T_3]; /* Shared memory allocation for a tile of matrix M */
         __shared__ double       Ns[T_3][T_3]; /* Shared memory allocation for a tile on matrix N */

         /* We will use a for loop to go through each phase */
         for(int m=0;m<(width+T_3-1)/T_3;m++){
                /* Loading Operations */
                Ms[ty][tx] = M[Row*width+m*T_3+tx];
                Ns[ty][tx] = N[(m*T_3+ty)*width+Col];

                /* We now need to synchronize as all threads in a block have to finidh loading for the multiplication to continue */
                __syncthreads();

                /* Iterations to perform the partial matrix multiplication from the tile */
                for(int i=0;i<T_3;i++){
                        pSum += Ms[ty][i]*Ns[i][tx];
                }
                /* Another synchronization need to be performed so that all threads finish one phase and then start loading the next phase together */
                __syncthreads();
         }
         P[Row*width+Col] = pSum;
}

__global__ void matrixMuld(double *M,double *N,double *P,int width){
        /* Function to find the matrix multiplication using the tiling algorithm 
         * Shared memory has been used to avoid high latency global memory accesses
         * Parameters : M - matrix multiplicand on the device
         *              N - matrix multiplicand on the device
         *              P - output of the matrix multiplication M*N
         *              width - dimensions of the matrix(it is a square matrix)
         */
         int            bx = blockDim.x; /* Number of threads in the block in the x-direction */
         int            by = blockDim.y; /* Number of threads in the block in the y-direction */
         int            tx = threadIdx.x; /* x-Thread-ID in the block */
         int            ty = threadIdx.y; /* y-Thread-ID in the block */
         double          pSum = 0.0; /* Sum to store the partial matrix multiplication */

         int    Row = blockIdx.y*by+ty; /* Specific row of the cell in the output matrix */
         int    Col = blockIdx.x*bx+tx; /* Column of the cell in the output matrix */

         /* Decalring the shared memory variables */
         __shared__ double       Ms[T_4][T_4]; /* Shared memory allocation for a tile of matrix M */
         __shared__ double       Ns[T_4][T_4]; /* Shared memory allocation for a tile on matrix N */

         /* We will use a for loop to go through each phase */
         for(int m=0;m<(width+T_4-1)/T_4;m++){
                /* Loading Operations */
                Ms[ty][tx] = M[Row*width+m*T_4+tx];
                Ns[ty][tx] = N[(m*T_4+ty)*width+Col];

                /* We now need to synchronize as all threads in a block have to finidh loading for the multiplication to continue */
                __syncthreads();

                /* Iterations to perform the partial matrix multiplication from the tile */
                for(int i=0;i<T_4;i++){
                        pSum += Ms[ty][i]*Ns[i][tx];
                }
                /* Another synchronization need to be performed so that all threads finish one phase and then start loading the next phase together */
                __syncthreads();
         }
         P[Row*width+Col] = pSum;
}

int main(int argc,char **argv){
	if(argc!=1){
		printf("error : invalid number of arguments");
		exit(EXIT_FAILURE);
	}

	/********************************************** Variable Declaration *********************************/
        double           *h_M; /* Matrix multiplicand M on the host */
        double           *h_N; /* Matrix multiplicand N on the host */
        double           *h_P; /* Matrix product M*N on the host */
        double           *d_M; /* Matrix multiplicand M on the device */
        double           *d_N; /* Matrix multiplicand N on the device */
        double           *d_P; /* Matrix product M*N on the device */
        size_t          size; /* Size in bytes of the square matrices */
        cudaEvent_t     start,stop; /* CUDA events to measure the time of the matrix multiplication execution */
	cudaEvent_t	start_1,stop_1;
	cudaEvent_t	start_2,stop_2;
	cudaEvent_t	start_3,stop_3;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventCreate(&start_1);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&start_2);
	cudaEventCreate(&stop_2);
	cudaEventCreate(&start_3);
	cudaEventCreate(&stop_3);

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
	float	run_time = 0.0;

	dim3    threads_1(T_1,T_1);
	dim3    blocks_1((SIZE+T_1-1)/T_1,(SIZE+T_1-1)/T_1);

 	cudaEventRecord(start_1);
        matrixMula<<<blocks_1,threads_1>>>(d_M,d_N,d_P,SIZE);
	cudaEventRecord(stop_1);

	ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
	print_matrix_to_file(h_P,SIZE,SIZE);

 	dim3    threads_2(T_2,T_2);
        dim3    blocks_2((SIZE+T_2-1)/T_2,(SIZE+T_2-1)/T_2);

        cudaEventRecord(start_2);
        matrixMulb<<<blocks_2,threads_2>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_2);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

 	dim3    threads_3(T_3,T_3);
        dim3    blocks_3((SIZE+T_3-1)/T_3,(SIZE+T_3-1)/T_3);

        cudaEventRecord(start_3);
        matrixMulc<<<blocks_3,threads_3>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop_3);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

	dim3    threads_4(T_4,T_4);
        dim3    blocks_4((SIZE+T_4-1)/T_4,(SIZE+T_4-1)/T_4);

        cudaEventRecord(start);
        matrixMuld<<<blocks_4,threads_4>>>(d_M,d_N,d_P,SIZE);
        cudaEventRecord(stop);

        ERROR_HANDLER(cudaMemcpy(h_P,d_P,size,cudaMemcpyDeviceToHost),__LINE__);
        print_matrix_to_file(h_P,SIZE,SIZE);

        cudaEventSynchronize(stop);	

        cudaEventElapsedTime(&run_time,start_1,stop_1);
        printf("Run-Time(seconds)(4*4) : %.10f\n",run_time/1000);
	cudaEventElapsedTime(&run_time,start_2,stop_2);
        printf("Run-Time(seconds)(8*8) : %.10f\n",run_time/1000);
	cudaEventElapsedTime(&run_time,start_3,stop_3);
        printf("Run-Time(seconds)(16*16) : %.10f\n",run_time/1000);
	cudaEventElapsedTime(&run_time,start,stop);
        printf("Run-Time(seconds)(32*32) : %.10f\n",run_time/1000);

        /*********************************** Free the memory *********************************************/
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        free(h_P);
}
