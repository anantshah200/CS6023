/* Program : To add two randomly generated vectors with the help of a GPU
 * Author : Anant Shah
 * Roll Number : EE16B105
 * Date : 14-8-2018
 **/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<math.h>
#include<stdlib.h>

#define MAX_NUM 10000000 /* Maximum integer upto which random numbers are produced */
#define ERROR_HANDLER(error_msg) error_handler(error_msg)

void error_handler(cudaError_t error_msg){
	/* Handles error messages */
	if(error_msg != cudaSuccess){
		printf("%s in %s at line %d\n",cudaGetErrorString(error_msg),__FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
}

__global__ void vecAdd(int *A,int *B,double *C,int vector_size,int num_ops_per_thread) {
	
	/* One of the parameters of this kernel function is the number of operations per thread.
	 * This code follows the cyclic distribution of elements to each thread.
	 * eg : num_ops_per_thread = 2. 
	 * Then C[0] and C[vector_size/2] will be calculated by thread with Id = 0.
	 * C[1] and C[vector_size/2+1] will be calculated by thread with Id = 1 and so on.
	 */
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	
	/* Since we are performing the cyclic distribution, we need to make sure thread Ids grater than (vector_size/num_ops_per_thread) do not execute as the	       * y will only repeat the addition performed by another thread. Also, if the size of the vector and number of operations per thread are not perfectly 
	 * , then some threads will have to do some extra operations. 
	 */
	if(id<vector_size/num_ops_per_thread){
		for(int i=id;i<vector_size;i+=(vector_size/num_ops_per_thread)){
				C[i] = A[i] + B[i];
		}
	}
}

void readFileData(char *file,int size,int *vector){
	/* Function to read the integers from a .txt or .dat file */
	FILE	*fp;
	fp = fopen(file,"r+");
	if(fp!=NULL){
		for(int i=0;i<size;i++){
			fscanf(fp,"%d",(vector+i));
		}
		fclose(fp);
	}
	else{
		printf("error : File %s not found",file);
		exit(EXIT_FAILURE);
	}
}

int main(int argc,char **argv) {

	if(argc!=6){
		printf("Error : Invalid number of arguments \n");
		exit(EXIT_FAILURE);
	}

	/*************************** Variable Declaration ****************************/

	int		*h_A; /* Vector declared on the host */
	int		*h_B; /* Vector declared on the host */
	double		*h_C; /* Vector which will be the sum of previously deined vectors */
	int		*d_A; /* Vector which will be a copy of <h_A> but on the device(GPU) */
	int		*d_B; /* Vector which will be a copy of <h_B> but on the device(GPU) */
	double		*d_C; /* Vector which stores the sum of <d_A> and <d_B> on the device */
	size_t 		size_input; /* Bytes required to store the vectors */
	size_t		size_output; /* Bytes required to store the output vector */
	clock_t		start; /* Start time of the program */
	clock_t		stop; /* Stop time of the program */
	int		num_ops_per_thread; /* Number of operations per thread */
	int		num_blocks; /* Number of blocks in the grid */
	int		num_threads; /* Number of threads per block */	
	int		vector_size; /* Size of the vector(number of data elements) */
	char		*file_A; /* File which contains the data elements of the integer vector A */
	char		*file_B; /* File which contains the data elements of the integer vector B*/	

	num_threads = atoi(argv[1]);
	num_ops_per_thread = atoi(argv[2]);
	vector_size = atoi(argv[3]);
	file_A = argv[4];
	file_B = argv[5];	

	size_input = vector_size*sizeof(int);
	size_output = vector_size*sizeof(double);

	/******************************************************************************/

	/************************** Memory Allocation *********************************/	
	
	h_A = (int *)malloc(size_input); /* Allocate memory to the vector on the host  */
	h_B = (int *)malloc(size_input);	
	h_C = (double *)malloc(size_output);

	/************************** Store the integers from the input files ***************************/

	readFileData(file_A,vector_size,h_A);
	readFileData(file_B,vector_size,h_B);	

	/******************************************************************************/	

	ERROR_HANDLER(cudaMalloc((void **)&d_A,size_input));
	ERROR_HANDLER(cudaMalloc((void **)&d_B,size_input));
	ERROR_HANDLER(cudaMalloc((void **)&d_C,size_output));

	/************************* Copy vectors to the host ***************************/

	ERROR_HANDLER(cudaMemcpy(d_A, h_A, size_input, cudaMemcpyHostToDevice));
	ERROR_HANDLER(cudaMemcpy(d_B, h_B, size_input, cudaMemcpyHostToDevice));

	/************************ Add the vectors *************************************/
	num_blocks = (vector_size+num_threads*num_ops_per_thread-1)/(num_threads*num_ops_per_thread); 
	/* The ceiling function for the number of blocks is not applied as the number of threads is exactly equal to the number of data items to work on */
	
	start = clock();	

	vecAdd<<<num_blocks,num_threads>>>(d_A, d_B, d_C, vector_size, num_ops_per_thread);
	ERROR_HANDLER(cudaDeviceSynchronize());
	
	stop = clock();

	ERROR_HANDLER(cudaMemcpy(h_C, d_C, size_output, cudaMemcpyDeviceToHost)); /* Copy the result back to the host */

	printf("Time for the vector addition : %f (milli-seconds)  \n",1000*(stop-start)/(float)CLOCKS_PER_SEC);

	/************************* Free Memory ****************************************/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);
}
