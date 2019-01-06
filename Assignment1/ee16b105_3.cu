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

#define SIZE 32768
#define NUM_THREADS 256
#define NUM_BLOCKS 128
#define ERROR_HANDLER(error_msg) error_handler(error_msg)

void error_handler(cudaError_t error_msg){
	/* Handles error messages */
	if(error_msg != cudaSuccess){
		printf("%s in %s at line %d\n",cudaGetErrorString(error_msg),__FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
}

__global__ void vecAdd(int *A,int *B,double *C,int length) {
	
	/* Kernel definition. Each thread will implement addition for one data item in the vectors. */
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i<length){
		C[i] = A[i] + B[i];
	}
}

void readFileData(char *file,int size,int *vector){
        /* Function to read the integers from a .txt or .dat file */
        FILE    *fp;
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

	if(argc!=3){
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
	size_t 		size_input = SIZE*sizeof(int);
	size_t		size_output = SIZE*sizeof(double);
	clock_t		start; /* Start time of the program */
	clock_t		stop; /* Stop time of the program */
	char		*file_A; /* File containing the data elements of vector A */
	char		*file_B; /* File containing the data elements of vector B */
	char		*file_out = "ee16b105_3_out.txt";	
	FILE		*fp; /* File pointer which manages the output text file */	

	file_A = argv[1];
	file_B = argv[2];

	/******************************************************************************/

	/************************** Memory Allocation *********************************/	
	
	h_A = (int *)malloc(size_input); /* Allocate memory to the vector on the host  */
	h_B = (int *)malloc(size_input);	
	h_C = (double *)malloc(size_output);

	/************************** Read the vector elements from the file ***************************/

	readFileData(file_A,SIZE,h_A);
	readFileData(file_B,SIZE,h_B);

	/******************************************************************************/	

	ERROR_HANDLER(cudaMalloc((void **)&d_A,size_input));
	ERROR_HANDLER(cudaMalloc((void **)&d_B,size_input));
	ERROR_HANDLER(cudaMalloc((void **)&d_C,size_output));

	/************************* Copy vectors to the host ***************************/

	ERROR_HANDLER(cudaMemcpy(d_A, h_A, size_input, cudaMemcpyHostToDevice));
	ERROR_HANDLER(cudaMemcpy(d_B, h_B, size_input, cudaMemcpyHostToDevice));

	/************************ Add the vectors *************************************/

	/* The ceiling function for the number of blocks is not applied as the number of threads is exactly equal to the number of data items to work on */
	start = clock();	

	vecAdd<<<NUM_BLOCKS,NUM_THREADS>>>(d_A, d_B, d_C, SIZE);
	ERROR_HANDLER(cudaDeviceSynchronize());

	stop = clock();

	ERROR_HANDLER(cudaMemcpy(h_C, d_C, size_output, cudaMemcpyDeviceToHost)); /* Copy the result back to the host */

	printf("Time for the vector addition : %f (seconds)  \n",(stop-start)/(float)CLOCKS_PER_SEC);
	/************************* Print the Result ***********************************/
	
	fp = fopen(file_out,"w");
	if(fp!=NULL){
		for(int i=0;i<SIZE;i++){
			fprintf(fp,"%d \n",h_A[i]);
			fprintf(fp,"%d \n",h_B[i]);
			fprintf(fp,"%.4f \n",h_C[i]);
		}
	}else{
		printf("error : Could not write to file");
		exit(EXIT_FAILURE);
	}
	fclose(fp);
	/************************* Free Memory ****************************************/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);
}
