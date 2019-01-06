/* Program : To query the device properties of the Tesla K40c GPU
 * Author : Anant Shah
 * Roll Number : EE16B105
 * Date : 14-8-2018
 **/

#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
	
#define DEVICE_ID 0
#define ERROR_HANDLER(error_msg) error_handler(error_msg)

void error_handler(cudaError_t error_msg){
	/* Function to handle an error in CUDA; will be used as a macro */
	if(error_msg != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(error_msg),__FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
}

int main(int argc,char *argv[]){
	
	cudaDeviceProp*		device_properties; /* cudaDeviceProp is a struct whose variable fields consist data about the device */
	FILE			*fp; /* Pointer to a file to which the data is written */
	/* To query the specifications of each device, we run them through a for loop */
		device_properties = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp));
		ERROR_HANDLER(cudaGetDeviceProperties(device_properties,DEVICE_ID));

		fp = fopen("ee16b105_1_out.txt","w");
		if(fp!=NULL){
			fprintf(fp,"%d\n",(device_properties)->localL1CacheSupported);
			fprintf(fp,"%d \n",(device_properties)->globalL1CacheSupported);
			fprintf(fp,"%d \n",(device_properties)->l2CacheSize);
			fprintf(fp,"%d \n",(device_properties)->maxThreadsPerBlock);
			fprintf(fp,"%d \n",(device_properties)->regsPerBlock);
			fprintf(fp,"%d \n",(device_properties)->regsPerMultiprocessor);
			fprintf(fp,"%d \n",(device_properties)->warpSize);
			fprintf(fp,"%zu \n",device_properties->totalGlobalMem);

		}else{
			printf("Error : File not found");
			exit(EXIT_FAILURE);
		}
}	
