/*
Template code for convolution. CS6023, IITM */
#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define W 1024 // Input DIM
#define OW (W-4) // Output DIM
#define D 8   // Input and Kernel Depth
#define T 5  // Kernel DIM
#define N 128 // Number of kernels
#define BLOCK_DIM_Z 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_X 8

void fillMatrix(unsigned char *matrix){

unsigned char (*m)[W][D]=(unsigned char (*)[W][D])matrix;

for(int i=0;i<W;i++){
	for(int j=0;j<W;j++){
		for(int k=0;k<D;k++){
			m[i][j][k]=(i*j+j*k+i*k+i*2+j*3+k*4)%255;
				}
			}
		}
}



void fillKernel(float *kernel){

float (*t)[T][T][D]=(float (*)[T][T][D])kernel;

for(int i=0;i<N;i++){
	for(int j=0;j<T;j++){
		for(int k=0;k<T;k++){
			for(int l=0;l<D;l++){
			t[i][j][k][l]=fmod(-(i+1)*2.1+(j+1)*3.2-(k+1)*4.8+(l+1)*7.1,1.0);
				}
			}
		}
	}
}

void print_matrix_to_file(float *m){

	const char *fname = "assignment4_out";
	FILE *f = fopen(fname, "w");

	float (*mat)[OW][OW]=(float (*)[OW][OW])m;		

	for(unsigned i=0; i < N; i++) {
		for(unsigned j=0; j < OW; j++)
			for(unsigned k=0;k<OW;k++)
				fprintf(f,"%4.4f ", mat[i][j][k]);
		fprintf(f,"\n");
	}
	fclose(f);
}

__global__ void convolution_3d(unsigned char *matrix,float* kernel,float *output){

	__shared__ float	s_matrix[BLOCK_DIM_Z+(T-1)][BLOCK_DIM_Y+(T-1)][D]; /* Shared memory for the matrix values */	
	__shared__ float	s_conv[BLOCK_DIM_Z][BLOCK_DIM_Y][BLOCK_DIM_X];
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;
	int gid_x = blockDim.x*bx+tx;
	int gid_y = blockDim.y*by+ty;
	int gid_z = blockDim.z*bz+tz;
	
	int num_tasks = D/BLOCK_DIM_X;

	s_conv[tz][ty][tx] = 0;
	for(int k=0;k<(T-1)/2;k++){
		for(int i=0;i<(T-1)/2;i++){
			for(int j=0;j<num_tasks;j++){
				if( ((ty+i*BLOCK_DIM_Y) < (BLOCK_DIM_Y+T-1)) && (tz+k*BLOCK_DIM_Z < BLOCK_DIM_Z+T-1) ){
					s_matrix[tz+k*BLOCK_DIM_Z][ty+i*BLOCK_DIM_Y][tx+j*BLOCK_DIM_X] = matrix[(gid_z+k*BLOCK_DIM_Z)*W*D+(gid_y+i*BLOCK_DIM_Y)*D+tx+j*BLOCK_DIM_X];
				}
				
			}
		}
	}
	__syncthreads();

	/* Now perform the multiplication to find the convolution */
	if(gid_z<OW && gid_y<OW){
	
		for(int id=bx*BLOCK_DIM_X;id<(bx+1)*BLOCK_DIM_X;id++){
			if( id<N ){
				float conv = 0;
				for(int k=0;k<num_tasks;k++){
					for(int i=-(T-1)/2;i<=(T-1)/2;i++){
						for(int j=-(T-1)/2;j<=(T-1)/2;j++){
							conv += s_matrix[tz+i+(T-1)/2][ty+j+(T-1)/2][tx+BLOCK_DIM_X*k] * kernel[id*T*T*D+(i+(T-1)/2)*T*D+(j+(T-1)/2)*D+tx+BLOCK_DIM_X*k];
						}
					}
				}
				atomicAdd(&(s_conv[tz][ty][id-bx*BLOCK_DIM_X]),conv);
			}
		}
		output[gid_x*OW*OW+gid_z*OW+gid_y] = s_conv[tz][ty][tx];
	}
}

int main()
{

	unsigned char *matrix=(unsigned char*)malloc(sizeof(unsigned char)*W*W*D);
	float *kernel=(float*)malloc(sizeof(float)*T*T*D*N);
	float *output=(float *)malloc(sizeof(float)*N*OW*OW);


	fillMatrix(matrix);
	fillKernel(kernel);


	unsigned char *Dmatrix;cudaMalloc((void **)&Dmatrix,sizeof(unsigned char)*W*W*D);
	float *Dkernel;cudaMalloc((void **)&Dkernel,sizeof(float)*N*T*T*D);
	float *Doutput;cudaMalloc((void **)&Doutput,sizeof(float)*N*OW*OW);

	cudaMemcpy(Dmatrix, matrix, sizeof(unsigned char)*W*W*D,cudaMemcpyHostToDevice);
	cudaMemcpy(Dkernel, kernel, sizeof(float)*T*T*D*N,cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	//Make your cuda kernel call

	dim3 blockd(BLOCK_DIM_X,BLOCK_DIM_Y,BLOCK_DIM_Z);
	dim3 gridd((N+BLOCK_DIM_X-1)/BLOCK_DIM_X,(OW+BLOCK_DIM_Y-1)/(BLOCK_DIM_Y),(OW+BLOCK_DIM_Z-1)/BLOCK_DIM_Z);

	convolution_3d<<<gridd,blockd>>>(Dmatrix,Dkernel,Doutput);

	cudaDeviceSynchronize();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n",milliseconds);

	cudaMemcpy(output, Doutput, sizeof(float)*N*OW*OW,cudaMemcpyDeviceToHost);

	//Use print_matrix_to_file function only 
	print_matrix_to_file(output);

	cudaFree(Dmatrix);
	cudaFree(Dkernel);
	cudaFree(Doutput);
	free(matrix);
	free(kernel);
	free(output);
}
