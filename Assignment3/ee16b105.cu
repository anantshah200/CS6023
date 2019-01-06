/* Program : To find the histogram of N count words in a text file
 * Author : Anant Shah
 * Date : 17-9-2018
 * Roll Number : EE16B105
 **/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<ctype.h>

#define MAXWORDS 20000
#define MAX_WORD_LEN 20
#define ERROR_HANDLER(error_msg,line) error_handler(error_msg,line)
#define SHARED_BIN_SIZE 512
#define NUM_TASKS 8
#define NUM_THREADS 256

void error_handler(cudaError_t error_msg,int line){
	/* Function to detect an error in a cuda command */
	if( error_msg != cudaSuccess){
		printf("%s in %s at %d",cudaGetErrorString(error_msg),__FILE__,line);
		exit(EXIT_FAILURE);
	}
}

__host__ int str_len(char *str){
	/* Function to find the length of a string */
	int len=0;
	while( *(str+len) != '\0'){
		len++;
	}
	return len;
}

__host__ __device__ void initialize_bins(int *bin,unsigned size){
	/* Function to initialize the array of bins */
	for(int i=0;i<size;i++){
		bin[i] = 0;
	}
}

void checkWord(char *curWord,int *count,int *words_len){
        /* Function to modify the words so as to remove the punctuations */
        int     length = str_len(curWord); /* Length of the string */
        int     i=0;
        int     let_found=0; /* Letter found (used to test if apostrophe occurs in the middle of the word) */
        char    letter;
        int     size=0;
        char    *new_curWord;
        new_curWord = (char *)malloc(MAX_WORD_LEN*sizeof(char));

        while(i<length){
                letter = curWord[i];
                if(letter==45){
                        words_len[(*count)]=size;
                        (*count)++;
                        int k=i+1;
                        while(k<length){
                                new_curWord[k-i-1] = curWord[k];
                                k++;
                        }
                        checkWord(new_curWord,count,words_len);
                        return;
                }
                else if(isalnum(letter)==0){
                        if(let_found!=0){
                                /* When a punctuation exists but is not the last element */
                                char    let_inc;
                                let_inc = curWord[i+1];
                                if(isalnum(let_inc)!=0) size++;
                        }
                        i++;
                }
                else{
                        i++;
                        size++;
                        let_found=1;
                }
        }
        words_len[(*count)] = size;
}

__host__ int store_words(char *filename,int *words_len){
	/* Function to read all the words from a file and store them */
	FILE	*ipf = fopen(filename,"r");
	int	*totalWordCount;
	char	curWord[40]; /* Reading in the current word from the file */

	totalWordCount = (int *)malloc(sizeof(int));
	(*totalWordCount)=0;

	while(fscanf(ipf,"%s",curWord)!=EOF && *totalWordCount<MAXWORDS){
		checkWord(curWord,totalWordCount,words_len);	
		(*totalWordCount)++;
	}
	return *totalWordCount;
}

__host__ __device__ int power(int base,int exp){
	int power = 1;
	for(int i=0;i<exp;i++){
		power=power*base;
	}
	return power;
}

__global__ void n_count(int *words_len,int *bins,int N,int num_words){
	/* Kernel function to count the n-count grams
	 * Parameters : words_len - Array consisting of the lengths of the words
	 *		bins - Array consisting of the bins of the histogram
	 *		N - value of N in n-count 
	 *		start_bin - Start bin value to be subtracted to obtain the bin value
	 */
	
	extern __shared__ int	s_words_len[]; /* Storing the world lengths in the shared memory */
	__shared__ int	s_hist[SHARED_BIN_SIZE]; /* Pointer in the shared memory to the privatized histogram */
	__shared__ int	s_hist_ind; /* Pointer in the shared memory which maps to the bin indices in the global memory(starting index) */
	__shared__ int	ps_hist_ind; /* keep a track of the previous pointer which maps the global memory bins to the shared memory  */
	__shared__ int	s_hist_hit; /* Number of times the shared memory histogram is hit */
	int  	*global_ind = (int *)&s_words_len[NUM_TASKS*NUM_THREADS+N-1];
	int	bdim = blockDim.x; /* Number of threads in the block in the x-direction */
	int 	bx = blockIdx.x; /* Block ID  */
	int 	tid = threadIdx.x; /* Thread ID in a block */
	int	bin; /* Bin of the set of words read by a thread */

	/* Initialize the private histogram with 0's */
	for(int i=0;i<(SHARED_BIN_SIZE+bdim-1)/bdim;i++){
		if( (i*bdim+tid) < (SHARED_BIN_SIZE)){
			s_hist[i*bdim+tid] = 0;
			s_hist_ind = 0;
			s_hist_hit = 0;
		}
	}
	if(tid < power(20,N)/SHARED_BIN_SIZE)		global_ind[tid] = 0;
	__syncthreads(); 

	/* Load the words from the global memory into the shared memory */
	for(int i=0;i<=NUM_TASKS;i++){
		/* Only 1 warp will have control divergence */
		if( (bx*NUM_TASKS*bdim+i*bdim+tid) < num_words){
			if( (i*bdim+tid) < (NUM_TASKS*bdim+N-1))		s_words_len[i*bdim+tid] = words_len[bx*NUM_TASKS*bdim+i*bdim+tid];
		}
	}
	__syncthreads();

	/* Read the lengths of the corresponding N words */
	/* Keep a condition so that it dosent read invalid words */
	for(int i=0;i<NUM_TASKS;i++){
		if( (bx*NUM_TASKS*bdim+i*bdim+tid) <= (num_words-N)){
			bin = 0;
			for(int j=0;j<N;j++){
				bin+=(s_words_len[i*bdim+j+tid]-1)*power(20,N-1-j); 
			}
			/* Check which bin they lie in and add accordingly */
			if( (bin>=s_hist_ind) && (bin<(s_hist_ind+SHARED_BIN_SIZE)) ){
				atomicAdd(&(s_hist[bin-s_hist_ind]),1);
				atomicAdd(&s_hist_hit,1);
			}
			else {								
				atomicAdd(&(bins[bin]),1);
				atomicAdd(&global_ind[bin/SHARED_BIN_SIZE],1);
			}
		}
		__syncthreads();
		if(i==NUM_TASKS/4){
			if(tid==0){
				/* Only 1 thread is required to perform this computation */
				ps_hist_ind = s_hist_ind;
				for(int j=0;j<power(20,N)/SHARED_BIN_SIZE;j++){
					if(global_ind[j]>s_hist_hit)	s_hist_ind = SHARED_BIN_SIZE*j;
				}
				s_hist_hit = 0;
			}
			__syncthreads();

		/* Now I synchronously need to load the existing histogram into the global memory and move the other histogram into the shared memory */
			if(ps_hist_ind != s_hist_ind){
				for(int j=0;j<(SHARED_BIN_SIZE+bdim-1)/bdim;j++){
					if( (j*bdim+tid) < (SHARED_BIN_SIZE)){
						if(ps_hist_ind+j*bdim+tid<power(20,N))		atomicAdd(&(bins[ps_hist_ind+j*bdim+tid]),s_hist[j*bdim+tid]);
						s_hist[j*bdim+tid] = 0;
					}
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();

	/* Copy the privatized histogram back to the global memory */
	for(int i=0;i<(SHARED_BIN_SIZE+bdim-1)/bdim;i++){
		if( (i*bdim+tid) < (SHARED_BIN_SIZE)){
			if(s_hist_ind+i*bdim+tid<power(20,N))		atomicAdd(&(bins[s_hist_ind+i*bdim+tid]),s_hist[i*bdim+tid]);
		}
	}
}

void print_hist_to_file(int *hist,int N){
	FILE *fp;
	const char *file = "ee16b105_out.txt";
	fp = fopen(file,"w");

	for(int i=0;i<power(MAX_WORD_LEN,N);i++){
                int val = i;
		if(hist[i]!=0){
                	for(int j=0;j<N;j++){
                        	fprintf(fp,"%d ",(val/power(20,N-1-j))+1);
                        	val = val%power(20,N-1-j);
                	}
                	fprintf(fp,"%d\n",hist[i]);
		}
        }
	fclose(fp);
}

int main(int argc,char **argv){
	if(argc!=3){
		printf("error : Invalid number of arguments\n");
		exit(EXIT_FAILURE);
	}

	int 		N = atoi(argv[1]); /* N-grams to be found */
	char 		*filename = argv[2]; /* File which contains the text */

	int		*h_words_len; /* Store all the lengths of the words */
	int		*d_words_len; /* Store all the lengths of the words on the device */
	int		*h_bins; /* An array to store the bins. Each index will map to one bin. This will reside on the host */	
	int		*h_hist; /* Output histogram on the host */
	int		*d_bins; /* An array to store the bins on the device */
	size_t		size_bins; /* Size of the bin array */
	size_t		size_words_len; /* Size required to store all the words length */
	size_t		size_sharedmem; /* Size of the shared memory required by a block */
	int		total_count; /* Total number of words in the text file */

	/**************************** Allocating memory to the bins on the host ****************************/
	size_bins = sizeof(int)*power(MAX_WORD_LEN,N); /* There will be MAX_WORD_LEN^N combinations of N-count grams */
	size_words_len = sizeof(int)*MAXWORDS;

	h_bins = (int *)malloc(size_bins); /* Allocate memory to the bins on the device */
	h_words_len = (int *)malloc(size_words_len);
	h_hist = (int *)malloc(size_bins);	

	/********************************* Initialize the bins ********************************************/

	initialize_bins(h_bins,pow(MAX_WORD_LEN,N));
	total_count = store_words(filename,h_words_len);

	/********************************  Allocate memory on the device ***********************************/

	ERROR_HANDLER(cudaMalloc((void **)&d_words_len,size_words_len),__LINE__);
	ERROR_HANDLER(cudaMalloc((void **)&d_bins,size_bins),__LINE__);
	
	/******************************** Copy Data from the host to the device ****************************/

	ERROR_HANDLER(cudaMemcpy(d_bins,h_bins,size_bins,cudaMemcpyHostToDevice),__LINE__);
	ERROR_HANDLER(cudaMemcpy(d_words_len,h_words_len,size_words_len,cudaMemcpyHostToDevice),__LINE__);

	/******************************* Invoke the kernel to find the histogram ***************************/

	int	blocks = (total_count+NUM_THREADS*NUM_TASKS-1)/(NUM_THREADS*NUM_TASKS);
	size_sharedmem = sizeof(int)*(NUM_TASKS*NUM_THREADS+N-1)+sizeof(int)*(power(20,N)/SHARED_BIN_SIZE);	

	n_count<<<blocks,NUM_THREADS,size_sharedmem>>>(d_words_len,d_bins,N,total_count);

	ERROR_HANDLER(cudaMemcpy(h_hist,d_bins,size_bins,cudaMemcpyDeviceToHost),__LINE__);

	print_hist_to_file(h_hist,N);
	/******************************** Free the allocated memory ***************************************/

	cudaFree(d_bins);
	cudaFree(d_words_len);
	free(h_hist);
	free(h_bins);
	free(h_words_len);
}
