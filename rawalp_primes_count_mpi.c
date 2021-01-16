/*Pranav Rawal
McMaster University - Electrical and Computer Engineering */

//##################################################################

/* Objective is to convert a serial code to count the number of prime
   numbers within a given interval into MPI (32-bit, int version).

   All prime numbers can be expressed as 6*k-1 or 6*k+1, k being an
   integer. The range of k to probe is provided as macro parameters
   KMIN and KMAX (see below).

   Check the parallel code correctness - it should produce the same number of prime
   numbers as the serial version, for the same range KMIN...KMAX. (The result
   is 3,562,113 for K=1...10,000,000.)

MPI Implementation specifications:

* No assumptions about the KMAX-KMIN+1 being integer dividable by the number of ranks should be made.
  In other words, the final solution should give the correct result and be efficient with any number 
  of ranks.
* Reduce the number of communications to a minimum.
* It should be the master rank's (rank 0) responsibility to call gettimeofday() function, and print
  all the messages.
* Place the first timer right after MPI_Init/Comm_size/Comm_rank functions; the second timer
  should go right before master rank prints the final results.

Dynamic Workload Balancing Specifications:
* Master rank must take part in the prime number search 
* Introduce a chunk parameter dK (can be a macro parameter: "#define dK ..."). Find the dK value
  resulting in best performance (for a given number of ranks - 32 on graham).
* Master rank not only distributes the workload to slaves, chunk by chunk,
  on a "first come - first served" basis, and collects the results, but also takes part in
  counting the number of primes itself. The mode of operation for the master could be:
  (a) check if there was a request from a slave for the next chunk, if yes - go to (c), if no - go to (b)
  (b) process a single K prime candidate (or a small number of candidates) itself, then go to (a)
  (c) send the next chunk to the slave, and go to (a).
  
Compiling instructions:

 - MPI Compilation:
  mpicc -O2 rawalp_primes_count_mpi.c -o rawalp_primes_count_mpi
  
 - MPI execution:
  mpirun -np 32 ./rawalp_primes_count_mpi

*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>


// Range of k-numbers for primes search:
#define KMIN 1
// Should be smaller than 357,913,941 (because we are using signed int)
#define KMAX 10000000
//Large value constant for MPI_Test polling
#define Nlarge 100000000

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, success;
  int xmax, ymax, x, y, k, j, count;

  int dK = 200; /*chunksize; default value is 200 
				(experimentally determined to be the most optimal 
				for the specified KMIN and KMAX range above)*/
				
  int ind[2] = {0,0}; //index defining the work load,
					  // ind[0] --> start 
					  // ind[1] --> end
					  
  int flag, flag2;
  int my_rank, p, i;
  int local_count, global_count, root_count;
  MPI_Status status;
  MPI_Request req;
  
  global_count = 0;
  local_count = 0;
  flag = 1;
  
  // option to enter custom int chunksize at the command line 
  if (argc == 2) dK = atoi(argv[1]);
  
  ///////// Starting Parallel Region /////////
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  //starting stop watch
  gettimeofday (&tdr0, NULL);
  
  
  if (my_rank == 0) { //root rank operations
  
  printf("\n\nchunksize: %d\n", dK); //printing the chunksize for dynamic workload balancing
	
  ind[0] = KMIN;				//intializing the start index of the first work load
	ind[1] = ind[0] + dK - 1; 	//intializing the end index of the first work load
	
	
	//////// STAGE 1: Initiation ////////
	/*
	Send the starting signal to all the worker ranks in the form of the portion of work
	they must do to contribute to the global_count of prime numbers. The workload is 
	represented as start (ind[0]) and end (ind[1]) indices. 
	*/
	for (i = 1; i < p; i++){	//Send the starting signal to all worker ranks 
								//with the dK sized portion of work they must perform 
								//ranging from ind[0] to ind[1]
    
	MPI_Send(&ind[0], 2, MPI_INT, i, 0, MPI_COMM_WORLD); //sending the work load indices 
    
	ind[0] = ind[1] + 1; 	//Updating the start index for the next work load 
	ind[1] = ind[1] + dK;	//Updating the end index for the next work load
    
	//accounts for any remaining work loads that don't fit into the dK chunksizes
	if (flag && (ind[1] > KMAX)){flag = 0; ind[1] = KMAX;}	 
	} 
 
	//////// STAGE 2: Main Body ////////
	/*
	Any work not completed in stage 1 completed in the following while loop. 
	The root polls a nonblocking receive for any completed workload and the 
	associated local count of prime numbers which are then added to the global 
	count. Repeated until the work is complete. 
	*/
	while (ind[1] < KMAX+dK){ //offsetted to account for remainders
		
      //Non-Blocking Receive
	  MPI_Irecv(&local_count, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &req);
     
	 //Polling loop
    for(i = 0; i <= Nlarge; i++){
		
		//Check if received a new completed work order
      MPI_Test(&req, &flag2, &status);
      if(flag2) break; /* terminate polling loop and go down to message received*/
      
	  else{				/*if no completion message received then search for 
						prime numbers at the root rank, one number at a time*/
						
        root_count = 0; //initializing root count to 0
        k = ind[0];		//setting k to the value of the start index and testing it
		
        // testing "-1" and "+1" cases:
        for (j=-1; j<2; j=j+2)
	      {
	        // Prime candidate:
	        x = 6*k + j;
	        // We should be dividing by numbers up to sqrt(x):
	        ymax = (int)ceil(sqrt((double)x));

	        // Primality test:
	        for (y=3; y<=ymax; y=y+2)
	          {
	          // Tpo be a success, the modulus should not be equal to zero:
	          success = x % y;
	          if (!success)break;
	          }
	        if (success) root_count++;
	      }
		  
        ind[0]++; //incrementing the start index
        ind[1]++; //incrementing the end index
        global_count = global_count + root_count;//updating global count
        }
    }
    
	//If completion message received then add the local count of the completed order to the global count
    global_count = global_count + local_count;
	
      if (ind[1] <= KMAX){ // if the end index is less than or equal to the KMAX
        MPI_Send(&ind[0], 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }
	  else {	// if there are remaining work loads smaller than dK chunksize
	    ind[1] = KMAX;
	    MPI_Send(&ind[0], 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD); 
	  }
	  
	  //updating the start and end indices for the next work order
 	  ind[0] = ind[1] + 1;
	  ind[1] = ind[1] + dK;
	}
		
	//////// STAGE 3: Termination ////////
	/*
	At the end of stage 2 the worker rank took the last work order to complete KMIN to KMAX orders.
	In this stage all the local_counts from the pending work orders must be accounted for from the 
	p-1 worker ranks. After that worker ranks will be sent a terminating end index to complete the 
	function of the worker ranks.
	*/		
	  for (i = p; i > 1; i--){
	  //Receive a completion message with an updated local_count 	   
      MPI_Recv(&local_count, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status); 
	  //update global count
	  global_count = global_count + local_count;
      //send terminating end index
	  MPI_Send(&ind[0], 2, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
	  }
	
	//stopping stopwatch 
	
    gettimeofday (&tdr1, NULL);
    tdr = tdr0;
    timeval_subtract (&restime, &tdr1, &tdr);
    printf ("N_primes: %d\n", global_count);
    printf ("time: %e\n", restime);
    
  }

  if (my_rank != 0) { //worker rank operations
    
    while (ind[1] <= KMAX) { //workers start out in this loop from the beginning
      if (flag) // initially the flag is set to receive the start and end indices of the initial work order
        {MPI_Recv(&ind[0], 2, MPI_INT, 0, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
		 flag = 0;/*flag is unset*/}

      local_count = 0; //reset the local_count variable

      if (ind[1] <= KMAX){ //make sure that the indices received from root are in the valid range
							//and not the terminating end index
        for (k=ind[0]; k<=ind[1]; k++) //compute the prime number count for the work order
          {
          // testing "-1" and "+1" cases:
          for (j=-1; j<2; j=j+2)
	        {
	        // Prime candidate:
	        x = 6*k + j;
	        // We should be dividing by numbers up to sqrt(x):
	        ymax = (int)ceil(sqrt((double)x));
	    
	        // Primality test:
	        for (y=3; y<=ymax; y=y+2)
	          {
	            // Tpo be a success, the modulus should not be equal to zero:
	            success = x % y;
	            if (!success)
		        break;
	          }
	        if (success)
	          {
	            local_count++;
	          }
	        }
          }
        }
		//Send the updated local prime number count variable to the root as a completion message.
      MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	  //Worker rank will wait until it receives indices for a new work order and go back 
	  //to the start of the while loop
      MPI_Recv(&ind[0], 2, MPI_INT, 0, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
	}

  }
	///End of Parallel Region
  MPI_Finalize();

  return 0;

}
