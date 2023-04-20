// 优化三+小循环欧拉筛
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef long long LL;

#define MIN(a, b) ((a)<(b) ? (a) : (b))
#define BLOCK_LOW(id, p, n) ((long long)(id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index, p, n) ((((p)*(index)+1)-1)/(n))
#define VALUE_TO_INDEX(value) (((value)-2))
#define INDEX_TO_VALUE(index) ((index)+2)
#define ODD_TO_INDEX(value) (((value)-3)/2)
#define INDEX_TO_ODD(index) (2*(index)+3)

int main(int argc, char *argv[]) {
    LL count;               // 单进程count
    double elapsed_time;     //并行时间
    LL first;               // 某个chunk中的第一个被筛素数
    LL global_count;        // 总的count
    LL high_value;          // 某个chunk中的最高值
    LL i, j;
    int id;                 // 第id个进程
    LL index;               // 索引
    LL low_value;           // 某个chunk的最低值
    char *marked;           // 埃氏筛标记数组
    LL n;                   // 程序的输入n
    int p;                  // 进程总数
    LL proc0_size;          // 0号进程分到的数的数量
    LL prime;               // 当前素数
    LL size;                // marked数组大小
    LL m;                    /* Size of search list 奇数大小*/
    int *sub_primes;        // 存用来筛别的数的 质数数组
    LL sub_primes_size;         // 质数数组大小，小循环中的
    LL chunk = 2.2*(1<<20);               /* chunk size for *marked to adapt cache size */
    LL low_value_chunk; /* Lowest value in a chunk */
    // bool *is_prime;     //欧拉筛时的bool数组
    unsigned char* is_prime;

    MPI_Init(&argc, &argv);

    /* Start the timer */

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    if (argc != 2) {
        if (!id) printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    n = atoll(argv[1]);         
    m = ODD_TO_INDEX(n) + 1;    //奇数的数量

    low_value = INDEX_TO_ODD(BLOCK_LOW(id, p, m));
    high_value = INDEX_TO_ODD(BLOCK_HIGH(id, p, m));
    size = BLOCK_SIZE(id, p, m);

    proc0_size = m / p;

    if (INDEX_TO_ODD(proc0_size - 1) < (int) sqrt((double) n)) {
        if (!id) printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    /* Allocate this process's share of the array. */

    marked = (char *) calloc(size,1);
    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

// ============================================================== 小循环预处理primes，用欧拉筛

    sub_primes_size = ODD_TO_INDEX(sqrt(n)) + 1;      // 用于筛的数中 奇数的总数
    sub_primes = (int *) calloc(sub_primes_size,sizeof(int));     //存prime
    is_prime = (unsigned char*) calloc(sub_primes_size,1);      //欧拉筛的标记数组

    if (sub_primes == NULL) {
        printf("Cannot allocate enough memory\n");
        // free(marked);
        MPI_Finalize();
        exit(1);
    }

    // 欧拉筛，最后将所有素数存在sub_primes中，后面大循环可以直接查到
    int cnt = 0;
    for(int i = 3; ODD_TO_INDEX(i) < sub_primes_size; i +=2)      //从3开始找，每次+2，只找奇数
    {
        if(is_prime[ODD_TO_INDEX(i)] == 0) sub_primes[cnt++] = i;
        for(int j = 0; ODD_TO_INDEX(sub_primes[j] * i) < sub_primes_size; j++)
        {
            LL t = sub_primes[j] * i;
            is_prime[ODD_TO_INDEX(t)] = 1;
            if(i % sub_primes[j] == 0) break;
        }
    }

//  ======================================================================== 分块，算法主体，埃式筛 

    for(i = 0; i < size; i += chunk){   // chunking，i表示第i个chunk
        index = 0;
        prime = 3;
        // 每个chunk中的第一个值
        low_value_chunk = INDEX_TO_ODD(ODD_TO_INDEX(low_value) + i);
        
        do {
            //first 表示一个chunk里面第一个被标记的素数
            if (prime * prime > low_value_chunk)
                first = ODD_TO_INDEX(prime * prime) - ODD_TO_INDEX(low_value_chunk);
            else {
                if (!(low_value_chunk % prime)) first = 0;
                else {
                    first = prime - (low_value_chunk % prime);
                    if (!((low_value_chunk + first) & 1))
                        first += prime;
                    first >>= 1;
                }
            }
    
            // 快速标记 in first+i..min(first+i+chunk-1, size-1)
            for (j = first + i; j < first + i + chunk && j < size; j += prime)  
                marked[j] = 1;
            if(++index >= cnt){
                break;
            }else{
                prime = sub_primes[index];
            }
        } while (prime * prime <= n);
    }

    count = 0;
    for (i = 0; i < size; i++) {
        if (!marked[i])
            count++;
    }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Stop the timer */

    elapsed_time += MPI_Wtime();

    /* Print the results */

    if (!id) {
        printf("There are %lld primes less than or equal to %lld\n", global_count + 1, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
