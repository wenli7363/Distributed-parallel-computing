// 偶数优化
#include "mpi.h"
#include <math.h>
#include <stdio.h>

typedef long long LL;

#define MIN(a, b) ((a)<(b) ? (a) : (b))
#define BLOCK_LOW(id, p, n) ((long long)(id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index, p, n) ((((p)*(index)+1)-1)/(n))
#define VALUE_TO_INDEX(value) ((value)-2)
#define INDEX_TO_VALUE(index) ((index)+2)

#define ODD_TO_INDEX(value) (((value)-3)/2) //奇数值转索引值,第一个奇数是3
#define INDEX_TO_ODD(index) (2*(index)+3)   //索引值转奇数值

int main(int argc, char *argv[]) {
    int count;              // 单进程count
    double elapsed_time;    // 并行时间
    LL first;               // 某个chunk中的第一个被筛素数
    int global_count;       // 总的count
    LL high_value;          // 某个chunk中的最高值
    LL i;
    int id;                 // 第id个进程
    LL index;               // 索引   
    LL low_value;           // 某个chunk的最低值
    char *marked;           // 埃氏筛标记数组
    LL n;                   // 程序的输入n
    int p;                  // 进程总数
    LL proc0_size;          // 0号进程分到的数的数量
    LL prime;               // 当前素数
    LL size;                // marked数组大小
    LL m;                   // 奇数个数

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
    m = ODD_TO_INDEX(n) + 1; // odds in 3..n   奇数个数

    /* Figure out this process's share of the array, as
       well as the integers represented by the first and
       last array elements */

    low_value = INDEX_TO_ODD(BLOCK_LOW(id, p, m));
    high_value = INDEX_TO_ODD(BLOCK_HIGH(id, p, m));
    size = BLOCK_SIZE(id, p, m);

    /* Bail out if all the primes used for sieving are
       not all held by process 0 */

    proc0_size = m / p;

    if (INDEX_TO_ODD(proc0_size - 1) < (int) sqrt((double) n)) {
        if (!id) printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    /* Allocate this process's share of the array. */

    marked = (char *) malloc(size);

    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < size; i++) marked[i] = 0;

    if (!id) index = 0;
    prime = 3;      //从3开始筛，只考虑奇数
    do {
        if (prime * prime > low_value)      //如果区间包含prime*prime
            //从prime*prime开始，将值映射到下标
            first = ODD_TO_INDEX(prime * prime) - ODD_TO_INDEX(low_value);
        else {
            if (!(low_value % prime)) first = 0;        // 如果最小值能够整除prime，说明就是prime对应的合数
            else {
                first = prime - (low_value % prime);
                if (!((low_value + first) & 1)) // 如果是偶数，就要选择下一个奇数
                    first += prime;
                first >>= 1;        //divide by 2
            }
        }

        for (i = first; i < size; i += prime) marked[i] = 1;    //快速标记

        // 0号进程查找下一个素数
        if (!id) {
            while (marked[++index]);
            prime = INDEX_TO_ODD(index);
        }
        // 广播这个prime
        if (p > 1) MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (prime * prime <= n);

    //统计count
    count = 0;
    for (i = 0; i < size; i++) {
        if (!marked[i])
            count++;
    }
    // 规约
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Stop the timer */

    elapsed_time += MPI_Wtime();

    /* Print the results */

    if (!id) {
        printf("There are %d primes less than or equal to %lld\n", global_count + 1, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
