//优化2，消除通信
#include "mpi.h"
#include <math.h>
#include <stdio.h>

typedef long long LL;
#define MIN(a, b) ((a)<(b) ? (a) : (b))
#define BLOCK_LOW(id, p, n) ((long long)(id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(index, p, n) ((((p)*(index)+1)-1)/(n))
#define VALUE_TO_INDEX(value) (((value)-2))
#define INDEX_TO_VALUE(index) ((index)+2)

// 优化数值到对应marked数组的索引映射
#define ODD_TO_INDEX(value) (((value)-3)/2)     //某个奇数-3/2就是他在marked数组的下标
#define INDEX_TO_ODD(index) (2*(index)+3)       // mark数组中下标为i的数组为

int main(int argc, char *argv[]) {
    int count;                  // 单进程count
    double elapsed_time;        // 并行时间
    LL first;                   // 第一个被筛素数
    LL global_count;            // 总的count
    LL high_value;              // 该进程分配到的最高值
    LL i;
    int id;                     // 第id个进程
    LL index;                   // 索引
    LL low_value;               // 该进程分配中的最低值
    char *marked;               // 埃氏筛标记数组
    LL n;                       // 程序的输入n
    int p;                      // 进程总数
    LL proc0_size;              // 0号进程分到的数的数量
    LL prime;                   // 当前素数
    LL size;                    // marked数组大小
    LL m;                       // 奇数个数
    char *primes;               // 预处理得到的标记数组（3-sqrt(n)）
    LL primes_size;             // primes数组元素个数有限

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
    m = ODD_TO_INDEX(n) + 1;

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

    // =======================================================  分配空间，并初始化
    marked = (char *) malloc(size);
    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < size; i++) marked[i] = 0;

    primes_size = ODD_TO_INDEX(sqrt(n)) + 1;
    primes = (char *) malloc(primes_size);
    if (primes == NULL) {
        printf("Cannot allocate enough memory\n");
        free(marked);
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i < primes_size; i++) primes[i] = 0;
    //===================================================================

    // ======================================================   预处理素数 3..sqrt(n) 
    index = 0;
    prime = 3;
    do {
        for (i = ODD_TO_INDEX(prime * prime); i < primes_size; i += prime)
            primes[i] = 1;
        while (primes[++index]);
        prime = INDEX_TO_ODD(index);
    } while (prime * prime <= sqrt(n));

    // ============================================================== 算法主体，埃式筛

    index = 0;
    prime = 3;
    do {
        if (prime * prime > low_value)
            first = ODD_TO_INDEX(prime * prime) - ODD_TO_INDEX(low_value);
        else {
            if (!(low_value % prime)) first = 0;
            else {
                first = prime - (low_value % prime);
                if (!((low_value + first) & 1))    //如果是偶数，就找下一个奇数
                    first += prime;
                first >>= 1;
            }
        }
        // 从first开始，快速标记
        for (i = first; i < size; i += prime) marked[i] = 1;
        // 找到下一个素数prime
        while (primes[++index]);
        prime = INDEX_TO_ODD(index);
    } while (prime * prime <= n);

    // 统计素数个数
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
        printf("There are %lld primes less than or equal to %lld\n", global_count + 1, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}
