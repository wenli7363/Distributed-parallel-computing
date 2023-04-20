#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "error.h"

using namespace std;

typedef long long LL;

#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id) + 1) - BLOCK_LOW(id))
#define BLCOK_OWNER(index, p, n) (((p)* (index) +1 ) -1 / (n))
#define MIN(a, b) ((a)<(b)?(a):(b))

int main(int argc, char *argv[]) {
    LL count;              // 单进程count
    double elapsed_time;   //并行时间
    LL first;              // 某个chunk中的第一个被筛素数
    LL global_count;       // 总的count
    LL high_value;         // 某个chunk中的最高值
    int id;                // 第id个进程
    LL index;              // 索引
    LL low_value;          // 某个chunk的最低值
    char *marked;          // 埃氏筛标记数组
    LL n;                  // 程序的输入n
    int p;                  // 进程总数
    LL proc0_size;         // 0号进程分到的数的数量
    LL prime;              // 当前素数
    LL size;               // marked数组大小
    LL low_index;          //该进程最小index
    LL high_index;         //该进程最大index

    // 初始化
    MPI_Init(&argc, &argv);

    // MPI_COMM_RANK 得到本进程的进程号，进程号取值范围为 0, …, np-1
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // MPI_COMM_SIZE 得到所有参加运算的进程的个数
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // MPI_Barrier是MPI中的一个函数接口
    // 表示阻止调用直到communicator中所有进程完成调用
    MPI_Barrier(MPI_COMM_WORLD);
   // 计时
    elapsed_time = -MPI_Wtime();

    // 参数个数为2：文件名以及问题规模n
    if (argc != 2) {
        if (!id) printf("Command line: %s <m> \n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    // 输入
    n = atoll(argv[1]);

    LL N = n - 1;
    low_index = id * (N / p) + MIN(id, N % p); // 进程的第一个数的索引
    high_index = (id + 1) * (N / p) + MIN(id + 1, N % p) - 1; // 进程的最后一个数的索引
    low_value = 2 + low_index; //进程的第一个数
    high_value = 2 + high_index;//进程的最后一个数
    size = high_value - low_value + 1;    //进程处理的数组大小


    // Bail out if all the primes used for sieving are not all held by process 0
    proc0_size = (n - 1) / p;

    // 如果有太多进程
    if ((2 + proc0_size) < (LL) sqrt((double) n)) {
        if (!id) printf("Too many processes \n");
        MPI_Finalize();
        exit(1);
    }

    // allocate this process 's share of the array
    marked = (char *) malloc(size);
    if (marked == NULL) {
        printf("Cannot allocate enough memory \n");
        MPI_Finalize();
        exit(1);
    }

    // 先假定所有的整数都是素数
    for (LL i = 0; i < size; i++) marked[i] = 0;

    // 索引初始化为0
    if (!id) index = 0;

    // 从2开始搜寻
    prime = 2;
    do {
        /*确定该进程中素数的第一个倍数的下标 */
        // 如果该素数n*n>low_value，n*(n-i)都被标记了
        // 即n*n为该进程中的第一个素数
        // 其下标为n*n-low_value
        if (prime * prime > low_value) {
            first = prime * prime - low_value;
        } else {
            // 若最小值low_value为该素数的倍数
            // 则第一个倍数为low_value，即其下标为0
            if (!(low_value % prime)) first = 0;
                // 若最小值low_value不是该素数的倍数
                // 那么第一个倍数的下标为该素数减去余数的值
            else first = prime - (low_value % prime);
        }

        // 从第一个素数开始，标记该素数的倍数为非素数
        for (LL i = first; i < size; i += prime) marked[i] = 1;

        // 只有id=0的进程才调用，用于找到下一素数的位置
        if (!id) {
            while (marked[++index]); // 先自加再执行
            prime = index + 2; // 起始加偏移
        }

        // 将下一个素数广播出去
        if (p > 1) {
            MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

    } while (prime * prime <= n);

    // 将标记结果发给0号进程
    count = 0;
    for (LL i = 0; i < size; i++)
        if (marked[i] == 0) {
            count++;
        }
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // stop the timer
    elapsed_time += MPI_Wtime();

    // print the results
    if (!id) {
        printf("There are %lld primes less than or equal to %lld\n", global_count + 1, n);
        printf("SIEVE (%d) %10.6f\n", p, elapsed_time);
    }
    MPI_Finalize();
    return 0;
}

