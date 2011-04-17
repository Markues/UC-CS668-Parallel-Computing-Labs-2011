#define maxn %(max_n)d

__kernel void part1(__global unsigned int* a)
{ 
    unsigned int f = maxn;
    unsigned int idx = get_group_id(0) * get_global_size(0) + get_local_id(0); 
    if (idx > 1) {
        for(unsigned int i=idx+idx;i < f;i+=idx) 
            a[i] = 0;
    }
}
