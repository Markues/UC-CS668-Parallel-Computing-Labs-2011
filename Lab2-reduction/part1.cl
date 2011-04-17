#define maxn %(max_n)d

__kernel void part1(__global int *a)
{ 
    uint f = maxn;
    uint idx = get_group_id(0) * get_global_size(0) + get_local_id(0); 
    if (idx > 1) {
        for(uint i=idx+idx;i < f;i+=idx) 
            a[i] = get_group_id(0);
    }
}
