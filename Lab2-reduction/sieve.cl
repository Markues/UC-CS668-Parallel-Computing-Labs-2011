#define maxn %(max_n)d
#define hash %(hash)d

__local int proposed_storage(__local int value, __global int* primer __local int bound) {
    int tmp = value % hash;
    if (primer[tmp] == 0) return tmp;
    while (bound < 1000) {
        bound += 1;
        tmp = proposed_storage(value, primer, bound);
        if (tmp > 0) return tmp;
    }
    return -1;
}

__kernel void findsmallest(__global int* primer)
{ 
	int idx = get_group_id(0) * get_global_size(0) + get_local_id(0); 
	if (idx > 1) {
        int i = idx;
		for(i+=idx; i < maxn; i+=idx) {
            int bound = 0
			int tmp = proposed_storage(i, primer, bound)
            if (tmp < 0) continue;
            primer[tmp] = i;
        }
	}
}
