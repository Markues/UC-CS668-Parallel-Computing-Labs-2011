#define maxn 100
#define hash 10

__local int proposed_storage(__local int value, __global int* primer) {
    int tmp = value % hash;
    if (primer[tmp] == 0) return tmp;
    int bound = 1;
    while (bound < hash) {
        if (primer[tmp + bound] == 0) return tmp + bound;
        bound += 1;
    }
    return -1;
}

__kernel void findsmallest(__global int* primer)
{ 
	int idx = get_group_id(0) * get_global_size(0) + get_local_id(0); 
	if (idx > 1) {
        int i = idx;
		for(i+=idx; i < maxn; i+=idx) {
            int bound = 0;
			int tmp = proposed_storage(i, primer);
            if (tmp < 0) continue;
            primer[tmp] = i;
        }
	}
}
