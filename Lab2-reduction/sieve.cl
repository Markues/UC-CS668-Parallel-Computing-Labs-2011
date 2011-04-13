#define max %(max_n)d
#define BLOCK_SIZE %(block_size)d

__kernel void findsmallest(__global int* primer)
{ 
	int f = max;
	int idx = get_group_id(0) * get_global_size(0) + get_local_id(0); 
	if (idx > 1) {
		for(int i=idx+idx;i < f;i+=idx) 
			primer[i] = 1;
	}
}
