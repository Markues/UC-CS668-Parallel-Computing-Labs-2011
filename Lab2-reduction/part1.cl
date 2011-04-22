#define maxn %(max_n)d

__kernel void sieve(__global uint *a)
{ 
    int gid = get_group_id(0) * get_global_size(0) + get_local_id(0);
    
    // the sieve
    if (gid < 2) {
        return;
    }
    for (int i = gid + gid; i < maxn; i += gid){
        a[i] = 0;
    }
}

__kernel void pfilter(__global uint *prime_list, __global uint *a, __global uint offset) {
    int gid = get_group_id(0) * get_global_size(0) + get_local_id(0);
    
    //each work item filters out its assigned prime
    int prime = prime_list[gid];
    
    for (int i = offset % prime; i < maxn; i = i + prime;) {
        a[i] = 0;
    }
    
}