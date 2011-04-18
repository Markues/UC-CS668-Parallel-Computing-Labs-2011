#define maxn %(max_n)d

__kernel void part1(__global uint *a)
{ 
    int gid = get_group_id(0) * get_global_size(0) + get_local_id(0);
    if (gid < 2) {
        return;
    }
    int i = gid * 2;
    while (i < maxn) {
        a[i] = 0;
        i += gid;
    }
}
