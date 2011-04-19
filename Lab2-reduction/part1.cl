#define maxn %(max_n)d

__kernel void part1(__global uint4 *a)
{ 
    int gid = get_group_id(0) * get_global_size(0) + get_local_id(0);
    if (gid < 2) {
        return;
    }
    int i = gid * 2;
    while (i < maxn) {
        int sub_block = i % 4;
        int my_block = i/4;
        if (sub_block == 0) {
            a[my_block].w = 0;
        }
        if (sub_block == 1) {
            a[my_block].x = 0;
        }
        if (sub_block == 2) {
            a[my_block].y = 0;
        }
        if (sub_block == 3) {
            a[my_block].z = 0;
        }
        i += gid;
    }
}
