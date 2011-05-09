#include <stdio.h>
#include <cstdlib>
#include <ctime>

#define N 4000

void printArray(int *a,int n)
{
	int i;		
	for(i=0;i<n;i++){
		printf("%d\t",a[i]);
	}
}
   
void radixSort(int *a,int n)
{
	int i,b[N],m=0,exp=1;
	for(i=0;i<n;i++){
		if(a[i]>m){
			m=a[i];
		}
	}
		
	while(m/exp>0){
		int bucket[10]={0};
		for(i=0;i<n;i++){
			bucket[a[i]/exp%10]++;
		}
		for(i=1;i<10;i++){
			bucket[i]+=bucket[i-1];
		}
		for(i=n-1;i>=0;i--){
			b[--bucket[a[i]/exp%10]]=a[i];
		}
		for(i=0;i<n;i++){
			a[i]=b[i];
		}
		exp*=10;
	}		
}


int main()
{
	int arr[N];

	srand((unsigned)time(0)); 
	for(int i=0; i<N; i++){ 
		arr[i] = (rand()%100)+1; 
	} 

	printf("\nORIGINAL ARRAY  : ");
    printArray(&arr[0],N);
           
    radixSort(&arr[0],N);
           
    printf("\nSORTED ARRAY: ");
    printArray(&arr[0],N);

    return 0;
}
