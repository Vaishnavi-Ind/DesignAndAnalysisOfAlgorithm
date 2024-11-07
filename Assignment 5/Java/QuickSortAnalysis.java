import java.util.Random;
public class QuickSortAnalysis {
 
 public static void deterministicQuickSort(int[] arr, int low, int high) {
 if (low < high) {
 int pi = partition(arr, low, high);
 deterministicQuickSort(arr, low, pi - 1);
 deterministicQuickSort(arr, pi + 1, high);
 }
 }
 
 public static void randomizedQuickSort(int[] arr, int low, int high) {
 if (low < high) {
 int pi = randomizedPartition(arr, low, high);
 randomizedQuickSort(arr, low, pi - 1);
 randomizedQuickSort(arr, pi + 1, high);
 }
 }
 
 private static int partition(int[] arr, int low, int high) {
 int pivot = arr[high]; 
 int i = low - 1; 
 for (int j = low; j < high; j++) {
 if (arr[j] < pivot) {
 i++;
 swap(arr, i, j);
 }
 }
 swap(arr, i + 1, high);
 return i + 1;
 }
 
 private static int randomizedPartition(int[] arr, int low, int high) {
 Random rand = new Random();
 int randomIndex = low + rand.nextInt(high - low + 1);
 swap(arr, randomIndex, high); 
 return partition(arr, low, high);
 }
 
 private static void swap(int[] arr, int i, int j) {
 int temp = arr[i];
 arr[i] = arr[j];
 arr[j] = temp;
 }
 
 public static int[] generateRandomArray(int size, int range) {
 Random rand = new Random();
 int[] arr = new int[size];
 for (int i = 0; i < size; i++) {
 arr[i] = rand.nextInt(range);
 }
 return arr;
 }
 
 public static int[] copyArray(int[] arr) {
 int[] newArr = new int[arr.length];
 System.arraycopy(arr, 0, newArr, 0, arr.length);
 return newArr;
 }
 
 public static void main(String[] args) {
 int size = 10000; 
 int range = 10000; 
 
 int[] originalArray = generateRandomArray(size, range);
 
 int[] arr1 = copyArray(originalArray);
 long startTime = System.nanoTime();
 deterministicQuickSort(arr1, 0, arr1.length - 1);
 long endTime = System.nanoTime();
 System.out.println("Deterministic QuickSort time: " + (endTime - startTime) + " ns");
 
 int[] arr2 = copyArray(originalArray);
 startTime = System.nanoTime();
 randomizedQuickSort(arr2, 0, arr2.length - 1);
 endTime = System.nanoTime();
 System.out.println("Randomized QuickSort time: " + (endTime - startTime) + " ns");
 }
}