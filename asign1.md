# assignment 1
- Write up a short report (max 2 pages) with your results, plots, and a description of your work. For each algorithm, we look for answers to the following questions:
  - How does the algorithm work on a technical level and what kind of machine learning problems is it suited for?
  - What is its inductive bias, i.e., what assumptions does it make about the data in order to generalize?
  - What happens in the second dataset that makes it harder than the first and how does this problem relate to the algorithm’s inductive bias?
  - What modifications did you do to get around this problem?

For this assignment i have selected to implement K-means clustering and Decision tree.

To test the code, python unittest was used. These testfiles are called test.py and are located in the corresponding folder

## K Means algorithm
To begin with i quickly realized it would be much more efficient to maintain two different datastructures to store the datapoint-cluster assignment. I had a 1d list which stored the cluster assignment for each datapoint. Also i had a 2d list where the 1-d list at index i, coinained all the indexes for cluster i.

This is more efficient since the algorithm needs to quickly know every point a cluster contains, and which cluster a specific datapoint is assigned to. The algorithm needs to quickly return the 1d list.

I originally implemented 3 different types of initialisation methods. These were Frogy, first K and random-partition. However random-partition was occasionally putting the centroids unreasonably far away, so that it had no points assigned to it. Therefore it was removed.

I implemented different types of initialisation methods.
random partition
upscaling
