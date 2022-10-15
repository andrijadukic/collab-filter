# collab-filtering

This repository contains a python implementation of [*collaborative
filtering*](https://en.wikipedia.org/wiki/Collaborative_filtering).

Although the implementation is meant for education purposes, since it heavily utilizes *numpy* and entirely avoids doing
computation in python, it isn't terribly slow
either.

### Example

The [testdata](testdata) folder contains two files:

- [R.in](testdata/R.in), which contains input for the [main.py](main.py) test program
- [R.out](testdata/R.out), which contains the expected output

The [R.in](testdata/R.in) is constructed as follows:

- single line containing two integers, representing the (*n* x *m*) dimension of the user-item matrix
- *n* following lines, containing the matrix
- single line containing a single integer, *k*, representing the number of queries
- *k* following lines, each containing a single query. Each query consists of 4 integers:
    - The first two are the row and column indices of the query target
    - The third integer is the mode of the query (user-user or item-item), represented by 0 and 1, respectively
    - The last integer is the *sensitivity* of the query (how many nearest neighbours are taken into account)