import cv2
import numpy as np

digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)
# in the digits.png image we have many numbers so we have to divide the image in cells to get each number
# we have 50 rows and 50 columns => 250 of each digits from 0 to 9 let s split them
rows = np.vsplit(digits, 50) # split vertically  the array to 50 rows
cells = []
for row in rows: # we want to split the rows to cells to get each digits from it
    row_cells = np.hsplit(row, 50) # split horizontally the row in 50 parts
    for cell in row_cells:
        cell = cell.flatten() #  we need to convert the images in one single array : The flatten() function is used to get a copy of a given array collapsed into one dimension
        cells.append(cell) #  to have all the numbers splitted 50 by 50
cells = np.array(cells, dtype=np.float32) #transform the cells list into a numpy array

k = np.arange(10) # simple 1D array from 0 to 9
cells_labels = np.repeat(k, 250) # we are gonna to repeat each one of this numbers in k 250 times and we ll move to the next one

# let s do the same for the test digits
test_digits = np.vsplit(test_digits, 50) # split vertically  to 50 rows
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells, dtype=np.float32) #transform the test_cells list into a numpy array


# KNN algorithm
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3) # the knn.findNearest() is returning many results so we ll give them a name; k = K value indicates the count of the nearest neighbors


print(result) # the only thing that we are intrested in from the knn.findNearest() function which are the result of each test cell from the test set image