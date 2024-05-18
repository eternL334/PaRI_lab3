import numpy as np
import cv2 as cv
from itertools import product

class DSU:
    def __init__(self, size):
        self.representative = [i for i in range(size)]
        self.size = [1] * size
        self.groups = size
    
    def find_representative(self, x):
        if x == self.representative[x]:
            return x
        self.representative[x] = self.find_representative(self.representative[x])
        return self.representative[x] 
    
    def union(self, x, y):
        x = self.find_representative(x)
        y = self.find_representative(y)
        if x == y:
            return

        if self.size[x] >= self.size[y]:
            self.size[x] += self.size[y]
            self.representative[y] = x
        else:
            self.size[y] += self.size[x]
            self.representative[x] = y
        
        self.groups -= 1



def segmentate(img, low, high):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, low, high)

    mask = cv.medianBlur(mask, 7)

    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # kernel = np.ones((13, 13), dtype=np.uint8)
    # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask

def remove_components(img, min_size):
    img = img.copy()
    n, labels, stats, centroids = cv.connectedComponentsWithStats(img)
    size_stat = stats[:, -1]
    small, = np.where(size_stat < min_size)
    img[np.isin(labels, small)] = 0
    return img

def get_vertices(skeleton):
    kernel = np.ones((3, 3), dtype=np.uint8)
    nearby = cv.filter2D(skeleton, -1, kernel)
    return np.nonzero((nearby == 2) & skeleton)

def get_graph(vertices, radius):
    dsu = DSU(len(vertices[0]))
    for (i, (x1, y1)), (j, (x2, y2)) in product(enumerate(zip(*vertices)), repeat=2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 < radius:
            dsu.union(i, j)
    
    sizes = []
    taken = [False] * len(vertices[0])
    for i in range(len(dsu.size)):
        representative = dsu.find_representative(i)
        if not taken[representative]:
            taken[representative] = True
            sizes.append(dsu.size[representative])

    return sizes

def squeeze_sizes(sizes):
    if len(sizes) == 0:
        return [0]
    squeezed = [0] * max(sizes)
    for size in sizes:
        squeezed[size - 1] += 1
    return squeezed

def mse(x, y):
    ans = 0
    for i in range(max(len(x), len(y))):
        if i < len(x) and i < len(y):
            ans += (x[i] - y[i]) ** 2
        elif i < len(x):
            ans += x[i]
        else:
            ans += y[i]
    
    return ans

def get_class(sizes):
    classes = [
        [3, 4, 3, 3],
        [4, 4, 4, 1],
        [4, 3, 5, 2, 1],
        [6, 3, 3, 2]
    ]
    idx = None
    min_score = None
    for i, class_ in enumerate(classes):
        score = mse(class_, sizes)
        if min_score is None or score < min_score:
            min_score = score
            idx = i
    
    return idx

