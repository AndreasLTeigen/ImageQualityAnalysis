import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from Utils import utils

#Values: Mean, standard deviation, 95-percentile, kurtosis


class FocusAnalysis:
    def __init__(self):
        self.loadFirstOrderDerivativeOperators()
        self.block_size = 10#60

    def brennerFilter(self, image):
        src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.filter2D(src, ddepth=-1, kernel=self.difference3x3h)
        return dst

    def getAdjacentPixelBasedcontrastMapping(self, src):
        diff_h_kernel = np.array( [ [0,0,0],
                                    [0,1,-1],
                                    [0,0,0] ])
        diff_v_kernel = np.array( [ [0,0,0],
                                    [0,1,0],
                                    [0,-1,0] ])
        diff_h = np.absolute(cv2.filter2D(src, ddepth=-1, kernel=diff_h_kernel))
        diff_v = np.absolute(cv2.filter2D(src, ddepth=-1, kernel=diff_v_kernel))
        diff = diff_h + diff_v
        #print(np.sum(diff))
        return diff

    def getDiffSumGrid(self, bw_src):
        diff = self.getAdjacentPixelBasedcontrastMapping(bw_src)
        local_diff_grid = utils.getBlockSumGrid(diff, self.block_size, self.block_size)
        return local_diff_grid
    
    def getSumGrid(self, bw_src):
        local_diff_grid = utils.getBlockSumGrid(bw_src, self.block_size, self.block_size)
        return local_diff_grid

    def addcontrastHeatmap(self, image):
        bw_src = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        local_diff_grid = self.getDiffSumGrid(bw_src)
        local_diff_grid = self.ignoreRegions(local_diff_grid, ignore_value=np.min(local_diff_grid))
        local_diff_grid_norm = utils.numpyMaxNormalize(local_diff_grid)
        local_diff_grid_norm_os = utils.overSampleBlockSumGrid(local_diff_grid_norm, bw_src.shape[0], bw_src.shape[1])
        contrast_heatmap = utils.addHeatMap(image, local_diff_grid_norm_os)
        return contrast_heatmap


    def getcontrastDataArray(self, frame):
        # Remove ignored blocks and restructure into 1d array
        bw_src = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grid = self.getDiffSumGrid(bw_src)
        valid_marked_grid = self.ignoreRegions(grid)
        valid_marked_array = valid_marked_grid.flatten()
        valid_array = valid_marked_array[valid_marked_array < 256*self.block_size*self.block_size]
        return valid_array

    def getLuminescenceDataArray(self, frame):
        bw_src = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grid = self.getSumGrid(bw_src)
        valid_marked_grid = self.ignoreRegions(grid)
        valid_marked_array = valid_marked_grid.flatten()
        valid_array = valid_marked_array[valid_marked_array < 256*self.block_size*self.block_size]
        return valid_array

    def get2dFourierTransform(self, frame, size=60, thresh=10):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        (h,w) = frame.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(frame)
        fftShift = np.fft.fftshift(fft)
        magnitude = 20* np.log(np.abs(fftShift))
        return magnitude
        


    def ignoreRegions(self, local_diff_grid, ignore_value=-1, block_size_ignore=None):
        # Ignoring regions of video with  overlay, hardcoded with scale 0.5 in mind
        if block_size_ignore == None:
            block_size_ignore = self.block_size

        #TODO: Make this work for block size = 40, this can possibly be done by using the rest (%) operator
        # Top border
        for i in range(int(100/block_size_ignore)):
            local_diff_grid[i,:] = ignore_value
        #Bottom border
        for i in range(int(80/block_size_ignore)):
            local_diff_grid[-i,:] = ignore_value

        # Occasionally appearing numbers down left side
        for i in range(int(-140/block_size_ignore), int(-120/block_size_ignore)):
            for j in range(int(60/block_size_ignore)):
                local_diff_grid[i, j] = ignore_value

        #Right side as it is taking the difference with a 0 border
        local_diff_grid[:,local_diff_grid.shape[1]-1] = ignore_value
        return local_diff_grid



    def loadFirstOrderDerivativeOperators(self):
        self.black3x3 = np.array( [ [0,0,0],
                                    [0,0,0],
                                    [0,0,0] ])
        self.difference3x3h = np.array( [   [0,0,0],
                                            [-1,0,1],
                                            [0,0,0] ])
        self.difference3x3v = np.array( [   [0,-1,0],
                                            [0,0,0],
                                            [0,1,0] ])
        self.sobel3x3h = np.array( [[-1,0,1],
                                    [-2,0,2],
                                    [-1,0,1] ])
        self.sobel3x3v = np.array( [[-1,-2,-1],
                                    [0,0,0],
                                    [1,2,1] ])
        self.scharr3x3h = np.array( [   [-3,0,3],
                                        [-10,0,10],
                                        [-3,0,3] ])
        self.scharr3x3v = np.array( [   [-3,-10,-3],
                                        [0,0,0],
                                        [3,10,3] ])
        self.roberts3x3h = np.array( [  [0,0,0],
                                        [0,1,0],
                                        [-1,0,0] ])
        self.roberts3x3v = np.array( [  [0,0,0],
                                        [0,1,0],
                                        [0,0,-1] ])
        self.prewitt3x3h = np.array( [  [-1,0,1],
                                        [-1,0,1],
                                        [-1,0,1] ])
        self.prewitt3x3v = np.array( [  [-1,-1,-1],
                                        [-0,0,0],
                                        [1,1,1] ])
        self.sobel5x5h = np.array( [[-1,-2,0,2,1],
                                    [-4,-8,0,8,4],
                                    [-6,-12,0,12,6],
                                    [-4,-8,0,8,4],
                                    [-1,-2,0,2,1]])
        self.sobel5x5v = np.array( [[0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0]])
