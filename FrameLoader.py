import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

from FocusAnalysis import FocusAnalysis
from blur_detector_video import detectBlurFft

def scaleResizeFrame(frame, scale):
    if scale != None:
        return cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
    else:
        return frame

def displayFrame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('video', frame)
    cv2.waitKey(0)


def getcontrastDataArray(frame, focus_analysis):
    # Remove ignored blocks and restructure into 1d array
    bw_src = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grid = focus_analysis.getDiffSumGrid(bw_src)
    valid_marked_grid = focus_analysis.ignoreRegions(grid)
    valid_marked_array = valid_marked_grid.flatten()
    valid_array = valid_marked_array[valid_marked_array < 256*focus_analysis.block_size*focus_analysis.block_size]
    return valid_array

def getLuminescenceDataArray(frame, focus_analysis):
    bw_src = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grid = focus_analysis.getSumGrid(bw_src)
    valid_marked_grid = focus_analysis.ignoreRegions(grid)
    valid_marked_array = valid_marked_grid.flatten()
    valid_array = valid_marked_array[valid_marked_array < 256*focus_analysis.block_size*focus_analysis.block_size]
    return valid_array
    
def main():
    scale = 0.5
    frame_cnt = 0
    video_name = "HerkulesMission.mp4"
    desired_frame = None

    vidcap = cv2.VideoCapture(video_name)
    focus_analysis = FocusAnalysis()
    num_total_frames = int(vidcap.get(7))

    contrast_log = []
    mean_contrast_log = []
    std_contrast_log = []
    percentile_95_contrast_log = []
    kurtosis_contrast_log = []

    luminescence_log = []
    mean_luminescence_log = []
    std_luminescence_log = []
    percentile_95_luminescence_log = []
    kurtosis_luminescence_log = []

    blur_log = []

    print("Total frames: ", num_total_frames)

    if not vidcap.isOpened():
        raise Exception("Could not open video device: " + str(video_name) )

    if desired_frame == None:
        ret = True
        while ret == True:
            ret, frame = vidcap.read()

            if ret == False:
                print('EOF!')
                break

            frame = scaleResizeFrame(frame, scale)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #heat_map = focus_analysis.addcontrastHeatmap(frame)
            #displayFrame(heat_map)

            # Remove ignored blocks and restructure into 1d array
            contrast_array = getcontrastDataArray(frame, focus_analysis)
            luminescence_array = getLuminescenceDataArray(frame, focus_analysis)

            # Calculate blur
            blur_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #blur_gray_valid_marked = focus_analysis.ignoreRegions(blur_gray, block_size_ignore=1)
            #blur_gray_valid_marked[blur_gray_valid_marked < 256*focus_analysis.block_size*focus_analysis.block_size] = 0
            blur_score, blurry = detectBlurFft(blur_gray)#_valid_marked)
            blur_log.append(blur_score)
            #print(blur_score)


            # Basic statistics calucluations - contrast
            frame_contrast = np.sum(contrast_array)
            mean_frame_contrast = frame_contrast/len(contrast_array)
            std_frame_contrast = np.std(contrast_array)
            percentile_95_frame_contrast = np.percentile(contrast_array, 95)
            kurtosis_frame_contrast = kurtosis(contrast_array)

            # Basic statistics calucluations - luminescence
            frame_luminescence = np.sum(luminescence_array)
            mean_frame_luminescence = frame_luminescence/len(luminescence_array)
            std_frame_luminescence = np.std(luminescence_array)
            percentile_95_frame_luminescence = np.percentile(luminescence_array, 95)
            kurtosis_frame_luminescence = kurtosis(luminescence_array)

            # Record values - contrast
            contrast_log.append(frame_contrast)
            mean_contrast_log.append(mean_frame_contrast)
            std_contrast_log.append(std_frame_contrast)
            percentile_95_contrast_log.append(percentile_95_frame_contrast)
            kurtosis_contrast_log.append(kurtosis_frame_contrast)

            # Record values - luminescence
            luminescence_log.append(frame_luminescence)
            mean_luminescence_log.append(mean_frame_luminescence)
            std_luminescence_log.append(std_frame_luminescence)
            percentile_95_luminescence_log.append(percentile_95_frame_luminescence)
            kurtosis_luminescence_log.append(kurtosis_frame_luminescence)

            print("Frame: ", frame_cnt, " / ", num_total_frames)
            frame_cnt += 1
        
        # Plot values - contrast
        plt.plot(mean_contrast_log)
        plt.title("contrast Mean")
        plt.ylabel('contrast Mean')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(std_contrast_log)
        plt.title("contrast STD")
        plt.ylabel('contrast STD')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(percentile_95_contrast_log)
        plt.title("contrast 95th percentile")
        plt.ylabel('contrast 95th percentile')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(kurtosis_contrast_log)
        plt.title("contrast kurtosis")
        plt.ylabel('contrast kurtosis')
        plt.xlabel('Frame nr')
        plt.show()

        # Plot values - luminescence
        plt.plot(mean_luminescence_log)
        plt.title("Luminescence Mean")
        plt.ylabel('Luminescence Mean')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(std_luminescence_log)
        plt.title("Luminescence STD")
        plt.ylabel('Luminescence STD')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(percentile_95_luminescence_log)
        plt.title("Luminescence 95th percentile")
        plt.ylabel('Luminescence 95th percentile')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(kurtosis_luminescence_log)
        plt.title("Luminescence kurtosis")
        plt.ylabel('Luminescence kurtosis')
        plt.xlabel('Frame nr')
        plt.show()

        plt.plot(blur_log)
        plt.title("Blur")
        plt.ylabel('Blur score')
        plt.xlabel('Frame nr')
        plt.show()

    else:
        print('Showing frame: ', desired_frame)
        vidcap.set(1, desired_frame)
        ret, frame = vidcap.read()
        frame = scaleResizeFrame(frame, scale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #displayFrame(frame)
        contrast_array = getcontrastDataArray(frame, focus_analysis)
        plt.hist(contrast_array, bins=100)
        plt.ylabel('Num blocks')
        plt.xlabel('contrast value')
        plt.show()


main()