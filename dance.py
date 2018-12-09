import numpy as np
import cv2 as cv
import time

conf = 0.2
alpha = 0.5
beta = 1.0 - alpha
writer = None

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
frame_no = 0
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]
# load our serialized model from disk

mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
          [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
          [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
          [37, 38], [45, 46]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    # find the blobs
    _, contours, _ = cv.findContours(mapMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv.resize(pafA, (frameWidth, frameHeight))
        pafB = cv.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if (nA != 0 and nB != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(nA):
                max_j = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else:  # If no keypoints are detected
            # f.write("No Connection : k = {}".format(k) + '\n')
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:, 0]
            partBs = valid_pairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][
                        2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


net = cv.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
net2 = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

cap = cv.VideoCapture('americano.mp4')
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
old_points = None
firstFrame = True
while (1):
    ret, frame = cap.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    blank_image = np.zeros((frameHeight, frameWidth, 3), np.uint8)
    cv.imwrite('images/' + str(frame_no) + '.jpg', blank_image)
    roi = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    # print("[INFO] computing object detections...")
    t = time.time()
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                area = np.abs(startX - endX) * np.abs(startY - endY)
                if area < 10000:
                    frame_no += 1
                    continue

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                # print("[INFO] {}".format(label))
                cv.rectangle(frame, (startX, startY), (endX, endY),
                             COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv.putText(frame, label, (startX, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Set  ROI to the detected persons to minmize the movement lookup area
                roi = frame[startY-40:endY+40, startX-40:endX+40]

                image1 = frame.copy()

                frameWidth = roi.shape[1]
                frameHeight = roi.shape[0]

                dancer = np.zeros((frameHeight, frameWidth, 3), np.uint8)

                cv.rectangle(dancer, (startX, startY), (endX, endY),
                             COLORS[idx], 2)
                cv.putText(dancer, label, (startX, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Fix the input Height and get the width according to the Aspect Ratio
                inHeight = 368
                inWidth = int((inHeight / frameHeight) * frameWidth)

                inpBlob = cv.dnn.blobFromImage(roi, 1.0 / 255, (inWidth, inHeight),
                                               (0, 0, 0), swapRB=False, crop=False)

                net2.setInput(inpBlob)
                output = net2.forward()

                detected_keypoints = []
                keypoints_list = np.zeros((0, 3))
                keypoint_id = 0
                threshold = 0.1

                for part in range(nPoints):
                    probMap = output[0, part, :, :]
                    probMap = cv.resize(probMap, (roi.shape[1], roi.shape[0]))
                    keypoints = getKeypoints(probMap, threshold)
                    keypoints_with_id = []
                    for i in range(len(keypoints)):
                        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                        keypoint_id += 1

                    detected_keypoints.append(keypoints_with_id)

                valid_pairs, invalid_pairs = getValidPairs(output)
                personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)


                for i in range(17):
                    for n in range(len(personwiseKeypoints)):
                        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                        if -1 in index:
                            continue

                        B = np.int32(keypoints_list[index.astype(int), 0])
                        A = np.int32(keypoints_list[index.astype(int), 1])
                        cv.circle(dancer, (B[0], A[0]), 5, colors[i], -1, cv.LINE_AA)
                        cv.circle(dancer, (B[1], A[1]), 5, colors[i], -1, cv.LINE_AA)
                        cv.line(dancer, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv.LINE_AA)

                if writer is None:
                    # initialize our video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter("dance1.avi", fourcc, 30,
                                            (blank_image.shape[1], blank_image.shape[0]), True)

                # To save the separted dancer movement
                blank_image[startY-40:endY+40, startX-40:endX+40] = dancer
                writer.write(frame)
                cv.imwrite('images/' + str(frame_no) + '.jpg', blank_image)
                cv.imshow('image1', blank_image)
                cv.waitKey(1)
                print("Time Taken in forward pass = {}".format(time.time() - t))
                firstFrame = False
    frame_no += 1

cv.destroyAllWindows()
writer.release()
cap.release()
