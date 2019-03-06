# Date: Wednesday 07 June 2017 11:28:11 AM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: tracker manager

import cv2
from ..helper.BoundingBox import BoundingBox
import MOT17testbench.benchmark as bm

opencv_version = cv2.__version__.split('.')[0]


class tracker_manager:

    """Docstring for tracker_manager. """

    def __init__(self, videos, regressor, tracker, logger):
        """This is

        :videos: list of video frames and annotations
        :regressor: regressor object
        :tracker: tracker object
        :logger: logger object
        :returns: list of video sub directories
        """

        self.videos = videos
        self.regressor = regressor
        self.tracker = tracker
        self.logger = logger

    def trackAll(self, start_video_num, pause_val):
        """Track the objects in the video
        """

        videos = self.videos
        objRegressor = self.regressor
        objTracker = self.tracker

        video_keys = list(videos.keys())
        #print(opencv_version);
        for i in range(start_video_num, len(videos)):
            video_frames = videos[video_keys[i]][0]
            annot_frames = videos[video_keys[i]][1]

            num_frames = min(len(video_frames), len(annot_frames))

            # Get the first frame of this video with the intial ground-truth bounding box
            frame_0 = video_frames[0]
            #bbox_0 = annot_frames[0]
            sMatImage = cv2.imread(frame_0)
            bbox_0 = cv2.selectROI(sMatImage)
            x, y, w, h = bbox_0
            bbox_0 = BoundingBox(x, y, x+w, y+h)
            print(bbox_0.x1, bbox_0.y1, bbox_0.x2, bbox_0.y2)
            objTracker.init(sMatImage, bbox_0, objRegressor)

            #print(video_frames[1])
            # cv2.imshow("initial", sMatImage)
            # cv2.waitKey(0)

            for i in range(1, num_frames):
                frame = video_frames[i]
                sMatImage = cv2.imread(frame)
                sMatImageDraw = sMatImage.copy()
                bbox = annot_frames[i]

                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)

                bbox2 = objTracker.track(sMatImage, objRegressor)
                #print(bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2, " | ", bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2)
                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox2.x1), int(bbox2.y1)), (int(bbox2.x2), int(bbox2.y2)), (255, 0, 0), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox2.x1), int(bbox2.y1)), (int(bbox2.x2), int(bbox2.y2)), (255, 0, 0), 2)

                cv2.imshow('Results', sMatImageDraw)
                cv2.waitKey(1)

#Moiz's trackall for MOT116
    def trackAll2(self, start_video_num, pause_val):
        """Track the objects in the video
        """

        objRegressor = self.regressor
        objTracker = self.tracker

        loader = bm.DataLoader("/home/mrasheed/Downloads/MOT16/")
        #print(opencv_version);
        for seq in loader:

            prev_img, bboxes = next(iter(seq))
            bounding_boxes = [];
            for box in bboxes:
                bounding_boxes.append(BoundingBox(box[0], box[1], box[0] + box[2], box[1] + box[3]))

            frame = 1

            for next_img, bbox_truths in seq:

                sMatImageDraw = next_img.copy()

                for i in range(0, len(bounding_boxes)):
                    bounding_boxes[i] = objTracker.track2(next_img, prev_img, bounding_boxes[i], objRegressor)
                    bbox = bounding_boxes[i]
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)

                if frame % 20 == 0:
                    self.addNewDetections(bounding_boxes, bbox_truths)

                for bbox in bbox_truths:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 255, 255), 2)

                prev_img = next_img
                frame += 1
                cv2.imshow('Results', sMatImageDraw)
                cv2.waitKey(1)

    def addNewDetections(self, current_bboxes, detected_bboxes):
        leftover = [i for i in detected_bboxes if self.bboxAccountedFor(i, current_bboxes)]
        for i in leftover:
            current_bboxes.append(BoundingBox(i[0], i[1], i[0] + i[2], i[1] + i[3]))

    def bboxAccountedFor(self, box, curr_bboxes):
        box_area = box[2] * box[3]
        for b in curr_bboxes:
            area = self.calcSharedArea(box, b)
            if area/box_area > .6:
                return 0
        return 1

#first argument is array of [x, y, width, height] and second is BoundingBox
    def calcSharedArea(self, box1, box2):
        x1 = max(box1[0], box2.x1)
        y1 = max(box1[1], box2.y1)
        x2 = min(box1[0] + box1[2], box2.x2)
        y2 = min(box1[1] + box1[3], box2.y2)
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        else:
            return -1
