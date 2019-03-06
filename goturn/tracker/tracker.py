# Date: Friday 02 June 2017 05:04:00 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Basic regressor function implemented

from __future__ import print_function
from ..helper.image_proc import cropPadImage
from ..helper.BoundingBox import BoundingBox
import cv2

class tracker:
    """tracker class"""

    def __init__(self, show_intermediate_output, logger):
        """TODO: to be defined. """
        self.show_intermediate_output = show_intermediate_output
        self.logger = logger

    def init(self, image_curr, bbox_gt, objRegressor):
        """ initializing the first frame in the video
        """
        self.image_prev = image_curr
        self.bbox_prev_tight = bbox_gt
        self.bbox_curr_prior_tight = bbox_gt
        #print(bbox_gt.x1, bbox_gt.y1, bbox_gt.x2, bbox_gt.y2);
        # objRegressor.init()

    def track(self, image_curr, objRegressor):
        """TODO: Docstring for tracker.
        :returns: TODO

        """
        target_pad, _, _,  _ = cropPadImage(self.bbox_prev_tight, self.image_prev)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(self.bbox_curr_prior_tight, image_curr)

        #print(self.bbox_prev_tight.y2 - self.bbox_prev_tight.y1, self.bbox_prev_tight.x2 - self.bbox_prev_tight.x1, target_pad.shape)
        # target_pad = cv2.rectangle(target_pad, (int(search_location.x1), int(search_location.y1)), (int(search_location.x2), int(search_location.y2)), (0,0,255), 2)
        # cv2.imshow('target_pad', target_pad)
        # cv2.imshow('cur_search', cur_search_region)

        bbox_estimate = objRegressor.regress(cur_search_region, target_pad)
        bbox_estimate = BoundingBox(bbox_estimate[0, 0], bbox_estimate[0, 1], bbox_estimate[0, 2], bbox_estimate[0, 3])

        # image_after = cur_search_region.copy()
        # image_after = cv2.rectangle(image_after, (int(bbox_estimate.x1), int(bbox_estimate.y1)), (int(bbox_estimate.x2), int(bbox_estimate.y2)), (255, 0, 0), 2)
        # cv2.imshow('after_image', image_after)
        # cv2.waitKey(0)

        # Inplace correction of bounding box
        bbox_estimate.unscale(cur_search_region)
        bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)

        self.image_prev = image_curr
        self.bbox_prev_tight = bbox_estimate
        self.bbox_curr_prior_tight = bbox_estimate

        return bbox_estimate

#Moiz's track for MOT16
    def track2(self, image_curr, image_prev, bbox_prev, objRegressor):
        """TODO: Docstring for tracker.
        :returns: TODO

        """
        target_pad, _, _,  _ = cropPadImage(bbox_prev, image_prev)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_prev, image_curr)

        #print(self.bbox_prev_tight.y2 - self.bbox_prev_tight.y1, self.bbox_prev_tight.x2 - self.bbox_prev_tight.x1, target_pad.shape)
        # target_pad = cv2.rectangle(target_pad, (int(search_location.x1), int(search_location.y1)), (int(search_location.x2), int(search_location.y2)), (0,0,255), 2)
        # cv2.imshow('target_pad', target_pad)
        # cv2.imshow('cur_search', cur_search_region)

        bbox_estimate = objRegressor.regress(cur_search_region, target_pad)
        bbox_estimate = BoundingBox(bbox_estimate[0, 0], bbox_estimate[0, 1], bbox_estimate[0, 2], bbox_estimate[0, 3])

        # image_after = cur_search_region.copy()
        # image_after = cv2.rectangle(image_after, (int(bbox_estimate.x1), int(bbox_estimate.y1)), (int(bbox_estimate.x2), int(bbox_estimate.y2)), (255, 0, 0), 2)
        # cv2.imshow('after_image', image_after)
        # cv2.waitKey(0)

        # Inplace correction of bounding box
        bbox_estimate.unscale(cur_search_region)
        bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)

        #self.image_prev = image_curr
        #self.bbox_prev_tight = bbox_estimate
        #self.bbox_curr_prior_tight = bbox_estimate

        return bbox_estimate
