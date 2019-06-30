import os
import cv2
import numpy as np
from detector import YOLO3
from detector.util import COLORS_10, draw_bboxes
from utils.timer import Timer


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},-1,{x1},{y1},{w},{h},{score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, cls_scores in results:
            for tlwh, cls_score in zip(tlwhs, cls_scores):
                xc, yc, w, h = tlwh
                x1, y1 = xc - w/2, yc - h/2
                line = save_format.format(frame=frame_id, id=-1, x1=x1, y1=y1, w=w, h=h,score =cls_score)
                f.write(line)


class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLO3("detector/cfg/yolo_v3.cfg","detector/yolov3.weights","detector/cfg/coco.names", is_xywh=True)
        self.class_names = self.yolo3.class_names
        self.write_video = False
        self.timer = Timer()
            
    def tracking(self,data_root,seq):
        xmin = 0
        ymin =0
        directory = os.path.join(data_root,seq,'img1')
        TEST_IMAGE_PATHS = os.listdir(directory)
        TEST_IMAGE_PATHS.sort(key=str.lower)
        img_path_size=os.path.join(directory,TEST_IMAGE_PATHS[0])
        im = cv2.imread(img_path_size)
        #im  = im[:,:,::-1]
        im_width,im_height = im.shape[1],im.shape[0]
        area = 0, 0, im_width, im_height
        frame = 0
        results = []
        print ("============{}===========".format(seq))
        for image in TEST_IMAGE_PATHS:
            #start = time.time()
            self.timer.tic()
            frame=frame+1
            image_path=os.path.join(directory,image)
            image = cv2.imread(image_path)
            #image  = image[:,:,::-1]
            #image_num = self.load_image_into_numpy_array(image)
            #print("before detector")
            bbox_xywh, cls_conf, cls_ids = self.yolo3(image)
            if bbox_xywh is not None:
                mask = cls_ids==0
                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3] *= 1.15
                #print(bbox_xyxy, identities)
                #image = draw_bboxes(image, bbox_xywh, None, offset=(xmin,ymin))
                #end = time.time()
                self.timer.toc()
                results.append((frame , bbox_xywh, cls_conf))
            if frame % 50 ==0:
                print("frame: {}, fps: {}".format(frame, 1./max(1e-5, self.timer.average_time)))
            #cv2.imshow("test", image)
            #cv2.waitKey(1)
            
        #filename = 'MOT16-13.txt'
        #write_results(os.path.join(data_root,seq,'det_yolo','det.txt'), results, 'mot')

if __name__=="__main__":
    import sys
    if len(sys.argv) == 1:
        print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    else:
        
        data_root='/home/SharedData/swapnil/tracker/MOTDT/data/MOT16/train'
        seqs=('MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13')
        det = Detector()
        for seq in seqs:
            
            det.tracking(data_root,seq)
        '''
        det = Detector()
        #det.open(sys.argv[1])
        #det.detect()
        det.tracking(sys.argv[1],None)
        '''

