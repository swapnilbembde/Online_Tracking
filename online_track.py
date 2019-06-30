import os
import cv2
import logging
#import imageio
import numpy as np
import sys
from models.tracker.tracker import OnlineTracker

from detector import YOLO3
from detector.util import COLORS_10, draw_bboxes
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))



def tracking(data_root,seq, result_root, save_dir=None, show_image=True,from_video = False,video_path =None,write_video = False,live_demo = False):
    if save_dir is not None:
        mkdirs(save_dir)
    yolo3 = YOLO3("detector/cfg/yolo_v3.cfg","detector/yolov3.weights","detector/cfg/coco.names", is_xywh=True)
    tracker = OnlineTracker()
    timer = Timer()
    wait_time = 1
    frame_no = 1
    
    
    if from_video:
        assert os.path.isfile(video_path), "Error: path error"
        if live_demo:
            vdo = cv2.VideoCapture("http://10.196.30.16:8081") ##http://10.3.0.24:8081
        else:
            vdo = cv2.VideoCapture()
            vdo.open(video_path)
        im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))         
        if write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            output = cv2.VideoWriter(os.path.join(result_root,'{}.avi'.format(seq)), fourcc, 10, (im_width,im_height))
        while vdo.grab(): 
            if frame_no % 10 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_no, 1./max(1e-5, timer.average_time)))
            timer.tic()
            _, image = vdo.retrieve()
            #image = ori_im[0:im_height, 0:im_width, (2,1,0)]
            bbox_xywh, det_scores, cls_ids = yolo3(image)
            if bbox_xywh is not None:
                mask = cls_ids==0
                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3] *= 1.15
                tlwhs = np.empty_like(bbox_xywh[:,:4])
                tlwhs[:,2] = bbox_xywh[:,2] # w
                tlwhs[:,3] = bbox_xywh[:,3] # h
                tlwhs[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2 # x1
                tlwhs[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2 # y1
                #frame, det_tlwhs, det_scores, _, _ = batch

                # run tracking
                #timer.tic()
                online_targets = tracker.update(image, tlwhs, det_scores)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    online_tlwhs.append(t.tlwh)
                    online_ids.append(t.track_id)
                timer.toc()
                #results.append((frame_no, online_tlwhs, online_ids))
                frame_no +=1
                online_im = vis.plot_tracking(image, online_tlwhs, online_ids, frame_id=frame_no,fps=1. / timer.average_time)
                #online_im = vis.plot_trajectory(frame, online_tlwhs, online_ids)
                if show_image:
                    cv2.imshow('online_im', online_im)
                if save_dir is not None:
                    cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_no)), online_im)
                #online_im = online_im[:,:,::-1]
                #images.append(online_im)
                if write_video:
                    output.write(online_im)
                key = cv2.waitKey(wait_time)
                key = chr(key % 128).lower()
                if key == 'q':
                    exit(0)
                elif key == 'p':
                    cv2.waitKey(0)
                elif key == 'a':
                    wait_time = int(not wait_time)
    else:

        directory = os.path.join(data_root,seq,'img1')
        TEST_IMAGE_PATHS = os.listdir(directory)
        TEST_IMAGE_PATHS.sort(key=str.lower)
        #images =[]
        results = []
        if write_video:
            image_path=os.path.join(directory,TEST_IMAGE_PATHS[0])
            image = cv2.imread(image_path)
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            output = cv2.VideoWriter(os.path.join(result_root,'{}.avi'.format(seq)), fourcc, 10, (image.shape[1],image.shape[0]))
        for image in TEST_IMAGE_PATHS:
            
            if frame_no % 50 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_no, 1./max(1e-5, timer.average_time)))
            image_path=os.path.join(directory,image)
            timer.tic()
            image = cv2.imread(image_path)
            #image  = image[:,:,::-1]
            #image_num = self.load_image_into_numpy_array(image)
            #print("before detector")
            bbox_xywh, det_scores, cls_ids = yolo3(image)
            mask = cls_ids==0
            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:,3] *= 1.15
            ## convert to top left width height
            tlwhs = np.empty_like(bbox_xywh[:,:4])
            tlwhs[:,2] = bbox_xywh[:,2] # w
            tlwhs[:,3] = bbox_xywh[:,3] # h
            tlwhs[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2 # x1
            tlwhs[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2 # y1
            #frame, det_tlwhs, det_scores, _, _ = batch

            # run tracking
            #timer.tic()
            online_targets = tracker.update(image, tlwhs, det_scores)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                online_tlwhs.append(t.tlwh)
                online_ids.append(t.track_id)
            timer.toc()

            # save results
            results.append((frame_no, online_tlwhs, online_ids))
            frame_no +=1
            online_im = vis.plot_tracking(image, online_tlwhs, online_ids, frame_id=frame_no,
                                          fps=1. / timer.average_time)

            #online_im = vis.plot_trajectory(frame, online_tlwhs, online_ids)
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_no)), online_im)
            #online_im = online_im[:,:,::-1]
            #images.append(online_im)
            if write_video:
                fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
                output = cv2.VideoWriter(os.path.join(result_root,'{}.avi'.format(seq)), fourcc, 10, (online_im.shape[1],online_im.shape[0]))
                output.write(online_im)
            key = cv2.waitKey(wait_time)
            key = chr(key % 128).lower()
            if key == 'q':
                exit(0)
            elif key == 'p':
                cv2.waitKey(0)
            elif key == 'a':
                wait_time = int(not wait_time)

        # save results
        #imageio.mimsave('MOT16-04.gif', images, fps=10)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        write_results(result_filename, results)

## dont give arg if not video 

#def main(data_root='/home/SharedData/swapnil/tracker/MOTDT/data/MOT16/train', det_root=None,
#         seqs=('MOT16-13',), exp_name='demo', save_image=True, show_image=False):

def main(data_root='/home/SharedData/swapnil/tracker/MOTDT/data/MOT16/train', det_root=None,
         seqs=('MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13',), exp_name='demo', save_image=False, show_image=True,from_video = True,live_demo = False,write_video = True):

    #logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    ## give path of video
    if from_video:
        video_path = sys.argv[1]
    else:
        video_path =None
    # run tracking
    for seq in seqs:
        result_root = os.path.join('RESULT/MOT16/dets_yolo')
        mkdirs(result_root)
        output_dir = os.path.join(result_root,seq,'img') if save_image else None
        #logger.info('start seq: {}'.format(seq))
        tracking(data_root,seq,result_root,save_dir=output_dir, show_image=show_image,from_video =from_video,video_path = video_path,write_video =write_video,live_demo = live_demo)


if __name__ == '__main__':
    import fire
    fire.Fire(main)

