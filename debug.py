import os
import cv2
import logging
#import imageio
from models.tracker.tracker import OnlineTracker

from utils.mot_seq import get_loader
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results, data_type):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def run(dataloader, result_filename, save_dir=None, show_image=True):
    if save_dir is not None:
        mkdirs(save_dir)

    tracker = OnlineTracker()
    timer = Timer()
    results = []
    wait_time = 1
    #images =[]
    for frame_id, batch in enumerate(dataloader):
        if frame_id % 50 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        frame, det_tlwhs, det_scores, _, _ = batch

        # run tracking
        timer.tic()
        online_targets = tracker.update(frame, det_tlwhs, None)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh)
            online_ids.append(t.track_id)
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=1. / timer.average_time)
        #online_im = vis.plot_trajectory(frame, online_tlwhs, online_ids)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        #online_im = online_im[:,:,::-1]
	#images.append(online_im)
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
    write_results(result_filename, results)



def main(data_root='/home/SharedData/swapnil/tracker/MOTDT/data/MOT16/train', det_root=None,
         seqs=('MOT16-05',), exp_name='demo', save_image=True, show_image=True):

#def main(data_root='/home/SharedData/swapnil/tracker/MOTDT/data/MOT16/train', det_root=None,
#         seqs=('MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13',), exp_name='demo', save_image=False, show_image=False):
#def main(data_root='/home/SharedData/swapnil/tracker/MOTDT/data/MOT16/test', det_root=None,
#         seqs=('MOT16-01','MOT16-03','MOT16-06','MOT16-07','MOT16-08','MOT16-12','MOT16-14',), exp_name='demo', save_image=False, show_image=False):
    #logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # run tracking
    for seq in seqs:
        result_root = os.path.join('RESULT/mot16')
        mkdirs(result_root)
        output_dir = os.path.join(result_root,seq,'img') if save_image else None
        #logger.info('start seq: {}'.format(seq))
        loader = get_loader(data_root, det_root, seq)
        run(loader, os.path.join(result_root, '{}.txt'.format(seq)),
                 save_dir=output_dir, show_image=show_image)


if __name__ == '__main__':
    import fire
    fire.Fire(main)

