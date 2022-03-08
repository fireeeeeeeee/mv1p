
import socket
from easymocap.config import load_object_from_cmd
from threading import Thread
from easymocap.socket.utils import encode_annot
from easymocap.mytools import Timer
from easymocap.annotator.file_utils import save_annot
from os.path import join
import cv2
import torch
from  SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
import matplotlib.cm as cm



def handleIMG(display=True):

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.5,
            'max_keypoints': -1,
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    if display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    cameras = load_object_from_cmd('.\\usb_phone.yml',[])
    detector = load_object_from_cmd('.\\mediapipe-holistic.yml', [])



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matching = Matching(config).eval().to(device)
    for nf in range(10000):
        images = cameras.capture()
        def getAnnots():
            if len(images) < 0:
                print('Stop')
            annots = detector(images)
            for i in range(len(images)):
                cam = cameras.camnames[i]
                if cam in cameras.params.keys():
                    camera = cameras.params[cam]
                    #annots[i].update(camera)
            # print(annots[i]['annots'][0]['keypoints'])
            
            return (annots[0]['annots'][0]['keypoints'],annots[1]['annots'][0]['keypoints'])
        def getKeypoints():
            frame1=cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
            frame2=cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY)
            frame_tensor1 = frame2tensor(frame1, device)
            frame_tensor2= frame2tensor(frame2,device)
            pred = matching({'image0':frame_tensor1, 'image1': frame_tensor2})
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().detach().numpy()
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            if display:
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0))
                ]
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                ]
                color = cm.jet(confidence[valid])            
                out = make_matching_plot_fast(
                frame1, frame2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=display, small_text=small_text)
                cv2.imshow('SuperGlue matches', out)
            return (mkpts0,mkpts1)
            
               
        yield getKeypoints(),getAnnots()



if __name__ == '__main__':
    for keypoints,annots in handleIMG():
        pass
