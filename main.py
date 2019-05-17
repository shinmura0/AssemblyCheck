import sys
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    sys.exit(1)
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading

lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
camera_width = 320
camera_height = 240
window_name = ""
ssd_detection_mode = 1
face_detection_mode = 0
elapsedtime = 0.0
map_ref = np.zeros((camera_height, camera_width))##
map_result = np.zeros((camera_height, camera_width))##
map_flag = "wait"##
message1 = "Push [m] to measure reference."##
message2 = "Push [s] to start inspection."##


LABELS = [['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'],
          ['background', 'face']]

def camThread(LABELS, results, frameBuffer, camera_width, camera_height, vidfps, number_of_camera):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name
    global map_ref##
    global map_result##
    global map_flag##
    global message1##
    global message2##


    cam = cv2.VideoCapture(number_of_camera)
    if cam.isOpened() != True:
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')##movie save
    #writer = cv2.VideoWriter('output.avi',fourcc, vidfps, (camera_width,camera_height))##movie save
    window_name = "USB Camera"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        # USB Camera Stream Read
        s, color_image = cam.read()
        if not s:
            continue
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        height = color_image.shape[0]
        width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        res = None

        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res, LABELS)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults, LABELS)

        cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
        #writer.write(imdraw)##movie save

        key = cv2.waitKey(1)&0xFF##
        
        if key == ord('q'):
            # Stop streaming
            #writer.release()##movie save
            sys.exit(0)
            
        if key == ord('m'):##
            # measure reference heatmap
            if map_flag == "wait":##
                map_flag = "measure"##
                map_ref = np.zeros((camera_height, camera_width))##
                message1 = "Push [e] to stop measuring."##
                message2 = " "##

        if key == ord('s'):##
            # start inspection
            if map_flag == "wait":##
                map_flag = "start"##
                map_result = np.zeros((camera_height, camera_width))##
                message1 = " "##
                message2 = "Push [e] to finish inspection."##

        if key == ord('e'):##
            # end measure reference heatmap and inspection
            if map_flag == "measure":##
                map_flag = "measure_finish"##
                map_ref /= np.max(map_ref) + 1e-7##
            elif map_flag == "start":##
                map_flag = "inspection_finish"##
                map_result /= np.max(map_result) + 1e-7##
                
            message1 = "Push [m] to measure reference."##
            message2 = "Push [s] to start inspection."##

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):

    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, frameBuffer, results, camera_width, camera_height, number_of_ncs):
        self.devid = devid
        self.frameBuffer = frameBuffer
        self.model_xml = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.xml"
        self.model_bin = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.bin"
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device="MYRIAD")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs


    def image_preprocessing(self, color_image):

        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg - 127.5
        prepimg = prepimg * 0.007843
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            prepimg = self.image_preprocessing(self.frameBuffer.get())
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1
                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            cnt, dev = heapq.heappop(self.heap_request)

            if self.exec_net.requests[dev].wait(0) == 0:
                self.exec_net.requests[dev].wait(-1)
                out = self.exec_net.requests[dev].outputs["detection_out"].flatten()
                self.results.put([out])
                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev))

        except:
            import traceback
            traceback.print_exc()


def inferencer(results, frameBuffer, ssd_detection_mode, face_detection_mode, camera_width, camera_height, number_of_ncs):

    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(NcsWorker(devid, frameBuffer, results, camera_width, camera_height, number_of_ncs),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()
        
def heatmap(box_left, box_top, box_right, box_bottom):##
    global map_ref
    global map_result
    
    if map_flag == "measure":
        temp = np.zeros(map_ref.shape)
        temp[box_top:box_bottom, box_left:box_right] = 1
        map_ref += temp
    elif map_flag == "start":
        temp = np.zeros(map_result.shape)
        temp[np.max([box_top,0]):box_bottom, np.max([box_left,0]):box_right] = 1
        map_result += temp


def overlay_on_image(frames, object_infos, LABELS):
    global map_flag##

    try:

        color_image = frames

        if isinstance(object_infos, type(None)):
            return color_image

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        img_cp = color_image.copy()
        
        #show inspection result
        if map_flag == "measure_finish":##
            map_flag = "wait"##
            heat_map = cv2.applyColorMap(np.uint8(255 * map_ref), cv2.COLORMAP_JET)##
            heat_map = cv2.addWeighted(heat_map,0.5,img_cp,0.5,2.2)##
            cv2.imshow("Reference",heat_map)##
            cv2.imwrite("Reference.jpg",heat_map)##
        elif map_flag == "inspection_finish":##
            map_flag = "wait"##
            heat_map = cv2.applyColorMap(np.uint8(255 * np.abs(map_result-map_ref)), cv2.COLORMAP_JET)##
            heat_map = cv2.addWeighted(heat_map,0.5,img_cp,0.5,2.2)##
            cv2.imshow("Result",heat_map)##
            cv2.imwrite("Result.jpg",heat_map)##


        for (object_info, LABEL) in zip(object_infos, LABELS):

            drawing_initial_flag = True

            for box_index in range(100):
                if object_info[box_index + 1] == 0.0:
                    break
                base_index = box_index * 7
                if (not np.isfinite(object_info[base_index]) or
                    not np.isfinite(object_info[base_index + 1]) or
                    not np.isfinite(object_info[base_index + 2]) or
                    not np.isfinite(object_info[base_index + 3]) or
                    not np.isfinite(object_info[base_index + 4]) or
                    not np.isfinite(object_info[base_index + 5]) or
                    not np.isfinite(object_info[base_index + 6])):
                    continue

                object_info_overlay = object_info[base_index:base_index + 7]

                min_score_percent = 30##

                source_image_width = width
                source_image_height = height

                base_index = 0
                class_id = object_info_overlay[base_index + 1]
                percentage = int(object_info_overlay[base_index + 2] * 100)
                if (percentage <= min_score_percent):
                    continue

                box_left = int(object_info_overlay[base_index + 3] * source_image_width)
                box_top = int(object_info_overlay[base_index + 4] * source_image_height)
                box_right = int(object_info_overlay[base_index + 5] * source_image_width)
                box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

                label_text = LABEL[int(class_id)] + " (" + str(percentage) + "%)"

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
                if "person" in label_text:##
                    label_background_color = (0, 0,255)##
                    heatmap(box_left, box_top, box_right, box_bottom)##
                else:##
                    label_background_color = (125, 175, 75)##                
                label_text_color = (0, 0, 0)##
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


        cv2.putText(img_cp, fps,       (width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, message1,  (width-280,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,38), 1, cv2.LINE_AA)##
        cv2.putText(img_cp, message2,  (width-280,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,38), 1, cv2.LINE_AA)##
        return img_cp

    except:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cn','--numberofcamera',dest='number_of_camera',type=int,default=0,help='USB camera number. (Default=0)')
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=320,help='Width of the frames in the video stream. (Default=320)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=240,help='Height of the frames in the video stream. (Default=240)')
    parser.add_argument('-sd','--ssddetection',dest='ssd_detection_mode',type=int,default=1,help='[Future functions] SSDDetectionMode. (0:=Disabled, 1:=Enabled Default=1)')
    parser.add_argument('-fd','--facedetection',dest='face_detection_mode',type=int,default=0,help='[Future functions] FaceDetectionMode. (0:=Disabled, 1:=Full, 2:=Short Default=0)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (Default=30)')

    args = parser.parse_args()

    number_of_camera = args.number_of_camera
    camera_width  = args.camera_width
    camera_height = args.camera_height
    ssd_detection_mode = args.ssd_detection_mode
    face_detection_mode = args.face_detection_mode
    number_of_ncs = args.number_of_ncs
    vidfps = args.fps_of_video

    if ssd_detection_mode == 0 and face_detection_mode != 0:
        del(LABELS[0])

    try:

        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(LABELS, results, frameBuffer, camera_width, camera_height, vidfps, number_of_camera),
                       daemon=True)
        p.start()
        processes.append(p)

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, ssd_detection_mode, face_detection_mode, camera_width, camera_height, number_of_ncs),
                       daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

        print("\n\nFinished\n\n")
