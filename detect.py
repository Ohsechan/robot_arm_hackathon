import numpy as np
from sklearn.linear_model import LinearRegression

import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
from time import time

import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from time import sleep
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode




def return_destination_pxl(color_box):
    pxl_list = None
    if (color_box[2] == "red_center"):
        pxl_list = ((68, 196), (125, 123))
    elif ((color_box[2] == "green_center")):
        pxl_list = ((437, 369), (176, 369), (307, 212))
    elif ((color_box[2] == "yellow_center")):
        pxl_list = ((565, 202), (538, 112))
    else:
        print("ERROR!")
        return
    max_distance = 0
    max_pxl = (0, 0)
    for each in pxl_list:
        dist = math.sqrt((each[0] - color_box[0])**2 + (each[1] - color_box[1])**2)
        if (dist > max_distance):
            max_distance = dist
            max_pxl = each
    return max_pxl

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global g_source, g_device, model, stride, names, pt, g_imgsz, dataset, bs, dt, pred, annotator, g_augment, g_visualize, g_conf_thres, g_iou_thres, g_classes, g_agnostic_nms, g_max_det, g_line_thickness, color_center
    g_augment, g_visualize, g_conf_thres, g_iou_thres, g_classes, g_agnostic_nms, g_max_det, g_line_thickness = augment, visualize, conf_thres, iou_thres, classes, agnostic_nms, max_det, line_thickness
    g_source = str(source)

    # Load model
    g_device = select_device(device) # select_device 안에 LOGGER.info(s) 주석처리 필요
    model = DetectMultiBackend(weights, device=g_device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    g_imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # view_img = check_imshow(warn=True)
    dataset = LoadStreams(g_source, img_size=g_imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        i=0
        det = pred[i]
        p, im0, frame = path[i], im0s[i].copy(), dataset.count

        p = Path(p)  # to Path
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            class_xyxy = []
            center_xyxy = []
            color_center = []
            # Write results
            for *xyxy, conf, cls in reversed(det):

                c = int(cls)  # integer class
                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                # annotator.box_label(xyxy, label, color=colors(c, True))

                # Make xyxy list
                tmp = []
                for each in xyxy:
                    tmp.append(int(each.cpu()))
                tmp.append(names[c])
                
                if (names[c] == "center"):
                    center_xyxy.append(tmp)
                else:
                    class_xyxy.append(tmp)
            
            # Make color_center
            for each in center_xyxy:
                center_x = (each[0] + each[2]) // 2
                center_y = (each[1] + each[3]) // 2
                for j in range(len(class_xyxy)):
                    if (class_xyxy[j][0] < center_x and center_x < class_xyxy[j][2] and class_xyxy[j][1] < center_y and center_y < class_xyxy[j][3]):
                        tmp = [center_x, center_y, class_xyxy[j][4]+"_center"]
                        color_center.append(tmp)
                        break
            return color_center
        else:
            return None

def picture():
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *g_imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=g_augment, visualize=g_visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, g_conf_thres, g_iou_thres, g_classes, g_agnostic_nms, max_det=g_max_det)

        # Process predictions
        i=0
        det = pred[i]
        p, im0, frame = path[i], im0s[i].copy(), dataset.count

        p = Path(p)  # to Path
        annotator = Annotator(im0, line_width=g_line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            class_xyxy = []
            center_xyxy = []
            color_center = []
            # Write results
            for *xyxy, conf, cls in reversed(det):

                c = int(cls)  # integer class
                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                # annotator.box_label(xyxy, label, color=colors(c, True))

                # Make xyxy list
                tmp = []
                for each in xyxy:
                    tmp.append(int(each.cpu()))
                tmp.append(names[c])
                
                if (names[c] == "center"):
                    center_xyxy.append(tmp)
                else:
                    class_xyxy.append(tmp)
            
            # Make color_center
            for each in center_xyxy:
                center_x = (each[0] + each[2]) // 2
                center_y = (each[1] + each[3]) // 2
                for j in range(len(class_xyxy)):
                    if (class_xyxy[j][0] < center_x and center_x < class_xyxy[j][2] and class_xyxy[j][1] < center_y and center_y < class_xyxy[j][3]):
                        tmp = [center_x, center_y, class_xyxy[j][4]+"_center"]
                        color_center.append(tmp)
                        break
            return color_center
        return []

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=ROOT / 'yolov3-tiny.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def return_models():
    # Assuming you have a list of 60 pairs of known coordinates, where each pair is a tuple (pixel, ground)
    known_coordinates = (
        ((49, 237), (275.6326074911072, -122.67898718933081, -56.72599792480469)),
        ((48, 59), (193.33179693590466, -124.15807267624096, -56.923912048339844)),
        ((46, 146), (233.04035966160154, -126.21468612554683, -57.32183074951172)),
        ((51, 419), (357.99339041179, -127.60228957817546, -56.71638870239258)),
        ((52, 331), (318.69598450890174, -125.96023812624524, -58.1599983215332)),

        ((151, 329), (318.82759867164276, -79.31259283222408, -56.711204528808594)),
        ((154, 241), (278.8913096442727, -77.98629355583003, -56.785301208496094)),
        ((151, 422), (360.9724296061843, -81.36392778716511, -56.59962844848633)),
        ((160, 57), (195.93847596676267, -71.57828750354697, -57.488525390625)),
        ((158, 150), (237.92552342215046, -74.02532754186778, -57.416236877441406)),

        ((254, 59), (197.82054218083493, -28.884789036475297, -57.77962875366211)),
        ((252, 152), (238.6083555199741, -33.05244236354166, -57.14307403564453)),
        ((252, 246), (281.0684551672538, -33.51637681693395, -56.89265060424805)),
        ((253, 333), (320.8732292941727, -35.07890122961732, -56.529727935791016)),
        ((250, 427), (362.9623744945026, -36.49080789533954, -56.55120849609375)),

        ((377, 420), (362.55032854754546, 22.336274763893577, -56.603377532958984)),
        ((379, 235), (277.76179475639145, 23.14904544421453, -57.064698791503906)),
        ((384, 331), (320.25661913749514, 22.87218497624237, -57.03407897949219)),
        ((383, 51), (196.20794735105102, 27.97977207564024, -57.089073181152344)),
        ((386, 143), (237.569047621159, 26.71573858814299, -57.03708572387695)),

        ((422, 150), (240.74455208187052, 44.38188921150644, -57.21449279785156)),
        ((420, 56), (199.98930120255366, 43.491470235160875, -57.52277755737305)),
        ((426, 239), (280.51405177262035, 43.90568204806508, -56.82350158691406)),
        ((434, 333), (323.1341089146995, 46.22427840209013, -56.43318557739258)),
        ((430, 423), (365.64963763401573, 45.49095830407009, -56.74821090698242)),

        ((511, 420), (364.47403072618675, 81.52369906739922, -56.953285217285156)),
        ((511, 328), (322.1858761664633, 82.25364566194064, -56.46592330932617)),
        ((503, 234), (279.55130673711005, 79.72543239800842, -56.93348693847656)),
        ((502, 51), (197.62287767435467, 81.20530911491541, -57.51274490356445)),
        ((507, 144), (239.0765518754956, 81.31993472402644, -57.31686782836914)),

        ((582, 56), (199.40120182548998, 116.24588583891197, -57.526893615722656)),
        ((592, 414), (362.78325385996004, 119.20442208280471, -56.921932220458984)),
        ((585, 230), (277.8775120627365, 117.86349430891397, -56.88685607910156)),
        ((588, 320), (319.52822119459205, 118.5026321882211, -56.6039924621582)),
        ((587, 141), (239.69878720624266, 117.68782589137868, -57.08183288574219))
    )

    # Separate the known pixel coordinates (x, y) and ground coordinates (X, Y)
    pixel_coords, ground_coords = zip(*known_coordinates)

    # Fit a linear regression model
    layer1_model = LinearRegression()
    layer1_model.fit(pixel_coords, np.array(ground_coords))

    # Assuming you have a list of 60 pairs of known coordinates, where each pair is a tuple (pixel, ground)
    known_coordinates = (
        ((37, 59),(195.28713205606297, -124.5209664982722, -41.28672790527344)),
        ((32, 151),(236.39005692252258, -126.5501077115287, -41.05376052856445)),
        ((35, 343),(322.058382296425, -130.57761592957863, -39.90864181518555)),
        ((40, 445),(367.470730164378, -130.486408889597, -40.251922607421875)),
        ((37, 220),(268.3891290626609, -126.77010809092913, -40.65782928466797)),

        ((151, 241),(278.1630752313755, -76.06997486542944, -40.105838775634766)),
        ((153, 53),(196.72369639118827, -74.07507574244899, -41.527713775634766)),
        ((138, 436),(363.45159037991704, -86.28466147995861, -40.16398620605469)),
        ((144, 348),(324.836243672344, -80.94526286969335, -39.72987747192383)),
        ((157, 132),(230.90719725313667, -72.21132701888503, -41.01066970825195)),

        ((244, 54),(196.8290632834665, -33.54005270936796, -41.69313430786133)),
        ((248, 228),(273.5359651792135, -34.43978626152793, -40.322635650634766)),
        ((237, 433),(363.74669130958875, -41.2832749424232, -40.18488311767578)),
        ((240, 333),(319.2155709873734, -39.19048054834911, -39.87953567504883)),
        ((251, 130),(232.04170914533265, -31.256630777484798, -41.250423431396484)),

        ((329, 435),(364.79498219583496, -2.0515115595646227, -40.07523727416992)),
        ((333, 246),(283.09399104015324, 2.25044227865015, -40.445003509521484)),
        ((324, 46),(195.325624581086, 0.15183337705012964, -41.70359420776367)),
        ((327, 347),(326.3122695730275, -1.8048225354934546, -40.23997116088867)),
        ((320, 142),(237.1448583298254, -1.7425457357750676, -40.76787567138672)),

        ((421, 243),(283.11706660841406, 40.66159449428351, -40.302635192871094)),
        ((413, 49),(198.44173174989513, 39.2949083510117, -41.75190734863281)),
        ((414, 435),(366.88969647627937, 35.63417025591277, -40.39707946777344)),
        ((418, 341),(325.1636007710754, 38.273543397592974, -39.95843505859375)),
        ((418, 142),(238.4688880301115, 39.089197109378745, -41.027713775634766)),

        ((506, 420),(360.8367787371981, 75.08162278747305, -40.27121353149414)),
        ((494, 58),(202.42517828268683, 75.17310077816994, -41.54117965698242)),
        ((500, 233),(279.72513499234464, 76.56352551558084, -40.34137725830078)),
        ((502, 333),(322.5557172159432, 74.35243461235285, -40.23277282714844)),
        ((501, 136),(236.57037333686264, 75.40211099713297, -41.107994079589844)),

        ((602, 421),(363.10724735241763, 118.66386933628772, -40.55376434326172)),
        ((601, 244),(285.26076790131793, 120.5249071443946, -40.6328125)),
        ((601, 56),(203.89447707745896, 120.90696694526115, -41.00460433959961)),
        ((597, 349),(330.4345758181348, 116.6035918368565, -40.183162689208984)),
        ((603, 154),(244.40684286565505, 120.40914355061928, -40.98419952392578))
    )

    # Separate the known pixel coordinates (x, y) and ground coordinates (X, Y)
    pixel_coords, ground_coords = zip(*known_coordinates)

    # Fit a linear regression model
    layer2_model = LinearRegression()
    layer2_model.fit(pixel_coords, np.array(ground_coords))

    # Assuming you have a list of 60 pairs of known coordinates, where each pair is a tuple (pixel, ground)
    known_coordinates = (
        ((31, 419),(352.98523990776454, -128.17017067006987, -23.948150634765625)),
        ((20, 48),(194.0012194955304, -127.53062905049634, -24.99176025390625)),
        ((21, 239),(275.83308761166603, -128.69139259354884, -24.300718307495117)),
        ((19, 339),(318.6381068191546, -133.52811878585078, -24.191919326782227)),
        ((28, 146),(235.81402064100683, -125.15582157179799, -24.42872428894043)),

        ((120, 39),(191.5618100798726, -83.81247419570431, -25.718109130859375)),
        ((123, 244),(279.2405064729096, -86.4075826291423, -24.252424240112305)),
        ((122, 428),(357.4415115988443, -89.92310541302952, -24.28862190246582)),
        ((125, 333),(315.988019327132, -84.78981779731356, -23.941251754760742)),
        ((127, 144),(236.99425108778453, -81.36517871825158, -24.982908248901367)),

        ((224, 47),(197.15865726261325, -40.64651730763791, -25.761760711669922)),
        ((218, 239),(278.6925190733572, -45.3878166752974, -24.16609001159668)),
        ((217, 435),(361.1268710592698, -48.7306046245689, -23.924392700195312)),
        ((219, 337),(319.95465559424343, -44.13083491218314, -23.671985626220703)),
        ((226, 136),(235.00398084025215, -41.40795926871197, -25.067873001098633)),

        ((311, 235),(277.3038622793824, -6.7643580524983635, -24.395158767700195)),
        ((308, 427),(359.4392274106933, -9.728641518577465, -23.566667556762695)),
        ((322, 43),(197.72002153429034, -1.5470974236956887, -25.7122802734375)),
        ((311, 329),(317.6867640058492, -6.16839974888237, -24.095294952392578)),
        ((316, 141),(238.46821758441868, -4.137284128408377, -24.785207748413086)),

        ((403, 243),(281.68874243560003, 31.200058869491784, -24.4703311920166)),
        ((400, 429),(361.10731850757475, 28.277878982521948, -23.565038681030273)),
        ((409, 44),(197.56790978469283, 34.699617516512305, -25.78985023498535)),
        ((410, 140),(238.49557862138732, 33.89567433905474, -24.798324584960938)),
        ((399, 333),(319.77872097418305, 28.66973784177278, -24.142314910888672)),

        ((492, 38),(197.355420820956, 70.51000068047024, -25.712974548339844)),
        ((509, 242),(283.5009565344336, 75.60312176294251, -24.2554988861084)),
        ((494, 420),(358.9464735243272, 69.00838829895963, -23.865901947021484)),
        ((504, 322),(316.5386882840356, 74.3338372338904, -23.788305282592773)),
        ((505, 138),(239.17407476768724, 74.59341103426185, -25.014881134033203)),

        ((606, 44),(199.73475849960136, 118.32375181969728, -25.355175018310547)),
        ((592, 415),(358.2469351038052, 110.50491211049766, -24.137409210205078)),
        ((596, 235),(280.6246208003357, 113.30519352063199, -24.318349838256836)),
        ((592, 334),(323.6209706182783, 109.86686267675134, -24.179370880126953)),
        ((603, 129),(236.8638812681343, 115.58679457230727, -24.733108520507812)),
    )

    # Separate the known pixel coordinates (x, y) and ground coordinates (X, Y)
    pixel_coords, ground_coords = zip(*known_coordinates)

    # Fit a linear regression model
    layer3_model = LinearRegression()
    layer3_model.fit(pixel_coords, np.array(ground_coords))
    
    return (layer1_model, layer2_model, layer3_model)

def pixel2xyz(layer, x, y):
    pixel = np.array([[x, y]])
    if (layer==1):
        ground = layer1_model.predict(pixel)
        return tuple(ground[0])
    elif (layer==2):
        ground = layer2_model.predict(pixel)
        return tuple(ground[0])
    elif (layer==3):
        ground = layer3_model.predict(pixel)
        return tuple(ground[0])

def connect_robot(ip):
    try:
        dashboard_p = 29999
        move_p = 30003
        feed_p = 30004
        # print("연결 설정 중...")

        dashboard = DobotApiDashboard(ip, dashboard_p)
        move = DobotApiMove(ip, move_p)
        feed = DobotApi(ip, feed_p)
        # print("연결 성공!!")

        return dashboard, move, feed

    except Exception as e:
        print("연결 실패")
        raise e

def robot_clear(dashboard : DobotApiDashboard):
    dashboard.ClearError()

def robot_speed(dashboard : DobotApiDashboard, speed_value):
    dashboard.SpeedFactor(speed_value)

def gripper_DO(dashboard : DobotApiDashboard, index, status):
    dashboard.ToolDO(index, status)

def get_Pose(dashboard : DobotApiDashboard):
    dashboard.GetPose()

def run_point(move: DobotApiMove, point_list: list):
    move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])

def run_point2(move: DobotApiMove, point_list1: list, point_list2: list):
    wait_near(point_list1)
    move.MovL(point_list2[0], point_list2[1], point_list2[2], point_list2[3])

def get_feed(feed: DobotApi):
    global current_actual
    hasRead = 0

    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0
        a = np.frombuffer(data, dtype=MyType)

        if hex((a['test_value'][0])) == '0x123456789abcdef':
            current_actual = a["tool_vector_actual"][0]     # Refresh Properties
        sleep(0.001)

def wait_arrive(point_list):
    global current_actual
    while True:
        is_arrive = True
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive:
                return
        sleep(0.001)

def wait_near(point_list):
    global current_actual
    while True:
        is_arrive = True
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 15):
                    is_arrive = False
            if is_arrive:
                return
        sleep(0.001)

def main():
    global color_center

    # 입력 파라미터
    ip = "192.168.1.6"              # Robot의 IP 주소
    gripper_port = 1                # 그리퍼 포트 번호
    speed_value = 100                # 로봇 속도 (1~100 사이의 값 입력)

    ## 로봇의 원위치 좌표 (x, y, z, yaw) unit : mm, degree
    point_home = [175.67116488, -128.80691014, 32.12535477, 86.67063904]

    # 로봇 연결
    dashboard, move, feed = connect_robot(ip)
    dashboard.EnableRobot()
    # print("이제 로봇을 사용할 수 있습니다!")


    # 쓰레드 설정
    feed_thread = threading.Thread(target=get_feed, args=(feed,))
    feed_thread.setDaemon(True)
    feed_thread.start()

    # 로봇 상태 초기화 1 : 로봇 에러 메시지 초기화
    robot_clear(dashboard)

    # 로봇 상태 초기화 2 : 로봇 속도 조절
    robot_speed(dashboard, speed_value)


    # 로봇 원위치하기!
    if (current_actual[2] < 0):
        run_point(move, (current_actual[0],current_actual[1], 0, 115))
        wait_arrive((current_actual[0],current_actual[1], 0, 115))
    run_point(move, point_home)

    # 블록 순서
    block_seq = []

    ## 1단계: 3층 센터 인식 후, 해당색깔로 이동
    thread1.join()
    print(color_center)

    # 사진 안 찍혔을 때 다시 찍기
    while (color_center is None or len(color_center) == 0):
        run_point(move, point_home)
        wait_arrive(point_home)
        color_center = picture()
        print(color_center)

    # print(color_center)
    if (color_center is not None) :
        for each in color_center:
            print(each[2], tuple(each[:2]))
    max_pxl = None

    print("dobot_xyz\t", tuple(current_actual[:3]))

    # 블록 순서 추가
    for each in color_center:
        check = True
        for color in block_seq:
            if (color[2] == each[2]):
                check = False
        if (check):
            block_seq.append(each)

    # print(color_center)
    if block_seq is not None and len(block_seq) > 0:
        max_pxl = return_destination_pxl(block_seq[-1])
        print(max_pxl)
    
    # 시작지점, 끝지점 알아내기
    start_layer, end_layer = 3, 1
    start_x, start_y, start_z = pixel2xyz(start_layer, block_seq[-1][0], block_seq[-1][1])
    end_x, end_y, end_z = pixel2xyz(end_layer, max_pxl[0], max_pxl[1])
    stage1_xy = (max_pxl[0], max_pxl[1])

    # 블럭 들고, 로봇 이동
    run_point(move, [start_x, start_y, start_z+1.1, 115])    # 블럭 위치로 손 이동
    wait_arrive([start_x, start_y, start_z+1.1, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 1)    # 자력 발동

    run_point(move, [end_x, end_y, start_z+5, 115])    # 목적지로 이동
    wait_arrive([end_x, end_y, start_z+5, 115])    # 도착까지 대기
    # run_point(move, [(start_x+end_x)/2, (start_y+end_y)/2, start_z+5, 115])    # 목적지로 이동
    # wait_near([(start_x+end_x)/2, (start_y+end_y)/2, start_z+5, 115])   # move softly

    # run_point(move, [end_x, end_y, -30, 115])    # 떨어뜨리기(첫 state에서만 끄기)
    # wait_arrive([end_x, end_y, -30, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거

    # run_point(move, [end_x, end_y, -15, 115])    # 목적지로 이동
    # gripper_DO(dashboard, gripper_port, 0)    # 자력 제거
    # wait_arrive([end_x, end_y, -15, 115])    # 도착까지 대기
    # gripper_DO(dashboard, gripper_port, 0)    # 자력 제거


    ## 2단계: 2층 센터 인식 후, 해당색깔로 이동

    color_center = picture()
    print(color_center)
    max_pxl = None

    # 블록 순서 추가
    while_check = True
    while (while_check):
        for each in color_center:
            check = True
            for color in block_seq:
                if (color[2] == each[2]):
                    check = False
            if (check):
                block_seq.append(each)
                while_check = False
                break
        if (while_check):
            run_point(move, point_home)
            wait_arrive(point_home)
            color_center = picture()

    # print(color_center)
    if block_seq is not None and len(block_seq) > 0:
        max_pxl = return_destination_pxl(block_seq[-1])
        print(max_pxl)

    # 시작지점, 끝지점 알아내기
    start_layer, end_layer = 2, 1
    start_x, start_y, start_z = pixel2xyz(start_layer, block_seq[-1][0], block_seq[-1][1])   # 현재 2층에 놓여있는 블럭의 위치
    end_x, end_y, end_z = pixel2xyz(end_layer, max_pxl[0], max_pxl[1])   # 도착할 블럭의 위치는 2층, 이미 도착한 블럭의 센터 찾는 함수로 변경 
    stage2_xy = (max_pxl[0], max_pxl[1])

    # 블럭 들고, 로봇 이동
    run_point(move, [start_x, start_y, start_z+1.1, 115])    # 블럭 위치로 손 이동
    wait_arrive([start_x, start_y, start_z+1.1, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 1)    # 자력 발동

    run_point(move, [end_x, end_y, -30, 115])    # 목적지로 이동
    # run_point(move, [(start_x+end_x)/2, (start_y+end_y)/2, start_z+5, 115])    # 목적지로 이동
    # wait_near([(start_x+end_x)/2, (start_y+end_y)/2, start_z+5, 115])   # move softly
    wait_arrive([end_x, end_y, -30, 115])    # 도착까지 대기

    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거

    # run_point(move, [end_x, end_y, end_z+2.3, 115])    # 떨어뜨리기
    # wait_arrive([end_x, end_y, end_z+2.3, 115])    # 도착까지 대기
    # gripper_DO(dashboard, gripper_port, 0)    # 자력 제거

    # run_point(move, [end_x, end_y, -15, 115])    # 목적지로 이동
    # gripper_DO(dashboard, gripper_port, 0)    # 자력 제거
    # wait_arrive([end_x, end_y, -15, 115])    # 도착까지 대기
    # gripper_DO(dashboard, gripper_port, 0)    # 자력 제거



    ## 3단계: 1층 센터 인식 후, 해당색깔로 이동

    color_center = picture()
    print(color_center)
    max_pxl = None

    # 블록 순서 추가
    while_check = True
    while (while_check):
        for each in color_center:
            check = True
            for color in block_seq:
                if (color[2] == each[2]):
                    check = False
            if (check):
                block_seq.append(each)
                while_check = False
                break
        if (while_check):
            run_point(move, point_home)
            color_center = picture()

    # print(color_center)
    if block_seq is not None and len(block_seq) > 0:
        max_pxl = return_destination_pxl(block_seq[-1])
        print(max_pxl)

    # 시작지점, 끝지점 알아내기
    start_layer, end_layer = 1, 1
    start_x, start_y, start_z = pixel2xyz(start_layer, block_seq[-1][0], block_seq[-1][1])   # 현재 2층에 놓여있는 블럭의 위치
    end_x, end_y, end_z = pixel2xyz(end_layer, max_pxl[0], max_pxl[1])   # 도착할 블럭의 위치는 2층, 이미 도착한 블럭의 센터 찾는 함수로 변경
    stage3_xy = (max_pxl[0], max_pxl[1])

    # 블럭 들고, 로봇 이동
    run_point(move, [start_x, start_y, start_z+1.1, 115])    # 블럭 위치로 손 이동
    wait_arrive([start_x, start_y, start_z+1.1, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 1)    # 자력 발동

    # run_point(move, [start_x, start_y, start_z+40, 115])    # 블럭 위치로 손 이동
    # wait_near([start_x, start_y, start_z+40, 115])   # move softly
    # wait_arrive([start_x, start_y, start_z+40, 115])    # 도착까지 대기

    run_point(move, [start_x, start_y, -13, 115])    # 목적지로 이동
    wait_arrive([start_x, start_y, -13, 115])    # 도착까지 대기

    run_point(move, [end_x, end_y, end_z+2.3, 115])    # 떨어뜨리기(첫 state에서만 끄기)
    wait_arrive([end_x, end_y, end_z+2.3, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거

    # run_point(move, [end_x, end_y, -15, 115])    # 목적지로 이동
    # wait_arrive([end_x, end_y, -15, 115])    # 도착까지 대기
    # gripper_DO(dashboard, gripper_port, 0)    # 자력 제거


    ## 4단계: 두번째 블럭과 세번째 블럭의 센터위치 인식 & 두번째 블럭을 세번째 블럭의 위(2층)로 이동
    
    color_center = picture()
    print(color_center)
    max_pxl = None

    c_list = []
    for each in color_center:
        c_list.append(each[2])
    while (len(color_center) == 0 or not(block_seq[1][2] in c_list)):
        run_point(move, point_home)
        wait_arrive(point_home)
        color_center = picture()
        c_list = []
        for each in color_center:
            c_list.append(each[2])

    c_list = []
    pix_x, pix_y = None, None
    for each in color_center:
        c_list.append(each[2])
    for i in range(len(c_list)):
        if (block_seq[1][2] == c_list[i]):
            pix_x, pix_y = color_center[i][0], color_center[i][1]

    # 시작지점, 끝지점 알아내기
    start_layer, end_layer = 1, 2
    # start_x, start_y, start_z = pixel2xyz(start_layer, stage2_xy[0], stage2_xy[1])
    start_x, start_y, start_z = pixel2xyz(start_layer, pix_x, pix_y)
    end_x, end_y, end_z = pixel2xyz(end_layer, stage3_xy[0], stage3_xy[1])


    # 블럭 들고, 로봇 이동
    run_point(move, [start_x, start_y, start_z+1.1, 115])    # 블럭 위치로 손 이동
    wait_arrive([start_x, start_y, start_z+1.1, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 1)    # 자력 발동
    
    run_point(move, [(start_x+end_x)/2, (start_y+end_y)/2, end_z+2.3, 115])    # 목적지로 이동
    wait_arrive([(start_x+end_x)/2, (start_y+end_y)/2, end_z+2.3, 115])    # 도착까지 대기
    # run_point(move, [start_x, start_y, end_z+2.3, 115])    # 목적지로 이동
    # wait_near([start_x, start_y, end_z+2.3, 115])   # move softly
    # wait_arrive([start_x, start_y, end_z+2.3, 115])    # 도착까지 대기
    
    run_point(move, [end_x, end_y, end_z+2.3, 115])    # 떨어뜨리기
    wait_arrive([end_x, end_y, end_z+2.3, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거
    
    run_point(move, [end_x, end_y, -20, 115])    # 목적지로 이동
    wait_arrive([end_x, end_y, -20, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거

    ## 5단계: 첫번째 블럭과 두번째 블럭의 센터위치 인식 & 첫번째 블럭을 두번째 블럭의 위(3층)로 이동

    color_center = picture()
    print(color_center)
    max_pxl = None

    c_list = []
    for each in color_center:
        c_list.append(each[2])
    while (len(color_center) == 0 or not(block_seq[0][2] in c_list)):
        run_point(move, point_home)
        wait_arrive(point_home)
        color_center = picture()
        c_list = []
        for each in color_center:
            c_list.append(each[2])

    c_list = []
    pix_x, pix_y = None, None
    for each in color_center:
        c_list.append(each[2])
    for i in range(len(c_list)):
        if (block_seq[0][2] == c_list[i]):
            pix_x, pix_y = color_center[i][0], color_center[i][1]

    # 시작지점, 끝지점 알아내기
    start_layer, end_layer = 1, 3
    # start_x, start_y, start_z = pixel2xyz(start_layer, stage1_xy[0], stage1_xy[1])
    start_x, start_y, start_z = pixel2xyz(start_layer, pix_x, pix_y)
    end_x, end_y, end_z = pixel2xyz(end_layer, stage3_xy[0], stage3_xy[1])

    # 블럭 들고, 로봇 이동
    gripper_DO(dashboard, gripper_port, 1)    # 자력 발동
    run_point(move, [start_x, start_y, start_z+1.1, 115])    # 블럭 위치로 손 이동
    wait_arrive([start_x, start_y, start_z+1.1, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 1)    # 자력 발동
    
    run_point(move, [(start_x+end_x)/2, (start_y+end_y)/2, end_z+2.3, 115])    # 목적지로 이동
    wait_arrive([(start_x+end_x)/2, (start_y+end_y)/2, end_z+2.3, 115])    # 도착까지 대기
    # run_point(move, [start_x, start_y, end_z+2.3, 115])    # 목적지로 이동
    # wait_near([start_x, start_y, end_z+2.3, 115])   # move softly
    # wait_arrive([start_x, start_y, end_z+2.3, 115])    # 도착까지 대기
    
    run_point(move, [end_x, end_y, end_z+2.3, 115])    # 떨어뜨리기(첫 state에서만 끄기)
    wait_arrive([end_x, end_y, end_z+2.3, 115])    # 도착까지 대기
    
    run_point(move, [end_x, end_y, -23, 115])    # 목적지로 이동
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거
    wait_arrive([end_x, end_y, -23, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거

    run_point(move, [end_x, end_y, -21, 115])    # 목적지로 이동
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거
    wait_arrive([end_x, end_y, -21, 115])    # 도착까지 대기
    gripper_DO(dashboard, gripper_port, 0)    # 자력 제거


    # 로봇 끄기
    dashboard.DisableRobot()

    return

color_center = None

g_source, g_device, model, stride, names, pt, g_imgsz, dataset, bs, dt, pred, annotator = [0] * 12
g_augment, g_visualize, g_conf_thres, g_iou_thres, g_classes, g_agnostic_nms, g_max_det, g_line_thickness = [0] * 8

layer1_model, layer2_model, layer3_model = return_models()
current_actual = None

opt = parse_opt()

# 사진 찍기
# check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
opt_dict = vars(opt)  # Convert opt to a dictionary
# Create a thread with the `run` function and unpacked keyword arguments
thread1 = threading.Thread(target=run, kwargs=opt_dict)
thread1.start()
# run(**vars(opt))

main()






















