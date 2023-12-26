import time
import numpy as np
# import tensorflow as tf
from PIL import Image
import cv2
from tflite_inference.functions import process_image_onnx , non_max_suppression , xywh2xyxy , segmap_to_color , rescale , box_convert , plot_box_and_label , generate_colors , CalcFPS
import torch.nn.functional as F
import glob
import onnxruntime as ort

def onnx_inference(conf_thres, iou_thres, classes, agnostic_nms , max_det , save_seg , save_dir , save_img) :

    fps_calculator = CalcFPS()
    model_path = 'ckpts/teacher0324_18_12_last_ckpt_opset12.onnx'
    session = ort.InferenceSession(model_path)

    img_path = "/home/re6101029/MTK2023comp/yolov6/test_inference/itp_1.jpg"
    img_src = cv2.imread(img_path)
    img , img_src = process_image_onnx(img_src , False)
    # img_src = Image.open(img_path)
    # img = img_src.copy().convert('RGB').resize((1280 , 1280), resample=Image.BILINEAR)
    # img_src = np.array(img_src)
    # print("img_src shape :" , img_src.shape)
    # img = np.array(img).transpose((2, 0, 1)).astype(np.float32)
    # img = np.expand_dims(img, axis=0)
    # img /= 255.0
    # img -= np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    # img /= np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    
    print(type(img))
    print(img.shape)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})

    # print("*"*30 , "\n" , outputs , "\n" , "*"*30)
    output_data_od = outputs[0]
    output_data_ss = outputs[1]

    print("od type :" , type(output_data_od))
    print("ss type :" , type(output_data_ss))



    t1 = time.time()

    # 取得輸出結果
    segmap = output_data_ss
    pred_results = output_data_od

    # segmap = torch.from_numpy(segmap)
    # pred_results = torch.from_numpy(pred_results)

    pred_results = torch.tensor(pred_results)

    print(type(segmap))
    print(segmap.shape)    

    print(type(pred_results))
    print(pred_results.shape)


    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    # segmap = F.interpolate(segmap , size=(img.shape[:2]), mode='bilinear', align_corners=True)
    t2 = time.time()

    print("*"*20 , "\n" , "det")
    print(det.shape)
    print(det)

    if save_seg:
        segmap = segmap[0].cpu().numpy()
        segmap = np.argmax(segmap,0)
        print(os.getcwd())
        # cv2.imwrite('test_inference/output/'+os.path.basename(path).replace('jpg', 'png'), segmap.astype(np.uint8))
        # color_pred_segmap = segmap_to_color(segmap)
    
    # Create output files in nested dirs that mirrors the structure of the images' dirs
    rel_path = "."
    save_path = "test_inference/output/onnx_infer/0413_itp_1.jpg"
    #txt_path = osp.join(save_dir, rel_path, osp.splitext(osp.basename(img_path))[0])
    txt_path = os.path.join(save_dir, "submission")
    # os.makedirs(os.path.join(save_dir, rel_path), exist_ok=True)

    # fps_calculator.update(1.0 / (t2 - t1))
    # avg_fps = fps_calculator.accumulate()
    # print(avg_fps)

    # print("output : \n" , output_data)
    # print("output shape : \n" , output_data.shape)
    
    save_txt = False
    hide_labels = False
    hide_conf = False
    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]
    class_names = ['vehicle', 'pedestrian', 'scooter', 'bicycle']

    if len(det):
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # Write to file -> False
                xywh = (box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf)
                with open(txt_path+'.txt', 'a') as f:
                    f.write(osp.splitext(osp.basename(img_path))[0]+'.jpg ')
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img:
                class_num = int(cls)  # integer class
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')

                plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color = generate_colors(class_num, True))

        img_src = np.asarray(img_src)

    fps_calculator.update(1.0 / (t2 - t1))
    avg_fps = fps_calculator.accumulate()
    print(avg_fps)

    if save_img:
        # if self.files.type == 'image':
        cv2.imwrite(save_path , img_src)
        # else:  # 'video' or 'stream'
        #     if vid_path != save_path:  # new video
        #         vid_path = save_path
        #         if isinstance(vid_writer, cv2.VideoWriter):
        #             vid_writer.release()  # release previous video writer
        #         if vid_cap:  # video
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #         else:  # stream
        #             fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
        #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #     vid_writer.write(img_src)

import faulthandler
faulthandler.enable()
import os
import torch

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

conf_thres = 0.4
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000
save_seg = False
save_dir = "/home/re6101029/MTK2023comp/yolov6/test_inference/output"
save_img = True


onnx_inference(conf_thres, iou_thres, classes, agnostic_nms , max_det , save_seg , save_dir , save_img)