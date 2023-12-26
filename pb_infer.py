import time
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tflite_inference.functions import non_max_suppression , xywh2xyxy , segmap_to_color , rescale , box_convert , plot_box_and_label , generate_colors , CalcFPS
import torch.nn.functional as F
import glob

def pb_inference(conf_thres, iou_thres, classes, agnostic_nms , max_det , save_seg , save_dir , save_img) :

    fps_calculator = CalcFPS()

    # 載入.pb模型
    model_path = '/home/re6101029/MTK2023comp/yolov6/ckpts/teacher0324_18_12_last_ckpt_opset11'
    model = tf.saved_model.load(model_path)
    signature = list(model.signatures.values())[0]
    print(signature.structured_outputs.keys() , "\n" , "*"*30)

    # 獲取輸入張量
    # input_tensor_name = model.signature_def['serving_default'].inputs['input_tensor'].name
    # input_tensor = model.graph.get_tensor_by_name(input_tensor_name)
    inputs = model.signatures['serving_default'].inputs

    # 獲取輸出張量
    # output_tensor_name = model.signature_def['serving_default'].outputs['output_tensor'].name
    # output_tensor = model.graph.get_tensor_by_name(output_tensor_name)
    outputs = model.signatures['serving_default'].outputs


    #################################################################################
    
    input_shape = [1 , 3 , 1280 , 1280]
    # 處理輸入圖片
    img_folder = "/home/re6101029/MTK2023comp/yolov6/test_inference"
    img_list = glob.glob(img_folder)
    # for i in 
    path = "/home/re6101029/MTK2023comp/yolov6/test_inference/itp_1.jpg"
    img_src = Image.open(path)  # 假設您有一個名為 test_image.jpg 的 RGB 圖片
    # print(np.array(img).shape)
    img = img_src.resize((input_shape[2], input_shape[3]) , resample = Image.BILINEAR)  # 將圖片大小調整為模型所需的大小
    img_src = np.array(img_src)
    assert img_src.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
    # print(np.array(img).shape)
    img = np.array(img).astype('float32') / 255.0
    img = img.transpose((2, 0, 1))[::-1]
    img = np.expand_dims(img , axis = 0)
    print("img shape:\n" , img.shape)

    t1 = time.time()
    infer = model.signatures['serving_default']

    # 執行推理
    input_data = ...  # 填入您的輸入數據
    output_data = infer(images = tf.constant(img)) # ['output_tensor'].numpy()
    print("*"*30 , "\n" , output_data["outputs"].shape , "\n" , "*"*30)
    print(output_data["1677"].shape , "\n" , "*"*30)

    # 取得輸出結果
    segmap = output_data["1677"]
    pred_results = output_data["outputs"].numpy()

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
    save_path = "test_inference/output/pb_infer/0408_itp_1.jpg"
    #txt_path = osp.join(save_dir, rel_path, osp.splitext(osp.basename(img_path))[0])
    txt_path = os.path.join(save_dir, "submission")
    # os.makedirs(os.path.join(save_dir, rel_path), exist_ok=True)
    
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


pb_inference(conf_thres, iou_thres, classes, agnostic_nms , max_det , save_seg , save_dir , save_img)