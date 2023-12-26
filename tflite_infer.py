import time
import numpy as np
import tensorflow as tf
# from PIL import Image
import cv2
from tflite_inference.functions import process_image , non_max_suppression , xywh2xyxy , segmap_to_color , rescale , box_convert , plot_box_and_label , generate_colors , CalcFPS
import torch.nn.functional as F
import glob

def tflite_inference(conf_thres, iou_thres, classes, agnostic_nms , max_det , save_seg , save_dir , save_img) :

    fps_calculator = CalcFPS()

    # 載入 tflite 模型
    path = "0417/saved_model/last_ckpt_opset11_0417_float32.tflite" # "./ckpts/teacher0324_18_12_last_ckpt_opset11.tflite"
    interpreter = tf.lite.Interpreter(model_path = path)
    interpreter.allocate_tensors()

    # 取得輸入和輸出 tensor 引用
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 準備輸入數據
    input_shape = input_details[0]['shape']
    print("input shape :" , input_shape)

    #################################################################################
    
    # # 用Pillow讀image
    # # 處理輸入圖片
    # img_folder = "/home/re6101029/MTK2023comp/yolov6/test_inference"
    # img_list = glob.glob(img_folder)
    # # for i in 
    # path = "/home/re6101029/MTK2023comp/yolov6/test_inference/itp_1.jpg"
    # img_src = Image.open(path)  # 假設您有一個名為 test_image.jpg 的 RGB 圖片
    # # print(np.array(img).shape)
    # img = img_src.resize((input_shape[2], input_shape[3]) , resample = Image.BILINEAR)  # 將圖片大小調整為模型所需的大小
    # img_src = np.array(img_src)
    # assert img_src.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
    # # print(np.array(img).shape)
    # img = np.array(img).astype('float32') / 255.0
    # img = img.transpose((2, 0, 1))[::-1]
    # img = np.expand_dims(img , axis = 0)

    #################################################################################

    # 用cv2讀image
    # 處理輸入圖片
    img_folder = "/ssd3/MTK_ODSScomp2023_Dataset/ivslab_test_private_final"
    both = "ivslab_test_private_final_both/jpg/*.jpg"
    det_only = "ivslab_test_private_final_only_detection/ivslab_test_private_final/JPEGImages/All/*.jpg"
    both_path =os.path.join(img_folder , both)
    det_only_path = os.path.join(img_folder , det_only)
    # for i in 

    # n = 0
    # for item in glob.glob(both_path) :
    #     n += 1
    #     print(item)
    # print(n)
    # assert 1 == 2

    path = "/home/re6101029/MTK2023comp/yolov6/test_inference/itp_1.jpg"
    img_src = cv2.imread(path)  # 假設您有一個名為 test_image.jpg 的 RGB 圖片
    # print(np.array(img).shape)
    img , img_src = process_image(img_src , False)
    print(img.shape)
    # img = img_src.resize((input_shape[2], input_shape[3]) , resample = Image.BILINEAR)  # 將圖片大小調整為模型所需的大小
    # img_src = np.array(img_src)
    assert img.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
    # print(np.array(img).shape)
    # img = np.array(img).astype('float32') / 255.0
    # img = img.transpose((2, 0, 1))[::-1]
    # img = np.expand_dims(img , axis = 0)

    #################################################################################

    # print(img.shape)
    # input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0).transpose(1 , 3 , 1280 , 1280)
    # print(input_data.shape)

    # path = "/home/re6101029/MTK2023comp/yolov6/test_inference/itp_1.jpg"
    # img_src = cv2.imread(path)
    # print("img_src shape :" , img_src.shape)
    # print(img_src)
    # img , img_src = process_image(img_src , img_size = [1280 , 1280] , stride = 64 , half = False)
    # print("img shape" , img.shape)
    # if len(img.shape) == 3:
    #     img = img[None]
    
    # print("img shape :" , img.shape)
    # print("img_src shape :" , img_src.shape)

    # print("input detail :" , interpreter.get_input_details())
    # print("output detail :" , interpreter.get_output_details())


    t1 = time.time()
    # 將數據傳遞給輸入 tensor
    interpreter.set_tensor(input_details[0]['index'] , img)

    # 執行推論
    interpreter.invoke()

    # 取得輸出結果
    segmap = interpreter.get_tensor(output_details[0]['index'])
    pred_results = interpreter.get_tensor(output_details[1]['index'])

    segmap = torch.from_numpy(segmap)
    pred_results = torch.from_numpy(pred_results)

    # print(type(segmap))
    # print(segmap.shape)

    # print("#"*80)
    # print(type(pred_results))
    # print(pred_results.shape)
    # print(pred_results)
    # print("#"*80)
    # assert 1 == 2


    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    # segmap = F.interpolate(segmap , size=(img.shape[:2]), mode='bilinear', align_corners=True)
    t2 = time.time()

    print("*"*20 , "\n" , "det")
    print(det.shape)
    print(det)
    # print("*"*20 , "\n" , "segmap")
    # print(segmap.shape)
    # print(segmap)

    if save_seg:
        segmap = segmap[0].cpu().numpy()
        segmap = np.argmax(segmap,0)
        # print(os.getcwd())
        # cv2.imwrite('test_inference/output/'+os.path.basename(path).replace('jpg', 'png'), segmap.astype(np.uint8))
        # color_pred_segmap = segmap_to_color(segmap)
    
    # Create output files in nested dirs that mirrors the structure of the images' dirs
    rel_path = "."
    save_path = "test_inference/output/tflite_infer/last_ckpt_opset11_0417_itp_1.jpg"
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
    # print("det: " , det)

    if len(det):
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round() # 
        for *xyxy, conf, cls in reversed(det):
            # xyxy = np.array(xyxy).astype(np.int16)
            
            if save_txt:  # Write to file -> False
                xywh = (box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf)
                with open(txt_path+'.txt', 'a') as f:
                    f.write(osp.splitext(osp.basename(img_path))[0]+'.jpg ')
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img:
                class_num = int(cls)  # integer class
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                print("#"*80)
                print(len(xyxy))
                print(xyxy)
                print("#"*80)
                plot_box_and_label(img_src, max(round(sum(img_src.shape) / 2 * 0.003), 2), xyxy, label, color = generate_colors(class_num, True))

        img_src = np.asarray(img_src)

    fps_calculator.update(1.0 / (t2 - t1))
    avg_fps = fps_calculator.accumulate()
    print("avg_fps :\n" , avg_fps)

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

conf_thres = 0.2 # 0.4
iou_thres = 0.7 # 0.45
classes = None
agnostic_nms = False
max_det = 1000
save_seg = False
save_dir = "/home/re6101029/MTK2023comp/yolov6/test_inference/output"
save_img = True


tflite_inference(conf_thres, iou_thres, classes, agnostic_nms , max_det , save_seg , save_dir , save_img)