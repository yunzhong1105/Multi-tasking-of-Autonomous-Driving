python tools/infer.py --weights best_model/teacher0324_18_12_last_ckpt.pt --img 1280 1280 --source test_inference/itp_5.jpg

python tools/infer.py --weights runs/train/exp71/weights/last_ckpt.pt --yaml data/TT_100K_det.yaml --img 1280 1280 --source cvpr_infer/images/det/2.jpg --save-dir cvpr_infer/output/det

python tools/infer.py --weights runs/train/exp4/weights/last_ckpt.pt --yaml data/bdd100k_seg.yaml --img 1280 1280 --source cvpr_infer/images/seg --save-dir cvpr_infer/output/seg --save-seg

python tools/infer.py --weights runs/train/exp4/weights/last_ckpt.pt --yaml data/bdd100k_seg.yaml --img 1280 1280 --source cvpr_infer/images/seg/0a0a0b1a-7c39d841.jpg --save-dir cvpr_infer/output/seg --save-seg