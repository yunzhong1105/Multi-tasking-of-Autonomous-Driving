python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/seg.yaml --img 1280 --device 0,1,2,3 --resume "yolov6/runs/finetune_output" --segonly=True --eval-final-only --epochs 540 --freeze_backbone 0

# python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/dataset.yaml --img 1280 --device 0,1,2,3 --resume --detonly=True --eval-final-only --epochs 560 --freeze_backbone 0
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/dataset.yaml --img 1280 --device 0,1,2,3 --resume "yolov6/runs/finetune_output" --detonly=True --eval-final-only --epochs 560 --freeze_backbone 0


python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/seg.yaml --img 1280 --device 0,1,2,3 --resume --segonly=True --eval-final-only --epochs 580 --freeze_backbone 2

python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/dataset.yaml --img 1280 --device 0,1,2,3 --resume --detonly=True --eval-final-only --epochs 600 --freeze_backbone 2


python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/dataset.yaml --img 1280 --device 0,1,2,3 --resume --detonly=True --eval-final-only --epochs 620 --freeze_backbone 1

python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/seg.yaml --img 1280 --device 0,1,2,3 --resume --segonly=True --eval-final-only --epochs 640 --freeze_backbone 1

# python tools/train.py --batch_size 4 --conf configs/yolov6m6_finetune.py --data data/dataset.yaml --img 1280 --device 0 --resume --detonly=True

# #Evaluation for det (see dataset.yaml val)
# python tools/eval.py --data data/dataset.yaml --batch 1 --weights /work/u1657859/DCSNv2/ODSEG23/yolov6/runs/train/exp60/weights/last_ckpt.pt --task val --img 1280 --device 0 --detonly True

# #Evaluation for seg (see dataset.yaml val)
# python tools/eval.py --data data/segval.yaml --batch 1 --weights /work/u1657859/DCSNv2/ODSEG23/yolov6/runs/train/exp60/weights/last_ckpt.pt --task val --img 1280 --device 0 --segonly=True


# #Make submission for det 
# python tools/infer.py --weights runs/train/exp60/weights/last_ckpt.pt --img 1280 1280 --source ../data/testpriv/det --save-txt --save-dir det_ssub

# +
#Make submission for det python tools/infer.py --weights /work/u1657859/DCSNv2/ODSEG23/yolov6/runs/train/exp60/weights/last_ckpt.pt --img 1280 1280 --source /work/u1657859/DCSNv2/ODSEG23/data/testpriv/det/ --save-txt --save-dir sub/
# -

# #Make submission for seg 
# python tools/infer.py --weights runs/train/exp60/weights/last_ckpt.pt --img 1280 1280 --source /work/u1657859/DCSNv2/ODSEG23/data/testpriv/seg --save-dir ./sub/image

#Make submission for seg 
python tools/eval.py --data data/segval.yaml --batch 1 --weights /work/u1657859/DCSNv2/ODSEG23/yolov6/runs/train/exp60/weights/last_ckpt.pt --task val --img 1280 --device 0 --segonly=True
