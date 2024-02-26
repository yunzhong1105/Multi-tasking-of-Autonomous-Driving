python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch-size 48 --conf configs/yolov6m6_finetune.py --data data/seg.yaml --img 1280 --device 0,1,2,3 --resume "yolov6/runs/finetune_output" --segonly=True --eval-final-only --epochs 540 --freeze_backbone 0

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 4 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 1280 --device 0,1 --detonly=True --eval-final-only --epochs 540 --freeze_backbone 0

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 4 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 1280 --device 0,1 --resume "runs/train/exp89/weights/last_ckpt.pt" --detonly=True --eval-final-only --epochs 540 --freeze_backbone 0



python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 4 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 1280 --device 0,1 --segonly=True --eval-final-only --epochs 300 --freeze_backbone 2

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 4 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 1280 --device 0,1 --resume "runs/train/exp209/weights/last_ckpt.pt" --segonly=True --eval-final-only --epochs 300 --freeze_backbone 2

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 4 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 1280 --device 0,1 --resume "runs/train/exp210/weights/last_ckpt.pt" --segonly=True --eval-final-only --epochs 300 --freeze_backbone 2



python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 8 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 1280 --device 2,3 --segonly=True --eval-final-only --epochs 300 --freeze_backbone 1




# img size 640

# new det training
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --detonly=True --eval-final-only --epochs 500 --freeze_backbone 0

# resume det training
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --resume "runs/train/exp/weights/last_ckpt.pt" --detonly=True --eval-final-only --epochs 500 --freeze_backbone 0

# new seg training
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --device 0,1 --segonly=True --eval-final-only --epochs 500 --freeze_backbone 0

# resume seg training
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --device 0,1 --resume "runs/train/exp/weights/last_ckpt.pt" --segonly=True --eval-final-only --epochs 500 --freeze_backbone 0

# new cls training
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/stanfordcar_cls.yaml --img 640 --device 0,1 --clsonly=True --eval-final-only --epochs 500 --freeze_backbone 0

# resume cls training
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/stanfordcar_cls.yaml --img 640 --device 0,1 --resume "runs/train/exp/weights/last_ckpt.pt" --clsonly=True --eval-final-only --epochs 500 --freeze_backbone 0



# 1 gpu for checking
python tools/train.py --batch-size 8 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --segonly=True --eval-final-only --epochs 500 --freeze_backbone 0

python tools/train.py --batch-size 8 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --detonly=True --eval-final-only --epochs 500 --freeze_backbone 0

python tools/train.py --batch-size 8 --conf configs/yolov6m6_finetune.py --data data/stanfordcar_cls.yaml --img 640 --clsonly=True --eval-final-only --epochs 500 --freeze_backbone 0