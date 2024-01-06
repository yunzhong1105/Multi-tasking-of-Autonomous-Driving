python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --detonly=True --eval-final-only --epochs 500 --freeze_backbone 0

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --segonly=True --eval-final-only --epochs 520 --freeze_backbone 1

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --detonly=True --eval-final-only --epochs 540 --freeze_backbone 1

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --segonly=True --eval-final-only --epochs 560 --freeze_backbone 1

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --detonly=True --eval-final-only --epochs 580 --freeze_backbone 1

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --segonly=True --eval-final-only --epochs 600 --freeze_backbone 2

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --detonly=True --eval-final-only --epochs 620 --freeze_backbone 2

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/bdd100k_seg.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --segonly=True --eval-final-only --epochs 640 --freeze_backbone 2

python -m torch.distributed.launch --nproc_per_node 2 tools/train.py --batch-size 16 --conf configs/yolov6m6_finetune.py --data data/TT_100K_det.yaml --img 640 --device 0,1 --resume "runs/train/exp20/weights/last_ckpt.pt" --skip True --detonly=True --eval-final-only --epochs 660 --freeze_backbone 2