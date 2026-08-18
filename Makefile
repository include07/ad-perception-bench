.PHONY: smoke eval fps

smoke:  ## 2-min sanity check on coco128
	python -m src.evaluate --data coco128.yaml --model yolov8n.pt

eval:   ## real numbers on COCO val2017 (downloads ~1GB once)
	python -m src.evaluate --data coco.yaml --model yolov8n.pt

fps:    ## inference FPS on this machine
	python -m src.bench_fps --model yolov8n.pt --device mps
