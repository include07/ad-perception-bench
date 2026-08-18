.PHONY: smoke eval fps onnx cpp

smoke:  ## 2-min sanity check on coco128
	python -m src.evaluate --data coco128.yaml --model yolov8n.pt

eval:   ## real numbers on COCO val2017 (downloads COCO once — ~20GB incl. train2017)
	python -m src.evaluate --data coco.yaml --model yolov8n.pt

fps:    ## inference FPS on this machine
	python -m src.bench_fps --model yolov8n.pt --device mps

onnx:   ## export yolov8n to ONNX for the C++ node
	yolo export model=yolov8n.pt format=onnx imgsz=640

cpp:    ## build the C++ inference node (needs: brew install opencv cmake)
	cmake -S cpp -B cpp/build && cmake --build cpp/build
