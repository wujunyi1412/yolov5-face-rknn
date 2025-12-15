# YOLOv5-Face RKNN 推理 Demo

本项目为 [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) 的 RKNN 适配版本，基于原作者的开源工作进行开发。  
感谢原作者的贡献，并遵循原项目的开源许可（请参考原项目 LICENSE）。  

## 推理流程

### 1. 导出 ONNX 模型

```bash
cd yolov5-face-rknn

python export.py --weights <your_model>.pt --onnx_rknn

python -m onnxsim <your_model>.onnx <your_model>_sim.onnx
```

### 2. 下载瑞芯微推理demo 
[https://github.com/airockchip/rknn_model_zoo/tree/v2.3.2](https://github.com/airockchip/rknn_model_zoo/tree/v2.3.2)

```bash
cp -r yolov5-face-rknn/rknn_cpp/yolov5_face rknn_model_zoo-2.3.2/examples/

cp yolov5-face-rknn/xxx_sim.onnx rknn_model_zoo-2.3.2/examples/yolov5_face/model/
```
### 3. onnx转rknn：
```bash
cd rknn_model_zoo-2.3.2/examples/yolov5_face/python/

python convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>

例如：    

python convert.py ../model/xxx_sim.onnx rk3588
```
### 4. 模型编译
```bash
cd rknn_model_zoo-2.3.2

export GCC_COMPILER=<GCC_COMPILER_PATH>  # 这里是自己的编译链

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d yolov5_face  

例如：    

./build-linux.sh -t rk3588 -a aarch64 -d yolov5_face
  
编译好的文件都在install里面
```
### 5. 板上运行
```bash
cd /userdata/rknn_yolov5_demo

export LD_LIBRARY_PATH=./lib

./rknn_yolov5_demo model/yolov5.rknn model/bus.jpg
```





