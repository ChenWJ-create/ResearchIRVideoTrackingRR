# 模型与源码来源

## TFW 热成像权重

- 官方项目：[IS2AI/TFW](https://github.com/IS2AI/TFW)
- 模型：TFW YOLOv5n-Face
- 官方权重链接：[Google Drive](https://drive.google.com/file/d/1vXk9P3CfhUtRBGI44SqWbuiTJ7rAI4hP/view)
- 本地文件：`weights/yolov5n-face-tfw.pt`
- 文件大小：3,793,465 字节
- SHA-256：`5596275882839ab6e21177cc15572dd56c71c3fcafd2b0ea3b3ffa45d2c2677a`
- TFW 仓库声明许可证：MIT

TFW 权重提供热成像人脸框和 5 个面部关键点，不包含人员身份分类器或人脸特征库。

## YOLOv5-Face 推理源码

- 上游项目：[deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- 固定提交：`152c688d551aefb973b7b589fb0691c93dab3564`
- 本地目录：`vendor/yolov5-face`
- 上游许可证：GPL-3.0

当前实现不修改上游源码，只在 `run_inference.py` 中封装热像读取、位深归一化和结果导出。若后续用于闭源或商业交付，需要单独确认 GPL-3.0 源码依赖和模型权重的分发条件。
