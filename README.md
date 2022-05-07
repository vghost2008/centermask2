#CenterMask2

#运行Demo

使用摄像头作为输入源

```
python boedemo/video_demo.py 
```

指定摄像头编号

```
python boedemo/video_demo.py 1
```

使用文件作为输入源

```
python boedemo/video_demo.py ~/0day/test3p.webm
```

#测试环境

- cuda 11.1
- cudnn 11.4
- opencv 4.5.2
- OS: ubuntu 18.04/windows10
- GPU: 3090TI
- FPS: 35
- libtorch 1.9.0
