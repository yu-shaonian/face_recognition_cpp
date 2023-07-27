### 安装配置libtorch，opencv

```
参考这篇博客
https://blog.csdn.net/u013250861/article/details/127829590
opencv版本3.4.12
libtorch版本1.9.1cpu版本
```

## 使用方式

```
cd face_recognition_cpp
mkdir build
cd build
cmake ..
make
./demo ../centerface.onnx ../face.jpg
```

