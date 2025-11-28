# graphcast

## 论文

`GraphCast: Learning skillful medium-range global weather forecasting`

* https://arxiv.org/pdf/2212.12794

## 模型结构

网络是包含`编码-处理-解码`的GNN网络。

![alt text](asset/model_structure.png)

## 算法原理

基于GNN的学习模拟器在学习和模拟流体和其他材料的复杂物理动力学方面非常有效，因为它们的表示和计算结构类似于学习的有限元求解器。GNN的一个关键优势是，输入图的结构决定了通过学习消息传递相互作用的表示的哪些部分，允许任意范围的空间交互模式。

![alt text](asset/alg.png)

## 环境配置

### Docker（方法一）

    docker pull image.sourcefind.cn:5000/dcu/admin/base/jax:0.4.23-ubuntu20.04-dtk24.04-py310
    
    docker run --shm-size 10g --network=host --name=graphcast --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 项目地址(绝对路径):/home/ -v /opt/hyhal:/opt/hyhal:ro -it <your IMAGE ID> bash
    
    pip install -r requirements.txt --constraint constraints.txt
    pip install -e .
    pip uninstall shapely
    pip install shapely
    
    pip install --upgrade google-api-python-client
    pip install google.cloud.bigquery
    pip install google.cloud.storage


### Dockerfile（方法二）

    docker build -t <IMAGE_NAME>:<TAG> .
    
    docker run --shm-size 10g --network=host --name=graphcast --privileged --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v 项目地址(绝对路径):/home/ -v /opt/hyhal:/opt/hyhal:ro -it <your IMAGE ID> bash
    
    pip install -r requirements.txt --constraint constraints.txt
    pip install -e .
    pip uninstall shapely
    pip install shapely
    
    pip install --upgrade google-api-python-client
    pip install google.cloud.bigquery
    pip install google.cloud.storage

### Anaconda（方法三）

    DTK驱动：dtk24.04
    python：python3.10
    jax: 0.4.23

Tips：以上dtk驱动、python、jax等DCU相关工具版本需要严格一一对应

2、其他非特殊库

    pip install -r requirements.txt --constraint constraints.txt
    pip install -e .
    pip uninstall shapely
    pip install shapely
    
    pip install --upgrade google-api-python-client
    pip install google.cloud.bigquery
    pip install google.cloud.storage


​    
## 数据集

[Google Cloud](https://console.cloud.google.com/storage/browser/dm_graphcast) | [SCNet](http://113.200.138.88:18080/aidatasets/project-dependency/dm_graphcast)


注意：该数据集按需下载，在执行`graphcast-jax.ipynb`时，可自动下载示例数据。

## 推理

参考并执行`graphcast-jax.ipynb`。


## result

![alt text](asset/result.png)

### 精度

无

## 应用场景

### 算法类别

`天气预报`

### 热点应用行业

`气象,交通,环境`

## 预训练权重

[Google Cloud](https://console.cloud.google.com/storage/browser/dm_graphcast) | [SCNet](http://113.200.138.88:18080/aimodels/findsource-dependency/dm_graphcast) 高速下载通道 

## 源码仓库及问题反馈

* https://developer.hpccube.com/codes/modelzoo/graphcast_jax

## 参考资料

* https://github.com/google-deepmind/graphcast

