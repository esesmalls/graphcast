# inference 气象预测模型推理代码解析

## 一、代码整体功能概述
该代码实现了DeepMind提出的**GraphCast**气象预测模型的推理与可视化功能，核心包括：
1. 命令行参数解析，支持指定预训练模型路径、数据集、可视化变量等；
2. 加载GraphCast预训练模型或初始化随机模型参数；
3. 读取NetCDF格式的气象数据集，提取模型输入、目标和强制变量；
4. 执行自回归滚动预测（rollout）；
5. 可视化预测结果、真实目标值及两者的差值，并保存图片。

代码基于JAX/Haiku框架实现，充分利用了JAX的自动微分、向量化和JIT编译特性，同时结合Xarray处理气象数据的多维数组结构。

## 二、依赖库与核心模块说明
### 2.1 核心依赖库
| 库名 | 功能用途 |
|------|----------|
| `jax`/`haiku (hk)` | 深度学习框架，支持自动微分、JIT编译、向量化计算，是GraphCast的核心实现框架 |
| `xarray` | 处理带标签的多维气象数据（NetCDF格式），替代NumPy/Pandas实现气象数据的便捷索引 |
| `netCDF4` | 读取NetCDF格式的气象数据集 |
| `matplotlib` | 可视化气象变量的预测结果、真实值及差值 |
| `graphcast` | DeepMind开源的GraphCast模型核心模块（含模型结构、配置、数据处理等） |
| `dataclasses`/`typing` | 类型注解与数据类，规范配置参数的结构 |
| `argparse` | 命令行参数解析 |

### 2.2 GraphCast核心子模块
- `graphcast.graphcast`：定义ModelConfig/TaskConfig配置类和GraphCast模型主体；
- `graphcast.autoregressive`：实现自回归预测的封装；
- `graphcast.rollout`：提供分块预测（chunked_prediction）实现长序列滚动推理；
- `graphcast.normalization`：对气象数据进行标准化处理；
- `graphcast.checkpoint`：加载预训练模型的检查点；
- `graphcast.data_utils`：提取模型输入、目标和强制变量的工具函数。

## 三、代码结构与关键函数解析
### 3.1 命令行参数解析（`argsparser`/`print_arguments`）
```python
def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="./", help="pretrained model")
    parser.add_argument("--dataset", type=str, help="input data")
    parser.add_argument("--mode", type=str, help="the ways of getting model params")
    parser.add_argument("--var", type=str, default="2m_temperature", help="visualizing atmospheric variables")
    parser.add_argument("--level", type=int, default=500, help="atmospheric pressure level")
    return parser
```
**功能**：定义5个核心命令行参数，分别指定预训练模型路径、数据集路径、参数加载方式（Random/Checkpoint）、可视化变量（如2米气温）、气压层（如500hPa）。

**`print_arguments`**：按字母序打印参数配置，方便调试时确认参数是否正确。

### 3.2 数据处理工具函数
#### 3.2.1 `parse_file_parts`
```python
def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))
```
**功能**：解析数据集文件名的结构化信息（如分辨率、气压层数、数据源），返回键值对字典。
**示例**：若文件名为`res-0.25_levels-13_source-era5.nc`，则解析为`{"res": "0.25", "levels": "13", "source": "era5"}`。

#### 3.2.2 `select`
```python
def select(data: xarray.Dataset, variable: str, level: Optional[int] = None, max_steps: Optional[int] = None) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims: data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data
```
**功能**：从Xarray数据集中筛选指定的气象变量、气压层和时间步，移除batch维度（取第0个样本）。
**关键技巧**：使用Xarray的`isel`（按索引选择）和`sel`（按标签选择）实现多维数据的便捷筛选，比NumPy的切片更直观。

#### 3.2.3 `scale`
```python
def scale(data: xarray.Dataset, center: Optional[float] = None, robust: bool = False) -> tuple:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax), ("RdBu_r" if center is not None else "viridis"))
```
**功能**：对气象数据进行可视化前的归一化，返回归一化后的数据、Matplotlib的Normalize对象和配色方案。
**核心算法/技巧**：
- **稳健百分位归一化**：当`robust=True`时，使用2%和98%分位数替代最小值/最大值，避免异常值影响可视化；
- **中心对称归一化**：当`center`不为None时，保证vmin和vmax关于center对称，适合展示差值（如预测-真实）；
- **配色方案选择**：差值使用`RdBu_r`（红-蓝），绝对量使用`viridis`（彩虹色）。

### 3.3 可视化函数
#### 3.3.1 `plot_data`
```python
def plot_data(data: dict[str, xarray.Dataset], fig_title: str, plot_size: float = 5, robust: bool = False, cols: int = 4) -> tuple:
    # 计算子图行列数、创建画布
    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
    # 遍历数据绘制子图
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(plot_data.isel(time=0, missing_dims="ignore"), norm=norm, origin="lower", cmap=cmap)
        # 添加颜色条
        plt.colorbar(mappable=im, ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.75, cmap=cmap, extend=("both" if robust else "neither"))
        plt.savefig(f"{fig_title}.png")
```
**功能**：绘制多子图（真实值、预测值、差值），保存可视化图片。
**关键技巧**：
- 动态计算子图行列数，适配任意数量的可视化项；
- 使用`missing_dims="ignore"`忽略不存在的维度（如部分变量无time维度）；
- 隐藏坐标轴刻度，聚焦气象变量的空间分布；
- 颜色条的`extend`参数：稳健归一化时显示超出范围的极值（both）。

#### 3.3.2 `save_var_diff`
```python
def save_var_diff(eval_targets, predictions, plot_pred_variable, plot_pred_level, plot_max_steps=1):
    data = {
        "Targets": scale(select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps), robust=True),
        "Predictions": scale(select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps), robust=True),
        "Diff": scale((select(eval_targets, ...) - select(predictions, ...)), robust=True, center=0),
    }
    fig_title = plot_pred_variable + (f"_at_{plot_pred_level}_hPa" if "level" in predictions[plot_pred_variable].coords else "")
    plot_data(data, fig_title, plot_size, True)
```
**功能**：封装可视化逻辑，生成真实值、预测值、差值的字典，调用`plot_data`完成绘图。
**核心设计**：差值的`center=0`保证红-蓝配色对称，直观展示预测偏高（红）或偏低（蓝）的区域。

### 3.4 模型构建与JIT编译
#### 3.4.1 `construct_wrapped_graphcast`
```python
def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
    # 初始化GraphCast模型
    predictor = graphcast.GraphCast(model_config, task_config)
    # BFloat16精度转换
    predictor = casting.Bfloat16Cast(predictor)
    # 数据标准化
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_data()
    predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level=diffs_stddev_by_level, mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)
    # 自回归预测封装
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor
```
**功能**：构建带标准化、精度转换和自回归封装的GraphCast模型。
**核心算法/技巧**：
1. **混合精度训练/推理**：`casting.Bfloat16Cast`将模型输入输出转换为BFloat16，降低显存占用并加速计算；
2. **气象数据标准化**：`normalization.InputsAndResiduals`基于气压层的均值/标准差对输入和残差进行标准化，消除量纲影响；
3. **梯度检查点**：`gradient_checkpointing=True`在自回归预测时节省显存，适合长序列推理；
4. **自回归封装**：`autoregressive.Predictor`将单步预测模型封装为多步自回归预测模型。

#### 3.4.2 `run_forward`/`loss_fn`/`grads_fn`
```python
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(...)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(...)
    return loss, diagnostics, next_state, grads
```
**功能**：定义模型前向传播、损失计算和梯度计算函数。
**核心算法/技巧**：
1. **Haiku变换**：`hk.transform_with_state`将函数式模型转换为带参数和状态的模型，适配JAX的无状态设计；
2. **Xarray与JAX结合**：`xarray_tree.map_structure`和`xarray_jax.unwrap_data`实现Xarray数据集与JAX数组的转换；
3. **JAX自动微分**：`jax.value_and_grad`计算损失的梯度，`has_aux=True`保留辅助输出（诊断信息和状态）；
4. **闭包封装**：`_aux`函数封装损失计算，便于梯度计算时传递参数。

### 3.5 主函数（`main`）
主函数是代码的执行入口，分为**参数解析**、**模型加载/初始化**、**数据加载**、**JIT编译**、**滚动推理**和**可视化**六个步骤。

#### 步骤1：参数解析与打印
```python
parser = argsparser()
FLAGS = parser.parse_args()
print_arguments(FLAGS)
```

#### 步骤2：模型加载/初始化
```python
source = FLAGS.mode
if source == "Random":
    # 初始化随机模型配置
    params = None
    state = {}
    model_config = graphcast.ModelConfig(...)
    task_config = graphcast.TaskConfig(...)
else:
    # 加载预训练检查点
    with open(FLAGS.pretrained, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
```
**关键设计**：支持两种参数加载方式——随机初始化（用于训练）和加载预训练检查点（用于推理）。

#### 步骤3：数据加载与预处理
```python
dataset_file = FLAGS.dataset
with open(dataset_file, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# 提取输入、目标和强制变量
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(...)
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(...)
```
**核心函数**：`data_utils.extract_inputs_targets_forcings`根据TaskConfig提取模型所需的输入变量（如气温、气压）、目标变量（预测的气象变量）和强制变量（如地形、海温）。

#### 步骤4：JIT编译优化
```python
def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)

def with_params(fn):
    return functools.partial(fn, params=params, state=state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

# JIT编译前向传播、损失和梯度函数
init_jitted = jax.jit(with_configs(run_forward.init))
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(with_configs(run_forward.apply)))
```
**核心技巧**：
1. **偏函数封装**：`functools.partial`将模型配置和参数固定为函数参数，避免JIT编译时重复传递；
2. **状态丢弃**：`drop_state`移除模型输出的状态（GraphCast无状态），简化推理接口；
3. **JIT编译**：`jax.jit`将函数编译为机器码，大幅提升推理和训练速度。

#### 步骤5：滚动推理
```python
predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
```
**核心算法**：`rollout.chunked_prediction`实现分块自回归预测，将长序列拆分为多个块，逐块预测并拼接结果，避免显存溢出。

#### 步骤6：结果可视化
```python
save_var_diff(eval_targets, predictions, FLAGS.var, FLAGS.level)
print("----------------------------graphcast inference results----------------------------")
print(predictions)
```
调用`save_var_diff`绘制真实值、预测值和差值的可视化图片，并打印预测结果的Xarray数据集信息。

## 四、核心算法与关键技巧总结
### 4.1 核心算法
1. **GraphCast图神经网络**：基于气象网格的图表示，通过消息传递实现气象变量的空间依赖建模；
2. **自回归滚动预测**：单步预测模型通过自回归封装实现多步气象预测；
3. **分块预测**：将长序列拆分为块，逐块推理并拼接结果，平衡显存占用和推理效率；
4. **稳健归一化**：使用2%/98%分位数对气象数据进行可视化归一化，避免异常值影响。

### 4.2 关键编程技巧
1. **JAX/Haiku函数式编程**：无状态模型设计，结合`hk.transform_with_state`实现参数管理；
2. **JIT编译优化**：`jax.jit`编译核心函数，提升计算效率；
3. **Xarray多维数据处理**：便捷的气象数据索引和筛选，替代NumPy的复杂切片；
4. **偏函数封装**：`functools.partial`固定函数参数，简化JIT编译接口；
5. **混合精度计算**：BFloat16精度转换降低显存占用并加速推理；
6. **梯度检查点**：在自回归预测时节省显存，支持长序列推理。

## 五、代码扩展与优化建议
1. **批量推理**：当前代码仅处理单样本（`isel(batch=0)`），可扩展为批量推理以提升效率；
2. **多变量可视化**：支持同时可视化多个气象变量（如气温、降水、风速）；
3. **动画生成**：使用`matplotlib.animation`生成时间序列的预测结果动画；
4. **量化推理**：添加模型量化（如INT8），进一步降低显存占用；
5. **分布式推理**：使用JAX的`pmap`实现多设备分布式推理，加速长序列预测；
6. **评估指标计算**：添加RMSE、MAE等气象预测评估指标，量化模型性能。

## 六、运行说明
1. **环境配置**：安装依赖库`pip install jax haiku xarray netCDF4 matplotlib graphcast`；
2. **数据准备**：下载NetCDF格式的气象数据集（如ERA5）和GraphCast预训练检查点；
3. **运行命令**：
   ```bash
   python graphcast_inference.py --pretrained ./graphcast_ckpt.npz --dataset ./era5_data.nc --mode Checkpoint --var 2m_temperature --level 500
   ```
4. **结果输出**：生成可视化图片（如`2m_temperature_at_500_hPa.png`）和预测结果的Xarray数据集打印信息。
