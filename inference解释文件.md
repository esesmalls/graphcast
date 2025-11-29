# GraphCast 天气预报模型推理脚本分析
*导出时间: 2025/11/28 16:04:07*

---

## 需求背景
用户提供了 `inference.py` 脚本文件，要求对该 GraphCast 天气预报模型的推理+可视化脚本进行详细的代码解析。

## 脚本核心功能
该脚本是 GraphCast 天气预报模型的推理与可视化工具，核心能力包括：
- 解析命令行传入的运行参数
- 加载预训练模型权重或初始化随机模型
- 读取气象数据集并完成数据预处理
- 执行 GraphCast 模型的滚动推理（rollout）
- 可视化真实值、预测值及两者差值并保存结果图片

---

## 代码模块详细解析
### 1. 命令行参数与工具函数
该模块包含参数解析、参数打印、文件名元数据解析三个基础工具函数，为脚本运行提供配置支撑。

#### 1.1 命令行参数解析 `argsparser()`
```python
def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=str,
        default="./",
        help="pretrained model")
    parser.add_argument(
        "--dataset",
        type=str,
        help="input data")
    parser.add_argument(
        "--mode",
        type=str,
        help="the ways of getting model params")
    parser.add_argument(
        "--var",
        type=str,
        default="2m_temperature",
        help="visualizing atmospheric variables")
    parser.add_argument(
        "--level",
        type=int,
        default=500,
        help="atmospheric pressure level")
    return parser
```
**功能说明**：定义脚本运行的核心命令行参数，包括模型路径、数据集路径、参数加载模式、可视化变量、气压层等。

#### 1.2 参数打印 `print_arguments(args)`
**功能说明**：打印解析后的命令行参数，方便开发者确认运行配置，无核心代码逻辑，仅为调试辅助。

#### 1.3 文件名元数据解析 `parse_file_parts(file_name)`
```python
def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))
```
**功能说明**：从文件名中提取元数据（如数据源、分辨率、压力层数），格式示例：`source-era5_res-0.25_levels-13` 解析为 `{"source": "era5", "res": "0.25"}`，用于后续数据与模型配置的匹配检查。

### 2. 数据选择与可视化函数
该模块实现气象数据的筛选、归一化及可视化绘图，是脚本的核心展示层逻辑。

#### 2.1 数据筛选 `select(...)`
```python
def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data
```
**功能说明**：从 xarray 数据集中筛选指定的气象变量、气压层和时间步，同时去除 batch 维度（仅保留第一个样本）。

#### 2.2 可视化归一化 `scale(...)`
```python
def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))
```
**功能说明**：计算可视化的颜色映射范围，支持鲁棒归一化（2-98 百分位）和对称中心归一化（适用于差值图），并选择对应的 colormap（差值图用 `RdBu_r`，普通图用 `viridis`）。

#### 2.3 绘图函数 `plot_data(...)`
```python
def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    # 核心逻辑：创建子图、绘制数据、添加色条、保存图片
    pass
```
**功能说明**：根据输入的真实值、预测值、差值数据创建子图，隐藏坐标轴并添加颜色条，最终保存可视化图片（循环保存会覆盖，不影响核心展示）。

#### 2.4 差值可视化封装 `save_var_diff(...)`
```python
def save_var_diff(eval_targets, predictions, plot_pred_variable, plot_pred_level, plot_max_steps=1):
    plot_size = 5
    plot_max_steps = min(predictions.sizes["time"], 1)

    data = {
        "Targets": scale(select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps), robust=True),
        "Predictions": scale(select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps), robust=True),
        "Diff": scale((select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps) -
                            select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps)),
                          robust=True, center=0),
    }
    fig_title = plot_pred_variable
    if "level" in predictions[plot_pred_variable].coords:
      fig_title += f"_at_{plot_pred_level}_hPa"

    plot_data(data, fig_title, plot_size, True)
```
**功能说明**：封装真实值、预测值、差值的可视化流程，自动生成图片名称（如 `2m_temperature_at_500_hPa.png`）并调用 `plot_data` 绘图。

### 3. 数据归一化与模型配置
该模块实现模型所需的统计数据加载及数据合法性检查。

#### 3.1 数据合法性检查 `data_valid_for_model(...)`
**功能说明**：检查文件名提取的元数据（分辨率、压力层数）与模型配置是否一致，目前未在主流程中调用，为预留的健壮性检查逻辑。

#### 3.2 归一化统计数据加载 `load_data()`
```python
def load_data():
    # Load normalization data
    with open("./stats/stats_diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open("./stats/stats_mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open("./stats/stats_stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level 
```
**功能说明**：从 `./stats/` 目录加载气象数据的归一化统计量（均值、标准差、差分标准差），为模型输入输出的归一化提供依据。

### 4. GraphCast 模型构建
该模块是脚本的核心模型层，实现 GraphCast 模型的封装与配置。

```python
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):

    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # BFloat16 cast
    predictor = casting.Bfloat16Cast(predictor)

    # 加载标准化数据
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_data()
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor
```
**功能说明**：按层级封装 GraphCast 模型：
1. 创建基础的单步预测器 `graphcast.GraphCast`
2. 转换为 BFloat16 精度以节省显存
3. 添加输入/残差归一化层
4. 封装为自回归预测器，支持多步滚动推理（rollout）

### 5. Haiku 模型转换与梯度计算
该模块基于 Haiku 框架实现模型的前向推理、损失计算与梯度求解。

#### 5.1 前向推理 `run_forward`
```python
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)
```
**功能说明**：通过 `hk.transform_with_state` 将模型转换为 Haiku 格式，支持参数初始化与前向计算分离。

#### 5.2 损失函数 `loss_fn`
```python
@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):

    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))
```
**功能说明**：计算模型损失与诊断指标，并对结果求均值后转换为纯 JAX 数组。

#### 5.3 梯度计算 `grads_fn`
```python
def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):

    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads
```
**功能说明**：通过 `jax.value_and_grad` 计算损失的梯度，为模型训练预留接口（主流程中训练代码被注释）。

### 6. 主流程 `main()`
该模块是脚本的入口，串联所有功能模块实现端到端的推理与可视化。

```python
def main():
    # 1. 解析命令行参数
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    # 2. 加载/初始化模型
    source = FLAGS.mode
    if source == "Random":
        params = None
        state = {}
        model_config = graphcast.ModelConfig(...)
        task_config = graphcast.TaskConfig(...)
    else:
        assert source == "Checkpoint"
        with open(FLAGS.pretrained, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config

    # 3. 加载数据并拆分 inputs/targets/forcings
    # ... 数据加载逻辑 ...

    # 4. JIT 加速模型函数
    # ... JIT 封装逻辑 ...

    # 5. 执行滚动推理
    # ... rollout 推理逻辑 ...

    # 6. 可视化结果并保存
    save_var_diff(eval_targets, predictions, FLAGS.var, FLAGS.level)
```
**核心步骤**：
1. 解析并打印命令行参数
2. 根据 `mode` 加载预训练模型或初始化随机模型
3. 读取气象数据集并拆分为输入、目标、强迫数据
4. 通过 JIT 加速模型推理函数
5. 执行 GraphCast 模型的滚动推理
6. 调用 `save_var_diff` 可视化并保存结果图片

---

## 脚本运行示例
```bash
python inference.py \
  --mode Checkpoint \
  --pretrained ./graphcast_ckpt \
  --dataset ./era5_sample.nc \
  --var 2m_temperature \
  --level 500
```
**运行结果**：生成指定气象变量的真实值、预测值、差值可视化图片，控制台打印运行参数与推理信息。
