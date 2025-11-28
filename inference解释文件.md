# 代码分析解释

*导出时间: 2025/11/28 16:04:07*

---

### User

📎 **inference.py** (Python)

详细解释分析这段代码

---

### ChatGPT

这段脚本整体上是一个 GraphCast 天气预报模型的推理+可视化脚本：


从命令行读取参数


加载预训练模型和数据


用 GraphCast 做一次滚动推理（rollout）


把真实值、预测值和它们的差画图并保存为图片


下面我按“从上到下 + 整体流程”来详细讲一遍。inference

1. 整体结构概览
文件大致分为几块：


参数和小工具函数：argsparser、print_arguments、parse_file_parts


数据处理和画图相关函数：select、scale、plot_data、save_var_diff


数据合法性检查和归一化数据加载：data_valid_for_model、load_data


构建 GraphCast 模型：construct_wrapped_graphcast


Haiku transform 的前向/损失/梯度函数：run_forward、loss_fn、grads_fn


main() 主流程：


解析命令行参数


加载模型（随机 or Checkpoint）


加载数据并拆分 inputs/targets/forcings


JIT 包装模型、损失和梯度函数


做推理 rollout


可视化结果





2. 命令行参数和工具函数
2.1 argsparser()
python复制代码def argsparser():
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

命令行传参：


--pretrained: 预训练模型 checkpoint 文件路径


--dataset: 输入数据（NetCDF 文件）路径


--mode: 获取模型参数的方式，代码中只用到了 "Random" 或 "Checkpoint"


--var: 想要可视化的气象变量名，默认 "2m_temperature"


--level: 可视化的气压层，默认 500 hPa


2.2 print_arguments(args)
简单打印参数，方便确认运行时配置。
2.3 parse_file_parts(file_name)
python复制代码def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

假设文件名形如：
source-era5_res-0.25_levels-13_...


按 _ 切分 -> 每块像 source-era5


再按 - 切分成 key/value -> {"source": "era5", "res": "0.25", "levels": "13", ...}


后面用它来检查数据是否匹配模型配置。

3. 数据选择与画图相关函数
3.1 select(...)：从 xarray Dataset 里选变量/层/时间步
python复制代码def select(
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

做了几件事：


选出指定变量 variable


如果有 batch 维度，只取 batch=0（只看第一样本）


如果指定 max_steps，并且时间步数大于 max_steps，则只取前 max_steps 个时间步


如果指定 level 且有 level 坐标，则选定指定气压层


3.2 scale(...)：计算画图时的归一化范围与 colormap
python复制代码def scale(
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



robust=True 时用 2–98 百分位，避免极端值影响色条范围


如果指定了 center（例如差值图中心为 0），会让 vmin/vmax 对称地围绕 center


返回：


原数据


Normalize 对象（控制颜色映射范围）


使用的 colormap：


有 center 用 RdBu_r（红蓝反转，适合偏差）


否则用 viridis






3.3 plot_data(...)：画多个子图（目标/预测/差值）
python复制代码def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

这里 data 结构类似：
python复制代码{
  "Targets": (data_xarray, norm, cmap),
  "Predictions": (...),
  "Diff": (...)
}

主要步骤：


从第一个数据中取出时间维大小，用来确认所有数据时间步数一致


根据 len(data) 和 cols 计算行列数，创建 figure


对每个 (title, (plot_data, norm, cmap))：


创建子图


隐藏坐标轴


图名用 title（如 "Targets", "Predictions", "Diff"）


用 imshow 把第 0 个时间步画出来


加 colorbar




每次循环里调用 plt.savefig(f"{fig_title}.png")（这里有点小问题：在循环里保存会重复覆盖同一个文件，多余，但不影响核心逻辑）


3.4 save_var_diff(...)：专门画目标 vs 预测 vs 差值
python复制代码def save_var_diff(eval_targets, predictions, plot_pred_variable, plot_pred_level, plot_max_steps=1):
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



使用上面的 select 和 scale：


Targets: 真实值


Predictions: 预测值


Diff: 真实值 - 预测值，center=0 用红蓝色差分色条




图片文件名类似 2m_temperature_at_500_hPa.png



4. 归一化和模型配置相关
4.1 data_valid_for_model(...)（目前没在 main 中用）
检查数据文件名提取出的信息是否与模型配置一致，比如：


分辨率 res


压力层数 levels


源数据 source 与是否使用 total_precipitation_6hr 对应


如果严格部署，这个函数可以在加载数据前做 sanity check。
4.2 load_data()：加载标准化统计量
python复制代码def load_data():
    # Load normalization data
    with open("./stats/stats_diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open("./stats/stats_mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open("./stats/stats_stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level 

从 ./stats/ 目录读三个 NetCDF 文件，包含：


每层的差分标准差


每层的均值


每层的标准差


用于输入输出的归一化和残差归一化。

5. 构建封装好的 GraphCast 模型
5.1 construct_wrapped_graphcast(...)
python复制代码def construct_wrapped_graphcast(
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

封装顺序：


GraphCast 原始一时刻预测器（single-step）


用 Bfloat16Cast 把内部计算转成 bfloat16（更省显存）


外面再包一层 InputsAndResiduals，负责：


对输入/输出做标准化（用之前 load_data() 得到的统计量）


预测残差等




最外层包成 autoregressive.Predictor：


可以根据单步模型反复滚动，得到多步预测序列


打开 gradient_checkpointing 减少显存（如果训练时用）





6. Haiku 模型 transform 与损失/梯度函数
6.1 run_forward：前向推理函数
python复制代码@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)



@hk.transform_with_state 会把这个函数转成 Haiku 的形式：


init(...) 负责初始化参数


apply(params, state, rng, ...) 负责前向计算




输入：


inputs: 历史输入变量（xarray -> jax array）


targets_template: 只用来提供目标的 shape（这里传入的是 eval_targets * np.nan）


forcings: 外部强迫（如太阳辐射等）




6.2 loss_fn：带诊断的损失函数
python复制代码@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):

    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))



调用 predictor.loss(...) 计算：


标量损失（带 time/space/variable 维度）


其他诊断指标




再用 xarray_tree.map_structure 把每个 xarray 里的数据取 mean（全局平均），并 unwrap 成纯 JAX 数组。


6.3 grads_fn：计算损失及其梯度
python复制代码def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):

    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config,
            i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads



_aux 封装了调用 loss_fn.apply 的逻辑，返回 (loss, (diagnostics, next_state))


用 jax.value_and_grad 对 params 求梯度


返回 loss、diagnostics、next_state 和 grads
（在 main 里这块训练代码被注释掉了）



7. main() 主流程
7.1 解析参数并打印
python复制代码parser = argsparser()
FLAGS = parser.parse_args()
print_arguments(FLAGS)

7.2 加载或构建模型
python复制代码source = FLAGS.mode
if source == "Random":
    params = None  # Filled in below
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

两种模式：


Random:


不加载 checkpoint，params=None，随后会用 run_forward.init 随机初始化


使用一个较小的模型配置（latent_size=32 等）




Checkpoint：


从 FLAGS.pretrained 指定的文件加载 CheckPoint


里面包含 params、model_config、task_config、描述和 license


打印模型信息




7.3 加载数据并拆分 inputs/targets/forcings
python复制代码dataset_file = FLAGS.dataset
with open(dataset_file, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()
assert example_batch.sizes["time"] >= 3
print(", ".join([...parse_file_parts...]))

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{1 * 6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{1 * 6}h"),
    **dataclasses.asdict(task_config))



xarray.load_dataset 读取 NetCDF 数据集


需要至少 3 个 time step（2 个输入 + >=1 个目标）


extract_inputs_targets_forcings 会：


根据 task_config 中的变量列表、时间长度等，从 example_batch 中拆分出：


inputs（比如过去 24 小时）


targets（未来 6 小时）


forcings（外部强迫）






这里 train 和 eval 用的是同一批数据，只是分别命名。
7.4 辅助包装函数：with_configs / with_params / drop_state
python复制代码def with_configs(fn):
    return functools.partial(
        fn, model_config=model_config, task_config=task_config)

def with_params(fn):
    return functools.partial(fn, params=params, state=state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

用途：


JAX 的 jit 一般不喜欢对复杂对象（如 config dataclass）做闭包捕获，因此用 partial 方式作为显式参数。


drop_state：有些函数返回 (output, state)，但这里模型其实无状态，所以只取 [0]。


7.5 初始化参数（如果是 Random 模式）
python复制代码init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
    params, state = init_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets,
        forcings=train_forcings)



用 run_forward.init 根据输入数据 shape 初始化 Haiku 参数


初始化时也要传入 model_config, task_config 作为静态参数


7.6 JIT 化损失、梯度、前向推理函数
python复制代码loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(with_configs(
    run_forward.apply)))



loss_fn_jitted(rng=..., inputs=..., targets=..., forcings=...) 返回损失和诊断


grads_fn_jitted(inputs=..., targets=..., forcings=...) 计算梯度


run_forward_jitted(rng=..., inputs=..., targets_template=..., forcings=...) 做前向预测


7.7 分辨率检查
python复制代码assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
    "Model resolution doesn't match the data resolution. ..."
)

确保模型和数据的经纬度分辨率一致，否则说明数据和模型不匹配。
7.8 rollout 推理
python复制代码predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)



rollout.chunked_prediction 会多步滚动预测：


初始用 inputs 预测 targets_template 形状的数据


然后把预测结果作为新的输入再往前推（如果需要多时间步）




这里 targets_template = eval_targets * np.nan 的目的是：


用真实目标的 shape 填充 NaN，用于告诉模型期望输出的大小




7.9 预测结果可视化
python复制代码save_var_diff(eval_targets, predictions, FLAGS.var, FLAGS.level)
print("----------------------------graphcast inference results----------------------------")
print(predictions)



画图（真实 vs 预测 vs 差值），图片保存到当前目录


控制变量名 FLAGS.var 和层次 FLAGS.level


打印整个 predictions 的 xarray 结构


7.10 训练相关代码（已注释）
下面一大段关于：


计算训练损失


反向传播求梯度


再用 run_forward_jitted 预测


都被注释掉了，保留作参考。

8. 如何使用这段脚本（示例）
假设你已经有：


预训练模型 checkpoint：./graphcast.ckpt


数据：./data/sample.nc


命令可能像这样：
bash复制代码python inference.py \
  --mode Checkpoint \
  --pretrained ./graphcast.ckpt \
  --dataset ./data/sample.nc \
  --var 2m_temperature \
  --level 500

运行后，你会得到类似：


2m_temperature_at_500_hPa.png：真实/预测/差值图


控制台打印模型信息、输入/目标维度结构，以及 predictions 的 xarray 数据



如果你愿意，我可以帮你逐行加中文注释版本，或者帮你改造成只做多步预测（比如连滚 4 个 6h 预测，总共 24h）并保存所有时间步的可视化。

