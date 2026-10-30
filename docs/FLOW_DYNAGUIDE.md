# Flow-DynaGuide：把 DynaGuide 的主动动态引导接到流匹配（rectified/consistency flow）策略上

> 本文档是代码 `core/flows/`、`core/flow_dynaguide.py`、`core/flow_energies.py`、
> `robomimic/robomimic/algo/flow_policy.py` 的说明与数学依据。
> 目标读者：要继续改这套代码的人。

---

## 1. 为什么不能把 DynaGuide 的三行补丁直接搬到流匹配上

DynaGuide 在 DDIM 上的引导注入是

```
ε̂(a_k, o_t) = ε(a_k, o_t) − s · sqrt(1 − ᾱ_k) · ∇_{a_k} d
```

其中 `s` 是引导强度（论文 Eq.3），`σ`（代码里的 `alpha`）是 Eq.2 里 `log-sum-exp` 的温度，`M`（代码里的 `ss`）是同一噪声等级上重复去噪的次数。
`sqrt(1 − ᾱ_k)` **不是**一个可自由调节的旋钮，它是"score 参数化 → ε 参数化"的雅可比因子：DDIM 前向

```
a_k = sqrt(ᾱ_k) a_0 + sqrt(1 − ᾱ_k) ε
```

给出 `∇_{a_k} log q(a_k) = −ε_θ / sqrt(1 − ᾱ_k)`，因此"在 score 上加一个能量梯度"翻译到 ε 空间时必须乘 `sqrt(1 − ᾱ_k)`：

```
ε̂ = −sqrt(1 − ᾱ_k) · ( ∇log q + s ∇d ) = ε_θ − s·sqrt(1 − ᾱ_k)·∇d
```

这个因子还顺带提供了**自动退火**：最噪的一步系数≈1，接近数据时系数→0（这就是论文里唯一的退火机制，论文全文没有 `anneal` 这个参数）。

流匹配的线性插值路径是

```
x_t = (1 − t)·ε + t·a_1 ,   t ∈ [0,1] ,   t = 0 是噪声、t = 1 是动作块
```

它的**边际噪声标准差是 σ_t = 1 − t**，所以 `sqrt(1 − ᾱ_k)` 在流匹配里的等价物是 `1 − t`；

- 严格照搬 Eq.3 会得到 λ(t) = (1 − t)/t，在 `t → 0` **发散**——而 FlowPolicy 的卖点恰恰是 1-step（t≈0），直接搬会炸；
- 从 Fokker–Planck 漂移 `f = ν + (g²/2)∇log p`（取 `g = σ_t = 1 − t`）得到有界且自退火的

```
λ(t) = (1 − t)² / 2        ← 本仓库默认（GuidanceWeightSpec("fp")）
```

于是本仓库的统一形式是

```
ν̂(x_t, t) = ν_θ(x_t, t) − s · λ(t) · ∇_{x_t} J(x̂_1)
x̂_1 = x_t + (1 − t)·ν_θ           （FlowPolicy 自己的 f_θ，即 Tweedie / 单步欧拉干净动作估计）
```

`J` 是**要最小化的能量**（DynaGuide 的 `d` 是 log 似然，所以 `J = −d`）。当 `J` 取潜空间距离时，`∇J` 与 DynaGuide 的 `∇d` 只差一个符号，其余（log-sum-exp 聚合、正负条件、温度 σ）完全复用。

---

## 2. 梯度到底在哪一点求：三种模式

| 模式 | 代码 | 语义 | 代价 |
|---|---|---|---|
| `endpoint`（默认） | `FlowDynaGuide(mode="endpoint")` | 在 `x̂_1 = x_t + (1−t)ν` 上算能量并按单位映射回传（`jacobian=False`，等价 QGF 的 `apply_jacobian=False`） | 1 次反向，最稳 |
| `endpoint` + 全雅可比 | `jacobian=True` | 显式含采样器雅可比 `I + (1−t)·∂ν/∂x_t`（RectifID 的 anchored guidance） | 每次采样多一次反传到策略网 |
| `velocity` | `mode="velocity"` | DynaGuide 论文的"直接把带噪动作当动作"字面移植，作消融 | 便宜但有偏（critic 只在干净动作上训练过） |

**为什么默认在干净动作上求梯度**：判别器/critic/动力学模型都是在**干净动作块**上训练的，在带噪的 `x_t` 上求梯度属于分布外查询，信号会有偏。`t` 很小时 `x̂_1 ≈ x_t + ν`，两者差别不大；中间时间段差别最大。

---

## 3. 与 DynaGuide 超参的对照

| DynaGuide | 本仓库 | 说明 |
|---|---|---|
| `s`（Eq.3，常用 1.0–2.0） | `FlowDynaGuide.scale` / `--scale` | 量纲随 λ(t) 变化，**必须重扫**（建议 0.25 / 0.5 / 1 / 2） |
| `σ`（Eq.2 温度，常用 15–40） | `LatentMatchEnergy.alpha` / `--alpha` | 只依赖潜空间尺度，可直接沿用（默认 30） |
| `M`（随机采样次数，=4） | `FlowSamplerConfig.ss` / `--ss` | `ss_mode="split"`：把一步拆成 `ss` 个子步并逐子步引导；`"refine"`：同一 t 上重复修正 |
| DDIM 10 步 | `--steps`（euler/midpoint/heun） | 1 步是 FlowPolicy 的主场，但带引导时 4 步更稳 |
| 内建 `sqrt(1−ᾱ_k)` 退火 | `--weight fp`（默认）/ `linear` / `ddim` / `noise_fraction` / `const` | 见第 1 节的推导 |
| 噪声动作预训练 | `SemanticDynamicsModel.noise_augment` | **必须保留**：用同一条插值路径加噪，50% 概率（见第 5 节） |

---

## 4. 代码地图

```
core/flows/
  paths.py        LinearFlowPath：(1−t)ε + t·a 的概率路径、x1/x0 投影、SNR/噪声比例
  consistency.py  rectified_flow_loss（默认目标）+ ConsistencyFlowMatcher（FlowPolicy 的一致性目标）
  samplers.py     FlowSampler：euler/midpoint/heun + 引导钩子 + ss 内层循环 + 轨迹记录
  nets.py         小型 ConditionalUnet1D（FiLM）+ ObsEncoder
  policy.py       FlowActionChunkPolicy：观测→速度场，训练/采样/存取
  normalizer.py   动作 min-max 归一化（[-1,1]）
core/flow_dynaguide.py   GuidanceWeightSpec（λ(t)）、FlowDynaGuide（能量→速度修正）、
                         calculate_flow_classifier_guidance（给 run_dynaguide.py 的 drop-in）
core/flow_energies.py    LatentMatchEnergy（DynaGuide 原文目标）+ SatellitePhysicsEnergy（微观重力物理能量）
robomimic/robomimic/algo/flow_policy.py   在 robomimic/CALVIN 栈里的 flow 策略（训练 + 引导采样）
robomimic/robomimic/config/flow_policy_config.py, configs/flow_policy_image_calvin.json
```

典型用法（独立栈）：

```python
from core.flows.policy import FlowActionChunkPolicy
from core.flows.samplers import FlowSampler, FlowSamplerConfig
from core.flow_dynaguide import make_flow_guidance

policy = FlowActionChunkPolicy.load("checkpoints/satellite/flow_policy_payload.pt")
sampler = FlowSampler(FlowSamplerConfig(n_steps=4, method="euler", ss=2))
guide = make_flow_guidance(energy, scale=1.0, weight_mode="fp", mode="endpoint", clip_norm=5.0)
out = policy.sample(obs_window, sampler=sampler, guidance=guide, ctx={"state": proprio, "proprio": proprio, "module_index": 0})
action_chunk = out["actions"]           # 反归一化后的真实动作尺度
```

在 robomimic/CALVIN 栈上（替换 DDIM 策略）：

```bash
python train.py --config configs/flow_policy_image_calvin.json --dataset calvin.hdf5 --name flow_run
python run_dynaguide.py ... --guidance_type flow --flow_weight fp --scale 1.0 --alpha 30 --ss 4
```

`run_dynaguide.py --guidance_type flow` 会自动改用
`core.flow_dynaguide.calculate_flow_classifier_guidance`（返回签名与 DDIM 版一致：
`(guidance, good_embeddings, bad_embeddings)`），策略侧由
`FlowPolicyUNet.get_action(..., guidance_type="flow")` 接收。

---

## 5. 两个"必须做对否则一定失效"的工程点

### 5.1 动力学模型必须见过"带噪动作"

DynaGuide 论文（Fig. 9）明确消融过：如果不给动力学模型做噪声动作预训练，引导能力直接崩。
原因很直接——引导在采样过程中会拿"尚未去噪完的动作块"去问 `h_θ`。

本仓库在流匹配里的等价做法（`SemanticDynamicsModel.augment_actions`）：

```
a_t = (1 − t)·ε + t·a_1 ,  t ~ U[0,1] ,  以 50% 概率替换原动作
```

即用**与采样器同一条插值路径**加噪，而不是另造一个高斯噪声。

### 5.2 能量/梯度必须做限幅

引导权重极其敏感（QGF 官方实现要逐值扫 `--guidance_weights`）。本仓库提供了三道护栏，默认开启第一道：

- `clip_norm`（默认 5.0）：OmniGuide 的 `α` clip，按 L2 范数裁剪；
- `clip_value`（默认 `None`）：逐元素裁剪；
- `ema`（默认 0）：跨采样步的引导方向动量平滑。

---

## 6. 与论文的关系（诚实说明）

- DynaGuide 原文**只**讨论 DDIM（"Like past work, we use DDIM as our sampling method"），没有任何一句讨论推广到 flow matching / ODE / 速度场。本仓库的推广是**从论文 Eq.2/Eq.3 的数学结构推出来的**，依据是 `sqrt(1−ᾱ_k)` 的雅可比含义（第 1 节），与 QGF / PreferenceFlow / OmniGuide 这类工作采用的形式一致（"单步欧拉估干净动作 → 在干净动作上求梯度 → 加到速度场"）。
- 因此论文里的 `s` 数值（1.0–2.0）**不能**直接搬到 flow 上；本仓库的 `scale` 需要重扫。
- 引用 DynaGuide 时请注意与同名的无监督分割工作 *DynaGuide: A Generalizable Dynamic Guidance Framework…* 消歧。

---

## 7. 复现实验（卫星拆解）

```bash
# 1) 采集示范（三种示范策略混合：safe / impatient / offset）
python scripts/satellite/make_demos.py --tasks payload_box --per_task 90 --noise 0.03 --tag payload
# 2) 训练流匹配基础策略（行为克隆，混合策略 → 会学到"平均"的不安全行为）
python scripts/satellite/train_flow_policy.py --demos data/satellite/demos_payload.npz \
    --out checkpoints/satellite/flow_policy_payload.pt --steps 3000
# 3) 训练潜空间动力学模型（含噪声动作增广）
python scripts/satellite/train_latent_dynamics.py --demos data/satellite/demos_payload.npz \
    --out checkpoints/satellite/latent_payload.pt --steps 2500
# 4) 评测：base / latent / physics / both / reject
python scripts/satellite/run_flow_dynaguide.py \
    --policy checkpoints/satellite/flow_policy_payload.pt \
    --dynamics checkpoints/satellite/latent_payload.pt \
    --methods base latent physics both reject --episodes 8 --steps 4 --scale 1.0 --alpha 30
```

结果写入 `results/satellite/*.json` 与 `*.md`（含逐方法/逐任务的表格）。
