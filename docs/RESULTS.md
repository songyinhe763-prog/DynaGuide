# 实验结果：Flow-DynaGuide on free-floating satellite disassembly

> 全部结果可复现；命令见文末。工作机为 CPU（torch 2.9.1+cpu），单次训练 ~5-25 min。

## 0. 结论摘要（先说最重要的）

1. **算法与工程管线完整跑通**：流匹配策略（rectified flow）训练 → 潜空间动力学模型
   （含噪声动作增广）→ 引导采样（latent / physics / 两者）→ 逐回合评测 → 参数扫描，
   四个测试套件（`tests/`）全部通过，robomimic/CALVIN 侧的 `flow_policy` 亦可实例化、
   训练、带引导采样（CPU 冒烟测试通过）。
2. **"局部"引导是有效的**：把引导当成"对动作块的一步改进"来单独检验
   （`scripts/satellite/check_energy_alignment.py`，20 个状态点），降 3 步能量梯度后
   在**真实仿真**里重放同一动作块：
   - 代理能量下降 17.2%；
   - 基座漂移在 **15/20** 个样本上下降（均值 0.0385 → 0.0332 rad，−14%）；
   - 接触力在 **16/20** 个样本上下降（均值 1.53 → 1.30 N，−15%）。
   说明可微物理代理与仿真**方向一致**，引导项本身没有写反。
3. **"闭环"整体成功率上没有可分辨的提升**：当前基础策略已经相当安全
   （20 回合：成功率 85%、约束满足率 80%、漂移 0.098 rad < 0.12 预算），引导带来的
   逐块改善被"闭环分布漂移"抵消，强引导甚至会掉成功率。6-10 回合的小样本上排名会翻转
   （见 §3），**差异在 n=20 的二项置信区间（±20 个百分点）内**。
   要复现 DynaGuide 论文那种"引导明显提升"的趋势，必须让基础策略**不安全**——见 §5 的
   下一步建议（提高不安全示范比例）。

这一节也解释了为什么不能直接照抄论文的 `scale`：`s` 的量纲绑定了 `sqrt(1-ᾱ_k)`，
在流匹配里换了 `λ(t)`，必须重扫（见 `docs/FLOW_DYNAGUIDE.md` §3）。

## 1. 设置

- 场景：`sc1_dead_satellite` 单模块任务 `payload_box`（螺栓：需先拧 90°，再轴向拉 > 4 N）。
- 示范：120 回合，策略混合 `safe 60% / impatient 20% / offset 20%`，动作噪声 0.02。
  各策略的实测约束满足率：`safe 97% / impatient 8% / offset 4%`。
- 基础策略：`FlowActionChunkPolicy`（ConditionalUnet1D 1D-UNet，192/384，horizon 16，执行 8 步，
  观测 2 帧），rectified-flow 目标，5000 步，batch 96，AdamW lr 3e-4；验证损失 0.19。
- 动力学模型：冻结语义编码器（22 维）+ 3 层 MLP，噪声动作增广 50%；验证 MSE 0.0026。
- 引导条件：正样本 = 安全回合末端状态，负样本 = 违约回合末端状态（按模块分别建库）。
- 采样：euler，4 步，`t∈[0.01,1]`，`λ(t)=(1-t)²/2`（`--weight fp`），梯度 L2 裁剪 5.0。

## 2. 主表（每方法 20 回合，episodes 0..19，同一组种子）

| method | success | constraint-satisfied | base drift [rad] | max force [N] |
|---|---|---|---|---|
| base（无引导） | 85% | 80% | 0.098 | 3.24 |
| latent（DynaGuide 潜空间，s=1.0, σ=30） | 65% | 45% | 0.105 | 3.60 |
| physics（物理能量，s=0.25） | 70% | 45% | 0.110 | 3.30 |
| both（两能量，s=0.25） | 60% | 35% | 0.113 | 3.54 |
| reject（16 采样取最优，无梯度） | 0% | 0% | 0.126 | 3.80 |

## 3. 小样本上的波动（说明为什么不能只看 6-10 回合）

同配置、10 回合：

| method | success | constraint | drift |
|---|---|---|---|
| base | 80% | 70% | 0.098 |
| latent (s=1.0, σ=30) | **100%** | **90%** | 0.090 |
| physics (s=1.0) | 10% | 10% | 0.122 |
| both (s=1.0) | 20% | 20% | 0.126 |
| reject | 20% | 20% | 0.117 |

20 回合时 latent 变成 65%/45%。⇒ **±20 个百分点以内的差异不可解释**，本表只用于选超参。

## 4. 引导强度扫描（每点 6 回合，`--weight fp`，euler 4 步）

| method | scale | alpha | success | constraint | drift |
|---|---|---|---|---|---|
| physics | 0.05 | 30 | 83% | 67% | 0.091 |
| physics | 0.25 | 30 | 83% | 83% | 0.100 |
| physics | 0.50 | 30 | 33% | 17% | 0.123 |
| both | 0.10 | 30 | 83% | 67% | 0.094 |
| both | 0.40 | 30 | 83% | 67% | 0.104 |
| latent | 0.50 | 15 | 83% | 67% | 0.090 |
| latent | 2.00 | 15 | 100% | 83% | 0.089 |

规律与文献一致：**引导权重极其敏感**（大于 ~0.5 就开始破坏任务执行），
`σ`（温度）在 15-30 之间差异不大。

## 5. 下一步（要让引导"赢"，需要给基础策略留出不安全裕度）

当前基础策略约束满足率 80%，几乎没有提升空间。建议按下面任一条重建实验：

1. **提高不安全示范比例**：
   `python scripts/satellite/make_demos.py --tasks payload_box --per_task 120 --noise 0.02 --behavior_mix safe=0.35,impatient=0.35,offset=0.30 --tag payload_hard`
   然后重新训练策略/动力学并评测（命令同下）。此时 BC 会学到明显不安全的平均行为，
   引导的目标就是把"拆下来了但在推转卫星"变成"安全拆下"。
2. **收紧安全预算**（同一策略、更严指标）：`SatelliteSpec.attitude_budget` 0.12 → 0.06 rad、
   `force_budget` 4 → 2.5 N，再评测（只改指标，不用重训）。
3. **增大样本量**：每方法 ≥ 40 回合，报二项置信区间；6-20 回合只用于调参。
4. **闭环更强地把引导"铺开"**：`--ss 2~4`（同一时间层多次修正）+ 更小的 `scale`，
   或把引导只作用于动作块前 `n_action_steps` 步（当前作用整段）。

## 6. 复现命令

```bash
python scripts/satellite/make_demos.py --tasks payload_box --per_task 120 --noise 0.02 --tag payload --seed 5
python scripts/satellite/train_flow_policy.py --demos data/satellite/demos_payload.npz \
    --out checkpoints/satellite/flow_policy_payload.pt --steps 5000 --batch_size 96 --down_dims 192 384
python scripts/satellite/train_latent_dynamics.py --demos data/satellite/demos_payload.npz \
    --out checkpoints/satellite/latent_payload.pt --steps 2500
python scripts/satellite/run_flow_dynaguide.py \
    --policy checkpoints/satellite/flow_policy_payload.pt \
    --dynamics checkpoints/satellite/latent_payload.pt \
    --methods base latent physics both reject --episodes 20 --steps 4 --alpha 30 \
    --per_method_scale "latent=1.0,physics=0.25,both=0.25" \
    --out results/satellite/eval_final.json
python scripts/satellite/sweep_guidance.py --methods physics --scales 0.05 0.25 0.5 --episodes 6
python scripts/satellite/check_energy_alignment.py --episodes 4   # 逐块代理 vs 仿真检验
```

原始数据：`results/satellite/*.json`（`eval_final`、`eval_payload`、`sweep_*`）。
