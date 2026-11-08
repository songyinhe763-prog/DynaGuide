"""Build the FULL code mind map (Mubu / 幕布 importable).

Coverage: every top-level folder, every file I added or touched, and — for the important
files — the main classes / functions / variables and how they connect.

Outputs (one source tree -> four formats):

    C:\\代码修改\\代码思维导图.md      Markdown 大纲（幕布「导入 Markdown」/ 备份）
    C:\\代码修改\\代码思维导图.opml    OPML 1.0
    C:\\代码修改\\代码思维导图.mm      FreeMind XML
    C:\\代码修改\\mubu_paste.txt      制表符缩进（直接在幕布 Ctrl+V）

Usage::

    python scripts/make_mindmap.py
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path("C:/代码修改")

TREE = (
    "DynaGuide → 流匹配改造（失效卫星拆解）｜全代码框架",
    [
        # ─────────────────────────────────────────────── 0
        ("① 项目总览", [
            ("原仓库 DynaGuide｜NeurIPS 2025", [
                "核心思想：不改策略权重，推理期用「潜空间动力学 + 能量梯度」引导 DDIM 去噪",
                "改造前只有一条通路：DDIM 噪声预测（ε）+ √(1−ᾱ_k) 加权梯度",
            ]),
            ("本次改造做了什么", [
                "新增 37 个文件、约 5450 行：core/flows、core/flow_dynaguide.py、core/flow_energies.py、core/satellite、robomimic/algo/flow_policy.py、scripts、tests、docs",
                "改动 5 个原文件：run_dynaguide.py（加开关）、algo/__init__.py、config/__init__.py、README.md、.gitignore",
                "新增一条通路：流匹配速度场（v）+ λ(t) 加权 Δv；两条通路并存，用 --guidance_type 切换",
            ]),
            ("一句话数据流", [
                "混合示范 → 训练流策略 v_θ → ODE 采样（每步加引导 Δv）→ 动作块 → 自由漂浮仿真 → 成功/安全成功",
                "旁路：混合示范 → 训练潜空间动力学 h_θ → 建 goal bank（正=安全回合末态，负=违约回合末态）",
            ]),
        ]),

        # ─────────────────────────────────────────────── 1
        ("② 顶层目录（先看框架，每个文件夹一句话）", [
            ("core/｜★ 算法核心（本次新增 13 个文件）", [
                "原有 5 个文件未改动：dynaguide.py、dynamics_models.py、calvin_utils.py、embedder_datasets.py、image_models.py",
                "新增核心 2 个：flow_dynaguide.py（引导）、flow_energies.py（能量）",
                "新增子包 2 个：flows/（流匹配底座 7 文件）、satellite/（拆解仿真 7 文件）",
                "→ 展开见 ④⑤⑥⑦",
            ]),
            ("robomimic/｜策略与算法库（第三方 + 本次新增一个算法）", [
                "robomimic/robomimic/algo/diffusion_policy.py｜原 DDIM 策略，**未改动**，是本次对照的基线",
                "robomimic/robomimic/algo/flow_policy.py｜★ 新增：流匹配策略（fm/cfm + 带引导采样）",
                "robomimic/robomimic/config/flow_policy_config.py｜★ 新增：配置类（algo.flow.*）",
                "algo/__init__.py、config/__init__.py｜各 +1 行注册",
                "robomimic/utils、models、envs…｜第三方基础设施，未改动",
                "→ 展开见 ⑧",
            ]),
            ("configs/｜训练配置（JSON）", [
                "diffusion_policy_image_calvin_ddim.json｜原 DDIM 基线配置（未动）",
                "★ flow_policy_image_calvin.json｜新增：流策略训练配置（algo_name=flow_policy + algo.flow.*）",
                "calvin.json / touchcubes.json / diffusion_policy_image_munk_ddim.json｜原有",
            ]),
            ("calvin_exp_configs_examples/｜引导实验配置（每个任务一个）", [
                "switch_on.json / door_left.json / …｜原 DDIM 实验配置（未动）",
                "★ switch_on_flow.json｜新增：流匹配引导实验配置（guidance_type=flow, flow_weight=fp）",
                "reset_poses/*.json｜机器人初始位姿（原论文实验用）",
            ]),
            ("scripts/｜★ 命令行脚本（本次新增 12 个）", [
                "satellite/｜拆解实验全套：采数→训练→评测→扫描→诊断→出图",
                "make_mindmap.py｜生成这份思维导图",
                "make_report_docx.py｜生成 Word 报告",
                "run_all_checks.ps1｜一键复现全部证据",
                "→ 展开见 ⑨",
            ]),
            ("tests/｜★ 单元与集成测试（本次新增 4 个，34 项断言）", [
                "→ 展开见 ⑩",
            ]),
            ("docs/｜★ 文档（本次新增 3 篇）", [
                "FLOW_DYNAGUIDE.md｜推导 + API + 与 DynaGuide 超参对照",
                "SATELLITE_DISASSEMBLY.md｜任务、物理、评测口径",
                "RESULTS.md｜实验表、波动分析、下一步",
            ]),
            ("calvin/｜第三方 CALVIN 仿真环境（未改动）", [
                "calvin_env/｜PyBullet 场景、机器人、相机、任务定义",
                "为什么不动它：改造只发生在「推理期引导」，环境与数据格式保持原样才能对照论文数字",
            ]),
            ("data_processing_calvin/｜CALVIN 数据预处理（未改动）", [
                "calvin_to_labeled_hdf5.py｜把 CALVIN 数据切成分段行为",
                "split_behavioral_validation_datasets_calvin.py｜拆成单行为文件（做引导条件用）",
                "hdf5_combiner.py / hdf5_filter.py / generate_goal_images.py｜拼条件、清洗、取目标图",
                "retrieve_initial_robot_states.py｜导出复位位姿",
            ]),
            ("paper_experiments/｜原论文 5 组实验脚本（未改动）", [
                "experiment_1~5_*.py + *.sh｜原论文的曲线生成与跑批脚本",
            ]),
            ("toy_squares_experiment/｜原玩具实验（未改动）", [
                "collect_scripted_data_pymunk.py｜pymunk 玩具数据采集",
                "train_end_state_classifier.py / classifier_guidance_pymunk.py｜分类器版引导",
            ]),
            ("运行产物目录（都在 .gitignore 里）", [
                "data/satellite/｜demos_*.npz 示范数据 + 元信息 JSON",
                "checkpoints/satellite/｜*.pt 权重 + 训练历史 JSON",
                "results/satellite/｜评测 JSON/MD、扫描结果",
                "logs/｜原始 stdout 日志",
                "models/｜（原仓库预留）",
            ]),
            ("根目录文件", [
                "README.md｜★ 改动：新增「流匹配版 DynaGuide」章节",
                "run_dynaguide.py｜★ 改动：加 --guidance_type / --flow_weight，flow 分支改用流版引导",
                "run_sampling_baseline.py｜原：拒绝采样基线",
                "run_trained_agent.py｜原：跑训练好的策略",
                "train_base_policy.py / train_dynaguide.py｜原：训练 DDIM 策略 / 动力学模型",
                "test_dynaguide_embedding.py｜原：动力学模型体检脚本",
                "analyze_calvin_touch.py｜原：行为分布统计",
                "requirements.txt / LICENSE / .gitignore｜★ .gitignore 改动：新增 checkpoints/",
            ]),
        ]),

        # ─────────────────────────────────────────────── 2
        ("③ core/ 原有文件（未改动，但要讲清它们和改造的关系）", [
            ("dynaguide.py｜★ 原 DDIM 引导函数", [
                "calculate_classifier_guidance｜潜空间 log-sum-exp 距离 → 对动作块求梯度",
                "calculate_position_guidance｜位置引导基线（ITPS）",
                "prepare_np｜numpy→torch 小工具",
                "与本改造关系：流版 calculate_flow_classifier_guidance 与它**同签名同返回**，所以 run_dynaguide.py 只换一个开关",
            ]),
            ("dynamics_models.py｜★ 原潜空间动力学 h_θ（DINOv2 版）", [
                "FinalStatePredictionDino｜冻结 DINOv2 patch + 位置编码 + 6 层 Transformer，预测末端 patch 嵌入",
                "  state_embedding / state_action_embedding｜给 goal bank 打嵌入 / 给引导打分（卫星版沿用同样两个接口）",
                "ActionEmbedding｜把动作块压成一个 token",
                "FinalStatePredictionDinoCLS / FinalStateClassification｜论文里的备选实现",
                "与本改造关系：卫星任务的 latent.py 就是「没有相机」时的同构替代",
            ]),
            ("calvin_utils.py｜CALVIN 场景复位（未动）", [
                "generate_reset_state / check_state_difference｜按实验配置摆场景、判断状态是否变化（算成功）",
            ]),
            ("embedder_datasets.py｜MultiviewDataset（未动）", [
                "按 hdf5 行为切窗口、取多相机帧与本体感知；引导条件的正负样本库就靠它读",
            ]),
            ("image_models.py｜VQVAE / ResNet18Dec（未动）", [
                "动力学模型训练时的重建分支（只用于可视化，不参与梯度）",
            ]),
        ]),

        # ─────────────────────────────────────────────── 3
        ("④ core/flows/｜★ 流匹配底座（7 文件，纯 torch，不依赖 diffusers）", [
            ("paths.py（约 200 行）｜流匹配的「坐标系」", [
                "作用：定义概率路径、速度、以及 x_t / 干净动作 / 噪声之间的互逆投影",
                ("LinearFlowPath｜核心类", [
                    "mu_t / sample_xt｜x_t = (1−t)ε + t·a₁（t=0 噪声，t=1 数据）",
                    "target_velocity｜真速度 u = a₁ − ε（沿直线恒定）",
                    "★ project_onto_data｜x̂₁ = x_t + (1−t)v ← 引导能量作用在这里",
                    "project_onto_noise｜x̂₀ = x_t − t·v（与上者互逆，有单测）",
                    "velocity_from_data｜反向：由干净动作反推速度",
                    "snr / sigma_t｜SNR=t²/(1−t)²，σ_t=1−t（→ 引导权重的来源）",
                ]),
                "flow_noise_fraction｜σ/√(1+σ²)：√(1−ᾱ) 在流匹配里的等价写法",
                "ddim_equivalence_table｜生成 DDIM 与流时间的等价对照表（写论文/答辩时查数）",
                "interpolate_endpoint｜单步把 x_t 推到段终点（一致性损失用）",
            ]),
            ("consistency.py（约 230 行）｜两个训练目标", [
                "rectified_flow_loss｜默认：L = E‖v_θ(x_t,t) − (a₁−ε)‖²（最朴素的回归）",
                ("ConsistencyFlowMatcher｜FlowPolicy 的一致性流匹配（可 1 步出动作）", [
                    "变量 eps / num_segments / boundary / delta / alpha｜时间下界、段数、贴边切换阈值、时间间隔、速度项权重",
                    "★ f_euler｜x_t + (段终点−t)·v ← 论文 f_θ，也是引导用的干净动作估计",
                    "threshold_based_f_euler｜贴近段终点时改用解析插值，避免跨段错误",
                    "loss｜位置一致性 + α·速度一致性（速度项会被双重 mask 掉）",
                ]),
                "consistency_flow_loss｜上面的便捷包装",
            ]),
            ("samplers.py（约 250 行）｜推理与「引导注入点」", [
                "FlowSamplerConfig｜n_steps / method(euler·midpoint·heun) / t_start·t_end / ss / ss_mode / time_scale",
                ("FlowSampler｜核心类", [
                    "sample｜主循环：宏步 × (ss 内层)；记录轨迹与引导范数",
                    "步骤｜k1=v(x,t) → x+=k1·dt（euler）；midpoint/heun 多估一次",
                    "★ _guided_velocity｜注入点 4 步：设可求导 → 网络给 v → 取 Δv → 相加",
                    "ss｜对应 DynaGuide Algorithm 1 的内层重复次数 M（split=再分步，refine=同层反复磨）",
                ]),
                "SamplerTrace｜每步的 state/time/guidance/energy，出图与诊断都靠它",
            ]),
            ("nets.py（约 180 行）｜网络结构", [
                "ConditionalUnet1D｜一维 UNet，FiLM 注入「时间嵌入 + 观测嵌入」",
                "ObsEncoder｜观测窗口 → 条件向量",
                "SinusoidalPosEmb / Conv1dBlock / ConditionalResidualBlock1D｜基础件",
            ]),
            ("policy.py（约 140 行）｜被引导的基础策略", [
                "FlowPolicyConfig｜obs_dim / horizon 16 / n_obs_steps 2 / n_action_steps 8 / objective fm|cfm / time_scale",
                ("FlowActionChunkPolicy｜核心类", [
                    "encode｜观测窗口 → 条件（obs_encoder）",
                    "velocity / velocity_fn｜(带噪动作, t, 条件) → v_θ",
                    "loss｜调用上面两个训练目标之一",
                    "sample｜建采样器 + 可选引导 → 反归一化 → 动作块",
                    "save / load｜连配置与归一化参数一起存取",
                ]),
                "★ 设计要点｜策略本身不知道「拆哪个模块、用什么手法」，所以学成三种示范的平均 → 这正是要引导的原因",
            ]),
            ("normalizer.py（约 60 行）｜动作归一化", [
                "LinearNormalizer.fit / fit_one / normalize / unnormalize / state_dict｜min-max 到 [-1,1]，保证训练稳定",
            ]),
            ("__init__.py｜把上面 6 个文件的常用符号导出，外部只 import 一次", []),
        ]),

        # ─────────────────────────────────────────────── 4
        ("⑤ core/flow_dynaguide.py（约 320 行）｜★ 本次改造的核心", [
            ("文件干什么｜把「能量」变成「速度场的修正 Δv」，并把原 DDIM 那套超参原样搬过来", [
                "输入：当前样本 x_t、时刻 t、当前速度 v、上下文（状态/模块号）",
                "输出：Δv 与日志（能量、梯度范数、权重）",
            ]),
            ("GuidanceWeightSpec｜λ(t) 四种调度（讲取舍）", [
                "fp = (1−t)²/2｜★默认：有界 + 自退火（从 Fokker-Planck 漂移 g²/2 推来）",
                "linear = 1−t｜只看一阶",
                "ddim = (1−t)/t｜严格照搬 DDIM 的 ε 参数化，t→0 发散 → 必须裁剪（clamp_max=50）",
                "const = 1｜纯强度旋钮（QGF 那类做法）",
                "power / clamp_max｜可调指数与上限",
            ]),
            ("FlowDynaGuide｜引导对象（dataclass）", [
                "字段｜energy、scale(s)、weight(λ)、mode、jacobian、clip_norm、clip_value、ema",
                "clean_estimate｜★ Tweedie 投影 x̂₁ = x_t+(1−t)v：能量只作用在干净动作上",
                "__call__｜★ 主流程：投影 → 算能量 → autograd 求梯度 → 护栏 → Δv",
                "  delta_v = −scale·λ(t)·grad｜★ 这一行就是原补丁的流匹配版（√(1−ᾱ) → λ(t)）",
                "_postprocess｜★ 三道护栏：按范数裁剪(5)、逐元素裁剪、跨步动量平滑",
                "guidance_fn｜适配 robomimic 的 (state, actions, t, velocity) 签名",
                "模式｜endpoint=Tweedie（默认）/ velocity=论文字面移植（消融）",
                "jacobian=False｜停梯度传输（单位映射，一次反向，稳）",
                "jacobian=True｜含 I+(1−t)∂v/∂x（RectifID 的 anchored guidance）",
            ]),
            ("make_flow_guidance｜工厂：把 --scale/--weight/--clip 变成引导对象", []),
            ("calculate_flow_classifier_guidance｜★ drop-in 接口", [
                "签名与 core/dynaguide.py 的 DDIM 版一致：返回 (guidance, good_emb, bad_emb)",
                "内部｜先用动力学模型给正/负样本库打嵌入，再包成 LatentMatchEnergy",
                "因此 run_dynaguide.py 只需换 --guidance_type flow",
            ]),
        ]),

        # ─────────────────────────────────────────────── 5
        ("⑥ core/flow_energies.py（约 360 行）｜★ 引导在优化什么", [
            ("两类能量｜都写成「标量代价、越小越好」，所以引导方向统一是 −∇J", []),
            ("LatentMatchEnergy｜沿用 DynaGuide 的思路", [
                "J = logΣexp(−d⁺/σ) − logΣexp(−d⁻/σ)｜d = 潜空间欧氏距离，σ 即 alpha",
                "log-sum-exp｜软最小值：多个正目标时自动挑「当前最可达」的那个，而不是平均",
                "正负相减｜对应论文「概率相除再取 log」，避免目标互斥",
                "字段｜latent_predictor、positives、negatives、alpha、mode(logsumexp|min|mean)",
            ]),
            ("DifferentiableDisassemblyModel｜可微的拆解代理", [
                "rollout(动作块, 状态, 模块号)｜沿动作块推演 16 步，返回 12 个可微标量",
                "返回量｜force_max / impulse / attitude_drift / pull_ratio / twist_ratio / blocked_pull / axis_err / capture_dist…",
                "协议状态机｜螺栓：转够 required_twist 才解锁；卡扣：压够 press_depth；销：拉力过破断力",
                "★ 反作用力矩｜ω += (r×(−F))·dt / I ← 微重力与地面任务的本质区别",
            ]),
            ("SatellitePhysicsEnergy｜6 项物理先验加权", [
                "protocol=−pull_ratio｜鼓励把模块真正拉出来",
                "premature_pull｜紧固件没解锁就拉：白费力还推转基座",
                "force=(F/force_budget)²｜接触力超预算",
                "momentum=(drift/attitude_budget)²+impulse｜基座被推转（微重力专属项）",
                "axis=切向位移｜拔出方向偏了会卡（jam）",
                "approach=离抓取点距离｜先得对着正确模块",
                "默认权重｜protocol 2、premature_pull 1.5、momentum 2、force 1、axis .25、approach .5",
            ]),
            ("CompositeEnergy｜按权重相加，支持「潜空间 + 物理」联合（both 方法）", []),
            ("张量版四元数工具｜quat_matrix_t / quat_multiply_t / rotvec_from_quat_t（与 numpy 版逐个对齐，有单测）", []),
        ]),

        # ─────────────────────────────────────────────── 6
        ("⑦ core/satellite/｜★ 任务载体：自由漂浮卫星拆解（7 文件）", [
            ("spec.py（约 210 行）｜场景与预算（所有数值的唯一来源）", [
                "JointSpec｜kind(bolt|latch|peg)、axis、required_twist、breakaway_force、press_depth、travel",
                "ModuleSpec｜name、size、mass、attach_pos、joint、capture_pos、priority、grasp_face",
                "SatelliteSpec｜base_mass 28.27、base_inertia(.62/.58/.41)、k_imp 900、force_limit 8、force_budget 4、attitude_budget 0.12、horizon 120、dp_scale .02、drot_scale .2",
                "make_sc1_spec｜三模块：payload_box(螺栓90°)、solar_panel(卡扣6mm)、battery_module(销7N)",
                "make_single_module_spec / TASK_NAMES / task_success_metrics｜消融与指标工具",
            ]),
            ("env.py（约 370 行）｜★ 仿真本体", [
                "proprio 布局｜0-2 基座位置、3-6 姿态四元数、7-9 线速度、10-12 角速度、13-15 末端位置、16-18 末端旋转、19 夹爪、20+ 每模块 5 维、末尾时间",
                "module_block(i)｜每模块 [attached, twist_ratio, press_ratio, pull_ratio, grasp_dist]",
                "reset｜随机初始姿态/速度、抖动末端；支持单任务与多任务序列",
                "step｜① 抓取判定 ② 末端运动 ③ 紧固件状态机 ④ 接触力 ⑤ 反作用到基座 ⑥ 判定",
                "  ★ 阻抗力｜只有被阻挡的位移才产生 F = k_imp·|dp_blocked|",
                "  ★ 反作用｜torque = r×(−F)，ω += torque·dt/I，姿态四元数积分 → 姿态漂移",
                "  success｜模块被拆下（pull ≥ travel，回合随即结束）",
                "  constraint_satisfied｜拆下 且 漂移 ≤ 0.12 rad 且 接触力 ≤ 4 N",
                "last_force / max_force / force_impulse｜逐步与累计力，供画图与逐块诊断",
                "clone()｜给「同一状态跑两条动作」的受控对比用",
            ]),
            ("expert.py（约 265 行）｜★ 三种示范策略（多模态来源）", [
                "ScriptedExpert｜按状态机分阶段：approach → grasp → prepare(拧/压) → extract → stow",
                "behavior=safe｜正规流程：先拧/先压 → 纯轴向拉（漂移小）",
                "behavior=impatient｜先拽 4 步再拧：力全变基座反作用（漂移大）",
                "behavior=offset｜拔出轴偏 0.25 rad：切向力持续推转基座（漂移最大）",
                "参数｜k_p、twist_step 0.1、press_step .0025、pull_ratio 1.7、impatient_steps 4、impatient_scale .18、tilt .25",
                "run_episode｜跑一个回合，返回 obs/actions/task_id/行为/是否达标",
                "collect_demonstrations｜按 behavior_mix 采样多回合，拼成数据集（含 behavior 与 ep_ok 标签）",
            ]),
            ("latent.py（约 210 行）｜★ 动力学模型 h_θ（本任务版）", [
                "SemanticLatentEncoder｜替代 DINOv2：22→14 维语义特征",
                "  特征｜2×四元数矢量(3)、角速度(3)、末端位置(3)、夹爪(1)、每模块 4 个进度量",
                "  ★ 固定名义尺度标准化｜attached 这类二值特征标准差≈0，用经验 std 会把潜空间尺度炸掉",
                "SemanticDynamicsModel｜(状态, 动作块) → 预测末端潜状态；3 层 MLP，hidden 256",
                "  state_embedding / state_action_embedding｜给 goal bank 打嵌入 / 给引导打分（与 DINO 版同接口）",
                "  ★ augment_actions｜50% 概率按同一条插值路径加噪（论文 Fig.9 的关键消融项）",
                "  loss｜潜空间 MSE（噪声动作→干净动作 的回归）",
                "build_goal_bank｜把一批成功末态嵌成 g⁺/g⁻ 库",
            ]),
            ("data.py（约 140 行）｜数据管道", [
                "build_chunk_dataset｜按回合切成 (观测窗口, 动作块, 末端观测)，不跨回合",
                "ChunkDataset / split_dataset｜torch 数据集与训练/验证切分",
                "end_state_bank｜正样本=安全回合末态（ep_ok=1），负样本=违约回合末态",
                "save_demos / load_demos｜npz 存取",
            ]),
            ("math3d.py（约 125 行）｜基础几何", [
                "四元数｜quat_multiply / quat_from_axis_angle / quat_to_rotvec / quat_matrix / integrate_quat",
                "工具｜unit / smoothstep / approach（专家控制用）",
            ]),
            ("__init__.py｜导出 SatelliteEnv、ScriptedExpert、collect_demonstrations、make_sc1_spec 等", []),
        ]),

        # ─────────────────────────────────────────────── 7
        ("⑧ robomimic/｜★ 接回 CALVIN 生态（证明方法是通用的）", [
            ("algo/flow_policy.py（约 330 行）｜★ 流匹配策略", [
                "与 diffusion_policy.py 的三处差异｜输出 ε→v；目标 MSE→fm/cfm；调度器 DDIM→FlowSampler",
                "FlowEMA｜轻量 EMA（避开 diffusers 版本差异）",
                "replace_bn_with_gn｜BN→GN，保证 EMA 与 batch=16 稳定",
                "FlowPolicyUNet._create_networks｜观测编码器 + ConditionalUnet1D(速度网) + EMA",
                "  train_on_batch｜freeze 观测编码 → 两条目标之一 → 反向 + EMA",
                "  get_action(guidance_type=\"flow\")｜采样 + 引导，出动作（动作队列缓存）",
                "  ★ _wrap_guidance｜兼容 4 参数(流) 与 2 参数(原版) 两种引导签名",
                "  _sample_chunk｜调 FlowSampler，记录 corrections_list / guidance_list（可视化用）",
                "serialize / deserialize｜权重存取",
                "边界｜只做了 CPU 合成观测集成测试，未跑真实 CALVIN 数据集",
            ]),
            ("config/flow_policy_config.py（约 60 行）｜配置类", [
                "algo.horizon｜观测 2 帧 / 动作 8 步 / 预测 16 步",
                "algo.unet｜时间嵌入 256、down_dims 256/512/1024、kernel 5、groups 8",
                "★ algo.flow｜objective(fm|cfm)、time_scale 99、t_min、t_start、sampler(euler)、num_inference_steps 4、cfm_*",
                "algo.ema｜power 0.75",
            ]),
            ("algo/__init__.py / config/__init__.py｜各 +1 行：注册 FlowPolicyUNet / FlowPolicyConfig", []),
            ("algo/diffusion_policy.py｜原 DDIM 策略，未改动", [
                "★ 讲解对照点：_get_action_trajectory_guidance 里那行 noise_pred −= √(1−ᾱ_k)·grad，就是被改造的对象",
            ]),
        ]),

        # ─────────────────────────────────────────────── 8
        ("⑨ scripts/｜★ 命令行（12 个）", [
            ("satellite/make_demos.py｜采集示范（三种行为混合）", [
                "关键参数｜--tasks、--per_task、--noise、--behavior_mix safe=0.6,impatient=0.2,offset=0.2、--tag",
            ]),
            ("satellite/train_flow_policy.py｜训练基础策略", [
                "关键参数｜--steps、--batch_size、--down_dims、--objective fm|cfm、--time_scale",
                "输出｜checkpoints/satellite/flow_policy_*.pt + 训练历史 JSON",
            ]),
            ("satellite/train_latent_dynamics.py｜训练动力学 + 建 goal bank", [
                "关键参数｜--steps、--hidden、--noise_prob 0.5",
                "输出｜latent_*.pt（含正/负样本嵌入库）",
            ]),
            ("satellite/run_flow_dynaguide.py｜★ 评测主程序", [
                "五方法｜base（不引导）/ latent（潜空间）/ physics（物理）/ both（联合）/ reject（best-of-16）",
                "关键参数｜--episodes、--steps、--alpha、--per_method_scale、--n_samples",
                "输出｜results/satellite/*.json + *.md（含逐方法逐任务表格）",
            ]),
            ("satellite/sweep_guidance.py｜引导强度扫描（s≥0.5 明显退化）", []),
            ("satellite/check_energy_alignment.py｜★ 逐块检验（最硬证据）", [
                "做法｜拿同一动作块，沿能量改进 3 步，在真实仿真里重放 → 对比漂移/接触力",
                "结果｜能量 −17.3%、漂移 18/25 改善、接触力 20/25 改善",
            ]),
            ("satellite/make_figures.py｜出 7 张彩色图（支持 --strip-title 供 Word）", []),
            ("satellite/debug_rollout.py｜打印一个回合的逐步状态（拧转/拉出/漂移）", []),
            ("satellite/common.py｜路径与种子等公共小工具", []),
            ("make_report_docx.py｜生成 Word 报告（数字读自运行产物）", []),
            ("make_mindmap.py｜生成这份思维导图（md/opml/mm/粘贴文本）", []),
            ("run_all_checks.ps1｜一键复现：测试 + 评测 + 诊断 + 从零管线 + 出图", []),
        ]),

        # ─────────────────────────────────────────────── 9
        ("⑩ tests/｜★ 34 项断言（CPU）", [
            ("test_flows.py（11 项）｜底座是否正确", [
                "投影互逆、真速度为零损失、采样器积分常数场、midpoint 优于 euler、CFM 可反传、归一化往返",
            ]),
            ("test_flow_dynaguide.py（10 项）｜★ 引导是否正确", [
                "λ(t) 各调度；引导确实降低二次能量；端点梯度与解析解逐元素相等；jacobian 含 I+(1−t)∂v",
                "潜空间能量与 DynaGuide Eq.2 公式一致；物理能量偏好「先拧后拉」；四元数 torch/numpy 一致",
            ]),
            ("test_satellite_env.py（9 项）｜仿真是否讲得通", [
                "观测布局；专家能解三模块；多目标回合；确定性；错误协议会把基座推转；预算能触发中止；安全/不安全示范可区分",
            ]),
            ("test_robomimic_flow_policy.py（4 项）｜集成是否成立", [
                "注册成功；fm/cfm 两目标可训练；get_action 带引导钩子被真正调用；非法 guidance_type 被拒",
            ]),
        ]),

        # ─────────────────────────────────────────────── 10
        ("⑪ 关键概念与变量速查（被问到就查这里）", [
            ("动作空间（7 维）｜a[0:3] 位移意图 ×0.02 m，a[3:6] 转动意图 ×0.2 rad，a[6] 夹爪", []),
            ("x_t / x̂₁ / v_θ / λ(t) / σ / s / ss / N", [
                "x_t｜t 时刻的带噪动作块；x̂₁｜干净动作估计；v_θ｜速度场",
                "λ(t)｜引导权重（fp=(1−t)²/2）；σ=alpha｜能量温度；s=scale｜引导强度",
                "ss｜同一宏步内重复次数（论文 M）；N｜ODE 步数",
            ]),
            ("goal bank g⁺ / g⁻｜正样本=安全回合末状态，负样本=违约回合末状态", []),
            ("安全预算｜姿态 0.12 rad、接触力 4 N（硬中止 8 N）、回合 120 步", []),
            ("三种紧固件｜螺栓转 90°、卡扣压 6 mm、销破断 7 N；travel 分别 0.09/0.07/0.10 m", []),
        ]),

        # ─────────────────────────────────────────────── 11
        ("⑫ 结果与证据在哪", [
            ("运行证据/01_tests｜四套测试的原始输出", []),
            ("运行证据/02_eval｜评测日志 + 全部 JSON", []),
            ("运行证据/03_train｜训练曲线 + 从零管线四步日志", []),
            ("运行证据/04_diag｜代理 vs 仿真诊断（含逐样本 CSV）", []),
            ("运行证据/06_figures｜7 张彩色图（report/ 子目录是给 Word 的无标题版）", []),
            ("主结果｜8 回合 base 88/75、latent 88/88；20 回合 base 85/80、latent 65/45", []),
            ("★ 逐块检验｜能量 −17.3%，漂移 18/25 改善，接触力 20/25 改善", []),
            ("示例回合｜seed=1002 漂移 0.070→0.014 rad", []),
        ]),

        # ─────────────────────────────────────────────── 12
        ("⑬ 讲解顺序（3 分钟走完）", [
            ("1｜fig7 框架图：训练 / 采样 / 引导三层，强调不改权重", []),
            ("2｜flow_dynaguide.py 顶部：不能照搬 → λ(t) → Tweedie", []),
            ("3｜该文件 delta_v 那行：对应原补丁", []),
            ("4｜samplers.py _guided_velocity：注入点 4 步", []),
            ("5｜flow_energies.py 反作用力矩：微重力难在哪", []),
            ("6｜fig2 + fig4：主结果与逐块检验（如实讲局限）", []),
        ]),
    ],
)


def _norm(node):
    if isinstance(node, str):
        return node, []
    title, *rest = node
    children = rest[0] if rest else []
    return title, [_norm(c) for c in children]


TREE = _norm(TREE)


def to_markdown(tree, depth=0) -> str:
    title, children = tree
    lines = [f"{'  ' * depth}- {title}"]
    for child in children:
        lines.append(to_markdown(child, depth + 1))
    return "\n".join(lines)


def to_paste(tree, depth=0) -> str:
    title, children = tree
    lines = ["\t" * depth + title]
    for child in children:
        lines.append(to_paste(child, depth + 1))
    return "\n".join(lines)


def to_opml(tree) -> str:
    def node(t, indent):
        title, children = t
        pad = "  " * indent
        if not children:
            return f'{pad}<outline text="{escape(title)}"/>'
        inner = "\n".join(node(c, indent + 1) for c in children)
        return f'{pad}<outline text="{escape(title)}">\n{inner}\n{pad}</outline>'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<opml version="1.0">\n  <head>\n'
        "    <title>DynaGuide 流匹配改造 · 全代码框架</title>\n  </head>\n  <body>\n"
        + node(tree, 3)
        + "\n  </body>\n</opml>\n"
    )


def to_freemind(tree) -> str:
    counter = [0]

    def node(t, indent):
        title, children = t
        pad = "  " * indent
        counter[0] += 1
        head = f'{pad}<node TEXT="{escape(title, {chr(34): "&quot;"})}" ID="ID_{counter[0]}">'
        if not children:
            return head + "</node>"
        inner = "\n".join(node(c, indent + 1) for c in children)
        return head + "\n" + inner + "\n" + pad + "</node>"

    return '<map version="1.0.1">\n<!-- FreeMind / 幕布 可导入 -->\n' + node(tree, 2) + "\n</map>\n"


def main() -> None:
    (OUT_DIR / "代码思维导图.md").write_text(to_markdown(TREE) + "\n", encoding="utf-8")
    (OUT_DIR / "mubu_paste.txt").write_text(to_paste(TREE) + "\n", encoding="utf-8")
    (OUT_DIR / "代码思维导图.opml").write_text(to_opml(TREE), encoding="utf-8")
    (OUT_DIR / "代码思维导图.mm").write_text(to_freemind(TREE), encoding="utf-8")

    def count(t):
        return 1 + sum(count(c) for c in t[1])

    def depth(t):
        return 1 + (max(depth(c) for c in t[1]) if t[1] else 0)

    print(f"nodes: {count(TREE)}  depth: {depth(TREE)}")
    for name in ("代码思维导图.md", "mubu_paste.txt", "代码思维导图.opml", "代码思维导图.mm"):
        p = OUT_DIR / name
        print(f"  {p}  ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
