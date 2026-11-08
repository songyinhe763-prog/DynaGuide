"""Build the code map mind map: folder -> file -> function -> one-line note.

Every folder and every file carries an annotation. For the files this project touched, the
important classes / functions are expanded and the modification is marked (★新增 / ★改动).
Third-party trees are collapsed into one short node.

Outputs (one source -> four formats):

    C:\\代码修改\\代码思维导图.md      Markdown 大纲（幕布「导入 Markdown」/ 直接粘贴）
    C:\\代码修改\\mubu_paste.txt      制表符缩进（幕布里 Ctrl+V 即成分层）
    C:\\代码修改\\代码思维导图.mm      FreeMind XML（幕布「导入 FreeMind」，天然可折叠）
    C:\\代码修改\\代码思维导图.opml    OPML 1.0

Usage::

    python scripts/make_codemap.py
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path("C:/代码修改")

TREE = (
    "DynaGuide-流匹配改造（失效卫星拆解）｜代码地图  ★=本次新增/改动",
    [
        ("根目录文件（9 个）", [
            ("README.md｜★改动｜新增「流匹配版 DynaGuide」章节：用法、推导摘要、命令", []),
            ("run_dynaguide.py（约 700 行）｜★改动｜引导实验主程序", [
                ("rollout()｜一个回合的循环：采样动作块 → 送入仿真 → 统计成功", []),
                ("★新增 --guidance_type {diffusion,flow}｜切换 DDIM 引导 / 流匹配引导", []),
                ("★新增 --flow_weight｜流匹配引导权重 λ(t)：fp|linear|ddim|const", []),
                ("★改动 分支｜guidance_type=flow 时改用 calculate_flow_classifier_guidance", []),
                ("原逻辑保留｜默认 diffusion，行为与改造前一致", []),
            ]),
            ("run_sampling_baseline.py｜原有｜拒绝采样基线（采多条按度量择优）", []),
            ("run_trained_agent.py｜原有｜加载训练好的策略跑回合", []),
            ("train_base_policy.py｜原有｜训练 DDIM 扩散策略", []),
            ("train_dynaguide.py｜原有｜训练潜空间动力学模型（DINOv2 + Transformer）", []),
            ("test_dynaguide_embedding.py｜原有｜动力学模型体检（预测/重建/方差）", []),
            ("analyze_calvin_touch.py｜原有｜统计各行为触发率", []),
            (".gitignore｜★改动｜新增 checkpoints/（权重不入库）", []),
            ("requirements.txt / LICENSE｜原有｜依赖与许可", []),
        ]),
        ("core/｜算法核心（原有 5 文件 + 新增 2 文件 + 2 子包）", [
            ("★新增 flow_dynaguide.py｜流匹配版引导（本次核心）", [
                ("GuidanceWeightSpec｜λ(t) 的四种调度", [
                    ("__call__()｜fp=(1−t)²/2 默认（有界+自退火）；linear；ddim=(1−t)/t（t→0 发散需裁剪）；const", []),
                ]),
                ("FlowDynaGuide｜把能量变成速度修正 Δv", [
                    ("clean_estimate()｜Tweedie 投影 x̂₁=x_t+(1−t)v：能量只作用在干净动作上", []),
                    ("__call__()｜投影 → 能量 → autograd 梯度 → 护栏 → Δv", []),
                    ("  ★ delta_v = −scale·λ(t)·grad｜原补丁 noise_pred −= √(1−ᾱ)·grad 的流匹配版", []),
                    ("_postprocess()｜三道护栏：L2 裁剪 5.0 / 逐元素裁剪 / 跨步动量平滑", []),
                    ("guidance_fn()｜适配 robomimic 的 (state, actions, t, velocity) 签名", []),
                ]),
                ("make_flow_guidance()｜工厂：命令行参数 → 引导对象", []),
                ("calculate_flow_classifier_guidance()｜★ drop-in：与原 DDIM 版同签名同返回", []),
            ]),
            ("★新增 flow_energies.py｜引导在优化什么（两类能量）", [
                ("LatentMatchEnergy｜沿用 DynaGuide：logΣexp(−d⁺/σ) − logΣexp(−d⁻/σ)", [
                    ("软最小值｜多个正目标自动挑最可达的那个；正负相减=概率相除再取 log", []),
                ]),
                ("DifferentiableDisassemblyModel.rollout()｜可微拆解代理，推演 16 步", [
                    ("返回 12 个可微标量｜force_max/impulse/attitude_drift/pull_ratio/blocked_pull/axis_err…", []),
                    ("协议状态机｜螺栓转够才解锁；卡扣压够才解锁；销拉力过破断力", []),
                    ("★ 反作用力矩 ω += (r×(−F))·dt/I｜微重力与地面的本质区别", []),
                ]),
                ("SatellitePhysicsEnergy｜6 项加权：protocol 2 / premature_pull 1.5 / momentum 2 / force 1 / axis .25 / approach .5", []),
                ("CompositeEnergy｜潜空间与物理能量加权相加（both 方法）", []),
                ("quat_*_t｜张量版四元数工具，与 numpy 版逐个对齐", []),
            ]),
            ("dynaguide.py（原有，未改）｜★原 DDIM 引导，对照用", [
                ("calculate_classifier_guidance()｜潜空间距离→动作块梯度（流版签名参照它）", []),
                ("calculate_position_guidance()｜位置引导基线（ITPS）", []),
            ]),
            ("dynamics_models.py（原有，未改）｜★原动力学 h_θ（DINOv2 版）", [
                ("FinalStatePredictionDino｜冻结 DINOv2 patch + 6 层 Transformer → 预测末端嵌入", []),
                ("  state_embedding / state_action_embedding｜卫星版 latent.py 沿用同接口", []),
                ("ActionEmbedding｜动作块压成一个 token", []),
            ]),
            ("calvin_utils.py（原有，未改）｜CALVIN 场景复位与成功判定", []),
            ("embedder_datasets.py（原有，未改）｜MultiviewDataset：切窗口、读正负样本库", []),
            ("image_models.py（原有，未改）｜VQVAE / ResNet18Dec：重建分支（仅可视化）", []),
            ("flows/｜★新增｜流匹配底座（7 文件，纯 torch）", [
                ("paths.py｜流匹配的坐标系", [
                    ("mu_t / sample_xt｜x_t=(1−t)ε+t·a₁（t=0 噪声，t=1 数据）", []),
                    ("target_velocity｜真速度 u=a₁−ε", []),
                    ("★ project_onto_data｜x̂₁=x_t+(1−t)v：引导能量作用点", []),
                    ("project_onto_noise｜x̂₀=x_t−t·v（与上者互逆，有单测）", []),
                    ("flow_noise_fraction｜σ/√(1+σ²)：√(1−ᾱ) 的流匹配等价写法", []),
                ]),
                ("consistency.py｜两个训练目标", [
                    ("rectified_flow_loss()｜默认：E‖v_θ−(a₁−ε)‖²", []),
                    ("ConsistencyFlowMatcher｜一致性流匹配（可 1 步）：eps/segments/boundary/delta/alpha", []),
                    ("  f_euler()｜x_t+(段终点−t)·v = 论文 f_θ，也是干净动作估计", []),
                ]),
                ("samplers.py｜推理与引导注入点", [
                    ("FlowSamplerConfig｜n_steps / method / t_start / ss / ss_mode / time_scale", []),
                    ("FlowSampler.sample()｜解 dx/dt=v_θ：宏步 × ss 内层，记录轨迹", []),
                    ("★ _guided_velocity()｜注入点 4 步：设可求导→网络给 v→取 Δv→相加", []),
                    ("SamplerTrace｜每步引导范数/能量，出图与诊断用", []),
                ]),
                ("nets.py｜网络结构", [
                    ("ConditionalUnet1D｜一维 UNet + FiLM（时间嵌入+观测）", []),
                    ("ObsEncoder｜观测窗口→条件向量", []),
                ]),
                ("policy.py｜被引导的基础策略", [
                    ("FlowActionChunkPolicy｜encode / velocity / loss / sample 三块结构", []),
                    ("★ 设计｜策略不知道要拆哪个模块→学到三种示范的平均→需要引导", []),
                    ("save / load｜连归一化参数一起存", []),
                ]),
                ("normalizer.py｜LinearNormalizer：动作 min-max 到 [−1,1]", []),
                ("__init__.py｜导出常用符号", []),
            ]),
            ("satellite/｜★新增｜拆解仿真与数据（7 文件）", [
                ("spec.py｜场景与预算（数值唯一来源）", [
                    ("JointSpec / ModuleSpec / SatelliteSpec｜紧固件、模块、整机参数", []),
                    ("make_sc1_spec()｜基座 28.27 kg；三模块：螺栓90°/卡扣6mm/销7N", []),
                    ("预算｜漂移 0.12 rad、接触力 4 N（硬中止 8 N）、120 步", []),
                ]),
                ("env.py｜★仿真本体", [
                    ("proprio 36 维｜基座位姿/速度+末端+夹爪+每模块 5 维+时间", []),
                    ("step()｜抓取→末端运动→紧固件状态机→接触力→反作用→判定", []),
                    ("  阻抗力｜只有被阻挡的位移产生 F=k·|dp_blocked|", []),
                    ("  ★反作用｜torque=r×(−F)，ω+=torque·dt/I → 基座姿态漂移", []),
                    ("  判定｜success=拆下；constraint=拆下且不超预算", []),
                ]),
                ("expert.py｜★三种示范策略（多模态来源）", [
                    ("safe｜先拧/先压→纯轴向拉；impatient｜先拽 4 步再拧；offset｜拔出轴偏 0.25 rad", []),
                    ("ScriptedExpert.act()｜状态机 approach→grasp→prepare→extract→stow", []),
                    ("collect_demonstrations()｜按 behavior_mix 采样并打 ep_ok 标签", []),
                ]),
                ("latent.py｜★动力学 h_θ（本任务版）", [
                    ("SemanticLatentEncoder｜替代 DINOv2：14 维语义特征，固定名义尺度标准化", []),
                    ("SemanticDynamicsModel｜(状态,动作块)→末端潜状态；3 层 MLP", []),
                    ("★ augment_actions()｜50% 概率按同一插值路径加噪（论文 Fig.9 关键项）", []),
                    ("state_embedding / state_action_embedding｜打 goal bank / 给引导打分", []),
                ]),
                ("data.py｜数据管道", [
                    ("build_chunk_dataset()｜按回合切 (观测窗口, 动作块, 末端观测)", []),
                    ("end_state_bank()｜正样本=安全回合末态；负样本=违约回合末态", []),
                ]),
                ("math3d.py｜四元数与旋转（姿态积分等）", []),
                ("__init__.py｜导出环境与专家", []),
            ]),
        ]),
        ("robomimic/｜策略库（第三方 + ★新增一个算法）", [
            ("robomimic/algo/flow_policy.py｜★新增｜流匹配策略（约 330 行）", [
                ("与 diffusion_policy 的三处差异｜输出 ε→v；目标 MSE→fm/cfm；DDIM→FlowSampler", []),
                ("FlowPolicyUNet._create_networks()｜观测编码器 + ConditionalUnet1D + EMA", []),
                ("train_on_batch()｜fm 或 cfm 两条目标之一 + 反向 + EMA", []),
                ("get_action(guidance_type=\"flow\")｜采样 + 引导 → 动作", []),
                ("★ _wrap_guidance()｜兼容 4 参数(流) 与 2 参数(原版) 两种引导签名", []),
                ("_sample_chunk()｜记录 corrections_list / guidance_list（可视化）", []),
                ("边界｜只做 CPU 合成观测集成测试，未跑真实 CALVIN", []),
            ]),
            ("robomimic/config/flow_policy_config.py｜★新增｜配置类", [
                ("algo.horizon｜观测 2 帧 / 动作 8 步 / 预测 16 步", []),
                ("★ algo.flow｜objective、time_scale 99、t_min、sampler、num_inference_steps 4、cfm_*", []),
            ]),
            ("robomimic/algo/__init__.py｜★改动 +1 行｜注册 FlowPolicyUNet", []),
            ("robomimic/config/__init__.py｜★改动 +1 行｜注册 FlowPolicyConfig", []),
            ("robomimic/algo/diffusion_policy.py｜原有（未改）｜★讲对照点", [
                ("_get_action_trajectory_guidance()｜noise_pred −= √(1−ᾱ_k)·grad 就是被改造的那行", []),
            ]),
            ("robomimic/utils、models、envs、scripts｜第三方基础设施（未改动）", []),
        ]),
        ("configs/ 与 calvin_exp_configs_examples/｜配置", [
            ("★ configs/flow_policy_image_calvin.json｜新增｜流策略训练配置（algo_name=flow_policy + algo.flow.*）", []),
            ("configs/diffusion_policy_image_calvin_ddim.json｜原有｜DDIM 基线配置", []),
            ("★ calvin_exp_configs_examples/switch_on_flow.json｜新增｜流引导实验配置", []),
            ("calvin_exp_configs_examples/*.json｜原有｜各任务引导配置", []),
        ]),
        ("scripts/｜★新增｜命令行（12 个）", [
            ("satellite/make_demos.py｜采示范：--tasks --per_task --noise --behavior_mix", []),
            ("satellite/train_flow_policy.py｜训练基础策略：--steps --batch_size --down_dims --objective", []),
            ("satellite/train_latent_dynamics.py｜训练动力学 + 建 goal bank：--steps --noise_prob", []),
            ("satellite/run_flow_dynaguide.py｜★评测主程序：base/latent/physics/both/reject", [
                ("build_energy()｜方法名→能量对象的开关（正负样本库在此挂上）", []),
                ("rollout_episode()｜每 8 步重规划；记录 success/constraint/drift/force", []),
            ]),
            ("satellite/sweep_guidance.py｜引导强度扫描（s≥0.5 明显退化）", []),
            ("satellite/check_energy_alignment.py｜★逐块检验：改进动作块后在真实仿真重放", []),
            ("satellite/make_figures.py｜出 7 张彩色图（--strip-title 供 Word）", []),
            ("satellite/debug_rollout.py｜打印一个回合的逐步状态", []),
            ("satellite/common.py｜路径与种子等公共工具", []),
            ("make_report_docx.py｜生成 Word 报告", []),
            ("make_codemap.py｜生成这份思维导图", []),
            ("run_all_checks.ps1｜一键复现：测试+评测+诊断+从零管线+出图", []),
        ]),
        ("tests/｜★新增｜34 项断言（CPU）", [
            ("test_flows.py（11）｜投影互逆、真速度零损失、采样器积分、midpoint 优于 euler、CFM 可反传", []),
            ("test_flow_dynaguide.py（10）｜★端点梯度=解析解、jacobian 含 I+(1−t)∂v、能量=Eq.2、物理能量偏好先拧后拉", []),
            ("test_satellite_env.py（9）｜★错误协议会把基座推转、预算能中止、安全/不安全示范可区分", []),
            ("test_robomimic_flow_policy.py（4）｜注册、fm/cfm 可训练、带引导 get_action、非法参数被拒", []),
        ]),
        ("docs/｜★新增｜3 篇", [
            ("FLOW_DYNAGUIDE.md｜推导（√(1−ᾱ) 的雅可比含义、λ(t) 取舍）+ API + 超参对照", []),
            ("SATELLITE_DISASSEMBLY.md｜任务、物理、四维评测口径", []),
            ("RESULTS.md｜实验表、波动分析、下一步", []),
        ]),
        ("第三方目录（未改动，只讲一句用途）", [
            ("calvin/｜CALVIN 仿真环境（PyBullet 场景、机器人、相机、任务）", []),
            ("data_processing_calvin/｜CALVIN 数据切片与引导条件生成", []),
            ("paper_experiments/｜原论文 5 组实验的跑批与曲线脚本", []),
            ("toy_squares_experiment/｜原玩具实验（pymunk 方块触碰）", []),
        ]),
        ("运行产物（gitignore，可复现）", [
            ("data/satellite/｜demos_*.npz 示范数据 + 元信息 JSON", []),
            ("checkpoints/satellite/｜*.pt 权重 + 训练历史 JSON", []),
            ("results/satellite/｜评测 JSON/MD、扫描结果", []),
            ("logs/｜原始 stdout 日志", []),
            ("运行证据/｜★测试日志、评测、诊断、06_figures 七张图（在 C:\\代码修改 下）", []),
        ]),
        ("讲解主线（3 分钟，按顺序指屏幕）", [
            ("1｜fig7 框架图：训练 / 采样 / 引导三层，强调「不改一个权重」", []),
            ("2｜flow_dynaguide.py 顶部：不能照搬 → λ(t)=(1−t)²/2 → Tweedie 求梯度", []),
            ("3｜同文件 delta_v 那行：原补丁的流匹配版", []),
            ("4｜samplers.py _guided_velocity：注入点 4 步", []),
            ("5｜flow_energies.py 反作用力矩：微重力难在哪", []),
            ("6｜fig2 + fig4：主结果与逐块检验（局限如实讲）", []),
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
        "    <title>DynaGuide 流匹配改造 · 代码地图</title>\n  </head>\n  <body>\n"
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
