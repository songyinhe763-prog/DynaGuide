"""Build the code mind map for the presentation (Mubu / 幕布 importable).

Defines one tree and emits four artefacts from it, so every format stays consistent:

    C:\\代码修改\\代码思维导图.md        Markdown 大纲（幕布"导入 Markdown"）
    C:\\代码修改\\代码思维导图.opml      OPML 1.0（幕布/XMind 通用）
    C:\\代码修改\\代码思维导图.mm        FreeMind XML（幕布"导入 FreeMind"）
    C:\\代码修改\\mubu_paste.txt        制表符缩进纯文本（直接在幕布里 Ctrl+V）

Usage::

    python scripts/make_mindmap.py
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path("C:/代码修改")

# 每个节点是 (标题, [子节点...])；标题里的 " | " 后面是给幕布的批注（相当于截图里的小字说明）
TREE = (
    "流匹配版 DynaGuide｜失效卫星拆解 · 代码思维导图（★ = 讲解重点）",
    [
        ("① 改造主线（3 分钟讲完这三句）", [
            ("问题：DynaGuide 只在 DDIM 的噪声预测上引导", [
                ("原来只有一行：noise_pred -= √(1−ᾱ_k) · s · grad | run_dynaguide.py / diffusion_policy.py"),
                ("√(1−ᾱ_k) 不是经验系数 | 它是 score→ε 参数化的雅可比因子 ∇log q = −ε/√(1−ᾱ)"),
                ("它顺带做了自动退火 | 噪声大时给满，接近干净数据时自动关掉"),
            ]),
            ("难点：流匹配预测的是速度 v，不是噪声 ε", [
                ("同位量是边际噪声标准差 σ_t = 1−t | 直接搬 √(1−ᾱ) 量纲不对"),
                ("严格换算得 λ(t)=(1−t)/t | 在 t→0（单步推理）发散 → 照搬会炸"),
                ("★ 采用 λ(t)=(1−t)²/2 | t→1 自动关断、t→0 有界，单步/四步都稳"),
            ]),
            ("★ 做法：v̂ = v_θ − s·λ(t)·∇J(x̂₁) | x̂₁ = x_t+(1−t)v_θ 是干净动作估计"),
        ]),

        ("② core/flows/｜流匹配底座（纯 torch，无 diffusers）", [
            ("paths.py｜流匹配的\"坐标系\"", [
                ("LinearFlowPath.mu_t / sample_xt | 直线插值 x_t=(1−t)ε+t·a₁"),
                ("target_velocity | 真速度 u = a₁ − ε（沿直线恒定）"),
                ("★ project_onto_data | x̂₁ = x_t+(1−t)v ：引导能量就作用在这个点上"),
                ("project_onto_noise | x̂₀ = x_t−t·v；与上者互逆（有单元测试）"),
                ("flow_noise_fraction | σ/√(1+σ²)：√(1−ᾱ) 在流匹配里的等价写法"),
            ]),
            ("consistency.py｜怎么训练速度场", [
                ("rectified_flow_loss | 默认目标：E‖v_θ − (a₁−ε)‖²"),
                ("ConsistencyFlowMatcher | FlowPolicy 的一致性流匹配（可单步）"),
                ("  参数 | delta=1e-2、boundary、alpha=1e-5、num_segments=2"),
                ("★ f_euler | x_t+(段终点−t)·v：论文 f_θ 的实现，也是引导用的干净动作估计"),
            ]),
            ("samplers.py｜推理与\"引导注入点\"", [
                ("FlowSamplerConfig | n_steps / method(euler|midpoint|heun) / ss / t_start"),
                ("FlowSampler.sample | 解 dx/dt=v_θ，从 t≈0 积到 1"),
                ("  宏步 + ss 内层 | ss 对应 DynaGuide Algorithm 1 的重复次数 M"),
                ("★ _guided_velocity | 引导注入点只有 4 步：设为可求导→网络给 v→拿 Δv→相加"),
                ("SamplerTrace | 记录每步引导范数与能量，供画图/分析"),
            ]),
            ("nets.py｜网络结构", [
                ("ConditionalUnet1D | 一维 UNet + FiLM 条件（时间 + 观测）"),
                ("ObsEncoder | 观测窗口 → 条件向量"),
            ]),
            ("policy.py｜被引导的基础策略", [
                ("encode / velocity / loss / sample | 三块结构，讲的时候按这三块走"),
                ("★ 设计：策略对\"拆哪个模块/用什么手法\"一无所知 | 所以会学成三种示范的平均 → 需要引导"),
                ("动作先归一化到 [-1,1] | 采样后反归一化再交给仿真"),
            ]),
            ("normalizer.py｜线性归一化 | 动作 min-max + 状态存取，纯 torch"),
        ]),

        ("③ core/flow_dynaguide.py｜★ 本次改造的核心", [
            ("GuidanceWeightSpec｜λ(t) 的四种调度", [
                ("fp = (1−t)²/2 | 默认：有界 + 自退火（从 Fokker-Planck 漂移推来）"),
                ("linear = 1−t | 只看一阶"),
                ("ddim = (1−t)/t | 严格照搬 DDIM，t→0 发散 → 必须裁剪（默认 50）"),
                ("const = 1 | 纯强度旋钮（QGF 那类做法）"),
            ]),
            ("FlowDynaGuide.clean_estimate｜Tweedie 投影", [
                ("x̂₁ = x_t + (1−t)·v | 能量/动力学模型只见过干净动作，不能对带噪点求梯度"),
            ]),
            ("FlowDynaGuide.__call__｜能量→梯度→Δv", [
                ("mode=endpoint | 默认：在干净动作上算能量（推荐）"),
                ("mode=velocity | DynaGuide 论文的字面移植，仅作消融"),
                ("jacobian=False | 停梯度传输（单位映射），一次反向，稳；True 才含 I+(1−t)∂v/∂x"),
                ("★ delta_v = −scale·λ(t)·grad | 这一行就是原补丁的流匹配版"),
            ]),
            ("_postprocess｜★ 三道梯度护栏", [
                ("clip_norm=5 | 按 L2 裁剪（默认开）"),
                ("clip_value / ema | 逐元素裁剪、跨步动量平滑（默认关）"),
            ]),
            ("make_flow_guidance｜工厂函数 | 把 --scale/--weight/--alpha 变成引导对象"),
            ("calculate_flow_classifier_guidance｜drop-in", [
                ("返回 (guidance, good_emb, bad_emb) | 与原 DDIM 版签名一致，run_dynaguide.py 只换一个开关"),
            ]),
        ]),

        ("④ core/flow_energies.py｜引导到底优化什么", [
            ("LatentMatchEnergy｜沿用 DynaGuide", [
                ("J = logΣexp(−d⁺/σ) − logΣexp(−d⁻/σ) | d = 潜空间欧氏距离"),
                ("log-sum-exp = 软最小值 | 多个正目标时自动挑\"当前最可达\"的那个，而不是平均"),
                ("正负相减 | 对应论文的\"概率相除再取 log\"，避免目标互斥"),
            ]),
            ("DifferentiableDisassemblyModel.rollout｜可微拆解代理", [
                ("协议状态机 | 螺栓:转够才解锁；卡扣:压够才解锁；销:拉力过破断力"),
                ("★ 反作用力矩 | ω += (r×(−F))·dt/I ：微重力与地面任务的本质区别"),
            ]),
            ("SatellitePhysicsEnergy｜6 项物理先验", [
                ("协议进度 / 过早拉拽 / 接触力 / 基座漂移 / 轴向偏差 / 是否对准模块"),
                ("默认权重 | protocol 2、premature_pull 1.5、momentum 2、force 1、axis .25、approach .5"),
            ]),
            ("CompositeEnergy｜按权重相加 | 支持把潜空间与物理能量合成一个总能量"),
        ]),

        ("⑤ core/satellite/｜任务载体（自由漂浮拆解）", [
            ("spec.py｜场景与预算", [
                ("make_sc1_spec | 基座 28.27 kg、惯量 0.62/0.58/0.41，三模块"),
                ("三种紧固 | 螺栓(转90°)、卡扣(压6mm)、摩擦销(7N)"),
                ("预算 | 姿态 0.12 rad、接触力 4N（硬中止 8N）、120 步"),
            ]),
            ("env.py｜★ 仿真本体", [
                ("step：动作→位移意图→阻抗力 | 只有被卡住的部分 F=k·|dp_blocked| 才产生接触力"),
                ("★ 反作用 | 力矩加回自由漂浮基座 → 姿态漂移；拆得越猛转得越多"),
                ("判定两档 | success=拆下；constraint=拆下且不超预算"),
                ("proprio 36 维 | 基座位姿/速度+末端+夹爪+每模块 5 维+时间"),
            ]),
            ("expert.py｜★ 三种示范策略（多模态来源）", [
                ("safe | 先拧/先压再纯轴向拉（漂移小、成功率高）"),
                ("impatient | 先拽 4 步再拧：力全变基座反作用"),
                ("offset | 拔出轴偏 0.25 rad：切向力持续推转基座"),
                ("为什么故意混不安全示范 | 复现\"示范质量参差→策略学到平均行为\"，引导把它拉回安全那支"),
            ]),
            ("latent.py｜动力学模型 h_θ（本任务版）", [
                ("SemanticLatentEncoder | 替代 DINOv2：14 维语义特征，固定名义尺度标准化"),
                ("SemanticDynamicsModel | (状态,动作块) → 预测末端潜状态，3 层 MLP"),
                ("★ augment_actions | 50% 概率按同一条插值路径加噪（论文 Fig.9 的关键消融项）"),
                ("state_embedding / state_action_embedding | 给 goal bank 打嵌入 / 给引导打分"),
            ]),
            ("data.py｜数据管道", [
                ("build_chunk_dataset | 按回合切 (观测窗口, 动作块, 末端观测)"),
                ("end_state_bank | 正样本=安全回合末状态；负样本=违约回合末状态"),
            ]),
            ("math3d.py｜四元数与旋转 | 姿态积分、最小旋转向量"),
        ]),

        ("⑥ robomimic 集成｜证明不是只能在玩具环境跑", [
            ("algo/flow_policy.py｜FlowPolicyUNet", [
                ("与 diffusion_policy 的三处差异 | 输出 ε→v、目标 MSE→fm/cfm、DDIM→FlowSampler"),
                ("★ _wrap_guidance | 兼容 4 参数(流) 与 2 参数(原版) 两种引导签名"),
                ("get_action(guidance_type=\"flow\") | 给 run_dynaguide.py 的接口"),
                ("边界 | 只做了 CPU 合成观测集成测试，未跑真实 CALVIN 数据集"),
            ]),
            ("config/flow_policy_config.py | algo.flow.* 配置类"),
            ("configs/flow_policy_image_calvin.json | CALVIN 训练配置（算法名 flow_policy）"),
            ("calvin_exp_configs_examples/switch_on_flow.json | 引导实验配置示例"),
        ]),

        ("⑦ scripts/satellite/｜实验与证据管线（10 个脚本）", [
            ("make_demos.py | 采集三种策略混合的示范"),
            ("train_flow_policy.py | 训练基础策略（fm 或 cfm）"),
            ("train_latent_dynamics.py | 训练动力学 + 建 goal bank"),
            ("★ run_flow_dynaguide.py | 五方法对比：base/latent/physics/both/reject"),
            ("sweep_guidance.py | 引导权重扫描（必须扫，s≥0.5 明显退化）"),
            ("check_energy_alignment.py | 逐块检验：改进动作块后在真实仿真重放"),
            ("make_figures.py | 出 7 张彩色图（支持 --strip-title 供 Word）"),
            ("debug_rollout.py | 打印一个回合的逐步状态（拧转/拉出/漂移）"),
            ("run_all_checks.ps1 | 一键复现：测试+评测+诊断+从零管线+出图"),
        ]),

        ("⑧ tests/｜34 项断言全过（CPU）", [
            ("test_flows.py (11) | 投影互逆、损失可训练到 0、采样器积分、midpoint 优于 euler"),
            ("★ test_flow_dynaguide.py (10) | 端点梯度=解析解、jacobian 含 I+(1−t)∂v、能量与 Eq.2 一致、物理能量偏好\"先拧后拉\""),
            ("test_satellite_env.py (9) | 专家能解各模块、错误协议会把基座推转、预算能触发中止"),
            ("test_robomimic_flow_policy.py (4) | 注册+训练+带引导采样+非法参数被拒"),
        ]),

        ("⑨ 结果（如实讲）", [
            ("主结果 8 回合 | base 88%/75%、latent 88%/88%、physics 62%/62%、reject 0%"),
            ("主结果 20 回合 | base 85%/80%、latent 65%/45%、physics 70%/45%"),
            ("★ 逐块检验（最硬证据） | 代理能量 −17.3%；漂移 18/25 改善（−14%）；接触力 20/25 改善（−16%）"),
            ("示例回合 seed=1002 | 漂移 0.070→0.014 rad，提前完成拧转"),
            ("局限 | CPU；样本量不足（排名会翻转）；闭环成功率暂无显著提升；reject 基线有已知缺陷"),
        ]),

        ("⑩ 讲解顺序（照着走 3 分钟）", [
            ("1. fig7 框架图 | 三层：训练 / 采样 / 推理期引导"),
            ("2. flow_dynaguide.py 顶部 | 三件事：不能照搬、λ(t)、Tweedie"),
            ("3. 该文件 delta_v 那行 | 对应原补丁"),
            ("4. samplers.py _guided_velocity | 注入点 4 步"),
            ("5. flow_energies.py 反作用力矩 | 微重力难在哪"),
            ("6. fig2 + fig4 | 主结果与逐块检验"),
        ]),
    ],
)


def _norm(node):
    """Accept both ``"title"`` and ``("title", [children])`` shapes."""
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

    body = node(tree, 3)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<opml version="1.0">\n  <head>\n    <title>流匹配版 DynaGuide 代码思维导图</title>\n'
        "  </head>\n  <body>\n" + body + "\n  </body>\n</opml>\n"
    )


def to_freemind(tree) -> str:
    def node(t, indent, nid=[0]):
        title, children = t
        pad = "  " * indent
        nid[0] += 1
        head = f'{pad}<node TEXT="{escape(title, {chr(34): "&quot;"})}" ID="ID_{nid[0]}">'
        if not children:
            return head + "</node>"
        inner = "\n".join(node(c, indent + 1, nid) for c in children)
        return head + "\n" + inner + "\n" + pad + "</node>"

    body = node(tree, 2)
    return (
        '<map version="1.0.1">\n'
        "<!-- FreeMind / 幕布 可导入 -->\n"
        + body
        + "\n</map>\n"
    )


def main() -> None:
    md = to_markdown(TREE)
    (OUT_DIR / "代码思维导图.md").write_text(md + "\n", encoding="utf-8")
    (OUT_DIR / "mubu_paste.txt").write_text(to_paste(TREE) + "\n", encoding="utf-8")
    (OUT_DIR / "代码思维导图.opml").write_text(to_opml(TREE), encoding="utf-8")
    (OUT_DIR / "代码思维导图.mm").write_text(to_freemind(TREE), encoding="utf-8")

    def count(t):
        return 1 + sum(count(c) for c in t[1])

    print(f"nodes: {count(TREE)}")
    for name in ("代码思维导图.md", "mubu_paste.txt", "代码思维导图.opml", "代码思维导图.mm"):
        p = OUT_DIR / name
        print(f"  {p}  ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
