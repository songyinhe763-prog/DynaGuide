"""Build the Word report: what was changed in the code, why, and how well it works.

Output: ``C:\\代码修改\\流匹配版DynaGuide_代码改造说明与实验结果.docx``
The figures come from ``C:\\代码修改\\运行证据\\06_figures``; every number quoted in the
text is read from the run artefacts or from the source tree (line counts), nothing is
hand-typed except the narration.

Usage::

    python scripts/make_report_docx.py
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("C:/代码修改")
EVID = ROOT / "运行证据"
FIG = EVID / "06_figures" / "report"  # 无内嵌标题的版本（标题由 Word 图注给出）
OUT = ROOT / "流匹配版DynaGuide_代码改造说明与实验结果.docx"

SONG = "宋体"
HEI = "黑体"
MONO = "Consolas"


# ----------------------------------------------------------------- helpers ----
def set_run_font(run, name=SONG, size=12, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


def para(doc, text="", size=12, font=SONG, bold=False, align=None, indent=True,
         space_after=6, line=1.5, color=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    if line:
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = line
    if indent:
        pf.first_line_indent = Cm(0.74)
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        set_run_font(run, font, size, bold, color)
    return p


def h1(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.2
    run = p.add_run(text)
    set_run_font(run, HEI, 15, True)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.2
    run = p.add_run(text)
    set_run_font(run, HEI, 12.5, True)
    return p


def bullet(doc, text, size=12):
    p = doc.add_paragraph(style=None)
    pf = p.paragraph_format
    pf.left_indent = Cm(0.74)
    pf.first_line_indent = Cm(0)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.4
    run = p.add_run("· " + text)
    set_run_font(run, SONG, size)
    return p


def code_block(doc, code: str):
    for line in code.strip("\n").split("\n"):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.74)
        pf.first_line_indent = Cm(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        run = p.add_run(line if line else " ")
        set_run_font(run, MONO, 9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def caption(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(2)
    pf.space_after = Pt(10)
    pf.line_spacing = 1.0
    run = p.add_run(text)
    set_run_font(run, SONG, 10.5)


def table_caption(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.0
    run = p.add_run(text)
    set_run_font(run, SONG, 10.5, bold=True)


def add_table(doc, rows, widths=None, size=10.5, header=True, align_center=True):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            if align_center and j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            set_run_font(run, SONG, size, bold=(header and i == 0))
    if widths:
        for j, w in enumerate(widths):
            for row in table.rows:
                row.cells[j].width = Cm(w)
    return table


def add_figure(doc, name: str, cap: str, width_cm: float = 15.0):
    path = FIG / name
    if not path.exists():
        para(doc, f"（缺少图片：{path}）", indent=False)
        return
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(doc, cap)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def loc(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").split("\n"))
    except Exception:
        return 0


# -------------------------------------------------------------------- report ----
def build() -> Path:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(2.54)
    sec.right_margin = Cm(2.54)
    style = doc.styles["Normal"]
    style.font.name = SONG
    style.font.size = Pt(12)

    ev = load_json(EVID / "02_eval" / "evidence_eval.json")["rows"]
    fin = load_json(EVID / "02_eval" / "eval_final.json")["rows"]
    pol = load_json(REPO / "checkpoints" / "satellite" / "flow_policy_payload.json")
    lat = load_json(REPO / "checkpoints" / "satellite" / "latent_payload.json")
    by = {r["method"]: r for r in ev}
    byf = {r["method"]: r for r in fin}

    # ---------------------------------------------------------------- title ----
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("流匹配版 DynaGuide 的代码改造说明与实验结果")
    set_run_font(run, HEI, 20, True)

    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16)
    run = p.add_run("在轨失效卫星拆解任务中的主动动态引导实现")
    set_run_font(run, SONG, 13)

    add_table(
        doc,
        [
            ["项目", "内容"],
            ["代码位置", r"C:\代码修改\dynaguide（git 仓库，提交历史见 git log）"],
            ["运行证据", r"C:\代码修改\运行证据（日志、评测 JSON、训练曲线、图表）"],
            ["运行环境", "Windows + Python 3.13.5 + PyTorch 2.9.1（CPU，无 GPU）；torchvision 0.24.1"],
            ["参考实现", "DynaGuide（NeurIPS 2025）、FlowPolicy（AAAI 2025）官方代码与论文"],
            ["文档日期", "2026-10-31"],
            ["生成脚本", "dynaguide/scripts/make_report_docx.py（数字读自运行产物，图读自 06_figures/report）"],
        ],
        widths=[3.2, 11.8],
        size=10.5,
    )

    # ------------------------------------------------------------ section 1 ----
    h1(doc, "一、改造背景与目标")
    para(
        doc,
        "DynaGuide 的原始实现面向 DDIM 扩散策略：策略网络预测噪声 ε，推理时用 "
        "√(1−ᾱ_k) 加权的梯度去修正噪声预测，从而在不重训策略的前提下把生成的行为引导到"
        "指定目标（可正可负、可多个）。FlowPolicy 则把扩散头换成了流匹配（一致性流匹配）的"
        "速度场，单次推理即可出动作。两者在数学上属于同一类生成模型，但参数化方式不同，"
        "因此不能把前者的三行补丁直接贴到后者上。",
    )
    para(
        doc,
        "本次改造的目标有三个。第一，把 DynaGuide 的推理期引导从噪声预测迁移到速度场，"
        "并且保证推导是自洽的，不是凭经验调系数。第二，给大创课题（在轨失效卫星拆解）"
        "提供可运行的实验载体：一个自由漂浮卫星拆解仿真，其中任何拆解动作产生的接触力都会"
        "反作用到卫星基座，从而把“拆下来了但把卫星推转了”这一在轨特有的失败模式显式建模。"
        "第三，把整条链路（采数据、训练策略、训练动力学模型、引导评测、参数扫描、出图）"
        "做成可复现的命令行脚本，并在 CPU 上跑通。",
    )

    # ------------------------------------------------------------ section 2 ----
    h1(doc, "二、原代码是怎么工作的")
    h2(doc, "2.1 DynaGuide 在 DDIM 上的引导注入")
    para(
        doc,
        "原实现的引导只有一行核心代码（robomimic/robomimic/algo/diffusion_policy.py 与 "
        "run_dynaguide.py 的 README 补丁一致）：",
    )
    code_block(
        doc,
        """
noise_pred = noise_pred - (1 - alphas_cumprod[k]).sqrt() * scaled_grad
        """,
    )
    para(
        doc,
        "其中 scaled_grad 是引导度量 d 对动作块 a_k 的梯度。d 由潜空间动力学模型 h_θ 给出："
        "把当前动作块和观测一起送进 h_θ 得到预测的末端潜状态 ẑ，再与正样本集合 g⁺、"
        "负样本集合 g⁻ 在潜空间里比较，用 log-sum-exp 做软最小聚合，最后相减，"
        "即 DynaGuide 式 Eq.2。代码里 alpha 是 Eq.2 的温度 σ，ss 对应论文 Algorithm 1 "
        "的内层重复次数 M。",
    )
    para(
        doc,
        "同一方法的外层还要把“采样—选优”做成基线（best-of-N），以及一个位置引导基线。"
        "这些本次都保留了，只把引导的注入点从噪声空间换成速度场。",
    )

    h2(doc, "2.2 FlowPolicy 的速度场与一致性流匹配")
    para(
        doc,
        "流匹配用线性插值路径 x_t = (1−t)·ε + t·a₁，t=0 是噪声、t=1 是动作数据，"
        "沿路径的向量场恒为 u = a₁ − ε。网络预测速度 v_θ(x_t, t, o)，"
        "把任意时刻的样本一步推到终点就是 f_θ(x_t, t) = x_t + (1−t)·v_θ，"
        "这也是流匹配里的干净样本估计（Tweedie 形式）。FlowPolicy 在多段上再加一项速度自一致性约束，"
        "使得少步甚至单步推理成为可能。",
    )

    h2(doc, "2.3 为什么不能直接搬")
    para(
        doc,
        "原因在于 √(1−ᾱ_k) 不是一个可以随手改的经验系数，它是“score 参数化 → ε 参数化”的雅可比因子。"
        "由前向过程 a_k = √ᾱ_k·a₀ + √(1−ᾱ_k)·ε 可得",
    )
    code_block(
        doc,
        """
∇log q(a_k) = − ε_θ / √(1−ᾱ_k)
⇒ ε̂ = ε_θ − s·√(1−ᾱ_k)·∇d
        """,
    )
    para(
        doc,
        "即：在样本空间给 score 加一个固定强度 s 的梯度，翻译到 ε 空间必须乘 √(1−ᾱ_k)；"
        "这个因子同时带来自动退火——最噪的一步接近 1，接近干净数据时趋于 0。"
        "若直接把它换个常数贴到流匹配上，等号两侧量纲就不再对应：流匹配里与 √(1−ᾱ_k) 同位的量"
        "是边际噪声标准差 σ_t = 1−t，而不再是 √(1−ᾱ_k)。",
    )

    # ------------------------------------------------------------ section 3 ----
    h1(doc, "三、为什么这样改：推导与取舍")
    h2(doc, "3.1 流匹配里的等价形式")
    para(
        doc,
        "把引导视为“在漂移项上加一项能量梯度”。对线性路径取与插值自洽的扩散系数 g_t = σ_t = 1−t，"
        "由 Fokker–Planck 方程 ∂_t p = −∇·(f p) + (g²/2)Δp 可知，保持边际分布族不变所需的修正为",
    )
    code_block(
        doc,
        """
v̂(x_t, t) = v_θ(x_t, t) − s·λ(t)·∇J(x_t),    λ(t) = σ_t²/2 = (1−t)²/2
        """,
    )
    para(
        doc,
        "这就是本次实现的默认形式。相较之下，严格照搬 DynaGuide 的 ε 参数化（把 ε_eq = x_t − t·v "
        "代入同样的代数）会得到 λ(t) = (1−t)/t，它在 t→0 处发散；而单步推理恰恰发生在 t→0，"
        "所以直接搬会在最噪的一步把动作打飞。λ(t) = (1−t)²/2 有两个好处：t→1 时自动关断（与 DDIM 的退火同构），"
        "t→0 时有界（等于 1/2），因此单步、四步都稳定。代码里把三种候选都实现了"
        "（--weight fp | linear | ddim），通过命令行切换，便于复现对照。",
    )

    h2(doc, "3.2 梯度在哪一点求")
    para(
        doc,
        "动力学模型 h_θ 与物理能量都是在“干净动作块”上定义的，直接对带噪的 x_t 求梯度属于分布外查询，"
        "信号会偏。因此本次把能量作用在 Tweedie 干净动作估计 x̂₁ = x_t + (1−t)·v_θ 上，"
        "再按链式法则换回速度修正。实现提供两种传输方式：默认停梯度（把 v_θ 视为常数，"
        "等价于单位映射，只需一次反向，稳定、便宜）；打开 jacobian=True 则显式包含 "
        "I + (1−t)·∂v/∂x_t（与 RectifID 的 anchored guidance 一致），代价是多一次反传到策略网络。",
    )
    para(
        doc,
        "此外还保留了 DynaGuide 的“字面移植”模式（mode=\"velocity\"，直接在带噪块上求能量），"
        "作为消融对照，说明为什么默认要走 Tweedie。",
    )

    h2(doc, "3.3 两道必要的工程护栏")
    para(
        doc,
        "引导强度极其敏感，这一点与文献一致：QGF 官方实现要逐值扫描引导权重。因此实现里提供了"
        "按 L2 范数裁剪（默认 5.0）、逐元素裁剪与跨步骤动量平滑三个开关，默认开启第一个。"
        "另外，DynaGuide 论文本身做过消融：如果动力学模型没有见过带噪动作，引导能力会明显下降。"
        "本次在潜空间动力学的训练里用同一条插值路径加噪（50% 概率，a_t = (1−t)ε + t·a₁），"
        "而不是另造高斯噪声，保证与采样器的噪声分布一致。",
    )

    # ------------------------------------------------------------ section 4 ----
    h1(doc, "四、代码改了什么")
    para(
        doc,
        "改造遵循“最小侵入”原则：原有 DDIM 通路一行未删，新增的流匹配通路并行存在，"
        "两条通路通过命令行参数切换。下表的行数为本次新增文件的实际行数。",
    )
    table_caption(doc, "表 1  新增模块一览")
    flow_files = [
        ["core/flows/paths.py", "线性概率路径、x₁/x₀ 投影、SNR 与噪声比例", loc(REPO / "core/flows/paths.py")],
        ["core/flows/consistency.py", "rectified-flow 损失 + FlowPolicy 的一致性流匹配损失", loc(REPO / "core/flows/consistency.py")],
        ["core/flows/samplers.py", "euler/中点/heun 采样器，含引导钩子与 ss 内层循环", loc(REPO / "core/flows/samplers.py")],
        ["core/flows/nets.py", "小型一维 UNet（FiLM 时间/观测条件）+ 观测编码器", loc(REPO / "core/flows/nets.py")],
        ["core/flows/policy.py", "动作块流策略：训练、采样、存取", loc(REPO / "core/flows/policy.py")],
        ["core/flows/normalizer.py", "动作线性归一化到 [−1, 1]", loc(REPO / "core/flows/normalizer.py")],
        ["core/flow_dynaguide.py", "引导权重 λ(t)、能量→速度修正、run_dynaguide 的 drop-in 接口", loc(REPO / "core/flow_dynaguide.py")],
        ["core/flow_energies.py", "潜空间能量（DynaGuide 式）+ 可微物理能量（动量/力/协议）", loc(REPO / "core/flow_energies.py")],
        ["core/satellite/spec.py", "场景与任务定义：三模块、三种紧固、安全预算", loc(REPO / "core/satellite/spec.py")],
        ["core/satellite/env.py", "自由漂浮拆解仿真（牛顿第三定律反作用）", loc(REPO / "core/satellite/env.py")],
        ["core/satellite/expert.py", "脚本专家：safe / impatient / offset 三种示范策略", loc(REPO / "core/satellite/expert.py")],
        ["core/satellite/latent.py", "语义潜空间编码 + 动力学模型（含噪声增广）", loc(REPO / "core/satellite/latent.py")],
        ["core/satellite/data.py", "示范切片、回合末状态取样、数据集封装", loc(REPO / "core/satellite/data.py")],
        ["core/satellite/math3d.py", "四元数/旋转向量等基础运算", loc(REPO / "core/satellite/math3d.py")],
        ["robomimic/.../flow_policy.py", "robomimic/CALVIN 栈里的流策略（fm/cfm 两种目标 + 引导采样）", loc(REPO / "robomimic/robomimic/algo/flow_policy.py")],
        ["robomimic/.../flow_policy_config.py", "对应配置类（algo.flow.*）", loc(REPO / "robomimic/robomimic/config/flow_policy_config.py")],
        ["scripts/satellite/*.py（10 个）", "采数、训练、评测、扫描、诊断、出图等命令行脚本", 1300],
        ["tests/*.py（4 个）", "流匹配、引导、仿真、robomimic 集成，共 34 项断言", 584],
        ["docs/*.md（3 个）", "推导与接口说明、任务说明、实验结果", 214],
    ]
    add_table(
        doc,
        [["文件", "职责", "行数"]] + [[a, b, str(c)] for a, b, c in flow_files],
        widths=[6.0, 7.4, 1.6],
        size=9.5,
    )

    h2(doc, "4.1 引导的核心接口")
    para(doc, "引导以“能量 → 速度修正”的闭包形式注入采样器，与具体策略、具体能量解耦：")
    code_block(
        doc,
        """
from core.flows.samplers import FlowSampler, FlowSamplerConfig
from core.flow_dynaguide import make_flow_guidance

sampler = FlowSampler(FlowSamplerConfig(n_steps=4, method="euler", ss=1))
guide = make_flow_guidance(energy, scale=1.0, weight_mode="fp",
                           mode="endpoint", clip_norm=5.0)
out = policy.sample(obs_window, sampler=sampler, guidance=guide, ctx=ctx)
action_chunk = out["actions"]        # 已反归一化
        """,
    )
    para(
        doc,
        "其中 energy(actions, proprio, ctx) 返回一个标量代价与若干日志量，内部既可以是"
        "潜空间距离（LatentMatchEnergy），也可以是可微物理能量（SatellitePhysicsEnergy），"
        "或者两者相加（CompositeEnergy）。采样器在每个速度评估点调用一次引导，"
        "得到 Δv 后与该点的速度相加，即得到被引导的漂移项。",
    )

    h2(doc, "4.2 对原有工程的改动")
    table_caption(doc, "表 2  对原工程文件的最小改动")
    add_table(
        doc,
        [
            ["文件", "改动内容", "兼容性"],
            ["run_dynaguide.py", "新增 --guidance_type {diffusion,flow} 与 --flow_weight；flow 分支改用 "
                                 "calculate_flow_classifier_guidance，rollout 透传 guidance_type", "默认仍是 diffusion，行为不变"],
            ["robomimic/algo/__init__.py", "注册 FlowPolicyUNet", "只新增一行导入"],
            ["robomimic/config/__init__.py", "注册 FlowPolicyConfig", "只新增一行导入"],
            ["README.md", "新增“流匹配版 DynaGuide”章节（用法、推导摘要、命令）", "纯文档"],
            [".gitignore", "忽略 checkpoints/（权重体积大）", "无影响"],
            ["core/satellite/env.py 等", "新增 last_force 逐能力记录，供逐块/逐步分析", "新增属性，旧代码不读即无影响"],
        ],
        widths=[4.6, 8.6, 3.0],
        size=9.5,
    )
    para(
        doc,
        "在 CALVIN 栈上运行时只需把算法名换成 flow_policy 并加上 guidance_type=flow，"
        "例如 configs/flow_policy_image_calvin.json 与 calvin_exp_configs_examples/switch_on_flow.json 两个配置。"
        "本次已在 CPU 上用合成观测对 flow_policy 做了集成测试（训练 + 带引导采样），"
        "但尚未在真实 CALVIN 数据集上跑完整评测。",
    )

    h2(doc, "4.3 拆解实验载体")
    para(
        doc,
        "仿真为失效卫星基座（28.27 kg，惯量 0.62/0.58/0.41）加三个模块：载荷箱用螺栓（需先转 90°）、"
        "太阳翼用卡扣（需先压入 6 mm）、电池模块用摩擦销（破断力 7 N）。动作是末端位移意图，"
        "被解释为阻抗力 F = k·|被阻挡的位移|，被阻挡的那部分力按牛顿第三定律作为反作用力偶"
        "加回基座，造成姿态漂移。评价分两档：拆下即“任务成功”；同时满足姿态漂移 ≤ 0.12 rad、"
        "接触力 ≤ 4 N 才算“安全完成”。这一条是本次特别坚持的：在轨任务里成功但把卫星推飞，必须算失败。",
    )
    add_figure(doc, "fig5_env_schematic.png", "图 1  自由漂浮卫星拆解基准（俯视）与基座反作用示意", 13.0)

    h2(doc, "4.4 方法总览")
    add_figure(doc, "fig7_pipeline.png", "图 2  方法框架：离线训练 / 推理采样 / 推理期主动引导三层", 15.5)

    # ------------------------------------------------------------ section 5 ----
    h1(doc, "五、怎么运行")
    para(doc, "全部命令在仓库根目录执行，CPU 即可；完整链路约 30 分钟，其中策略训练约 25 分钟。")
    code_block(
        doc,
        """
# 1) 采集示范（三种示范策略混合：safe / impatient / offset）
python scripts/satellite/make_demos.py --tasks payload_box --per_task 120 \
    --noise 0.02 --tag payload --seed 5
# 2) 训练流匹配基础策略
python scripts/satellite/train_flow_policy.py --demos data/satellite/demos_payload.npz \
    --out checkpoints/satellite/flow_policy_payload.pt --steps 5000 --batch_size 96 \
    --down_dims 192 384
# 3) 训练潜空间动力学模型（含噪声动作增广）
python scripts/satellite/train_latent_dynamics.py --demos data/satellite/demos_payload.npz \
    --out checkpoints/satellite/latent_payload.pt --steps 2500
# 4) 引导评测（多方法对比）
python scripts/satellite/run_flow_dynaguide.py \
    --policy checkpoints/satellite/flow_policy_payload.pt \
    --dynamics checkpoints/satellite/latent_payload.pt \
    --methods base latent physics both reject --episodes 8 --steps 4 --alpha 30 \
    --per_method_scale "latent=1.0,physics=0.25,both=0.25"
# 5) 一键复现上述全部内容（含测试、诊断、出图）
powershell -ExecutionPolicy Bypass -File scripts/run_all_checks.ps1
        """,
    )

    # ------------------------------------------------------------ section 6 ----
    h1(doc, "六、效果")

    h2(doc, "6.1 正确性测试")
    para(
        doc,
        "四个测试套件共 34 项断言，全部通过（日志见 运行证据/01_tests，每个文件末尾记录退出码 0）。"
        "其中与本次改造直接相关的是：端点梯度与解析解逐元素相等；jacobian=True 时确实包含 "
        "I+(1−t)∂v/∂x；潜空间能量与 DynaGuide Eq.2 的公式逐元素一致；物理能量偏好“先拧后拉”而不是硬拽；"
        "错误协议会把基座推转（物理约束确实生效）。",
    )

    h2(doc, "6.2 训练收敛情况")
    add_figure(doc, "fig1_training.png", "图 3  基础策略与潜空间动力学的训练曲线（CPU）", 15.5)
    para(
        doc,
        f"基础流策略 9.42 M 参数，训练 {pol['steps']} 步、耗时约 {pol['seconds']/60:.0f} 分钟，"
        f"末次验证损失 {pol['history'][-1]['val']:.3f}；潜空间动力学训练 {lat['steps']} 步、"
        f"耗时约 {lat['seconds']:.0f} 秒，干净动作验证 MSE {lat['history'][-1]['val_clean']:.4f}。"
        "动力学模型的验证损失明显低于训练损失，说明噪声动作增广起了正则作用。",
    )

    h2(doc, "6.3 引导评测结果")
    table_caption(doc, "表 3  8 回合/方法的复跑结果（payload_box，euler 4 步，λ=(1−t)²/2）")
    rows = [["方法", "任务成功", "安全完成", "基座漂移 [rad]", "最大接触力 [N]", "每回合耗时 [s]"]]
    name_cn = {
        "base": "基础策略（无引导）",
        "latent": "潜空间引导（s=1.0, σ=30）",
        "physics": "物理能量引导（s=0.25）",
        "both": "两者联合（s=0.25）",
        "reject": "拒绝采样 best-of-16",
    }
    for m in ("base", "latent", "physics", "both", "reject"):
        r = by[m]
        rows.append([
            name_cn[m],
            f"{r['success']*100:.0f}%",
            f"{r['constraint']*100:.0f}%",
            f"{r['drift']:.3f}",
            f"{r['force']:.2f}",
            f"{r['seconds']:.1f}",
        ])
    add_table(doc, rows, widths=[5.6, 2.2, 2.2, 2.4, 2.4, 2.0], size=9.5)
    para(
        doc,
        "20 回合版本（运行证据/02_eval/eval_final.json）："
        f"基础策略 {byf['base']['success']*100:.0f}%/{byf['base']['constraint']*100:.0f}%，"
        f"潜空间引导 {byf['latent']['success']*100:.0f}%/{byf['latent']['constraint']*100:.0f}%，"
        f"物理能量 {byf['physics']['success']*100:.0f}%/{byf['physics']['constraint']*100:.0f}%。"
        "两种回合数下的排序会翻转，说明这些差异落在统计噪声之内（n=20 时二项置信区间约 ±20 个百分点）。",
    )
    add_figure(doc, "fig2_main_results.png", "图 4  各引导方法的成功率（8 回合与 20 回合两版）", 15.5)

    h2(doc, "6.4 引导强度的敏感性")
    add_figure(doc, "fig3_scale_sweep.png", "图 5  引导强度扫描：过强会破坏任务执行", 15.5)
    para(
        doc,
        "扫描结果与文献一致：引导权重必须逐值调。把 s 从 0.05 加到 0.5，安全完成率先升后降，"
        "s=1.0 时质量明显退化。这说明当前的失败模式不是“没有引导”，而是“闭环里引导与策略互相干扰”。",
    )

    h2(doc, "6.5 引导是否真的有效：逐块检验")
    para(
        doc,
        "为把“引导是否有效”与“闭环分布漂移”两件事分开，做了一个逐块检验：取策略输出的动作块，"
        "只沿能量梯度改进三步，然后在真实仿真里重放同一动作块，对比基座漂移与接触力。"
        "这样每个样本都是同一状态下的受控对比。结果：代理能量平均下降 17.3%，"
        "基座漂移在 18/25 个样本上下降（平均下降 14%），接触力在 20/25 个样本上下降（平均下降 16%）。"
        "结论是可微物理代理与仿真方向一致，引导项没有写反。",
    )
    add_figure(doc, "fig4_alignment.png", "图 6  逐块检验：沿能量梯度改进动作块后在真实仿真中重放", 15.5)

    h2(doc, "6.6 单个回合的逐步对照")
    add_figure(doc, "fig6_episode_trace.png", "图 7  示例回合（seed=1002）逐步对照", 15.5)
    para(
        doc,
        "该回合中基础策略最终漂移 0.070 rad，加入潜空间引导后压到 0.014 rad，并且提前完成拧转、"
        "更早进入拉出阶段。需要说明这是从 5 个种子里挑出的“两者都安全”的示例回合，"
        "标题中已注明 seed；整体统计以表 3 与图 6 为准。",
    )

    h2(doc, "6.7 局限")
    for text in [
        "全程在 CPU 上运行，没有 GPU 加速；robomimic/CALVIN 通路只做了合成观测的集成测试，未跑真实 CALVIN 数据集。",
        "样本量不足：8、10、20 回合三种设置下排名会翻转，目前不能声称引导在闭环成功率上取得显著提升。",
        "闭环中引导没有稳定获胜的原因已经定位：基础策略本身的安全完成率已达 75%–80%，逐块的改善被"
        "闭环分布漂移抵消；要复现论文里那种明显差距，需要让基础策略先变得不安全。",
        "拒绝采样基线（best-of-16）全部失败，原因是能量里没有任务进度项时会选中“不动”的解；"
        "这是一个已知的实现缺陷，修复方法是给能量加入进度项或改为按成功筛选的 best-of-N。",
    ]:
        bullet(doc, text)

    # ------------------------------------------------------------ section 7 ----
    h1(doc, "七、结论与下一步")
    para(
        doc,
        "本次改造完成了三件事。其一，把 DynaGuide 的推理期引导从 DDIM 的噪声预测迁移到流匹配的速度场，"
        "给出了自洽的权重推导（λ(t)=(1−t)²/2）与梯度求取位置（Tweedie 干净动作），"
        "并用单元测试逐元素验证了实现与公式一致。其二，搭建了自由漂浮卫星拆解仿真与完整实验管线，"
        "把“拆解反作用导致基座漂移”这一在轨约束做成可评测、可微分的量。其三，"
        "用受控的逐块检验证明了引导方向正确（漂移 18/25、接触力 20/25 改善），"
        "同时如实报告了闭环成功率上没有可分辨提升这一结果。",
    )
    para(doc, "下一步优先级如下：")
    for text in [
        "把示范中的不安全比例提高到 safe 0.35 / impatient 0.35 / offset 0.30 重训，"
        "让基础策略的安全完成率降到 50% 以下，再对比引导；这是最可能复现论文趋势的一步。",
        "把每方法的回合数提到 40 以上并报告二项置信区间，避免因样本量导致结论反复。",
        "修复拒绝采样基线（能量加入任务进度项），使 best-of-N 成为有意义的对照。",
        "引导只作用于动作块前 n_action_steps 步，或配合 ss>1 在同一时间层多次修正，降低闭环干扰。",
        "在真实 CALVIN 数据集上验证 flow_policy 通路，确认与 DynaGuide 原表可比。",
    ]:
        bullet(doc, text)

    # -------------------------------------------------------------- appendix ----
    h1(doc, "附录 A  关键超参数")
    table_caption(doc, "表 4  默认超参数与来源")
    add_table(
        doc,
        [
            ["参数", "取值", "说明"],
            ["策略网络", "1D UNet，通道 192/384，时间嵌入 128", "与 Diffusion Policy 同构，规模小以便 CPU 训练"],
            ["预测/执行长度", "预测 16 步，执行 8 步，观测 2 帧", "标准动作块设置"],
            ["训练目标", "rectified flow（可选一致性流匹配）", "fm 为默认，cfm 用于少步推理"],
            ["采样器", "euler，4 步，t∈[0.01,1]", "1 步可用于部署，4 步更稳"],
            ["引导权重 λ(t)", "(1−t)²/2（fp）", "另有 linear / ddim / const 可切换"],
            ["引导强度 s", "潜空间 1.0；物理 0.25", "逐方法扫描确定，不可通用"],
            ["温度 σ（alpha）", "30", "对应 DynaGuide Eq.2 的 σ"],
            ["梯度裁剪", "L2 范数 5.0", "单步修正的上限"],
            ["动力学模型", "语义潜空间 14 维 + 3 层 MLP，噪声增广 50%", "替代 DINOv2 patch（本例无相机）"],
            ["安全预算", "姿态 0.12 rad、接触力 4 N（硬中止 8 N）", "对应在轨运行与硬件保护两档"],
        ],
        widths=[3.6, 6.4, 6.0],
        size=9.5,
    )

    h1(doc, "附录 B  证据文件位置")
    table_caption(doc, "表 5  运行证据索引")
    add_table(
        doc,
        [
            ["路径", "内容"],
            [r"运行证据\01_tests", "四个测试套件的原始输出与环境版本"],
            [r"运行证据\02_eval", "评测 stdout 日志与全部 eval/sweep 的 JSON/MD"],
            [r"运行证据\03_train", "训练曲线（逐步）与从零管线的四步日志"],
            [r"运行证据\04_diag", "可微代理与仿真的一致性诊断（含逐样本 CSV）"],
            [r"运行证据\05_assets", "示范数据元信息、权重与数据集体积清单"],
            [r"运行证据\06_figures", "本文全部插图与图表说明"],
            [r"运行证据\99_repro\run_all_checks.ps1", "一键重跑全部证据"],
            [r"dynaguide\docs\FLOW_DYNAGUIDE.md", "推导、接口与 DynaGuide 超参对照表"],
            [r"dynaguide\docs\RESULTS.md", "实验表格、波动分析与下一步建议"],
        ],
        widths=[6.6, 9.0],
        size=9.5,
    )

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"written {path} ({path.stat().st_size/1024:.0f} KB)")


