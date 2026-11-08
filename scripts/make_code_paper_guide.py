"""Build the code-reading manual: paper equations <-> real code <-> my modifications.

Every code block is read from the repository at build time, so the document always shows
the current source.  Every paper-side claim is tied to the equation / section it comes from
(DynaGuide, NeurIPS 2025).

Output:  C:\\代码修改\\1\\DynaGuide代码精读手册.docx
Usage :  python scripts/make_code_paper_guide.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(r"C:\代码修改\dynaguide")
OUT = Path(r"C:\代码修改\1\DynaGuide代码精读手册.docx")
SONG, HEI, MONO = "宋体", "黑体", "Consolas"

MISSING: list[str] = []


# ----------------------------------------------------------------- code snippets
def snip(rel: str, start: str, end: str | None = None, n: int | None = None) -> str:
    """Return the code block that starts at the first line containing ``start``.

    ``end`` (a substring of a later line) closes the block inclusively; otherwise ``n``
    lines are taken.  Common indentation is removed so the block fits the page width.
    """
    path = ROOT / rel
    if not path.exists():
        MISSING.append(f"{rel} (文件不存在)")
        return f"# 缺少 {rel}"
    lines = path.read_text(encoding="utf-8").split("\n")
    try:
        i = next(k for k, line in enumerate(lines) if start in line)
    except StopIteration:
        MISSING.append(f"{rel}:{start}")
        return f"# 未找到：{start}"
    if end is not None:
        try:
            j = next(k for k in range(i + 1, len(lines)) if end in lines[k])
        except StopIteration:
            MISSING.append(f"{rel}:{start}->{end}")
            j = min(i + 20, len(lines) - 1)
    else:
        j = min(i + (n or 12) - 1, len(lines) - 1)
    block = [l for l in lines[i:j + 1] if l.strip()]
    if not block:
        return ""
    pad = min(len(l) - len(l.lstrip()) for l in block)
    return "\n".join(l[pad:] for l in block)


def set_font(run, name=SONG, size=12, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rfonts.set(qn(attr), name)


def _repeat_header(table) -> None:
    """首行设为重复表头，并整表禁止跨页断行，避免表格被切口。"""
    from docx.oxml import OxmlElement

    for idx, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        tr_pr.append(cant)
        if idx == 0:
            header = OxmlElement("w:tblHeader")
            tr_pr.append(header)


# ----------------------------------------------------------------- document pieces
class Doc:
    def __init__(self) -> None:
        self.d = Document()
        sec = self.d.sections[0]
        sec.top_margin = sec.bottom_margin = Cm(2.3)
        sec.left_margin = sec.right_margin = Cm(2.5)

    def title(self, text: str, sub: str) -> None:
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        set_font(p.add_run(text), HEI, 19, True)
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(10)
        set_font(p.add_run(sub), SONG, 11)

    def h1(self, text: str) -> None:
        p = self.d.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(5)
        set_font(p.add_run(text), HEI, 14, True)

    def h2(self, text: str) -> None:
        p = self.d.add_paragraph()
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.2
        set_font(p.add_run(text), HEI, 12, True)

    def p(self, text: str, indent: bool = True, size: float = 12) -> None:
        para = self.d.add_paragraph()
        pf = para.paragraph_format
        pf.first_line_indent = Cm(0.74) if indent else Cm(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(4)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.4
        set_font(para.add_run(text), SONG, size)

    def bullet(self, text: str) -> None:
        para = self.d.add_paragraph()
        pf = para.paragraph_format
        pf.left_indent = Cm(0.8)
        pf.first_line_indent = Cm(0)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.35
        set_font(para.add_run("· " + text), SONG, 11.5)

    def formula(self, text: str) -> None:
        """A centred paper formula / derivation line."""
        para = self.d.add_paragraph()
        pf = para.paragraph_format
        pf.left_indent = Cm(0.9)
        pf.space_before = Pt(3)
        pf.space_after = Pt(3)
        pf.line_spacing = 1.25
        set_font(para.add_run(text), HEI, 11.5)

    def code(self, text: str, caption: str = "") -> None:
        if caption:
            cap = self.d.add_paragraph()
            cap.paragraph_format.space_before = Pt(4)
            cap.paragraph_format.space_after = Pt(2)
            cap.paragraph_format.line_spacing = 1.0
            set_font(cap.add_run(caption), SONG, 9.5)
        table = self.d.add_table(rows=1, cols=1)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        cell.text = ""
        para = cell.paragraphs[0]
        para.paragraph_format.space_before = Pt(2)
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.line_spacing = 1.05
        lines = text.split("\n")
        longest = max((len(l) for l in lines), default=0)
        size = 8.5 if longest <= 78 else (8.0 if longest <= 88 else 7.5)
        for idx, line in enumerate(lines):
            run = para.add_run(line)
            set_font(run, MONO, size)
            if idx != len(lines) - 1:
                run.add_break()

    def table(self, header: list[str], rows: list[list[str]], widths: list[float],
              size: float = 9.5, caption: str = "") -> None:
        if caption:
            cap = self.d.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_before = Pt(6)
            cap.paragraph_format.space_after = Pt(2)
            cap.paragraph_format.line_spacing = 1.0
            set_font(cap.add_run(caption), SONG, 9.5, True)
        t = self.d.add_table(rows=len(rows) + 1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for j, value in enumerate(header):
            cell = t.cell(0, j)
            cell.text = ""
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(cell.paragraphs[0].add_run(value), SONG, size, True)
        for i, row in enumerate(rows, start=1):
            for j, value in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                para = cell.paragraphs[0]
                para.paragraph_format.space_before = Pt(1)
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.line_spacing = 1.1
                set_font(para.add_run(value), SONG, size)
        total = sum(widths)
        usable = 16.0  # A4 21cm - 左右各 2.5cm
        scale = usable / total if total else 1.0
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w * scale)
        _repeat_header(t)

    def save(self) -> Path:
        self.d.save(OUT)
        return OUT


# ----------------------------------------------------------------- content
def build() -> Path:
    doc = Doc()
    doc.title(
        "DynaGuide 代码精读手册",
        "论文公式 ↔ 代码实现 ↔ 本次流匹配改造　｜　宋银贺　2026-11-08",
    )

    # ── 0 一页总览
    doc.h1("〇、一页看懂")
    doc.p("这份手册只回答三个问题：论文里的公式在代码哪一行；我改了哪些行、为什么改；"
          "改完之后怎么验证它是对的。每一处代码都是直接从仓库里读出来的，不是重写的示意。")
    doc.p("三条数据流贯穿全部代码，先记住这三条，再读任何文件都不会迷路：")
    doc.bullet("训练策略：示范数据 → 动作块 → 流匹配回归 → 权重文件（不涉及引导）")
    doc.bullet("训练动力学：示范数据 → 末端状态嵌入 → 正负样本库（不涉及引导）")
    doc.bullet("推理引导：噪声 → 常微分方程积分 → 每一步插入能量梯度 → 动作块（改动都在这里）")
    doc.table(
        ["论文里的概念", "论文位置", "代码位置", "这次是否改动"],
        [
            ["度量 d（潜空间距离的组合）", "Eq. 2, §3.1", "core/flow_energies.py：LatentMatchEnergy", "沿用，接口不变"],
            ["DDIM 引导更新", "Eq. 3, §3.2", "robomimic/algo/diffusion_policy.py:640", "★ 换成流匹配版"],
            ["每步重复 M 次稳定引导", "§3.2, Algorithm 1", "samplers.py：ss / ss_mode", "★ 新实现"],
            ["动力学模型 h_θ", "§3.1", "core/satellite/latent.py（本任务版）", "★ 新实现"],
            ["动作加噪训练", "§3.1 末段", "latent.py：augment_actions", "★ 新实现"],
            ["引导强度 s、方差 σ", "§3.2, App. A.4", "scale、alpha（命令行为 --scale / --alpha）", "沿用同名"],
        ],
        [4.6, 2.6, 7.2, 3.2],
        caption="表 1　论文概念与代码位置对照",
    )

    # ── 1 框架
    doc.h1("一、整体框架：先建立地图")
    doc.p("仓库可以分成三圈，读的顺序建议按圈走，不要一上来就翻 robomimic，那里是第三方实现。")
    doc.table(
        ["圈", "目录 / 文件", "角色", "精读优先级"],
        [
            ["外圈", "calvin/、data_processing_calvin/、paper_experiments/", "第三方仿真、数据切片、原论文跑批", "不读，知道用途即可"],
            ["中间圈", "robomimic/algo/diffusion_policy.py、core/dynaguide.py、core/dynamics_models.py", "原来的 DDIM 通路：策略、引导函数、动力学模型", "读三处：Eq.3 落点、梯度怎么求、嵌入怎么打"],
            ["内圈（本次新增）", "core/flow_dynaguide.py、core/flow_energies.py、core/flows/、core/satellite/", "流匹配引导、能量、底座、仿真", "全读，按第二节顺序"],
        ],
        [2.4, 6.4, 6.2, 4.0],
        caption="表 2　三圈结构与精读优先级",
    )
    doc.p("内圈四个部分的依赖关系是单向的：flows 提供路径与采样器，satellite 提供环境与动力学，"
          "flow_energies 提供标量代价，flow_dynaguide 把代价变成速度修正。读的时候按这个顺序，"
          "每个文件都能只依赖前面已经读过的内容。")

    # ── 2 论文公式 → 代码
    doc.h1("二、论文的三个关键点，在代码里是哪几行")

    doc.h2("2.1 度量 d：Eq. 2")
    doc.p("论文把“达成目标”写成一个软最大值的比值：正样本库越近越好，负样本库越远越好。"
          "原文式子是：")
    doc.formula("d = log Σᵢ exp( −‖φ(gᵢ⁺) − h_θ(φ(o_t), a)‖² / σ ) "
                "− log Σⱼ exp( −‖φ(gⱼ⁻) − h_θ(φ(o_t), a)‖² / σ )")
    doc.p("其中 φ 是视觉编码器，h_θ 是动力学模型，σ 是控制“软硬”的方差超参，"
          "论文明确说 log-sum-exp 起的是软最大值的作用，并在多样目标与低质量条件下更有用。"
          "代码里对应的就是这一小段：", indent=True)
    doc.code(
        snip("core/flow_energies.py", "def _aggregate(self, d: torch.Tensor)",
             end="return torch.logsumexp(-d / self.alpha, dim=0)"),
        "代码 1　core/flow_energies.py：LatentMatchEnergy._aggregate（log-sum-exp 三个实现之一）",
    )
    doc.table(
        ["论文符号", "代码里的名字", "说明"],
        [
            ["φ(gᵢ⁺), φ(gⱼ⁻)", "positives / negatives", "预先算好的正、负样本嵌入库"],
            ["h_θ(φ(o_t), a)", "state_action_embedding(state, actions)", "动力学模型对当前动作块的预测"],
            ["σ", "alpha（默认 30）", "方差超参，越大越“软”"],
            ["log Σ exp(−·)", "torch.logsumexp(-d / self.alpha, dim=0)", "负号在外部，等价于软最大值"],
        ],
        [4.0, 7.0, 6.0],
        caption="表 3　Eq.2 的符号与代码逐一对应",
    )

    doc.h2("2.2 DDIM 引导更新：Eq. 3（这是改造的起点）")
    doc.p("论文第 3.2 节把引导写进 DDIM 的每一步去噪，原文式子是：")
    doc.formula("ε̂(a_k, o_t) = ε(a_k, o_t) − s · √(1 − ᾱ_k) · ∇_{a_k} d(g⁺, g⁻, o_t, a_k)")
    doc.p("其中 ᾱ_k 是噪声调度器的累积乘积，s 是引导强度。论文同时指出：s 变大能更贴合条件，"
          "但梯度会把噪声预测拉出分布，误差会随去噪步累积，所以要用“每步重复 M 次”的随机采样技巧来稳住。"
          "这一行在代码里就是：")
    doc.code(
        snip("robomimic/robomimic/algo/diffusion_policy.py", "for ss_iter in range(ss):",
             end=".prev_sample # this gives the mu from the paper"),
        "代码 2　robomimic/algo/diffusion_policy.py:630-646（未改动；Eq.3 在第 640 行，外层 for k 是去噪步，"
        "内层 for ss_iter 就是论文的 M 次重复）",
    )
    doc.table(
        ["式中的量", "代码里的写法", "含义"],
        [
            ["ε(a_k, o_t)", "noise_pred（网络输出）", "当前步的噪声预测"],
            ["√(1 − ᾱ_k)", "(1 - alphas_cumprod[k]).sqrt()", "把“样本空间强度”换算到“噪声空间”的系数"],
            ["s", "scaled_grad 里的 scale", "引导强度，命令行 --scale"],
            ["∇_{a_k} d", "scaled_grad（= scale × 梯度）", "由 core/dynaguide.py 求出"],
        ],
        [3.4, 7.4, 6.2],
        caption="表 4　Eq.3 的符号与代码逐一对应",
    )
    doc.p("注意一个容易被忽略的点：这个根号不是超参，它是参数化的换算系数。"
          "由前向过程可以推出，噪声预测与 score 之间差一个负的 √(1−ᾱ_k)，"
          "所以式中的乘积实际上等于“样本空间里的引导步长”。这一认识正是第三节改造的依据。")

    doc.h2("2.3 每步重复 M 次：Algorithm 1")
    doc.p("论文为了抑制梯度把预测拉出分布，把每个去噪步重复 M 次（随机采样解），"
          "实践中允许把 s 推得更高。流匹配版把这一步实现成采样器里的 ss 与 ss_mode："
          "split 表示把当前步再细分、refine 表示在同一时刻反复磨，两种都保留，默认值与原实现一致。")
    doc.code(
        snip("core/flows/samplers.py", "ss: int = 1", n=6),
        "代码 3　core/flows/samplers.py：FlowSamplerConfig 里与 M 对应的字段",
    )

    doc.h2("2.4 动力学模型的加噪训练：论文 §3.1 末段")
    doc.p("论文说得很直接：因为引导时会用带噪动作去查询动力学模型，训练时必须用同一条噪声调度给动作加噪，"
          "噪声强度从几何分布里取。本任务的实现是 augment_actions：")
    doc.code(
        snip("core/satellite/latent.py", "def augment_actions", n=16),
        "代码 4　core/satellite/latent.py：augment_actions（与采样器噪声分布一致）",
    )

    # ── 3 改造
    doc.h1("三、我的改造：从噪声预测到速度场")
    doc.p("这部分是全文重点。改造只有一件事：Eq.3 是给 DDIM 的 ε 参数化写的，"
          "流匹配网络的输出是速度 v，两者对位关系不同，所以系数必须换、求梯度的位置也必须换。")

    doc.h2("3.1 为什么不能直接照搬那三行")
    doc.p("DDIM 与线性插值流匹配是两套时间参数化，同一个“时刻”对应的量并不相同。"
          "把两者的对位关系列出来，问题就很清楚：")
    doc.table(
        ["量", "DDIM（VP 调度）", "流匹配（线性插值）"],
        [
            ["网络输出", "噪声 ε", "速度 v"],
            ["带噪样本", "a_k = √ᾱ_k·a₀ + √(1−ᾱ_k)·ε", "x_t = (1−t)·ε + t·a₁"],
            ["边际噪声标准差", "√(1−ᾱ_k)", "1 − t"],
            ["引导换算系数", "√(1−ᾱ_k)", "严格对位应为 (1−t)/t"],
            ["单步推理所在时刻", "k 小，ᾱ_k 小，系数接近 1", "t→0，系数发散"],
        ],
        [3.6, 6.4, 7.0],
        caption="表 5　两套参数化的对位关系（发散就出在最后一行）",
    )
    doc.p("推导只需一步。流匹配的 score 与速度之间的关系是 v = ε − a₁（线性路径上速度恒定），"
          "把它与 Tweedie 投影联立可以得到 ∇log p_t(x) = −ε / σ_t，其中 σ_t = 1 − t；"
          "而 DDIM 里 √(1−ᾱ_k) 正是 σ_k。于是“把强度从样本空间换算到噪声空间”这个动作，"
          "在流匹配里写出来是 λ(t) 正比于 σ_t² / 某个量，严格照搬 DDIM 的写法就得到 (1−t)/t，"
          "它在 t→0（单步推理）处发散。")
    doc.p("所以要么裁剪、要么换一个在两端都良态的权重。我选择后者。")

    doc.h2("3.2 采用的权重：λ(t) = (1 − t)² / 2")
    doc.p("取平方除以二不是凑出来的：与线性插值自洽的扩散系数是 1 − t，"
          "由 Fokker-Planck 得到的漂移项正比于它的平方除以二，于是权重天然满足两个要求——"
          "t→1 时自动关断（与原实现的退火一致），t→0 时有界（等于二分之一，单步推理不会炸）。"
          "四种调度都实现了，命令行可切，方便做对照实验：")
    doc.table(
        ["模式", "式子", "t→0（单步）", "t→1（接近数据）", "用途"],
        [
            ["fp（默认）", "(1−t)²/2", "0.5，有界", "→0，自退火", "推荐，本实验默认"],
            ["linear", "1−t", "1", "→0", "一阶近似，消融"],
            ["ddim", "(1−t)/t", "发散，需裁剪 50", "→0", "严格照搬，仅作对照"],
            ["const", "1", "1", "1", "纯强度旋钮（QGF 风格）"],
        ],
        [3.0, 3.0, 3.6, 3.6, 4.0],
        caption="表 6　λ(t) 的四种调度与两端行为",
    )
    doc.code(
        snip("core/flow_dynaguide.py", "def __call__(self, t: torch.Tensor) -> torch.Tensor:",
             end="return w"),
        "代码 5　core/flow_dynaguide.py：GuidanceWeightSpec.__call__（四种 λ(t)）",
    )

    doc.h2("3.3 求梯度的位置：先投影成干净动作")
    doc.p("论文的 Eq.3 是对当前带噪样本求梯度。在流匹配里这样做的代价更大："
          "动力学模型和物理能量都只见过干净动作块，对带噪样本求导属于分布外查询，梯度会有偏。"
          "所以我先用 Tweedie 投影把当前样本换算成干净动作估计，再在它上面求梯度。"
          "这个算子和 FlowPolicy 论文里的 f_θ 是同一个：")
    doc.code(
        snip("core/flow_dynaguide.py", "def clean_estimate(self",
             end="return x_t + (1.0 - t_e) * v"),
        "代码 6　core/flow_dynaguide.py：clean_estimate（Tweedie 投影）",
    )
    doc.p("主流程把“投影、算能量、求梯度、过护栏、得到修正量”串在一起，"
          "其中 mode 决定用投影（endpoint，默认）还是照论文直接在带噪块上算（velocity，仅作消融）：")
    doc.code(
        snip("core/flow_dynaguide.py", 'if self.mode == "endpoint":',
             end="delta_v = -float(self.scale) * w * grad"),
        "代码 7　core/flow_dynaguide.py：FlowDynaGuide.__call__ 主流程",
    )

    doc.h2("3.4 改造的那一行")
    doc.p("把原实现的 Eq.3 和流匹配版并排放，就是这次改造的全部：")
    doc.table(
        ["", "原实现（DDIM）", "本改造（流匹配）"],
        [
            ["代码", "noise_pred = noise_pred − (1 − ᾱ_k).sqrt() × scaled_grad",
             "delta_v = −scale × λ(t) × grad"],
            ["被减的对象", "噪声预测 ε", "速度 v"],
            ["系数", "√(1−ᾱ_k)", "λ(t) = (1−t)²/2"],
            ["求梯度处", "当前带噪样本 a_k", "干净动作估计 x̂₁"],
            ["位置", "diffusion_policy.py:640", "flow_dynaguide.py:202"],
        ],
        [3.0, 7.6, 7.2],
        caption="表 7　Eq.3 与流匹配版的逐项对照",
    )
    doc.p("这一行在代码里长这样（注释是我后加的，说明两个式子的对应关系）：")
    doc.code(
        snip("core/flow_dynaguide.py", "w = self.weight(t).reshape", n=5),
        "代码 8　core/flow_dynaguide.py:197-203　λ(t) 加权与 delta_v",
    )

    doc.h2("3.5 三道梯度护栏")
    doc.p("论文已经指出引导强度极其敏感（附录里要逐值扫描），原因是梯度会把预测拉出分布。"
          "流匹配版额外加了三道护栏，默认只开第一道：")
    doc.code(
        snip("core/flow_dynaguide.py", "def _postprocess(self, g: torch.Tensor)",
             end="return g"),
        "代码 9　core/flow_dynaguide.py：_postprocess（范数裁剪 / 逐元素裁剪 / 动量平滑）",
    )

    doc.h2("3.6 注入点：只有四步")
    doc.p("改造要落进采样循环，但采样器本身不该知道能量长什么样。所以注入点做成一个小适配："
          "把当前样本设为可求导、网络给出原速度、引导函数返回修正量、两者相加交给积分。"
          "这也是唯一与原实现“结构等价”的位置：原来改的是噪声预测，现在改的是速度。")
    doc.code(
        snip("core/flows/samplers.py", "def _guided_velocity(self",
             end="return x_in, v + delta, delta, energy, 1"),
        "代码 10　core/flows/samplers.py：_guided_velocity（引导注入点）",
    )

    doc.h2("3.7 为了不动主程序：drop-in 接口")
    doc.p("原程序里引导函数是“给两个数据集、返回一个可调用对象”的形式。流匹配版保持同一签名与返回，"
          "只多了一组流匹配专用的可选参数，因此主程序只需换一个开关，两条通路随时可以对照：")
    doc.code(
        snip("core/flow_dynaguide.py", "def calculate_flow_classifier_guidance(",
             end="**kwargs,"),
        "代码 11　core/flow_dynaguide.py：与原 DDIM 版同签名（scale / alpha / 数据集）",
    )

    doc.h2("3.8 改造清单")
    doc.table(
        ["文件", "性质", "规模", "一句话"],
        [
            ["core/flow_dynaguide.py", "新增", "约 320 行", "流匹配引导：λ(t)、Tweedie、护栏、drop-in 接口"],
            ["core/flow_energies.py", "新增", "约 360 行", "两类能量：潜空间匹配 + 可微物理"],
            ["core/flows/（7 文件）", "新增", "约 1100 行", "路径、目标、采样器、网络、策略、归一化"],
            ["core/satellite/（7 文件）", "新增", "约 1500 行", "拆解环境、专家、动力学、数据、几何"],
            ["robomimic/algo/flow_policy.py", "新增", "约 330 行", "流匹配策略 + 兼容两种引导签名"],
            ["robomimic/config/flow_policy_config.py", "新增", "约 60 行", "algo.flow.* 配置"],
            ["run_dynaguide.py", "改动", "+2 开关", "--guidance_type / --flow_weight"],
            ["algo/__init__.py、config/__init__.py", "改动", "各 +1 行", "注册新算法与配置"],
        ],
        [6.4, 1.8, 2.4, 8.0],
        caption="表 8　本次新增与改动的清单（与代码地图一致）",
    )

    # ── 4 能量
    doc.h1("四、能量：从“距离”到“物理”")
    doc.p("论文的引导目标只有 Eq.2 一个度量 d。这套框架的问题在于：它需要“目标观测”。"
          "卫星拆解里没有相机、也很难给出“拆好之后长什么样”的观测，所以我做了两类能量，"
          "接口完全一致（标量代价，越小越好），可以单独用也可以加权合并。")
    doc.h2("4.1 沿用论文思路：潜空间匹配能量")
    doc.p("结构上就是 Eq.2，只是 φ 换成了本任务的语义编码器（十四维），h_θ 换成三层感知机。"
          "好处是正负样本的用法完全保留：正样本是安全回合的末状态，负样本是违约回合的末状态。")
    doc.h2("4.2 新增：可微拆解代理")
    doc.p("第二个能量不需要学，直接对一个“简化物理”的代理模型打分。代理沿动作块推演十六步，"
          "返回十二个可微标量。其中最关键的一段是反作用力矩：动作被解释为末端位移意图，"
          "只有被紧固件挡住的那部分才产生接触力，这个力按牛顿第三定律变成加在基座上的力偶。")
    doc.code(
        snip("core/flow_energies.py", "torque = torch.cross(R @ attach, -f_world",
             n=10),
        "代码 12　core/flow_energies.py：rollout 内的反作用力偶（微重力与地面的本质区别）",
    )
    doc.h2("4.3 六项权重")
    doc.p("物理能量把这些量按权重相加，取值不是随手写的：协议进度与动量反作用各给 2，"
          "因为它们决定“能不能拆下”和“拆完卫星还稳不稳”；过早拉拽 1.5，因为它是最典型的错误动作；"
          "接触力 1；轴向偏差 0.25、接近项 0.5，只作引导不作主导。")
    doc.table(
        ["项", "权重", "物理含义", "对应代码量"],
        [
            ["protocol", "2.0", "紧固件是否真的解锁（螺栓转够 / 卡扣压够 / 销拉断）", "−pull_ratio"],
            ["premature_pull", "1.5", "没解锁就拉：白费力气且把力矩全灌进基座", "blocked_pull"],
            ["momentum", "2.0", "基座姿态漂移与传递冲量（微重力专属）", "attitude_drift, impulse"],
            ["force", "1.0", "接触力超预算的程度", "force_max"],
            ["axis", "0.25", "拔出方向偏差，会卡住", "axis_err"],
            ["approach", "0.5", "末端到抓取点的距离", "capture_dist"],
        ],
        [3.4, 1.6, 8.6, 4.4],
        caption="表 9　物理能量的六项与权重",
    )
    doc.p("这六项里，前三项是论文那套“距离度量”无法表达的：论文的 d 只回答“像不像目标观测”，"
          "不回答“这一步的力有没有灌到基座上”。这也是我把卫星任务做进来的主要动机。")

    # ── 5 仿真
    doc.h1("五、仿真：为什么微重力是新的难点")
    doc.p("仿真本体只有三百多行，但它是全部指标的来源。核心的一段是反作用：")
    doc.code(
        snip("core/satellite/env.py", "torque = np.cross(r_arm, -f_world", n=8),
        "代码 13　core/satellite/env.py：接触力反作用到自由漂浮基座",
    )
    doc.p("基座惯量只有 0.6 这个量级，几牛的偏载在一步之内就能造成可观的姿态漂移。"
          "于是评价必须分两档：拆下算任务成功；拆下且姿态漂移与接触力都在预算内，才算安全完成。"
          "这一条是刻意的设计——在轨任务里，成功但把目标推飞必须算失败。")
    doc.table(
        ["预算项", "取值", "作用"],
        [
            ["姿态漂移", "0.12 rad", "安全完成的判据；超出即不安全"],
            ["接触力", "4 N（8 N 硬中止）", "运行预算与回合中止"],
            ["回合长度", "120 步", "超出即失败"],
            ["三种紧固件", "螺栓 90° / 卡扣 6 mm / 销 7 N", "决定“必须先做什么”"],
        ],
        [3.2, 5.6, 9.2],
        caption="表 10　安全预算与紧固件参数",
    )

    # ── 6 robomimic
    doc.h1("六、接回原有生态")
    doc.p("为了证明改造不是“只能在自己的环境里跑”，我在 robomimic 里加了一个流匹配策略。"
          "它与原来的扩散策略只有三处差异：输出从噪声变速度、目标从噪声回归变流匹配、"
          "推理从 DDIM 调度器变常微分方程积分。其余（观测编码器、EMA、动作块、权重存取）全部复用。")
    doc.p("这里有个小函数值得单独看：原程序的引导函数是两参数（状态、动作），流匹配版是四参数"
          "（多出当前时刻与当前速度）。这个适配层负责两者都能挂上来，所以老的引导函数不需要改写：")
    doc.code(
        snip("robomimic/robomimic/algo/flow_policy.py", "def _wrap_guidance(self",
             n=18),
        "代码 14　robomimic/algo/flow_policy.py：_wrap_guidance（兼容两种签名）",
    )

    # ── 7 跑起来
    doc.h1("七、跑起来是什么样")
    doc.p("整条链只有四个命令，按顺序执行即可复现（全部 CPU）：")
    doc.code(
        "python scripts/satellite/make_demos.py --per_task 40\n"
        "python scripts/satellite/train_flow_policy.py --steps 5000\n"
        "python scripts/satellite/train_latent_dynamics.py --steps 2500\n"
        "python scripts/satellite/run_flow_dynaguide.py --episodes 8 \\\n"
        "        --methods base latent physics both reject",
        "用法 1　从采数据到出评表的四个命令",
    )
    doc.p("评测输出是一张逐方法、逐任务的表，八回合一组的结果（成功 / 安全完成）为："
          "不引导 88/75、潜空间引导 88/88、物理能量 62/62、联合 62/50、拒绝采样 0。"
          "二十回合一组排序会变：不引导 85/80、潜空间 65/45、物理 70/45、联合 60/35。"
          "两组放在一起看，差异落在统计噪声里，所以我不声称闭环成功率显著提升。")
    doc.p("真正稳的证据是逐块检验：拿同一个动作块，只沿能量梯度改进三步，再到真实仿真里重放。"
          "代理能量平均下降 17.3%，基座漂移在二十五个样本里十八个改善（平均降一成四），"
          "接触力二十个改善（平均降一成六）。这证明代理与仿真同向，引导没有写反。")

    doc.h1("八、论文实验与本次实验的对照")
    doc.table(
        ["对比项", "论文（CALVIN）", "本次（卫星拆解）"],
        [
            ["策略", "DDIM 扩散策略（官方权重）", "流匹配策略（自训，9.42M 参数）"],
            ["引导对象", "噪声预测 ε（Eq.3）", "速度场 v（本手册第三节）"],
            ["度量", "潜空间距离组合 d（Eq.2）", "d + 六项物理能量，可加权"],
            ["条件来源", "目标观测图像 g⁺ / g⁻", "安全 / 违约回合的末状态嵌入"],
            ["主指标", "目标行为成功率（相对基线提升 8.7×）", "成功与安全完成（两档）"],
            ["样本量", "每行为多回合、报告置信区间", "每方法 8 / 20 回合，样本不足"],
            ["证据强度", "闭环成功率 + 多实验设置", "闭环无显著提升 + 逐块受控检验显著改善"],
        ],
        [3.0, 6.8, 8.0],
        caption="表 11　与论文实验的对照（第 6 行是最该主动说明的）",
    )

    doc.h1("九、建议的读代码路线（八步）")
    doc.table(
        ["步骤", "看哪里", "看什么"],
        [
            ["1", "core/flows/paths.py：project_onto_data", "两端的互逆关系，理解“投影”是后面一切的基础"],
            ["2", "core/flows/samplers.py：sample 与 _guided_velocity", "推理就是一个积分循环，引导只影响一步的速度"],
            ["3", "core/flow_energies.py：LatentMatchEnergy._aggregate", "对照论文 Eq.2，log-sum-exp 就是软最大值"],
            ["4", "core/flow_energies.py：rollout 的反作用段", "为什么这个任务的难点是基座漂移"],
            ["5", "core/flow_dynaguide.py：GuidanceWeightSpec", "λ(t) 的四种取法与两端行为"],
            ["6", "core/flow_dynaguide.py：__call__ 与 delta_v", "改造的那一行，与原 Eq.3 逐项对照"],
            ["7", "robomimic/algo/flow_policy.py：_wrap_guidance", "怎么在不动老代码的前提下挂上新引导"],
            ["8", "tests/test_flow_dynaguide.py", "端点梯度与解析解相等——这一条保证改造没有写错"],
        ],
        [1.6, 7.4, 9.0],
        caption="表 12　八步读法",
    )

    doc.h1("十、组会可能被问到的八个问题")
    qa = [
        ("为什么不直接沿用论文那三行？",
         "那三行的根号是 ε 参数化的换算系数；流匹配输出是速度，对位量是 1−t。"
         "直接搬量纲不对，严格换算得到 (1−t)/t，在单步推理的 t→0 处发散。"),
        ("权重为什么取平方除以二？",
         "与线性插值自洽的扩散系数是 1−t，漂移项正比于其平方除以二；"
         "结果是 t→1 自动关断、t→0 有界，两端都良态。"),
        ("为什么在干净动作估计上求梯度，而不是在当前样本上？",
         "动力学模型与物理能量只见过干净动作，对带噪样本求导是分布外查询，梯度有偏；"
         "投影（Tweedie）的计算量可忽略，属于纯收益。"),
        ("为什么不把整条去噪链反传？",
         "代价大、方差高，且没有必要：在干净动作上求梯度已经回到分布内，与近期几篇工作的做法一致。"),
        ("基础策略已经够安全，为什么还需要引导？",
         "逐块检验显示引导确实降低了漂移与接触力；闭环里被分布漂移抵消。"
         "要复现论文那种明显提升，需要先让基线不安全。"),
        ("样本量够吗？",
         "不够。八、十、二十回合三种设置的排名会翻转，二十分之一对应正负二十个百分点。"
         "所以主证据放在逐块受控检验上，下一步提到四十回合以上并报置信区间。"),
        ("拒绝采样为什么是零？",
         "已知缺陷：能量里没有任务进度项时，“最省力”的解是原地不动。修法是加进度项，或改成按成功筛选。"),
        ("哪些代码是你写的？",
         "原始两个仓库是公开的；新增三十七个文件约五千四百行，并对四个文件做了最小改动，"
         "清单见表 8 与代码地图。"),
    ]
    for i, (q, a) in enumerate(qa, start=1):
        doc.p(f"问{i}：{q}", indent=False, size=11.5)
        doc.p(f"答：{a}", indent=False, size=11.5)

    path = doc.save()
    return path


if __name__ == "__main__":
    out = build()
    print(f"written {out} ({out.stat().st_size/1024:.0f} KB)")
    if MISSING:
        print("以下代码片段未取到（需要检查定位串）：")
        for m in MISSING:
            print("  -", m)
    else:
        print("所有代码片段均从仓库实时读取，定位全部命中。")
