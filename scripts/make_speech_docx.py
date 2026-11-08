"""Build the presentation script (讲稿) that walks through 代码思维导图.md.

Output: ``C:\\代码修改\\代码讲解讲稿_10分钟.docx`` (+ a .md source next to it).

Speech length target: ~2500 字 ≈ 10 分钟（中文口播约 250 字/分钟）。段落里带
"【可跳过】"标记的，时间紧时直接略过。

Usage::

    python scripts/make_speech_docx.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

OUT_DOCX = Path("C:/代码修改/代码讲解讲稿_10分钟.docx")
OUT_MD = Path("C:/代码修改/代码讲解讲稿_10分钟.md")
SONG = "宋体"
HEI = "黑体"

SCHEDULE = [
    ("开场", "0:00–0:30", "导图总节点"),
    ("一、仓库结构", "0:30–1:30", "根目录 / core / robomimic"),
    ("二、核心改造", "1:30–4:00", "core/flow_dynaguide.py"),
    ("三、两类能量", "4:00–5:20", "core/flow_energies.py"),
    ("四、流匹配底座", "5:20–6:30", "core/flows/"),
    ("五、仿真与数据", "6:30–7:40", "core/satellite/"),
    ("六、接回原生态", "7:40–8:30", "robomimic / configs"),
    ("七、脚本与测试", "8:30–9:10", "scripts / tests"),
    ("八、结果与局限", "9:10–10:30", "fig2 / fig4"),
    ("九、收尾", "10:30–11:00", "—"),
]

SCRIPT: list[tuple[str, str]] = [
    ("h1", "开场（30 秒）"),
    ("cue", "（停在导图总节点，不展开分支）"),
    ("p", "各位老师好。我汇报的是对 DynaGuide 的代码改造。项目背景是失效卫星的在轨拆解，"
          "服务星贴着目标卫星，把一个模块从基座上拆下来带回去。"),
    ("p", "一句话说改造：DynaGuide 原本在 DDIM 扩散策略上做推理期引导，我把它搬到了流匹配的速度场上，"
          "并搭了一个自由漂浮拆解仿真来验证。整个改造没有动训练权重，只改了推理阶段。"),

    ("h1", "一、仓库结构（1 分钟）"),
    ("cue", "（展开前三个分支）"),
    ("p", "代码可以分成三圈。第一圈是原有的 DDIM 通路，一点没改：主程序 run_dynaguide.py、"
          "引导函数 core/dynaguide.py、动力学模型 core/dynamics_models.py，以及 calvin 和 robomimic 两个第三方库。"),
    ("p", "第二圈是这次新增的，也是重点：core/flow_dynaguide.py 是流匹配版引导，"
          "core/flow_energies.py 是引导用的能量，core/flows/ 是流匹配底座，"
          "core/satellite/ 是拆解仿真；另外在 robomimic 里新增一个算法文件 flow_policy.py，把改造接回它的生态。"),
    ("p", "第三圈是脚本、测试和文档：十个实验脚本把从采数据到出图整条链跑通，"
          "四个测试套件共三十四项断言，三篇说明文档。对原有代码只做了最小改动——"
          "主程序加两个命令行开关，两个初始化文件各加一行注册，这样两条通路可以随时切回去对照。"),

    ("h1", "二、核心改造：流匹配版引导（2 分 30 秒）"),
    ("cue", "（打开 core/flow_dynaguide.py）"),
    ("h2", "1 为什么不能照搬原来那三行"),
    ("p", "原来引导的核心只有一行：噪声预测减去一个带系数的梯度，系数里有一个根号下 1 减 alpha_bar。"
          "这个根号不是经验值。由前向过程可以推出，网络预测的噪声与 score 之间正好差一个负的这个根号，"
          "所以它是把“样本空间的引导强度”换算到“噪声空间”的换算系数；"
          "它顺带还起了退火作用——最噪一步接近 1，接近干净数据时趋近 0。"),
    ("p", "流匹配里网络输出的是速度，不是噪声。与之对位的量是边际噪声标准差，也就是 1 减 t。"
          "直接搬根号量纲不对；严格照搬会得到 1 减 t 除以 t，而单步推理恰好发生在 t 接近 0 处，"
          "这个系数会发散。这就是“照搬会炸”的原因。"),
    ("h2", "2 采用的权重"),
    ("p", "我取的是 1 减 t 的平方除以 2。t 趋近 1 时它自动关断，和原来退火一致；"
          "t 趋近 0 时它有界，等于二分之一，单步和四步都稳定。"
          "四种权重都实现了，命令行可切：默认的 fp、线性的 linear、严格照搬 DDIM 的 ddim、以及常数 const，方便做对照。"),
    ("h2", "3 梯度在哪一点求"),
    ("p", "第二件事是求导位置。动力学模型和物理能量只见过干净动作块，"
          "直接对带噪样本求梯度属于分布外查询，会有偏。所以代码先把当前样本投影成干净动作估计——"
          "x_t 加 1 减 t 乘速度，这个算子和 FlowPolicy 论文里的 f 是同一个，能量只作用在它上面。"),
    ("p", "投影之后算能量、反向求梯度、乘权重和强度，得到速度修正量。看这一行："
          "负号表示沿能量下降方向，lambda 承担原来根号的退火作用。"
          "这一行就是原来那行补丁在流匹配里的对应式。"),
    ("h2", "4 护栏与接口"),
    ("p", "引导强度非常敏感，文献里的实现都要逐值扫描，不加限制很容易一步把动作打飞。"
          "所以有三道护栏：按范数裁剪默认上限 5、逐元素裁剪默认关、跨步动量平滑默认关。"),
    ("ps", "文件里还有两个函数：make_flow_guidance 把命令行参数变成引导对象；"
          "calculate_flow_classifier_guidance 是替换接口，签名和返回值与原 DDIM 版完全一致，"
          "所以主程序只换一个开关就能切通路。"),

    ("h1", "三、引导在优化什么（1 分 20 秒）"),
    ("cue", "（打开 core/flow_energies.py）"),
    ("p", "能量都写成标量代价、越小越好，于是引导方向统一是负梯度。分两类。"),
    ("ps", "第一类沿用 DynaGuide：把动作块送进动力学模型预测末端潜状态，与正负样本库比距离，"
          "用 log-sum-exp 聚合。log-sum-exp 相当于软最小值，多个正目标时自动挑当前最可达的那个，"
          "而不是全部平均；正负相减对应论文里概率相除再取对数。"),
    ("p", "第二类是我新增的物理能量，不用学，直接对一个可微分解解代理打分，考核六项："
          "协议进度、是否过早拉拽、接触力、基座姿态漂移、拔出方向偏差、以及是否对着正确模块。"
          "看代理模型这一行：力矩等于位矢叉乘负的接触力，角速度按力矩乘时间除惯量累加，姿态再积分。"
          "基座惯量只有 0.6 这个量级，一次几牛的偏载就足以让姿态明显漂移——"
          "在这里“温柔地拆”不是风格问题，而是能不能完成的问题。"),
    ("p", "两类可以加权相加，就是实验里的 both。第一类负责“拆成什么样”，第二类负责“别把卫星推飞”。"),

    ("h1", "四、流匹配底座（1 分 10 秒）"),
    ("cue", "（展开 core/flows/）"),
    ("p", "这个子包是纯 torch 实现，不依赖 diffusers。三个要点。"),
    ("p", "paths.py 定义坐标系：加噪写成从噪声到动作的直线插值，沿直线速度恒定；"
          "知道当前样本、时刻和速度就能还原两端，干净动作是 x_t 加 1 减 t 乘速度，源噪声是 x_t 减 t 乘速度，"
          "两者互逆，有单元测试。引导用的就是这个投影。"),
    ("ps", "consistency.py 是训练目标：默认最朴素的流匹配回归，另外实现了一致性流匹配，"
          "把时间切段，要求把段内任意点一步外推到段终点、且段内速度一致，于是可以少步出动作。"),
    ("p", "samplers.py 是推理，也是引导注入点。推理就是解常微分方程，每步只是当前样本加速度乘步长。"
          "看这个方法：注入只有四步——设为可求导、网络给原速度、引导函数返回修正量、相加后交给积分。"
          "参数里还有一个重复次数，对应论文里在同一噪声层多迭代几次。"),

    ("h1", "五、仿真与数据（1 分 10 秒）"),
    ("cue", "（展开 core/satellite/）"),
    ("p", "spec.py 是唯一数值来源：基座 28.27 公斤，三个模块对应三种紧固方式——"
          "螺栓先转 90 度、卡扣先压 6 毫米、摩擦销拉过 7 牛；安全预算姿态 0.12 弧度、接触力 4 牛，8 牛硬中止。"),
    ("p", "env.py 是仿真本体：动作被解释为末端位移意图，只有被卡住那部分才产生接触力，"
          "这个力按牛顿第三定律作为反作用力偶加回自由漂浮基座，所以“没拧就拉”“斜着拉”会立刻产生大力矩。"
          "评价分两档：拆下算成功，拆下且不超预算才算安全完成——在轨任务里成功但把卫星推飞必须算失败。"),
    ("p", "expert.py 造示范：正规流程先拧再拉；第二种先拽四步再拧，力全变基座反作用；"
          "第三种拔出方向偏 0.25 弧度，切向力持续推转基座。故意混入不安全示范，"
          "是为了复现“示范参差、策略学到平均行为”，而引导就是把它拉回安全那一支。"),
    ("ps", "latent.py 是动力学模型的本任务版：没有相机，就把冻结的 DINOv2 换成十四维语义特征加三层感知机，结构同构。"
          "两个细节：每维用固定名义尺度标准化，因为“是否连接”这类二值特征标准差接近零，"
          "用经验标准差会把潜空间尺度算坏；训练时按一半概率给动作块加插值噪声，与采样器噪声分布一致。"),

    ("h1", "六、接回原有生态（50 秒）"),
    ("cue", "（展开 robomimic 与 configs）"),
    ("p", "flow_policy.py 是和原实现的三处差异：输出从噪声变速度、目标从噪声回归变流匹配、"
          "推理从 DDIM 调度器变常微分方程积分；观测编码器、EMA、动作块、权重存取全部复用。"
          "里面有个小函数负责兼容两种引导签名，四参数的是流匹配版，两参数的是原版，"
          "所以原有引导函数不用改就能挂上来。"),
    ("p", "配置新增一个训练配置和一个引导实验配置，算法名换成 flow_policy。"
          "边界要说明：这条通路我只在 CPU 上用合成观测做了集成测试，还没有在真实 CALVIN 数据集上跑完整评测。"),

    ("h1", "七、脚本与测试（40 秒）"),
    ("cue", "（展开 scripts 与 tests）"),
    ("p", "脚本是一条链：采示范、训练策略、训练动力学并建正负样本库、跑评测。"
          "评测五种方法：不引导、潜空间能量、物理能量、两者联合、采样多条取最优；另有两个分析脚本，一个做逐块检验、一个出图。"),
    ("ps", "测试三十四项断言全在 CPU 跑。与改造直接相关的几条：端点梯度与解析解逐元素相等；"
          "打开全雅可比时确实包含恒等项；潜空间能量与论文公式一致；物理能量偏好先拧后拉；"
          "以及错误协议确实会把基座推转——这一条保证物理约束真的生效。"),

    ("h1", "八、结果与局限（1 分 20 秒）"),
    ("cue", "（切到 fig2 与 fig4）"),
    ("p", "主结果两组。八回合：不引导成功 88、安全完成 75；潜空间引导都是 88；物理能量 62；拒绝采样基线 0。"
          "二十回合排序会变：不引导 85 和 80，潜空间 65 和 45。两组放在一起看，结论是差异落在统计噪声里——"
          "二十回合对应正负二十个百分点的置信区间，所以我不说它显著提升。"),
    ("p", "最硬的证据是逐块检验：取同一个动作块，只沿能量梯度改进三步，在真实仿真里重放，对比漂移和接触力。"
          "结果是代理能量下降 17.3%，漂移在二十五个样本里十八个改善、平均降一成四，"
          "接触力二十个样本改善、平均降一成六。这说明代理和仿真方向一致，引导没有写反。"
          "另外有个示例回合，漂移从 0.070 压到 0.014，并提前完成拧转，那只是示例。"),
    ("p", "局限三条：一是全程 CPU，CALVIN 通路只做了合成观测的集成测试；"
          "二是样本量不足，排名会翻转，目前不能声称闭环成功率显著提升；"
          "三是原因已定位——基础策略本身安全完成率就有七八成，逐块改善被闭环分布漂移吃掉，"
          "下一步是提高不安全示范比例，并把每方法回合数提到四十以上。"),

    ("h1", "九、收尾（30 秒）"),
    ("p", "总结三句：一，给出了从噪声预测迁移到速度场的自洽移植，包括权重推导、求导位置，"
          "并用单元测试逐元素核对；二，搭了自由漂浮拆解仿真，把反作用导致的基座漂移做成可评测、可微分的量；"
          "三，用受控的逐块实验证明引导方向正确，同时如实报告闭环成功率还无可分辨提升。"),
    ("p", "汇报到这里，请各位老师批评指正。"),

    ("h1", "附：可能的提问（不上台说，备用）"),
    ("bullet", "为什么不直接用原来那三行？——根号是参数化换算系数，流匹配对位的是 1 减 t；直接搬量纲不对，严格换算在单步处发散。"),
    ("bullet", "权重为什么取平方？——与插值自洽的扩散系数是 1 减 t，修正量正比于平方除以二；两端分别自动关断与有界。"),
    ("bullet", "为什么不对整条去噪链反传？——代价大、方差高；先在干净动作上求梯度就已进入分布内。"),
    ("bullet", "基础策略已经够安全还要引导？——逐块检验证明引导降低了漂移与接触力，闭环被分布漂移抵消；要看到提升需先造不安全基线。"),
    ("bullet", "样本量够吗？——不够，三种回合数排名会翻转；主证据是逐块检验，下一步提到四十回合以上并报置信区间。"),
    ("bullet", "拒绝采样为什么是零？——能量缺任务进度项时“最省力”的解是原地不动；修法是加进度项或按成功筛选。"),
    ("bullet", "算力？——全 CPU：策略九百四十万参数五千步约二十五分钟；动力学两千五百步四十一秒；单回合三到六秒。"),
    ("bullet", "哪些代码是你写的？——原始两个仓库公开；新增三十七个文件约五千四百行，原有四个文件最小改动，清单见思维导图标注。"),
]


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


def build() -> Path:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(2.8)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("流匹配版 DynaGuide 代码讲解讲稿"), HEI, 20, True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    set_font(p.add_run("（〔可跳〕5 段约 600 字；略过即约 10 分钟，全念约 12—13 分钟）"), SONG, 11)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    set_font(p.add_run("宋银贺　2026-11-08"), SONG, 10.5)

    # 时间分配表
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.line_spacing = 1.0
    set_font(cap.add_run("表 1　时间分配与看哪里"), SONG, 10.5, True)

    table = doc.add_table(rows=len(SCHEDULE) + 1, cols=3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header = ["段落", "时间", "屏幕上打开"]
    for j, value in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ""
        run = cell.paragraphs[0].add_run(value)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(run, SONG, 10, True)
    for i, (name, when, where) in enumerate(SCHEDULE, start=1):
        for j, value in enumerate((name, when, where)):
            cell = table.cell(i, j)
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_before = Pt(1)
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.line_spacing = 1.0
            if j:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(para.add_run(value), SONG, 9.5)
    widths = [4.6, 3.4, 7.4]
    for j, w in enumerate(widths):
        for row in table.rows:
            row.cells[j].width = Cm(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    for kind, text in SCRIPT:
        if kind == "h1":
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(12)
            para.paragraph_format.space_after = Pt(6)
            set_font(para.add_run(text), HEI, 14, True)
        elif kind == "h2":
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(3)
            para.paragraph_format.line_spacing = 1.3
            set_font(para.add_run(text), HEI, 12, True)
        elif kind == "cue":
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Cm(0.6)
            para.paragraph_format.space_after = Pt(4)
            para.paragraph_format.line_spacing = 1.2
            set_font(para.add_run(text), HEI, 10.5)
        elif kind == "bullet":
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Cm(0.74)
            para.paragraph_format.first_line_indent = Cm(0)
            para.paragraph_format.space_after = Pt(3)
            para.paragraph_format.line_spacing = 1.35
            set_font(para.add_run("· " + text), SONG, 11)
        elif kind == "ps":
            para = doc.add_paragraph()
            pf = para.paragraph_format
            pf.first_line_indent = Cm(0.74)
            pf.space_before = Pt(0)
            pf.space_after = Pt(5)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.5
            set_font(para.add_run("〔可跳〕"), HEI, 10.5, True)
            set_font(para.add_run(text), SONG, 12)
        else:
            para = doc.add_paragraph()
            pf = para.paragraph_format
            pf.first_line_indent = Cm(0.74)
            pf.space_before = Pt(0)
            pf.space_after = Pt(5)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.5
            set_font(para.add_run(text), SONG, 12)

    doc.save(OUT_DOCX)
    return OUT_DOCX


def build_md() -> Path:
    lines = ["# 流匹配版 DynaGuide 代码讲解讲稿", "", "约 10 分钟；按「代码思维导图」分支顺序。", ""]
    lines += ["| 段落 | 时间 | 屏幕 |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in SCHEDULE]
    lines += [""]
    for kind, text in SCRIPT:
        if kind == "h1":
            lines += ["", f"## {text}", ""]
        elif kind == "h2":
            lines += ["", f"### {text}", ""]
        elif kind == "cue":
            lines += [f"> {text}", ""]
        elif kind == "bullet":
            lines += [f"- {text}"]
        elif kind == "ps":
            lines += [f"**〔可跳〕**{text}", ""]
        else:
            lines += [text, ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    return OUT_MD


if __name__ == "__main__":
    d = build()
    m = build_md()
    chars = sum(len(t) for k, t in SCRIPT if k in ("p", "ps"))
    chars_core = sum(len(t) for k, t in SCRIPT if k == "p")
    print(f"written {d} ({d.stat().st_size/1024:.0f} KB)")
    print(f"written {m} ({m.stat().st_size/1024:.1f} KB)")
    print(f"口播字数：全念 {chars} 字（约 {chars/280:.1f} 分钟）；略过可跳段 {chars_core} 字（约 {chars_core/280:.1f} 分钟）")


