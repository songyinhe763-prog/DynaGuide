"""Build the speech script that walks *this mind map* branch by branch.

It reads ``C:\\代码修改\\1\\代码思维导图.md``, takes its ten top-level branches as the
section order, and verifies that every one of them is covered by the generated script.

Output:
    C:\\代码修改\\1\\代码讲解讲稿_按思维导图.docx
    C:\\代码修改\\1\\代码讲解讲稿_按思维导图.md

Usage::

    python scripts/make_speech_from_codemap.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

CODEMAP = Path("C:/代码修改/1/代码思维导图.md")
OUT_DIR = Path("C:/代码修改/1")
OUT_DOCX = OUT_DIR / "代码讲解讲稿_按思维导图.docx"
OUT_MD = OUT_DIR / "代码讲解讲稿_按思维导图.md"
SONG, HEI = "宋体", "黑体"

# h1 里的分支关键字 → 屏幕上展开哪一支（时长由字数自动算）
EXPAND = {
    "开场": "—",
    "根目录文件": "根目录文件（9 个）",
    "core/": "core/ 及其五个子分支",
    "robomimic": "robomimic/",
    "configs": "configs/ 与 calvin_exp_configs_examples/",
    "scripts": "scripts/",
    "tests": "tests/",
    "docs": "docs/",
    "第三方目录": "第三方目录",
    "运行产物": "运行产物",
    "讲解主线": "讲解主线（含 fig7 / fig2 / fig4）",
    "附": "（不上台说，备用）",
}

# kind: h1 / h2 / p / ps(可跳) / cue / bullet
SCRIPT: list[tuple[str, str]] = [
    ("h1", "开场（30 秒）"),
    ("cue", "（停在导图总节点）"),
    ("p", "各位老师好。我汇报的是对 DynaGuide 的代码改造，项目背景是失效卫星的在轨拆解——"
          "服务星贴着目标卫星，把一个模块从基座上拆下来带回去。"),
    ("p", "屏幕上这张是代码地图，按文件夹一路拆到文件、再到关键函数。下面我按它的分支顺序讲，"
          "讲到哪一支就展开哪一支，没有展开的部分是第三方代码，不动。"),
    ("p", "先给结论：改造只动推理阶段，不改任何训练权重；把原来 DDIM 噪声预测上的引导，"
          "搬到了流匹配的速度场上，并搭了一个自由漂浮拆解仿真来验证。"),

    ("h1", "一、根目录文件（1 分钟）"),
    ("cue", "（展开「根目录文件」这一支）"),
    ("p", "根目录九个文件，真正改动的是两个。run_dynaguide.py 是引导实验主程序，"
          "我在这里加了两个开关：--guidance_type 选引导类型，--flow_weight 选流匹配的权重调度。"
          "原来的 diffusion 分支一行没改，换开关就能切回去对照。"),
    ("p", "另一个是 .gitignore，只加了一行，把权重目录排除出版本库。"),
    ("ps", "其余都是原有的：拒绝采样基线、加载训练好的策略、训练扩散策略、训练动力学模型、"
          "动力学模型体检脚本、行为统计脚本，以及依赖与许可。这些没改，"
          "只需要知道它们各自负责什么——训练、评测、及统计各一条线。"),

    ("h1", "二、core/（4 分钟）"),
    ("cue", "（先只展开 core/ 的第一层，按五个子分支逐个展开）"),
    ("p", "core 是算法核心，原有五个文件，这次新增两个文件加两个子包。我按导图的顺序讲。"),

    ("h2", "1 新增 flow_dynaguide.py：流匹配版引导（2 分钟）"),
    ("cue", "（展开 flow_dynaguide.py 这一支）"),
    ("p", "这个文件只做一件事：把一个能量函数，变成速度场的修正量。"),
    ("p", "第一个类是 GuidanceWeightSpec，负责权重曲线 λ(t)，实现了四种调度："
          "默认的 fp 取 1 减 t 的平方除以 2；还有线性的、严格照搬 DDIM 的 ddim、以及常数。"
          "为什么不能直接照搬原来那三行？原来引导里那个根号下 1 减 alpha_bar，"
          "本质是“噪声参数化”和“score 参数化”之间的换算系数，它顺带起退火作用。"
          "流匹配里网络输出的是速度，对位的量是边际噪声标准差 1 减 t。"
          "如果把根号直接搬过来，量纲不对；严格换算会得到 1 减 t 除以 t，"
          "而单步推理恰好在 t 接近 0 处，这个系数会发散——这就是照搬会炸的原因。"
          "取平方除以 2 就两个好处：t 接近 1 时自动关断，t 接近 0 时有界，等于二分之一。"),
    ("p", "第二个类是核心的 FlowDynaGuide。它有三个要点，对应导图下面三行。"
          "一是 clean_estimate 做 Tweedie 投影：x_t 加 1 减 t 乘速度，得到干净动作估计；"
          "能量只作用在这个估计上，因为动力学模型和物理能量只见过干净动作块。"
          "二是 __call__ 的主流程：投影、算能量、求梯度、过护栏、得到修正量。"
          "导图里标了星号的那一行就是结果：速度的修正量等于负的强度乘以 λ(t) 再乘梯度，"
          "这一行就是原来那行补丁在流匹配里的对应式。"
          "三是 _postprocess 的三道护栏：按范数裁剪默认上限 5、逐元素裁剪默认关、跨步动量平滑默认关。"
          "引导强度非常敏感，不加限制很容易一步把动作打飞。"),
    ("ps", "最后两个函数：make_flow_guidance 把命令行参数装成引导对象；"
           "calculate_flow_classifier_guidance 是给原程序用的替换接口，"
           "签名和返回值和原 DDIM 版完全一致，所以主程序只换一个开关就能切通路。"),

    ("h2", "2 新增 flow_energies.py：引导在优化什么（1 分钟）"),
    ("cue", "（展开 flow_energies.py 这一支）"),
    ("p", "这里有两类能量，都写成标量代价、越小越好。第一类是 LatentMatchEnergy，"
          "沿用 DynaGuide 的做法：把动作块送进动力学模型预测末端潜状态，与正负样本库比距离，"
          "用 log-sum-exp 聚合。log-sum-exp 是软最小值，多个正目标时自动挑当前最可达的那个，"
          "正负相减对应论文里概率相除再取对数。"),
    ("p", "第二类是 DifferentiableDisassemblyModel.rollout，一个可微的拆解代理："
          "沿动作块推演十六步，返回十二个可微标量，包括最大接触力、冲量、基座漂移、拧转与拉出进度等。"
          "里面有两件事要说清：一是协议状态机，螺栓必须转够才解锁、卡扣必须压够、销必须拉力过破断力；"
          "二是导图上标星的那一行，反作用力矩按位矢叉乘负接触力算，角速度按力矩乘时间除惯量累加。"
          "基座惯量只有 0.6 这个量级，一次几牛的偏载就足以让姿态明显漂移。"),
    ("ps", "这上面再包一层 SatellitePhysicsEnergy，把六项加权：协议进度、过早拉拽、接触力、"
          "基座漂移、轴向偏差、是否对着正确模块。最后 CompositeEnergy 把两类能量相加，就是实验里的 both。"),

    ("h2", "3 core 原有五个文件（40 秒）"),
    ("cue", "（展开 dynaguide.py / dynamics_models.py 等五支）"),
    ("ps", "这五个都没改动，但和改造有关系。dynaguide.py 是原来的 DDIM 引导函数，"
          "流版就是照着它的签名写的，所以两者可以互换。dynamics_models.py 是原动力学模型，"
          "用冻结的 DINOv2 特征加六层 Transformer，卫星版沿用了它的两个接口："
          "一个给样本库打嵌入，一个给引导打分。另外三个分别是 CALVIN 场景复位、多视角数据集、"
          "以及重建分支，只在训练时看效果用。"),

    ("h2", "4 新增 flows/：流匹配底座（1 分钟）"),
    ("cue", "（展开 flows/ 这一支）"),
    ("p", "这个子包是纯 torch 实现，七个文件，按导图顺序说四个。"),
    ("p", "paths.py 定义坐标系：加噪写成从噪声到动作的直线插值，沿直线速度恒定；"
          "知道当前样本、时刻和速度就能还原两端——干净动作是 x_t 加 1 减 t 乘速度，"
          "源噪声是 x_t 减 t 乘速度，两者互逆，有单元测试。导图上标星的就是这个投影，引导的能量作用在它上面。"),
    ("ps", "consistency.py 是两个训练目标：默认最朴素的流匹配回归，另外实现了一致性流匹配，"
           "把时间切段，要求把段内任意点一步外推到段终点、且段内速度一致，于是可以少步出动作。"
           "里面那个 f_euler 就是论文里的 f，也是我们引导时用的干净动作估计。"),
    ("p", "samplers.py 是推理，也是引导的注入点。推理就是解常微分方程，每步是当前样本加速度乘步长；"
          "标星的那个 _guided_velocity 是注入点，只有四步：设为可求导、网络给原速度、"
          "引导函数返回修正量、相加后交给积分。参数里还有一个重复次数，"
          "对应论文里在同一噪声层多迭代几次的做法。"),
    ("ps", "剩下三个：nets.py 是一维 UNet 加 FiLM 条件；policy.py 是基础策略，"
           "关键是它的设计——策略本身不知道要拆哪个模块、用什么手法，所以会学成三种示范的平均，"
           "这正是需要推理期引导的原因；normalizer.py 把动作归一化到正负一之间。"),

    ("h2", "5 新增 satellite/：拆解仿真与数据（1 分钟）"),
    ("cue", "（展开 satellite/ 这一支）"),
    ("p", "spec.py 是所有数值的唯一来源：基座 28.27 公斤，三个模块对应三种紧固方式——"
          "螺栓先转 90 度、卡扣先压 6 毫米、摩擦销拉过 7 牛；预算分两档，"
          "姿态 0.12 弧度、接触力 4 牛是运行预算，8 牛是硬中止。"),
    ("p", "env.py 是仿真本体，导图里三行重点：观测是三十六维本体感知；"
          "step 的顺序是抓取、末端运动、紧固件状态机、接触力、反作用、判定；"
          "标星的是反作用——动作被解释为位移意图，只有被卡住那部分才产生接触力，"
          "这个力作为反作用力偶加回自由漂浮基座。判定分两档：拆下算成功，"
          "拆下且不超预算才算安全完成。在轨任务里，成功但把卫星推飞必须算失败。"),
    ("p", "expert.py 造示范，三种策略：正规流程先拧再拉；第二种先拽四步再拧，力全变基座反作用；"
          "第三种拔出方向偏 0.25 弧度，切向力持续推转基座。故意混入不安全示范，"
          "是为了复现“示范参差、策略学到平均行为”这个场景。"),
    ("ps", "latent.py 是动力学模型的本任务版：没有相机，就把冻结的 DINOv2 换成十四维语义特征加三层感知机，"
           "结构同构。两个细节：每维用固定名义尺度标准化，因为“是否连接”这种二值特征标准差接近零，"
           "用经验标准差会把潜空间尺度算坏；标星的 augment_actions 按一半概率给动作块加插值噪声，"
           "和采样器的噪声分布一致，这一项论文里明确做过消融。"),
    ("ps", "data.py 负责切数据：按回合切成观测窗口加动作块；正样本取安全回合的末状态，"
          "负样本取违约回合的末状态。math3d.py 是四元数工具。"),

    ("h1", "三、robomimic/（1 分钟）"),
    ("cue", "（展开 robomimic/ 这一支）"),
    ("p", "这里放的是把改造接回原有生态的一次验证。新增的 flow_policy.py 和原来的 diffusion_policy "
          "只有三处差异：输出从噪声变速度、目标从噪声回归变流匹配、推理从 DDIM 调度器变常微分方程积分；"
          "观测编码器、EMA、动作块、权重存取全部复用。"),
    ("p", "导图上几条重点：建网络时把批归一化换成组归一化以保证 EMA 稳定；"
          "训练时 fm 和 cfm 两条目标都支持；get_action 带引导类型参数；"
          "标星的 _wrap_guidance 是个小函数，负责兼容两种引导签名——四参数的是流匹配版，"
          "两参数的是原版，所以原有引导函数不用改就能挂上来。"),
    ("p", "配置类这边新增了 algo.flow 一组字段：objective、time_scale、采样步数、"
          "以及一致性流匹配的几个超参。要说明边界：这条通路我只在 CPU 上用合成观测做了集成测试，"
          "还没有在真实 CALVIN 数据集上跑完整评测。"),
    ("ps", "另外两个初始化文件各加一行注册；diffusion_policy 没改，"
          "但它是我们的对照点——那行噪声预测减去加权梯度的代码，就是被改造的对象。"),

    ("h1", "四、configs/ 与实验配置（20 秒）"),
    ("cue", "（展开 configs 与 calvin_exp_configs_examples）"),
    ("p", "新增两个 JSON：一个是流策略的训练配置，算法名换成 flow_policy，加一组 flow 字段；"
          "一个是引导实验配置，把引导类型设成 flow、权重设成 fp。其余的配置都是原有的，"
          "分别对应 DDIM 基线和各个任务的引导实验。"),

    ("h1", "五、scripts/（1 分钟）"),
    ("cue", "（展开 scripts/ 这一支）"),
    ("p", "十二个脚本，是一条链加两个分析工具。链条是：采示范、训练基础策略、训练动力学并建正负样本库、"
          "跑评测。评测主程序支持五种方法：不引导、只用潜空间能量、只用物理能量、两者联合、"
          "以及采样多条取最优；里面两个函数，build_energy 是方法到能量的开关，"
          "rollout_episode 是回合循环，每八步重规划一次。"),
    ("p", "分析工具两个：一个是引导强度扫描，能看到强度超过 0.5 之后明显退化；"
          "标星的是逐块检验，做法是拿同一个动作块沿能量改进三步、在真实仿真里重放，"
          "这是后面结果部分最硬的证据。还有一个出图脚本，把七张图一次生成。"),
    ("ps", "剩下几个是配套：debug_rollout 打印一个回合的逐步状态，common 放路径与种子，"
           "make_report_docx 出这份报告，make_codemap 生成屏幕上这张导图，"
           "run_all_checks 一键复现全部证据。"),

    ("h1", "六、tests/（40 秒）"),
    ("cue", "（展开 tests/ 这一支）"),
    ("ps", "四个套件三十四项断言，全在 CPU 上跑。第一组测底座：投影互逆、真速度零损失、"
          "采样器积分正确、中点法精度优于欧拉法、一致性损失可反传。"),
    ("p", "第二组是最关键的：端点梯度与解析解逐元素相等；打开全雅可比时确实包含恒等项；"
          "潜空间能量与论文公式一致；物理能量偏好先拧后拉而不是硬拽。"),
    ("ps", "第三组测仿真：专家能解各个模块、预算能触发中止、"
           "最重要的是错误协议确实会把基座推转——这一条保证物理约束是真的生效，不是摆设。"
           "第四组测集成：注册、两种目标可训练、带引导的 get_action 能被调用、非法参数被拒绝。"),

    ("h1", "七、docs/（20 秒）"),
    ("ps", "三篇说明。第一篇是推导与接口，包括根号系数的雅可比含义、λ(t) 的取舍、"
          "以及和 DynaGuide 超参的对照表。第二篇是任务说明，写清场景、物理、和四维评测口径。"
          "第三篇是实验结果，包括波动分析和下一步。"),

    ("h1", "八、第三方目录（20 秒）"),
    ("ps", "这四个没改，只讲用途：calvin 是仿真环境，data_processing_calvin 是数据切片与引导条件生成，"
          "paper_experiments 是原论文五组实验的跑批脚本，toy_squares_experiment 是原论文的玩具实验。"
          "不展开，改动都在前面的分支里。"),

    ("h1", "九、运行产物（20 秒）"),
    ("ps", "这些都是可以复现的、被 git 忽略的目录：data 下是示范数据，checkpoints 下是权重和训练历史，"
          "results 下是评测结果，logs 是原始日志。另外在代码目录外面还有一个运行证据目录，"
          "放测试日志、评测输出、诊断记录和七张图，评审要看原始记录可以直接去那里翻。"),

    ("h1", "十、讲解主线与结果（40 秒）"),
    ("cue", "（按导图末尾六步指屏幕：fig7 → flow_dynaguide 顶部 → delta_v 那行 → _guided_velocity → 反作用力矩 → fig2/fig4）"),
    ("p", "如果时间只够讲六步，就按导图末尾这个顺序：先看框架图，强调不改权重；"
          "再看 flow_dynaguide 顶部，讲清为什么不能照搬、λ(t) 怎么来的、梯度在哪求；"
          "然后指那一行修正量，说明它就是原补丁的对应式；接着指注入点那四步；"
          "再指反作用力矩那一行，说明微重力难在哪；最后切到两张结果图。"),
    ("p", "结果我先说数字，再说局限。八回合一组：不引导成功 88、安全完成 75；潜空间引导都是 88；"
          "物理能量 62；拒绝采样基线 0。二十回合排序会变：不引导 85 和 80，潜空间 65 和 45。"
          "两组放在一起看，差异落在统计噪声里，二十回合对应正负二十个百分点，所以我不说它显著提升。"),
    ("p", "最硬的证据是逐块检验：同一个动作块沿能量改进三步，在真实仿真里重放，"
          "结果是代理能量下降 17.3%，基座漂移在二十五个样本里十八个改善、平均降一成四，"
          "接触力二十个改善、平均降一成六。这说明代理和仿真方向一致，引导没有写反。"),
    ("p", "局限三条：全程 CPU，CALVIN 通路只做了合成观测的集成测试；样本量不足，排名会翻转；"
          "闭环上不去的原因已定位——基础策略本身安全完成率就有七八成，逐块改善被分布漂移吃掉。"
          "下一步是提高不安全示范比例，并把每方法回合数提到四十以上。汇报到这里，请各位老师批评指正。"),

    ("h1", "附：可能的提问"),
    ("bullet", "为什么不直接用原来那三行？——根号是参数化换算系数，流匹配对位的是 1 减 t；直接搬量纲不对，严格换算在单步处发散。"),
    ("bullet", "权重为什么取平方？——与插值自洽的扩散系数是 1 减 t，修正量正比于平方除以二；两端分别自动关断与有界。"),
    ("bullet", "为什么不对整条去噪链反传？——代价大、方差高；先在干净动作上求梯度就已进入分布内。"),
    ("bullet", "基础策略已经够安全还要引导？——逐块检验证明引导降低了漂移与接触力，闭环被分布漂移抵消；要看到提升需先造不安全基线。"),
    ("bullet", "样本量够吗？——不够，三种回合数排名会翻转；主证据是逐块检验。"),
    ("bullet", "拒绝采样为什么是零？——能量缺任务进度项时“最省力”的解是原地不动；修法是加进度项或按成功筛选。"),
    ("bullet", "算力？——全 CPU：策略九百四十万参数五千步约二十五分钟；动力学两千五百步四十一秒；单回合三到六秒。"),
    ("bullet", "哪些代码是你写的？——原始两个仓库公开；新增三十七个文件约五千四百行，原有四个文件最小改动。"),
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


def check_coverage() -> list[str]:
    """Every top-level branch of the mind map must appear as a speech section."""
    if not CODEMAP.exists():
        return ["（未找到导图文件，跳过覆盖检查）"]
    branches = []
    for line in CODEMAP.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^  - (.+)$", line)
        if m:
            branches.append(m.group(1))
    titles = [t for k, t in SCRIPT if k == "h1"]
    missing = []
    for branch in branches:
        key = branch.split("｜")[0].strip()
        if not any(key[:4] in t or key in t for t in titles):
            missing.append(branch)
    return [f"导图一级分支 {len(branches)} 个，缺失 {len(missing)} 个"] + [f"缺失：{m}" for m in missing]


def clean_title(title: str) -> str:
    """去掉标题里写死的时间（表格里已按字数算出）。"""
    return re.sub(r"（\d+\s*(分钟|秒)）$", "", title).strip()


def section_rows():
    """(分支, 字数, 时长字符串, 展开哪一支) —— 时长按 280 字/分钟由实际字数算出。"""
    rows, cur, buf = [], None, 0
    for kind, text in SCRIPT:
        if kind == "h1":
            if cur is not None:
                rows.append((cur, buf))
            cur, buf = clean_title(text), 0
        elif kind in ("p", "ps"):
            buf += len(text)
    if cur is not None:
        rows.append((cur, buf))
    out = []
    for title, chars in rows:
        key = next((k for k in EXPAND if k in title), "—")
        minutes = chars / 280.0
        when = f"{int(minutes)}:{int(round((minutes % 1) * 60)):02d}"
        out.append((title, chars, when, EXPAND.get(key, "—")))
    return out


def total_minutes(skip_skippable: bool = False) -> float:
    if not skip_skippable:
        return sum(len(t) for k, t in SCRIPT if k in ("p", "ps")) / 280.0
    return sum(len(t) for k, t in SCRIPT if k == "p") / 280.0


def build_docx() -> Path:
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
    set_font(
        p.add_run(
            f"（按《代码思维导图.md》的十个分支顺序；全念约 {total_minutes():.0f} 分钟，"
            f"略过标〔可跳〕的段落约 {total_minutes(True):.0f} 分钟）"
        ),
        SONG,
        10.5,
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    set_font(p.add_run("宋银贺　2026-11-08"), SONG, 10.5)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.line_spacing = 1.0
    set_font(cap.add_run("表 1　导图分支、字数与展开顺序"), SONG, 10.5, True)

    rows = section_rows()
    table = doc.add_table(rows=len(rows) + 1, cols=4)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for j, value in enumerate(["导图分支", "字数", "时长", "屏幕上展开"]):
        cell = table.cell(0, j)
        cell.text = ""
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(cell.paragraphs[0].add_run(value), SONG, 10, True)
    for i, (branch, chars, when, where) in enumerate(rows, start=1):
        for j, value in enumerate((branch, str(chars), when, where)):
            cell = table.cell(i, j)
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_before = Pt(1)
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.line_spacing = 1.0
            if j in (1, 2):
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(para.add_run(value), SONG, 9.5)
    for j, w in enumerate((3.2, 1.4, 1.4, 8.4)):
        for row in table.rows:
            row.cells[j].width = Cm(w)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    for kind, text in SCRIPT:
        if kind == "h1":
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(12)
            para.paragraph_format.space_after = Pt(6)
            set_font(para.add_run(clean_title(text)), HEI, 14, True)
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
    lines = ["# 流匹配版 DynaGuide 代码讲解讲稿", "",
             f"按《代码思维导图.md》的十个分支顺序；全念约 {total_minutes():.0f} 分钟，"
             f"略过〔可跳〕约 {total_minutes(True):.0f} 分钟。", "",
             "| 导图分支 | 字数 | 时长 | 屏幕上展开 |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in section_rows()]
    lines += [""]
    for kind, text in SCRIPT:
        if kind == "h1":
            lines += ["", f"## {clean_title(text)}", ""]
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
    d = build_docx()
    m = build_md()
    chars_all = sum(len(t) for k, t in SCRIPT if k in ("p", "ps"))
    chars_core = sum(len(t) for k, t in SCRIPT if k == "p")
    print(f"written {d} ({d.stat().st_size/1024:.0f} KB)")
    print(f"written {m} ({m.stat().st_size/1024:.1f} KB)")
    print(f"口播：全念 {chars_all} 字（约 {chars_all/280:.1f} 分钟）；略过可跳段 {chars_core} 字（约 {chars_core/280:.1f} 分钟）")
    for line in check_coverage():
        print("覆盖检查：", line)

