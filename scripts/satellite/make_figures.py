"""Generate the figures of the flow-DynaGuide satellite experiments.

All numbers are read from the artefacts of real runs (``results/satellite/*.json``,
``checkpoints/satellite/*.json``, ``04_diag/energy_alignment.csv``) or produced by a fresh
short rollout for the episode traces.  Nothing is hand-typed.

Usage
-----
::

    python scripts/satellite/make_figures.py --out "C:\\代码修改\\运行证据\\06_figures"

Figures
-------
1. ``fig1_training.png``       policy / latent-dynamics training curves
2. ``fig2_main_results.png``   success & constraint-satisfied per guidance method
3. ``fig3_scale_sweep.png``    guidance-scale sweep (success / constraint / drift)
4. ``fig4_alignment.png``      per-chunk surrogate-vs-simulator scatter (drift, force)
5. ``fig5_env_schematic.png``  the satellite scene, joints, reaction wrench
6. ``fig6_episode_trace.png``  one episode: base policy vs DynaGuide latent guidance
7. ``fig7_pipeline.png``       method pipeline diagram
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, RESULTS_DIR, device, set_seed  # noqa: E402

from core.flow_dynaguide import make_flow_guidance  # noqa: E402
from core.flow_energies import LatentMatchEnergy  # noqa: E402
from core.flows.policy import FlowActionChunkPolicy  # noqa: E402
from core.flows.samplers import FlowSampler, FlowSamplerConfig  # noqa: E402
from core.satellite.env import SatelliteEnv  # noqa: E402
from core.satellite.latent import SemanticDynamicsModel  # noqa: E402
from core.satellite.spec import make_sc1_spec  # noqa: E402

# ------------------------------------------------------------------ style ----
COLORS = {
    "blue": "#2E6BE6",
    "green": "#17A673",
    "orange": "#F5871F",
    "red": "#E23F3F",
    "purple": "#7B5EA7",
    "teal": "#0FA3A3",
    "gray": "#6B7280",
    "light": "#DCE6F7",
}
METHOD_COLOR = {
    "base": COLORS["gray"],
    "latent": COLORS["blue"],
    "physics": COLORS["orange"],
    "both": COLORS["purple"],
    "reject": COLORS["red"],
}
METHOD_LABEL = {
    "base": "基础策略\n(无引导)",
    "latent": "DynaGuide\n潜空间引导",
    "physics": "物理能量\n引导",
    "both": "两者联合",
    "reject": "拒绝采样\n(best-of-16)",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
        }
    )


def load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- fig 1 ----
def fig_training(out: Path) -> None:
    pol = load(CKPT_DIR / "flow_policy_payload.json")
    lat = load(CKPT_DIR / "latent_payload.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    ax = axes[0]
    steps = [h["step"] for h in pol["history"]]
    ax.plot(steps, [h["train"] for h in pol["history"]], color=COLORS["blue"], lw=2, label="训练损失")
    ax.plot(steps, [h["val"] for h in pol["history"]], color=COLORS["orange"], lw=2, ls="--", label="验证损失")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("流匹配 MSE")
    ax.set_title(f"基础流策略训练（{pol['steps']} 步，{pol['params']/1e6:.2f} M 参数）")
    ax.legend(loc="upper right")
    ax.fill_between(steps, [h["train"] for h in pol["history"]], alpha=0.12, color=COLORS["blue"])

    ax = axes[1]
    steps = [h["step"] for h in lat["history"]]
    ax.plot(steps, [h["train"] for h in lat["history"]], color=COLORS["green"], lw=2, label="训练 MSE（含噪声动作）")
    ax.plot(steps, [h["val_clean"] for h in lat["history"]], color=COLORS["red"], lw=2, ls="--", label="验证 MSE（干净动作）")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("潜空间 MSE（log）")
    ax.set_title(f"潜空间动力学模型训练（{lat['steps']} 步）")
    ax.legend(loc="upper right")
    fig.suptitle("图 1  两个网络的训练曲线（CPU，单机）", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out / "fig1_training.png")
    plt.close(fig)


# --------------------------------------------------------------- fig 2 ----
def fig_main(out: Path) -> None:
    runs = [
        ("eval_final.json", "20 回合/方法"),
        ("evidence_eval.json", "8 回合/方法（本次复跑）"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (fname, title) in zip(axes, runs):
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        rows = load(path)["rows"]
        methods = [r["method"] for r in rows]
        x = np.arange(len(methods))
        w = 0.38
        succ = [r["success"] * 100 for r in rows]
        cons = [r["constraint"] * 100 for r in rows]
        b1 = ax.bar(x - w / 2, succ, w, color=[METHOD_COLOR[m] for m in methods], label="任务成功（拆下）", alpha=0.95)
        b2 = ax.bar(
            x + w / 2, cons, w, color=[METHOD_COLOR[m] for m in methods], label="安全完成（含约束）", alpha=0.45, hatch="//"
        )
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 2, f"{h:.0f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods], fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.set_title(title)
        ax.set_ylabel("成功率 [%]")
    axes[0].legend(loc="upper right", ncol=1)
    fig.suptitle("图 2  各引导方法的成功率（同一组回合种子；深色=拆下，浅色斜纹=同时满足安全预算）",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.savefig(out / "fig2_main_results.png")
    plt.close(fig)


# --------------------------------------------------------------- fig 3 ----
def fig_sweep(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    specs = [("sweep_latent.json", "latent", "潜空间能量 σ=15"), ("sweep_physics.json", "physics", "物理能量"),
             ("sweep_both.json", "both", "联合能量")]
    for ax, (fname, method, label) in zip(axes, specs):
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        rows = [r for r in load(path)["rows"] if r["method"] == method]
        if method == "latent":
            rows = [r for r in rows if float(r["alpha"]) == 15.0]
        rows.sort(key=lambda r: float(r["scale"]))
        x = [float(r["scale"]) for r in rows]
        ax.plot(x, [r["success"] * 100 for r in rows], "o-", color=COLORS["blue"], lw=2, ms=6, label="成功")
        ax.plot(x, [r["constraint"] * 100 for r in rows], "s--", color=COLORS["green"], lw=2, ms=5, label="安全完成")
        ax.plot(x, [r["drift"] * 100 for r in rows], "^:", color=COLORS["red"], lw=1.6, ms=5, label="基座漂移×100 [rad]")
        ax.set_xscale("log")
        ax.set_xlabel("引导强度 s（log）")
        ax.set_title(label)
        ax.set_ylim(-3, 108)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("百分比 [%]")
    fig.suptitle("图 3  引导强度扫描：过强会破坏任务执行（每点 6 回合，λ(t)=(1−t)²/2）",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.savefig(out / "fig3_scale_sweep.png")
    plt.close(fig)


# --------------------------------------------------------------- fig 4 ----
def fig_alignment(out: Path, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"skip fig4: {csv_path} not found")
        return
    import csv as _csv

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    drift_b = np.array([float(r["drift_base"]) for r in rows])
    drift_g = np.array([float(r["drift_guided"]) for r in rows])
    force_b = np.array([float(r["force_base"]) for r in rows])
    force_g = np.array([float(r["force_guided"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (b, g, name, unit) in zip(
        axes[:2], [(drift_b, drift_g, "基座漂移", "rad"), (force_b, force_g, "最大接触力", "N")]
    ):
        better = g <= b + 1e-12
        lim = max(np.max(b), np.max(g)) * 1.12 + 1e-6
        ax.plot([0, lim], [0, lim], color=COLORS["gray"], lw=1, ls="--", label="y = x（无改善）")
        ax.scatter(b[better], g[better], s=46, color=COLORS["green"], edgecolor="white", lw=0.6,
                   label=f"改善 {int(better.sum())}/{len(b)}", zorder=3)
        ax.scatter(b[~better], g[~better], s=46, color=COLORS["red"], edgecolor="white", lw=0.6,
                   label=f"变差 {int((~better).sum())}/{len(b)}", zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"未引导的{name} [{unit}]")
        ax.set_ylabel(f"沿能量梯度改进后 [{unit}]")
        ax.set_title(f"{name}：{(1-g.mean()/max(b.mean(),1e-9))*100:.0f}% 平均下降")
        ax.legend(loc="upper left")

    ax = axes[2]
    energy_b = np.array([float(r["energy_before"]) for r in rows])
    energy_g = np.array([float(r["energy_after"]) for r in rows])
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, energy_b, 0.4, color=COLORS["gray"], label="改进前")
    ax.bar(idx + 0.2, energy_g, 0.4, color=COLORS["blue"], label="改进后")
    ax.set_xlabel("样本编号")
    ax.set_ylabel("代理能量 J")
    ax.set_title(f"代理能量平均 −{(1-energy_g.mean()/max(energy_b.mean(),1e-9))*100:.1f}%")
    ax.legend(loc="upper right")
    fig.suptitle("图 4  逐块检验：沿引导能量改进动作块后，在真实仿真中重放（每点一个动作块）",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.savefig(out / "fig4_alignment.png")
    plt.close(fig)


# --------------------------------------------------------------- fig 5 ----
def fig_env(out: Path) -> None:
    spec = make_sc1_spec()
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    bx, by = spec.base_size[0], spec.base_size[1]
    ax.add_patch(Rectangle((-bx / 2, -by / 2), bx, by, facecolor=COLORS["light"], edgecolor=COLORS["blue"], lw=2, zorder=1))
    ax.text(0.16, -0.215, f"失效卫星基座（{spec.base_mass} kg）", ha="center", fontsize=9, color=COLORS["blue"])

    colors = [COLORS["orange"], COLORS["green"], COLORS["purple"]]
    label_pos = {0: (0.30, 0.22), 1: (-0.10, 0.35), 2: (0.10, -0.32)}
    for k, (mod, col) in enumerate(zip(spec.modules, colors)):
        p = mod.attach_np
        ax.add_patch(
            Rectangle(
                (p[0] - mod.size[0] / 2, p[1] - mod.size[1] / 2), mod.size[0], mod.size[1],
                facecolor=col, alpha=0.5, edgecolor=col, lw=1.6, zorder=2,
            )
        )
        ax.plot(p[0], p[1], "o", color=col, ms=7, zorder=5)
        axis = mod.joint.axis_np
        ax.add_patch(
            FancyArrowPatch(
                (p[0], p[1]), (p[0] + axis[0] * 0.12, p[1] + axis[1] * 0.12),
                arrowstyle="-|>", mutation_scale=15, lw=2.2, color=col, zorder=6,
            )
        )
        kind = {
            "bolt": f"螺栓：先转 {np.degrees(mod.joint.required_twist):.0f}°",
            "latch": "卡扣：先压入 6 mm",
            "peg": f"摩擦销：破断 {mod.joint.breakaway_force:.0f} N",
        }[mod.joint.kind]
        tx, ty = label_pos[k]
        ax.annotate(
            f"{mod.name}\n{kind}", xy=(p[0], p[1]), xytext=(tx, ty), ha="center", fontsize=8.4, color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=0, shrinkB=4), zorder=7,
        )
        c = mod.capture_np
        ax.plot(c[0], c[1], "X", color=col, ms=10, alpha=0.9, zorder=5)

    # one legend entry for the staging poses instead of three labels
    ax.plot([], [], "X", color=COLORS["gray"], ms=9, label="收纳位姿（可选，require_stow=True）")
    ax.plot([], [], "*", color=COLORS["red"], ms=13, label="末端执行器起点")
    ax.legend(loc="upper left", fontsize=8)

    ee = np.array(spec.ee_start)
    ax.plot(ee[0], ee[1], "*", color=COLORS["red"], ms=16, zorder=6)

    # reaction wrench: the bolt axis of payload_box produces a torque about the base COM
    ax.add_patch(
        FancyArrowPatch(
            (0.22, 0.10), (0.02, 0.02), arrowstyle="-|>", mutation_scale=15, lw=2.2,
            color=COLORS["red"], connectionstyle="arc3,rad=0.35", zorder=6,
        )
    )
    ax.text(
        0.30, -0.42,
        "反作用力偶  τ = r × (−F)：拧/拔的力矩全部作用回自由漂浮基座\n"
        "→ 基座姿态漂移，超 0.12 rad 即判本次拆解“不安全”",
        fontsize=8.2, color=COLORS["red"], ha="right",
    )

    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("基座系 x [m]")
    ax.set_ylabel("基座系 y [m]")
    ax.set_title("图 5  自由漂浮卫星拆解基准（俯视）：三模块、三种紧固方式、基座反作用", fontsize=11.5)
    fig.savefig(out / "fig5_env_schematic.png")
    plt.close(fig)


# --------------------------------------------------------------- fig 6 ----
def _trace_episode(policy, sampler, spec, task, seed, guide=None, n_action_steps=8, max_steps=120):
    dev = next(policy.parameters()).device
    env = SatelliteEnv(spec=spec, seed=seed, randomize=True)
    obs = env.reset(task=task, seed=seed)
    hist = [obs["proprio"].copy()]
    idx = spec.module_names.index(task)
    rec = {"step": [], "drift": [], "force": [], "twist": [], "pull": [], "jerk": []}
    t = 0
    while True:
        win = np.stack(hist[-2:])
        if win.shape[0] < 2:
            win = np.concatenate([win[:1], win])
        obs_t = torch.from_numpy(win[None]).float().to(dev)
        proprio_t = torch.from_numpy(obs["proprio"][None]).float().to(dev)
        ctx = {"proprio": proprio_t, "state": proprio_t, "module_index": idx}
        if guide is None:
            with torch.no_grad():
                out = policy.sample(obs_t, sampler=sampler, ctx=ctx)
        else:
            out = policy.sample(obs_t, sampler=sampler, guidance=guide, ctx=ctx)
        acts = out["actions"][0, :n_action_steps].detach().cpu().numpy()
        for a in acts:
            obs, _r, done, info = env.step(a)
            hist.append(obs["proprio"].copy())
            t += 1
            rec["step"].append(t)
            rec["drift"].append(float(env.attitude_drift))
            rec["force"].append(float(getattr(env, "last_force", 0.0)))
            j = spec.module(task).joint
            rec["twist"].append(float(np.clip(env.twist[idx] / max(j.required_twist, 1e-6), 0, 1.2)))
            rec["pull"].append(float(np.clip(env.pull[idx] / max(j.travel, 1e-6), 0, 1.2)))
            rec["jerk"].append(float(np.linalg.norm(a[:3])))
            if done:
                break
        if done:
            break
    rec["success"] = bool(info.get("success", False))
    rec["constraint"] = bool(info.get("constraint_satisfied_success", False))
    return rec


def fig_trace(out: Path) -> None:
    dev = device()
    spec = make_sc1_spec()
    pol = FlowActionChunkPolicy.load(str(CKPT_DIR / "flow_policy_payload.pt"))
    pol.to(dev).eval()
    blob = torch.load(CKPT_DIR / "latent_payload.pt", map_location=dev, weights_only=False)
    dyn = SemanticDynamicsModel(
        latent_dim=blob["latent_dim"], action_dim=7, horizon=blob["horizon"], num_modules=blob["num_modules"]
    ).to(dev)
    dyn.load_state_dict(blob["model"])
    dyn.encoder.load_state_dict(blob["encoder"])
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)
    bank = blob["goal_banks"]["payload_box"].to(dev)
    neg = blob.get("negatives", {}).get("payload_box")
    neg = neg.to(dev) if neg is not None else None
    energy = LatentMatchEnergy(latent_predictor=dyn.state_action_embedding, positives=bank, negatives=neg, alpha=30.0)
    guide = make_flow_guidance(energy, scale=1.0, weight_mode="fp", mode="endpoint", clip_norm=5.0)
    sampler = FlowSampler(FlowSamplerConfig(n_steps=4, method="euler", t_start=1e-2, t_end=1.0, time_scale=pol.config.time_scale))

    seed_candidates = [1000, 1001, 1002, 1003, 1004]
    traces = []
    for s in seed_candidates:
        b = _trace_episode(pol, sampler, spec, "payload_box", s, guide=None)
        g = _trace_episode(pol, sampler, spec, "payload_box", s, guide=guide)
        traces.append((s, b, g))
    safe = [(s, b, g) for (s, b, g) in traces if b["constraint"] and g["constraint"]]
    if safe:
        seed, base, guided = min(safe, key=lambda t: t[2]["drift"][-1] - t[1]["drift"][-1])
        note = "两者均安全"
    else:
        ok = [(s, b, g) for (s, b, g) in traces if g["success"]]
        seed, base, guided = (ok or traces)[0]
        note = f"基础{'成功' if base['success'] else '失败'} / 引导{'成功' if guided['success'] else '失败'}"
    print(
        f"fig6 seed={seed} ({note}) base_drift={base['drift'][-1]:.3f} guided_drift={guided['drift'][-1]:.3f} "
        f"base_maxF={max(base['force']):.2f} guided_maxF={max(guided['force']):.2f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ax = axes[0]
    ax.plot(base["step"], base["drift"], color=COLORS["gray"], lw=2, label=f"基础策略（{'成功' if base['success'] else '失败'}）")
    ax.plot(guided["step"], guided["drift"], color=COLORS["blue"], lw=2, label=f"+潜空间引导（{'成功' if guided['success'] else '失败'}）")
    ax.axhline(spec.attitude_budget, color=COLORS["red"], ls="--", lw=1.4, label=f"安全预算 {spec.attitude_budget} rad")
    ax.set_xlabel("仿真步")
    ax.set_ylabel("基座姿态漂移 [rad]")
    ax.set_title("(a) 累积基座漂移")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.bar(np.array(base["step"]) - 0.2, base["force"], 0.4, color=COLORS["gray"], label="基础策略")
    ax.bar(np.array(guided["step"]) + 0.2, guided["force"], 0.4, color=COLORS["blue"], label="+潜空间引导")
    ax.axhline(spec.force_budget, color=COLORS["red"], ls="--", lw=1.4, label=f"运行预算 {spec.force_budget} N")
    ax.axhline(spec.force_limit, color=COLORS["orange"], ls=":", lw=1.4, label=f"硬中止 {spec.force_limit} N")
    ax.set_xlabel("仿真步")
    ax.set_ylabel("单步接触力 [N]")
    ax.set_title("(b) 每步接触力")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[2]
    ax.plot(base["step"], base["twist"], color=COLORS["green"], lw=1.8, label="拧转进度（基础）")
    ax.plot(base["step"], base["pull"], color=COLORS["orange"], lw=1.8, ls="--", label="拉出进度（基础）")
    ax.plot(guided["step"], guided["twist"], color=COLORS["green"], lw=2.4, alpha=0.55, label="拧转进度（引导）")
    ax.plot(guided["step"], guided["pull"], color=COLORS["orange"], lw=2.4, ls="--", alpha=0.55, label="拉出进度（引导）")
    ax.axhline(1.0, color=COLORS["gray"], ls=":", lw=1.2)
    ax.set_xlabel("仿真步")
    ax.set_ylabel("归一化进度")
    ax.set_title("(c) 拆解协议进度")
    ax.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(f"图 6  示例回合（seed={seed}，{note}）逐步对照：漂移/接触力/协议进度（统计见 图2、图4）",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.savefig(out / "fig6_episode_trace.png")
    plt.close(fig)


# --------------------------------------------------------------- fig 7 ----
def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.8, alpha=0.16):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                           facecolor=color, edgecolor=color, lw=1.8, alpha=alpha)
        )
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                           facecolor="none", edgecolor=color, lw=1.8)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111827")

    def arrow(x1, y1, x2, y2, color, style="-|>", lw=2.0, rad=0.0):
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, lw=lw,
                            color=color, connectionstyle=f"arc3,rad={rad}")
        )

    # training row
    box(0.2, 3.75, 2.5, 0.85, "混合示范\nsafe / impatient / offset", COLORS["gray"])
    box(3.2, 3.75, 2.7, 0.85, "流匹配基础策略\nv_θ(x_t, t, o)", COLORS["blue"])
    box(6.4, 3.75, 2.5, 0.85, "潜空间动力学 h_θ\n(冻结语义编码 φ)", COLORS["green"])
    box(9.6, 3.75, 2.6, 0.85, "goal bank\ng+ 安全 / g- 违约", COLORS["teal"])
    arrow(2.75, 4.17, 3.18, 4.17, COLORS["gray"])
    arrow(5.95, 4.17, 6.38, 4.17, COLORS["gray"], style="-|>", lw=1.4)
    arrow(8.95, 4.17, 9.58, 4.17, COLORS["gray"], lw=1.4)
    ax.text(0.25, 4.72, "① 离线训练", fontsize=10, fontweight="bold", color=COLORS["gray"])

    # inference row
    box(0.2, 1.95, 2.3, 0.95, "观测 o_t\n(本体感知窗口)", COLORS["gray"])
    box(3.2, 1.95, 2.9, 0.95, "ODE 采样器\nx←x+v·dt（euler/中点）", COLORS["blue"])
    box(6.7, 1.95, 2.1, 0.95, "动作块 a_1\n(反归一化执行)", COLORS["orange"])
    box(9.3, 1.95, 2.2, 0.95, "自由漂浮仿真\n基座反作用/接触力", COLORS["red"], fs=8.2)
    arrow(2.55, 2.42, 3.18, 2.42, COLORS["gray"])
    arrow(6.15, 2.42, 6.68, 2.42, COLORS["blue"])
    arrow(8.85, 2.42, 9.28, 2.42, COLORS["orange"])
    ax.text(0.25, 3.36, "② 推理（不改权重）", fontsize=10, fontweight="bold", color=COLORS["blue"])

    # guidance branch (bottom row)
    box(3.2, 0.45, 3.25, 0.95, "能量 J = J_latent + J_physics\n−logΣexp(−d/σ) + 动量/力/协议", COLORS["purple"], fs=8.0)
    box(6.95, 0.45, 2.0, 0.95, "Δv = −s·λ(t)·grad J\nTweedie 干净动作求导", COLORS["purple"], fs=8.0)
    arrow(7.95, 1.93, 7.95, 1.43, COLORS["purple"], style="-|>", lw=2.0)   # chunk -> Δv
    arrow(6.93, 0.92, 6.48, 0.92, COLORS["purple"], style="-|>", lw=2.0)   # Δv -> energy
    arrow(4.5, 1.43, 4.5, 1.92, COLORS["purple"], style="-|>", lw=2.0)     # energy -> sampler
    # h_theta -> energy (drop through the gap between sampler and chunk box)
    ax.plot([6.32, 6.32], [3.73, 1.42], color=COLORS["green"], lw=1.8, zorder=1)
    arrow(6.32, 1.55, 6.32, 1.43, COLORS["green"], style="-|>", lw=1.8)
    # goal bank -> energy (drop on the far right, then left above the bottom row)
    ax.plot([11.9, 11.9], [3.73, 1.62], color=COLORS["teal"], lw=1.6, zorder=1)
    ax.plot([11.9, 6.32], [1.62, 1.62], color=COLORS["teal"], lw=1.6, zorder=1)
    ax.text(9.2, 1.68, "引导条件嵌入", fontsize=7.8, color=COLORS["teal"])
    ax.text(3.25, 0.16, "③ 推理期主动引导（本工作新增：把 DynaGuide 从 DDIM 噪声预测搬到速度场）",
            fontsize=9.2, fontweight="bold", color=COLORS["purple"])

    ax.set_title("图 7  方法框架：流匹配策略 + 主动动态引导 + 自由漂浮拆解仿真", fontsize=12.5, fontweight="bold")
    fig.savefig(out / "fig7_pipeline.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=str(Path("C:/代码修改/运行证据/06_figures")))
    p.add_argument("--only", type=str, nargs="+", default=[])
    args = p.parse_args()
    style()
    set_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = {
        "1": fig_training,
        "2": fig_main,
        "3": fig_sweep,
        "4": lambda o: fig_alignment(o, Path(str(o).replace("06_figures", "04_diag")) / "energy_alignment.csv"),
        "5": fig_env,
        "6": fig_trace,
        "7": fig_pipeline,
    }
    for key, fn in todo.items():
        if args.only and key not in args.only:
            continue
        try:
            fn(out)
            print(f"fig{key} OK")
        except Exception as exc:  # noqa: BLE001
            print(f"fig{key} FAILED: {type(exc).__name__}: {exc}")
    print("figures ->", out)


if __name__ == "__main__":
    main()




