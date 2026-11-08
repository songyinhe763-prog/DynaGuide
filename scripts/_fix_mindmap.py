"""One-off fix: wrap 4 nested list items into tuples in make_mindmap.py."""

from pathlib import Path

p = Path(r"C:\代码修改\dynaguide\scripts\make_mindmap.py")
lines = p.read_text(encoding="utf-8").split("\n")

targets = {
    '                "LinearFlowPath｜核心类", [',
    '                "ConsistencyFlowMatcher｜FlowPolicy 的一致性流匹配（可 1 步出动作）", [',
    '                "FlowSampler｜核心类", [',
    '                "FlowActionChunkPolicy｜核心类", [',
}

for i, line in enumerate(lines):
    if line in targets:
        lines[i] = line.replace('", [', '", [')
        # add the opening paren
        lines[i] = "                (" + line.strip()

# find the matching closing "])," or "]," for each opened bracket... simpler: rely on
# the existing "            ])," pattern that follows; the closing already has "])" only
# when we opened with "(" — so we must also convert the matching close.
# Instead: do a targeted pass converting the *next* line that is exactly "            ]),"
# after each newly opened tuple? We only opened "(" so the close must become "            ]),"
# from "            ],"
out = []
open_stack = []
for line in lines:
    out.append(line)

text = "\n".join(out)

# repair the closers: for each opened tuple, the corresponding closer is the line
# "            ]," (12 spaces + ],) that terminates that block. We convert the first such
# closer after each opening, in order.
lines = out
starts = [i for i, l in enumerate(lines) if l.startswith("                (") and l.rstrip().endswith('", [')]
fixed = 0
for s in starts:
    for j in range(s + 1, len(lines)):
        if lines[j] == "            ],":
            lines[j] = "            ]),"
            fixed += 1
            break

p.write_text("\n".join(lines), encoding="utf-8")
print(f"opened {len(starts)} tuples, repaired {fixed} closers")
