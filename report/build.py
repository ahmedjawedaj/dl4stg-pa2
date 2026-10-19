"""Assemble the PDF report: title page + Q1 + Q2 (tables pulled from results/) + AI disclosure.

Usage (from repo root): python report/build.py           draft PDF, lists every gap
                        python report/build.py --final   refuses to write a PDF while any gap remains
Needs pandoc and playwright (Chromium). Placeholders {{name}} in the report parts are filled from
results/*.md and results/*.json.
"""
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
REP = os.path.join(ROOT, "report")


def read(p, default=""):
    p = os.path.join(ROOT, p)
    return open(p).read() if os.path.exists(p) else default


def placeholders():
    """Tables from results/*.md, raw JSON (json:file.key) and derived, formatted values."""
    from pa2.evaluate import fmt
    ph = {}
    for p in glob.glob(os.path.join(R, "*.md")):
        ph[os.path.basename(p)[:-3]] = open(p).read().strip()
    for p in glob.glob(os.path.join(R, "*.json")):
        ph["json:" + os.path.basename(p)[:-5]] = json.load(open(p))
    if "json:table1" in ph:
        rows = ["| Check | Expected | Yours |", "|---|---|---|"]
        for r in ph["json:table1"]:
            cells = [str(r[k]).replace("|", "\\|") for k in ("check", "expected", "yours")]  # |x| in a table
            rows.append("| " + " | ".join(cells) + " |")
        ph["table1"] = "\n".join(rows)
    f1 = ph.get("json:fig1_flow_stats")
    pc = f1["speed_percentiles"] if f1 else None
    ph.update({} if not f1 else {
        "FIG1_MEDIAN": fmt(f1["median_speed"]), "FIG1_NPIX": f"{f1['n_pixels']:,}",
        "FIG1_DIRMIN": fmt(f1["dir_min_frac"]), "FIG1_DIRMAX": fmt(f1["dir_max_frac"]),
        "FIG1_MIRL1": fmt(f1["mirror_l1"]), "FIG1_ROTL1": fmt(f1["rot90_l1"]),
        "FIG1_UNIL1": fmt(f1["uniform_l1"]), "FIG1_P10": fmt(pc["10"]), "FIG1_P90": fmt(pc["90"]),
        "FIG1_P99": fmt(pc["99"]), "FIG1_2X": fmt(f1["speed_2x"]), "FIG1_3X": fmt(f1["speed_3x"]),
        "FIG1_ABOVE6": fmt(100 * f1["frac_above_6px"]),
        "FIG1_OVL": fmt(100 * f1["overlap_pixel_frac"]),
        "FIG1_OVL_MED": fmt(f1["overlap_median_speed"]), "FIG1_SEP_MED": fmt(f1["nonoverlap_median_speed"]),
        "FIG1_OVL_LOW": fmt(100 * f1["overlap_frac_below_p10"]),
        "FIG1_SEP_DIRMIN": fmt(f1["dir_sep_min_frac"]), "FIG1_SEP_DIRMAX": fmt(f1["dir_sep_max_frac"]),
        "FIG1_SEP_L1": fmt(f1["dir_sep_vs_all_l1"]),
    })
    hw = ph.get("json:hardware")
    if hw:
        ph["HARDWARE"] = (f"{hw['cpu']}, {hw['ram_gb']:g} GB RAM, {hw['platform']}, PyTorch {hw['torch']}, "
                          f"device {hw['device']}")
    f2 = ph.get("json:fig2_videos")
    if f2:
        ph["FIG2_CAPTION"] = ("Videos " + ", ".join(map(str, f2["fig2_dataset_indices"])) +
                              f" (dataset index). Video {9000 + f2['overlap'][0]} has overlapping digits.")
    for gate, q, k1, k2 in (("phase2_gate", "Q2_5", "Q2_5_COMMIT", "PHASE2_START"),
                            ("phase3_gate", "Q2_11", "Q2_11_COMMIT", "PHASE3_START")):
        m = re.search(r"committed at (\S+) in ([0-9a-f]{40}), phase \d started at (\S+)",
                      read(f"logs/{gate}.txt"))
        if m:
            ph[k1], ph[k2] = f"{m.group(1)} (commit {m.group(2)[:7]})", m.group(3)
            # insert the version that was committed before the experiments, not the working tree
            blob = subprocess.run(["git", "show", f"{m.group(2)}:answers/{q}.md"], cwd=ROOT,
                                  capture_output=True, text=True)
            if blob.returncode == 0:
                ph[f"ANSWER_{q}_COMMITTED"] = blob.stdout.strip()
                if read(f"answers/{q}.md").strip() != blob.stdout.strip():
                    ph[f"PROBLEM_{q}"] = f"answers/{q}.md differs from its commit {m.group(2)[:7]}"
    f3 = ph.get("json:fig3_videos")
    if f3:
        ph["FIG3_GIFS"] = ", ".join(f"`{g}`" for g in f3["gifs"])
    f5 = ph.get("json:fig5_video")
    if f5:
        ph["FIG5_CAPTION"] = f"Test video {f5['dataset_idx']} (dataset index)."
    f6 = ph.get("json:fig6")
    if f6:
        ph["FIG6_GIFS"] = ", ".join(f"`{g}`" for g in f6["fig6a_gifs"])
    ph.update(json.load(open(os.path.join(REP, "manual.json"))) if os.path.exists(os.path.join(REP, "manual.json")) else {})
    return ph


def fill(text, ph):
    def rep(m):
        key = m.group(1).strip()
        if key.startswith("file:"):
            p = os.path.join(ROOT, key[5:])
            return open(p).read().strip() if os.path.exists(p) else f"**[missing: {key}]**"
        if key.startswith("json:"):
            parts = key.split(".")
            v = ph[parts[0]]
            for k in parts[1:]:
                v = v[int(k)] if isinstance(v, list) else v[k]
            return str(v)
        if key not in ph:
            return f"**[missing: {key}]**"
        return ph[key]
    return re.sub(r"\{\{(.+?)\}\}", rep, text)


def readiness(md, ph, absent):
    """Everything that would make the PDF incomplete. Checked before rendering."""
    gaps = [f"missing report part {n}" for n in absent]
    gaps += [f"unfilled {k}" for k in re.findall(r"\[missing: (.+?)\]", md)]
    gaps += [f"missing image {p}" for p in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
             if not os.path.exists(os.path.join(REP, p))]
    gaps += [f"missing GIF {g}" for g in re.findall(r"`(gifs/[^`]+\.gif)`", md)
             if not os.path.exists(os.path.join(ROOT, g))]
    url = ph.get("REPO_URL", "")
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", url) or "REPLACE" in url:
        gaps.append(f"repository URL not set ({url!r})")
    gaps += [v for k, v in ph.items() if k.startswith("PROBLEM_")]
    return gaps


def render(src):
    html = os.path.join(REP, "report.html")
    subprocess.run(["pandoc", src, "-f", "markdown+tex_math_dollars+pipe_tables+raw_html",
                    "-t", "html5", "--standalone", "--mathml", "--css", "style.css",
                    "--metadata", "pagetitle=DL4STG Assignment 2", "-o", html],
                   check=True, cwd=REP)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        executable = os.environ.get("PA2_CHROMIUM_EXECUTABLE")
        b = p.chromium.launch(**({"executable_path": executable} if executable else {}))
        pg = b.new_page()
        pg.goto("file://" + html)
        pg.wait_for_timeout(500)
        pg.pdf(path=os.path.join(REP, "DL4STG_PA2_report.pdf"), format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#777">'
                               '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        b.close()


def main():
    final = "--final" in sys.argv
    ph = placeholders()
    names = ["title.md", "q1.md", "q2_part1.md", "q2_part2.md", "q2_part3.md", "ai_disclosure.md"]
    absent = [n for n in names if not os.path.exists(os.path.join(REP, n))]
    md = "\n\n".join(p for p in (fill(read("report/" + n), ph) for n in names) if p)
    gaps = readiness(md, ph, absent)
    for g in gaps:
        print("GAP:", g)
    if final and gaps:
        sys.exit(f"NOT SUBMISSION READY: {len(gaps)} gaps, no PDF written.")
    src = os.path.join(REP, "report_full.md")
    open(src, "w").write(md)
    render(src)
    print("built report/DL4STG_PA2_report.pdf", "(DRAFT, not submission ready)" if gaps else "(submission ready)")


if __name__ == "__main__":
    main()
