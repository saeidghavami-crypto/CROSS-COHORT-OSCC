#!/usr/bin/env python3
"""Build and format the publication-oriented Word report from ML outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "203748"
MUTED = "666666"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
PALE_GOLD = "FFF4D8"
PALE_RED = "FBE9EA"
WHITE = "FFFFFF"
BLACK = "000000"
CONTENT_DXA = 9360


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--docx", type=Path, required=True)
    return p.parse_args()


def set_run_font(run, size=11, bold=False, italic=False, color=BLACK, name="Calibri"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def configure_styles(doc: Document):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    tokens = {
        "Title": (26, NAVY, 0, 8),
        "Subtitle": (14, MUTED, 0, 12),
        "Heading 1": (16, BLUE, 16, 8),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (11.5, DARK_BLUE, 8, 4),
    }
    for name, (size, color, before, after) in tokens.items():
        st = styles[name]
        st.font.name = "Calibri"
        st._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        st._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor.from_string(color)
        st.font.bold = name != "Subtitle"
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Calibri"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.append(fld)


def configure_page(doc: Document):
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.82)
    sec.bottom_margin = Inches(0.78)
    sec.left_margin = Inches(0.92)
    sec.right_margin = Inches(0.92)
    sec.header_distance = Inches(0.35)
    sec.footer_distance = Inches(0.35)
    sec.different_first_page_header_footer = True
    header = sec.header.paragraphs[0]
    header.text = "Oral microbiome cross-cohort ML reanalysis"
    set_run_font(header.runs[0], size=8.5, color=MUTED)
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer = sec.footer.paragraphs[0]
    add_page_number(footer)
    set_run_font(footer.runs[0], size=8.5, color=MUTED)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_table_borders(table, color="B7C5CF", size="4"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:color"), color)
        borders.append(tag)


def add_table(doc, headers, rows, widths, font_size=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = str(h)
        shade_cell(hdr.cells[i], LIGHT_BLUE)
        p = hdr.cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            set_run_font(run, size=font_size, bold=True, color=NAVY)
    set_repeat_table_header(hdr)
    for row_values in rows:
        row = table.add_row()
        for i, value in enumerate(row_values):
            row.cells[i].text = "" if value is None else str(value)
            p = row.cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                set_run_font(run, size=font_size)
    set_table_geometry(table, widths)
    set_table_borders(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_para(doc, text, bold_prefix=None, italic=False, align=None, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        set_run_font(r, bold=True)
        r2 = p.add_run(text[len(bold_prefix):])
        set_run_font(r2, italic=italic)
    else:
        r = p.add_run(text)
        set_run_font(r, italic=italic)
    return p


def add_callout(doc, label, text, fill=LIGHT_GRAY):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.18)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:color"), BLUE)
    borders.append(left)
    p_pr.append(borders)
    r = p.add_run(f"{label} ")
    set_run_font(r, bold=True, color=NAVY)
    r2 = p.add_run(text)
    set_run_font(r2)


def add_figure(doc, path: Path, caption: str, width=6.35):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    r = p.add_run()
    r.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(style="Caption")
    cap.paragraph_format.keep_with_next = False
    cap.add_run(caption)


def fmt(x, digits=2):
    return "NA" if pd.isna(x) else f"{x:.{digits}f}"


def task_name(task):
    return {"OLP": "OLP vs healthy", "OPMD": "All OPMDs vs healthy", "OSCC": "OSCC vs healthy"}.get(task, task)


def load_outputs(out: Path):
    t = out / "tables"
    p = out / "predictions"
    return {
        "coverage": pd.read_csv(t / "Table_ML1_Clinical_metadata_coverage.csv"),
        "metrics": pd.read_csv(t / "Table_ML2_External_cohort_performance.csv"),
        "summary": pd.read_csv(t / "Table_ML3_Macro_performance_summary.csv"),
        "stability": pd.read_csv(t / "Table_ML4_Elastic_net_feature_stability.csv"),
        "study": pd.read_csv(t / "Table_ML5_Residual_study_signal.csv"),
        "permutation": pd.read_csv(t / "Table_ML7_Permutation_summary.csv"),
        "predictions": pd.read_csv(p / "All_outer_fold_predictions.csv"),
    }


def build_report(out: Path, target: Path):
    d = load_outputs(out)
    coverage, metrics, summary, stability, study, perm = (
        d["coverage"], d["metrics"], d["summary"], d["stability"], d["study"], d["permutation"]
    )
    fig_dir = out / "figures"

    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    props = doc.core_properties
    props.title = "Cross-cohort oral microbiome machine-learning reanalysis"
    props.subject = "Leakage-controlled analysis and manuscript-ready text"
    props.author = "Prepared for Dr. Saeid Ghavami"
    props.keywords = "oral microbiome, OPMD, OLP, OSCC, machine learning, leave-one-cohort-out"

    # Editorial-cover opening pattern.
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("TECHNICAL ANALYSIS REPORT")
    set_run_font(r, size=11, bold=True, color=BLUE)
    p.paragraph_format.space_after = Pt(18)
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Cross-cohort oral microbiome\nmachine-learning reanalysis")
    p = doc.add_paragraph(style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Leakage-controlled modelling, clinical-covariate audit, stable bacterial signatures, and manuscript-ready text")
    add_para(doc, "OPMD, oral lichen planus, and oral squamous cell carcinoma", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=42)
    add_para(doc, "Prepared for integration into the uploaded cross-cohort manuscript", align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    add_para(doc, "Analysis date: August 10, 2026", align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    doc.add_page_break()

    doc.add_heading("Executive conclusion", level=1)
    add_callout(
        doc,
        "Main conclusion.",
        "The OSCC microbiome signal transported across cohorts with moderate discrimination, whereas OLP was weaker and the combined OPMD task was close to chance. The results support a hypothesis-generating cross-cohort microbial signature, not a clinically deployable diagnostic model.",
    )
    add_para(
        doc,
        "Smoking was not present as a participant-level variable in any uploaded abundance matrix. Age was absent from the OSCC matrices, and sex was available only in the Lee 2017 and Zhu 2023 OSCC cohorts. OLP age and sex were available for 85 of 203 participants across only two cohorts. Therefore, clinical-only and combined models could not be estimated validly for the primary OLP or OSCC analyses.",
    )
    add_para(
        doc,
        "A restricted all-OPMD complete-case analysis was possible in three cohorts with age and sex (n = 336), but smoking remained unavailable. This sensitivity analysis is explicitly exploratory because it excludes two cohorts and mixes different OPMD phenotypes. Its results cannot establish clinical incremental value.",
    )
    add_para(
        doc,
        "The strongest result was OSCC: mean held-out-cohort AUROC ranged from approximately 0.77 to 0.82 at genus level and 0.76 to 0.83 at species level. However, balanced accuracy and calibration were less stable than ranking performance, and microbiome profiles predicted study identity with nearly perfect accuracy. Residual cohort structure therefore remains a major limitation.",
    )

    doc.add_heading("1. Data and clinical-variable audit", level=1)
    add_para(
        doc,
        "The uploaded relative-abundance matrices contained 454 OPMD/healthy samples and 509 OSCC/healthy samples. The OLP-specific task contained 203 samples from four cohorts after excluding the Lee 2017 cohort, which contributed healthy controls but no OLP cases. Genus-level matrices contained 205 OPMD and 194 OSCC taxa; species-level matrices contained 535 OPMD and 480 OSCC taxa before training-specific prevalence filtering.",
    )
    rows = []
    for task in ["OLP", "OPMD", "OSCC"]:
        g = coverage[coverage.task == task]
        n = g.n.sum(); cases = g.cases.sum(); controls = g.controls.sum()
        age = g.age_available_n.sum(); sex = g.sex_available_n.sum(); smoke = g.smoking_available_n.sum()
        eligibility = {
            "OLP": "No valid clinical/combined LOCO model",
            "OPMD": "Exploratory age+sex complete-case model",
            "OSCC": "No valid clinical/combined LOCO model",
        }[task]
        rows.append([
            task_name(task), n, f"{cases}/{controls}", f"{age} ({100*age/n:.1f}%)",
            f"{sex} ({100*sex/n:.1f}%)", f"{smoke} ({100*smoke/n:.1f}%)", eligibility
        ])
    add_table(
        doc,
        ["Task", "n", "Cases/controls", "Age", "Sex", "Smoking", "Clinical-model decision"],
        rows,
        [1740, 650, 1120, 1050, 1050, 1050, 2700],
        font_size=7.9,
    )
    add_para(doc, "Table 1. Participant-level clinical-variable availability in the uploaded modelling matrices.", italic=True, after=8)
    add_callout(
        doc,
        "Important distinction.",
        "The manuscript tables contain pooled or study-level demographic summaries, including smoking frequencies. Those summaries cannot be assigned to individual sequencing samples and were therefore not used as predictors.",
        fill=PALE_GOLD,
    )
    add_figure(
        doc,
        fig_dir / "Figure_ML1_Workflow_and_clinical_coverage.png",
        "Figure ML1. Leakage-controlled modelling workflow and participant-level availability of age, sex, and smoking. Clinical availability is shown for each prediction task because OLP uses a subset of the broader OPMD datasets.",
    )

    doc.add_heading("2. Manuscript-ready Materials and Methods", level=1)
    doc.add_heading("2.1 Prediction tasks and input data", level=2)
    add_para(
        doc,
        "Machine-learning analyses were performed as cross-cohort prediction experiments using the genus- and species-level relative-abundance matrices generated by the standardized 16S rRNA processing workflow. Three binary tasks were defined a priori: oral lichen planus (OLP) versus healthy controls, all oral potentially malignant disorders (OPMDs) versus healthy controls, and oral squamous cell carcinoma (OSCC) versus healthy controls. The OLP analysis included only cohorts containing both OLP cases and their corresponding healthy controls. Genus-level abundance was designated the primary feature representation, and species-level abundance was evaluated as a secondary analysis. Study, geographic location, specimen type, and sequencing characteristics were treated as technical metadata and were not entered as biological predictors.",
    )
    doc.add_heading("2.2 Clinical covariates", level=2)
    add_para(
        doc,
        "Age, sex, and smoking status were considered for clinical-only and combined clinical–microbiome models only when participant-level values could be linked directly to sequencing sample identifiers and were sufficiently represented across cohorts. Smoking was unavailable in all uploaded participant-level matrices; age was unavailable for OSCC; and age and sex were available for only two OLP cohorts. Accordingly, primary clinical-only and combined models were not estimated for OLP or OSCC. An exploratory complete-case analysis was performed for all OPMDs using the three cohorts with complete participant-level age and sex data. Smoking could not be included in this sensitivity analysis. Missing values were not coded as absence or nonsmoking, and entire cohorts with unavailable covariates were not imputed because this would make missingness a surrogate for study membership.",
    )
    doc.add_heading("2.3 Leakage-controlled preprocessing", level=2)
    add_para(
        doc,
        "All preprocessing was estimated separately within the training data in every validation iteration. Taxa were retained when detected at non-zero abundance in at least 10% of the corresponding training samples and showed non-zero training variance. Retained abundances were renormalized within each sample. Zeros were replaced using one-half of the smallest positive abundance observed in the training data, with a lower bound of 1 × 10⁻⁸, and the centered log-ratio (CLR) transformation was then applied. CLR features were standardized using the training mean and standard deviation before model fitting. The same selected taxa, zero-replacement value, and scaling parameters were applied unchanged to the held-out samples. Previously pooled sPLSDA-corrected objects were not used for prediction because their construction preceded cohort holdout and could transmit information from the external test cohort into model development.",
    )
    doc.add_heading("2.4 Nested leave-one-cohort-out validation", level=2)
    add_para(
        doc,
        "Generalizability was evaluated using nested leave-one-cohort-out validation. In each outer iteration, one complete study was held out as an untouched external test cohort. Model hyperparameters were selected using an inner leave-one-cohort-out loop restricted to the remaining studies. The primary classifier was elastic-net logistic regression. Linear support vector machine, random forest, and XGBoost classifiers were evaluated as secondary sensitivity models. Class weighting was applied during training. Hyperparameter selection was based on the aggregate inner out-of-fold AUROC. Probability calibration was estimated from inner cohort-held-out predictions using logistic calibration, and the classification threshold was selected from the same inner predictions by maximizing Youden’s index. Neither calibration nor threshold selection used the outer test cohort.",
    )
    doc.add_heading("2.5 Performance, robustness, and stable features", level=2)
    add_para(
        doc,
        "Performance was reported separately for every held-out cohort using AUROC, balanced accuracy, sensitivity, specificity, Brier score, calibration intercept, and calibration slope. Stratified bootstrap resampling (1,000 iterations) was used to obtain 95% confidence intervals for discrimination and threshold-dependent metrics. Macro-average performance was calculated by assigning equal weight to each held-out cohort. For the primary elastic-net models, bacterial features were summarized by selection frequency across outer folds, coefficient direction and magnitude, and direction consistency. Independent random-effects meta-analysis of cohort-specific CLR standardized mean differences was used to estimate the direction, uncertainty, and I² heterogeneity of each taxon. Model-feature directions were additionally compared with the significant differential-abundance results already reported in the manuscript.",
    )
    doc.add_heading("2.6 Permutation and residual study-signal analyses", level=2)
    add_para(
        doc,
        "For genus-level elastic-net models, disease labels were permuted within each cohort 500 times while preserving cohort-specific class counts. Hyperparameters were locked to those selected in the real nested analysis, and the full outer cohort holdout was repeated to generate a null distribution of macro-average AUROC. An empirical P value was calculated as (1 + number of permuted statistics greater than or equal to the observed statistic)/(1 + number of permutations). As a diagnostic of residual technical structure, study identity was also predicted from microbiome profiles using five-fold stratified cross-validation. This diagnostic was not interpreted as disease prediction but as an estimate of the strength of cohort-specific microbiome structure.",
    )

    doc.add_heading("3. Results", level=1)
    doc.add_heading("3.1 Cross-cohort model performance", level=2)
    oscc_g = summary[(summary.task == "OSCC") & (summary.level == "genus") & (summary.view == "microbiome")]
    oscc_s = summary[(summary.task == "OSCC") & (summary.level == "species") & (summary.view == "microbiome")]
    olp_en_g = summary[(summary.task == "OLP") & (summary.level == "genus") & (summary.model == "elastic_net")].iloc[0]
    olp_en_s = summary[(summary.task == "OLP") & (summary.level == "species") & (summary.model == "elastic_net")].iloc[0]
    opmd_best_g = summary[(summary.task == "OPMD") & (summary.level == "genus") & (summary.view == "microbiome")].sort_values("macro_auroc_mean", ascending=False).iloc[0]
    add_para(
        doc,
        f"OSCC showed the most reproducible cross-cohort discrimination. At genus level, macro-average held-out AUROC ranged from {oscc_g.macro_auroc_mean.min():.2f} to {oscc_g.macro_auroc_mean.max():.2f} across the four classifiers; species-level AUROC ranged from {oscc_s.macro_auroc_mean.min():.2f} to {oscc_s.macro_auroc_mean.max():.2f}. The prespecified genus-level elastic-net model achieved a macro-average AUROC of {oscc_g[oscc_g.model=='elastic_net'].macro_auroc_mean.iloc[0]:.2f}, while random forest achieved {oscc_g[oscc_g.model=='random_forest'].macro_auroc_mean.iloc[0]:.2f}. Species-level XGBoost reached {oscc_s[oscc_s.model=='xgboost'].macro_auroc_mean.iloc[0]:.2f}. These differences were modest and do not justify replacing the prespecified interpretable primary model solely on the basis of the highest observed AUROC.",
    )
    add_para(
        doc,
        f"OLP discrimination was weaker and model-dependent. Elastic-net AUROC was {olp_en_g.macro_auroc_mean:.2f} at genus level and {olp_en_s.macro_auroc_mean:.2f} at species level. Combining all OPMDs produced little generalizable signal: the best genus-level model was {opmd_best_g.model.replace('_',' ')} with a macro-average AUROC of {opmd_best_g.macro_auroc_mean:.2f}, whereas several models were at or below chance. This result is consistent with the biological heterogeneity created by combining distinct OPMD phenotypes.",
    )
    perf_rows = []
    for row in summary[(summary.view == "microbiome") & (summary.task.isin(["OLP", "OPMD", "OSCC"]))].itertuples():
        perf_rows.append([
            task_name(row.task), row.level.title(), row.model.replace("_", " ").title(),
            f"{row.macro_auroc_mean:.2f} ({row.auroc_min:.2f}–{row.auroc_max:.2f})",
            f"{row.macro_balanced_accuracy_mean:.2f}", f"{row.macro_brier_mean:.2f}"
        ])
    add_table(doc, ["Task", "Level", "Model", "Mean AUROC (range)", "Balanced accuracy", "Brier"], perf_rows,
              [1840, 850, 1480, 2110, 1630, 1450], font_size=7.9)
    add_para(doc, "Table 2. Macro-average performance across untouched cohorts. Cohort-specific results and bootstrap confidence intervals are supplied in the accompanying CSV table.", italic=True)
    add_figure(doc, fig_dir / "Figure_ML2_Model_comparison.png",
               "Figure ML2. Macro-average held-out-cohort AUROC for the three disease tasks, two taxonomic levels, and four prespecified classifiers.")
    add_figure(doc, fig_dir / "Figure_ML3_External_cohort_AUROC.png",
               "Figure ML3. Cohort-specific AUROC and 95% bootstrap confidence intervals for the primary genus-level elastic-net models.")
    add_figure(doc, fig_dir / "Figure_ML4_Cohort_ROC.png",
               "Figure ML4. Receiver-operating-characteristic curves for the primary genus-level elastic-net model in each untouched cohort.")

    doc.add_heading("3.2 Threshold transport and calibration", level=2)
    primary = metrics[(metrics.model == "elastic_net") & (metrics.level == "genus") & (metrics.view == "microbiome")]
    add_para(
        doc,
        f"Ranking performance did not translate uniformly into threshold-dependent classification. Mean balanced accuracy was {primary[primary.task=='OLP'].balanced_accuracy.mean():.2f} for OLP, {primary[primary.task=='OPMD'].balanced_accuracy.mean():.2f} for all OPMDs, and {primary[primary.task=='OSCC'].balanced_accuracy.mean():.2f} for OSCC. Several held-out cohorts showed a strong sensitivity–specificity imbalance, demonstrating that decision thresholds selected in the training cohorts did not always transport to a new study. Calibration intercepts and slopes also varied substantially across cohorts. Accordingly, the observed AUROC should not be interpreted as evidence that the models are ready for individual clinical risk estimation.",
    )
    add_figure(doc, fig_dir / "Figure_ML5_Calibration.png",
               "Figure ML5. Descriptive calibration of pooled outer-fold predictions from the primary genus-level elastic-net models. Pooling is shown for visualization; formal interpretation remains cohort-specific.")

    doc.add_heading("3.3 Stable bacterial features", level=2)
    add_para(
        doc,
        "The OLP model identified several recurrent coefficients, but most were selected in only three of four outer folds and many showed substantial between-study heterogeneity. Corynebacterium was negatively associated with OLP in three of four outer folds and its direction agreed with the recalculated random-effects estimate. These observations remain exploratory because the OLP classifiers showed only weak-to-moderate transport.",
    )
    add_para(
        doc,
        "The all-OPMD model produced apparently stable coefficients for several taxa, but these features were accompanied by poor external discrimination, high heterogeneity, and limited agreement with the existing manuscript meta-analysis. In particular, apparent feature stability in a weak classifier should not be interpreted as biological validation. Propionibacterium showed a small negative elastic-net coefficient consistent with the manuscript direction, whereas Lancefieldella did not show a stable non-zero coefficient or reproduce the previously reported direction in the recalculated analysis.",
    )
    add_para(
        doc,
        "The OSCC model showed clearer agreement between predictive and differential-abundance results. Rothia, Oribacterium, and Corynebacterium had recurrent negative elastic-net coefficients, while Peptoanaerobacter and Parvimonas had recurrent positive coefficients. These directions were consistent with the taxa reported as depleted or enriched, respectively, in the manuscript meta-analysis. Other highly selected features, including Sphingomonas, showed discordant pooled directions or high heterogeneity and should not be prioritized without independent validation.",
    )
    add_figure(doc, fig_dir / "Figure_ML6_Stable_genus_features.png",
               "Figure ML6. Genus-level features ranked by elastic-net coefficient stability. Red indicates a positive disease-associated coefficient and blue a negative coefficient; labels show selection frequency across outer cohort folds.",
               width=5.15)

    doc.add_heading("3.4 Exploratory clinical incremental value", level=2)
    clin = summary[summary.task == "OPMD_complete_case"].set_index("view")
    add_para(
        doc,
        f"In the restricted all-OPMD complete-case analysis, the clinical-only age/sex model had a macro-average AUROC of {clin.loc['clinical','macro_auroc_mean']:.2f}; the microbiome-only model achieved {clin.loc['microbiome','macro_auroc_mean']:.2f}; and the combined model achieved {clin.loc['combined','macro_auroc_mean']:.2f}. The apparent combined-model improvement was not consistent across cohorts and was driven largely by one small cohort. Smoking was unavailable, two OPMD cohorts were excluded, and the three retained cohorts contained different disease compositions. This analysis therefore does not demonstrate incremental clinical utility.",
    )
    add_figure(doc, fig_dir / "Figure_ML7_Clinical_incremental_value.png",
               "Figure ML7. Exploratory complete-case comparison of age/sex, microbiome, and combined elastic-net models in three all-OPMD cohorts. Smoking was unavailable.")

    doc.add_heading("3.5 Permutation evidence and residual cohort structure", level=2)
    per_text = []
    for task in ["OLP", "OPMD", "OSCC"]:
        row = perm[perm.task == task].iloc[0]
        per_text.append(f"{task_name(task)}: observed macro-AUROC {row.observed_macro_auroc:.2f}, empirical P = {row.empirical_p_value:.3f}")
    add_para(doc, "Cohort-stratified locked-pipeline permutation testing produced the following results: " + "; ".join(per_text) + ".")
    study_g = study[study.level == "genus"]
    add_para(
        doc,
        f"Microbiome profiles classified study identity with balanced accuracy between {study_g.study_classification_balanced_accuracy.min():.2f} and {study_g.study_classification_balanced_accuracy.max():.2f} at genus level, compared with chance levels of 0.20–0.25. This near-perfect study signal confirms that standardized processing did not eliminate cohort-specific biological and technical structure. The LOCO estimates are therefore more informative than random sample-level cross-validation, but they should still be validated in additional independent populations.",
    )
    add_figure(doc, fig_dir / "Figure_ML8_Permutation_tests.png",
               "Figure ML8. Cohort-stratified locked-pipeline permutation distributions for the primary genus-level elastic-net models. The vertical line indicates observed macro-average AUROC.")

    doc.add_heading("4. Manuscript-ready Discussion", level=1)
    add_para(
        doc,
        "The machine-learning analysis provided an independent assessment of whether microbial abundance patterns identified by the cross-cohort meta-analysis could discriminate disease in an entirely unseen study. The results were strongly dependent on the disease definition. OSCC showed the most reproducible signal, with moderate held-out-cohort AUROC across linear and nonlinear classifiers at both genus and species levels. In contrast, OLP performance was weaker and varied substantially among cohorts, while combining all OPMDs produced models that were generally close to chance. These findings indicate that microbial alterations associated with established OSCC are more reproducible across the included datasets than those associated with heterogeneous potentially malignant disorders.",
    )
    add_para(
        doc,
        "The stronger OSCC discrimination should nevertheless be interpreted cautiously. Balanced accuracy, sensitivity, specificity, and calibration were not uniformly stable across held-out cohorts, indicating that probability scales and decision thresholds did not transport as reliably as rank ordering. Furthermore, study identity was recoverable from the microbiome profiles with nearly perfect accuracy. This confirms the persistence of major cohort-specific structure related to population, specimen type, amplified 16S region, laboratory procedures, and other unmeasured variables. Leave-one-cohort-out validation reduces the optimistic bias that would result from randomly mixing participants from the same studies across training and test sets, but it cannot by itself establish transport to populations that differ from all included cohorts.",
    )
    add_para(
        doc,
        "Feature stability provided a useful bridge between prediction and the differential-abundance meta-analysis. Several OSCC features showed concordant directions across the two analytical strategies, including depletion-associated coefficients for Rothia, Oribacterium, and Corynebacterium and enrichment-associated coefficients for Peptoanaerobacter and Parvimonas. These taxa are therefore reasonable candidates for targeted validation. However, coefficient stability alone is not evidence of causal involvement, and features showing substantial heterogeneity or disagreement with pooled abundance estimates may reflect correlated microbial communities, compositional substitution, or residual study effects.",
    )
    add_para(
        doc,
        "Clinical adjustment remained incomplete. Smoking was not available at participant level in the uploaded matrices, age was absent from the OSCC data, and OLP age and sex were restricted to two cohorts. Assigning pooled demographic values to individuals or coding unavailable smoking information as nonsmoking would introduce measurement error and potentially allow the model to identify study membership. The restricted all-OPMD complete-case analysis suggested that microbiome features contained information beyond age and sex in some cohorts, but the apparent gain was not consistent and cannot be generalized. Recovery of participant-level age, sex, smoking, alcohol exposure, betel use, periodontal status, and other relevant covariates—linked directly to sequencing sample identifiers—will be required to determine whether microbial profiles provide incremental information beyond established risk factors.",
    )
    add_para(
        doc,
        "Taken together, the analysis supports a hypothesis-generating OSCC-associated microbial signature but does not support clinical classification of all OPMDs as one group. Future work should prioritize subtype-specific OPMD models, additional external OSCC cohorts, prospective validation, and harmonized collection of clinical and behavioral covariates. Domain-adaptation approaches such as DEBIAS-M may be evaluated as secondary sensitivity analyses, provided that held-out disease labels remain inaccessible during model development and that any use of unlabeled target-cohort features is disclosed explicitly.",
    )

    doc.add_heading("5. Recommended integration into the paper", level=1)
    add_para(doc, "Insert the six Methods subsections after the current differential-abundance analysis and before the Results. Add the Results subsections after the existing differential-abundance meta-analysis. The Discussion text should follow the current biological interpretation of the differentially abundant taxa and precede the general limitations paragraph.")
    add_para(doc, "Recommended main-text figures: Figure ML2 as the overall model comparison; Figure ML3 for external cohort transport; and Figure ML6 for stable genus-level signatures. Recommended supplementary figures: ML1, ML4, ML5, ML7, and ML8. The complete cohort-level performance table, prediction file, and taxon-stability table should be supplied as supplementary data rather than compressed into the main manuscript.")
    add_callout(
        doc,
        "Claims to avoid.",
        "Do not describe the models as diagnostic, clinically validated, generalizable to all populations, independent of smoking, or capable of predicting malignant transformation. The analysis is cross-sectional and predicts current case status in the included datasets.",
        fill=PALE_RED,
    )

    doc.add_heading("6. Reproducibility and supplied files", level=1)
    add_para(doc, "The accompanying package contains the full Python analysis script, pinned software requirements, run instructions, sample-level outer-fold predictions, all cohort-specific performance estimates, stable-feature tables, random-effects taxon analyses, high-resolution PNG figures, vector PDF figures, and a machine-readable run manifest. The fixed random seed was 20260810.")
    add_para(doc, "The uploaded RData objects were inspected to confirm their contents but were not used for prediction. This was a deliberate leakage-control decision, not an omission. Reproducing the predictive analysis from the original relative-abundance tables ensures that feature filtering and transformations are recalculated independently inside each training split.")

    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target)


if __name__ == "__main__":
    a = args()
    build_report(a.output_dir, a.docx)
