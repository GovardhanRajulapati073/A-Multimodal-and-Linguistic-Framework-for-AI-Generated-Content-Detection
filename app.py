"""
Standalone Interactive Application
Title: A Multimodal and Linguistic Framework for AI-Generated Content Detection
Design: Academic / Enterprise Research Dashboard (Zero emojis, strict professional typography)
"""

import os
import sys
import math
import re
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from linguistic_engine import LinguisticAnalyzer, PerplexityBurstinessEstimator
from multimodal_engine import GatedMultimodalFusionNet, MultimodalInferencePipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing Framework on {DEVICE}...")
ling_analyzer = LinguisticAnalyzer()
ppl_engine = PerplexityBurstinessEstimator(use_neural=False)
pipeline = MultimodalInferencePipeline(device=DEVICE)

# Benchmark sample strings
s_ai = "Artificial intelligence is a transformative branch of computer science dedicated to developing computational systems capable of performing cognitive tasks. These tasks encompass natural language processing, visual object recognition, automated decision-making, and translation between human languages. In recent years, deep learning algorithms and transformer architectures have achieved unprecedented performance across diverse empirical benchmarks, fundamentally revolutionizing the technological and industrial landscape."
s_quillbot = "Undeniably, artificial intelligence stands as a revolutionary facet of computation focused on crafting mechanisms proficient in executing intellectual operations. These operations incorporate textual interpretation, optical recognition, and structured deduction. Recent evolutions within deep learning frameworks have established unrivaled outcomes over assorted testing baselines."
s_stealth = "It is generally acknowledged that global warming represents an acute ecological crisis requiring prompt multilateral coordination. Amplified greenhouse gas accumulations across the troposphere continue to instigate unpredictable meteorological anomalies. As a result, municipal authorities and legislative entities must proactively deploy comprehensive carbon offset strategies."
s_academic = "In this investigation, we evaluated the statistical significance of regional demographic variance across five hundred surveyed Midwestern households. The empirical results demonstrate a pronounced divergence in post-2020 economic mobility. Contrary to prevailing hypotheses, urban industrial clusters exhibited lower household savings rates than rural agricultural communities."
s_human = "I've been thinking a lot lately about how much technology has changed since I was a kid. Back then, we didn't have smartphones or high-speed internet in our pockets all day long. If you wanted to talk to a friend, you actually called their home landline and hoped their parents didn't answer first. Honestly, while it's amazing that we can look up anything in two seconds now, I kind of miss the quiet times when people weren't constantly staring at little glass screens during dinner."

def create_radar_chart(ling_feats, ppl_res):
    labels = ['Lexical Diversity (TTR)', 'Burstiness (CV)', 'Word Entropy', 'Syntactic Uniformity', 'Reading Ease (Norm)']
    
    input_vals = [
        min(1.0, ling_feats.get('lexical_diversity_ttr', 0.5)),
        min(1.0, ppl_res.get('burstiness_score', 0.2) * 2.0),
        min(1.0, ling_feats.get('shannon_entropy', 5.0) / 7.0),
        min(1.0, ling_feats.get('syntactic_uniformity', 0.5)),
        min(1.0, ling_feats.get('flesch_reading_ease', 50.0) / 100.0)
    ]
    
    human_baseline = [0.85, 0.75, 0.82, 0.25, 0.65]
    llm_baseline   = [0.60, 0.25, 0.62, 0.85, 0.45]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    input_vals += input_vals[:1]
    human_baseline += human_baseline[:1]
    llm_baseline += llm_baseline[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    ax.plot(angles, input_vals, color='#2563eb', linewidth=2.0, label='Target Text')
    ax.fill(angles, input_vals, color='#2563eb', alpha=0.18)
    
    ax.plot(angles, human_baseline, color='#16a34a', linestyle='--', linewidth=1.4, label='Human Baseline')
    ax.plot(angles, llm_baseline, color='#dc2626', linestyle='--', linewidth=1.4, label='LLM Baseline')
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, fontweight='600', color='#334155')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7.5)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def analyze_text_tab(input_text, domain_choice):
    try:
        if not input_text or not input_text.strip():
            error_msg = "<div style='color:#dc2626; font-size:13px; padding:12px; border:1px solid #fecaca; background:#fef2f2; border-radius:6px;'>Error: Input text area cannot be empty.</div>"
            return error_msg, "", None, ""

        res = pipeline.predict(input_text, image=None, ling_analyzer=ling_analyzer, ppl_estimator=ppl_engine, domain_mode=domain_choice)
        feats = res.get("features_extracted", {})
        ppl = res.get("linguistic_summary", {})
        h_rep = res.get("humanizer_report", {})

        v_code = res.get("verdict_code", "human")
        v_title = res.get("verdict", "HUMAN-AUTHORED CONTENT")
        tool_tag = res.get("tool_tag", "Natural Human Authorship")
        p_pure = res.get("prob_pure_ai", 0.0)
        p_hum_ai = res.get("prob_humanized_ai", 0.0)
        p_hum = res.get("prob_human", 100.0)

        # Verdict theme colors
        if v_code == "humanized_ai":
            badge_color = "#d97706"
            bg_color = "#fffbeb"
            border_color = "#f59e0b"
            header_tag = "ADVERSARIAL BYPASS IDENTIFIED"
        elif v_code == "pure_ai":
            badge_color = "#dc2626"
            bg_color = "#fef2f2"
            border_color = "#f87171"
            header_tag = "DIRECT GENERATIVE SYNTHESIS"
        else:
            badge_color = "#16a34a"
            bg_color = "#f0fdf4"
            border_color = "#4ade80"
            header_tag = "VERIFIED AUTHENTIC AUTHORSHIP"

        # Humanizer Diagnostics
        humanizer_html = ""
        if h_rep.get("is_humanized", False):
            m_hits = h_rep.get("marker_hits", [])
            m_str = ", ".join(m_hits[:8]) if m_hits else "Synonym substitution pattern"
            h_score = h_rep.get("humanizer_score", 0.0)
            humanizer_html = (
                f"<div style='background:#fffdf5; border:1px solid #f59e0b; border-left:4px solid #d97706; border-radius:6px; padding:12px; margin-top:12px; font-size:12.5px; color:#78350f; text-align:left;'>"
                f"<div style='font-weight:700; text-transform:uppercase; font-size:11px; letter-spacing:0.05em; color:#b45309;'>"
                f"Adversarial Humanizer Forensics ({tool_tag}) | Evasion Confidence: {h_score}%</div>"
                f"<div style='margin-top:5px; line-height:1.5;'>Lexical diversity is artificially inflated while sentence burstiness remains mechanically depressed. Identified characteristic paraphrasing markers.</div>"
                f"<div style='margin-top:6px; font-size:11.5px; color:#92400e;'>"
                f"<b>Identified Evasion Markers ({len(m_hits)}):</b> <code>{m_str}</code></div>"
                f"</div>"
            )

        verdict_html = (
            f"<div style='background:{bg_color}; border:2px solid {border_color}; border-radius:8px; padding:16px; margin-bottom:14px; text-align:center;'>"
            f"<div style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; color:{badge_color};'>{header_tag}</div>"
            f"<div style='font-size:20px; font-weight:800; color:{badge_color}; margin:4px 0 6px 0;'>{v_title}</div>"
            f"<div style='font-size:12.5px; color:#475569;'><b>Signature Profile:</b> {tool_tag} &nbsp;|&nbsp; <b>Domain Mode:</b> {domain_choice}</div>"
            f"<div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:14px;'>"
            f"<div style='background:#fef2f2; border:1px solid #fecaca; border-radius:6px; padding:10px; text-align:center;'>"
            f"<div style='font-size:10.5px; font-weight:700; color:#b91c1c;'>PURE AI (RAW LLM)</div>"
            f"<div style='font-size:19px; font-weight:800; color:#dc2626; margin:3px 0;'>{p_pure}%</div>"
            f"<div style='background:#fee2e2; border-radius:4px; height:5px; overflow:hidden;'><div style='background:#dc2626; width:{p_pure}%; height:100%;'></div></div>"
            f"</div>"
            f"<div style='background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:10px; text-align:center;'>"
            f"<div style='font-size:10.5px; font-weight:700; color:#92400e;'>HUMANIZED AI (BYPASS)</div>"
            f"<div style='font-size:19px; font-weight:800; color:#d97706; margin:3px 0;'>{p_hum_ai}%</div>"
            f"<div style='background:#fef3c7; border-radius:4px; height:5px; overflow:hidden;'><div style='background:#d97706; width:{p_hum_ai}%; height:100%;'></div></div>"
            f"</div>"
            f"<div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:10px; text-align:center;'>"
            f"<div style='font-size:10.5px; font-weight:700; color:#15803d;'>AUTHENTIC HUMAN</div>"
            f"<div style='font-size:19px; font-weight:800; color:#16a34a; margin:3px 0;'>{p_hum}%</div>"
            f"<div style='background:#dcfce7; border-radius:4px; height:5px; overflow:hidden;'><div style='background:#16a34a; width:{p_hum}%; height:100%;'></div></div>"
            f"</div>"
            f"</div>"
            f"{humanizer_html}"
            f"</div>"
        )

        heatmap_html = "<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:14px;'><div style='font-size:11px; font-weight:700; text-transform:uppercase; color:#475569; margin-bottom:8px;'>Sentence Conditional Perplexity Heatmap (XAI):</div><div style='line-height:1.8; font-size:13px;'>"
        for s in ppl.get("sentence_details", []):
            bg = s.get("bg_color", "#f8fafc")
            col = s.get("color_code", "#64748b")
            idx = s.get("sentence_index", 1)
            ppl_val = s.get("perplexity", 40.0)
            asmt = s.get("assessment", "Normal")
            txt = s.get("text", "")
            heatmap_html += f"<span style='background:{bg}; border-bottom:2px solid {col}; padding:3px 6px; margin:2px; border-radius:4px; display:inline-block;' title='Sentence {idx}: PPL={ppl_val} ({asmt})'>{txt} </span> "
        heatmap_html += "</div></div>"

        ov_ppl = ppl.get("overall_perplexity", 45.0)
        b_score = ppl.get("burstiness_score", 0.25)
        ttr_val = feats.get("lexical_diversity_ttr", 0.65)
        uni_val = feats.get("syntactic_uniformity", 0.30)
        fre_val = feats.get("flesch_reading_ease", 50.0)
        exp_sum = ppl.get("explanation_summary", "Analysis completed.")

        ppl_imp = "Constrained token likelihood (machine signature)" if ov_ppl < 30 else "High entropy vocabulary variation (human signature)"
        bst_imp = "Monotonous sentence rhythm (machine signature)" if b_score < 0.25 else "Dynamic rhythmic variance (human signature)"
        ttr_imp = "Elevated (Potential paraphrase synonym injection)" if ttr_val > 0.80 else "Balanced vocabulary distribution"

        metrics_html = (
            "<table style='width:100%; border-collapse:collapse; background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; font-size:12.5px; margin-top:8px;'>"
            + "<tr style='background:#f8fafc; border-bottom:1px solid #e2e8f0; text-align:left;'>"
            + "<th style='padding:9px 12px; color:#475569;'>Indicator</th><th style='padding:9px 12px; color:#475569;'>Value</th><th style='padding:9px 12px; color:#475569;'>Reference Norm</th><th style='padding:9px 12px; color:#475569;'>Implication</th></tr>"
            + f"<tr><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9; font-weight:600;'>Mean Perplexity (PPL)</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{ov_ppl}</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>Human: 45-80 | AI: 12-25</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{ppl_imp}</td></tr>"
            + f"<tr><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9; font-weight:600;'>Burstiness Index (CV)</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{b_score}</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>Human: &gt; 0.35 | AI: &lt; 0.22</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{bst_imp}</td></tr>"
            + f"<tr><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9; font-weight:600;'>Lexical Diversity (TTR)</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{ttr_val:.3f}</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>Standard: 0.55 - 0.75</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{ttr_imp}</td></tr>"
            + f"<tr><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9; font-weight:600;'>Syntactic Uniformity</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>{uni_val:.3f}</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>Standard: 0.10 - 0.45</td><td style='padding:8px 12px; border-bottom:1px solid #f1f5f9;'>Clause length consistency</td></tr>"
            + f"<tr><td style='padding:8px 12px; font-weight:600;'>Flesch Reading Ease</td><td style='padding:8px 12px;'>{fre_val:.1f} / 100</td><td style='padding:8px 12px;'>Academic: 30-50 | Casual: 60-80</td><td style='padding:8px 12px;'>Structural syllable complexity</td></tr>"
            + "</table>"
            + f"<div style='background:#f8fafc; border:1px solid #cbd5e1; border-radius:6px; padding:10px; margin-top:10px; font-size:12px; color:#334155;'><b>Diagnostic Summary:</b> {exp_sum}</div>"
        )

        radar_img = create_radar_chart(feats, ppl)
        return verdict_html, heatmap_html, radar_img, metrics_html

    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        err_box = f"<div style='color:#dc2626; padding:14px; border:1px solid #fca5a5; background:#fef2f2; border-radius:6px;'><b>Execution Exception:</b><pre style='font-size:11px; margin-top:6px;'>{err_trace}</pre></div>"
        return err_box, "<p style='color:#64748b; font-size:12px;'>Heatmap generation halted.</p>", None, "<p style='color:#64748b; font-size:12px;'>Metrics generation halted.</p>"

def analyze_multimodal_tab(text, image):
    try:
        text_clean = (text or "").strip()
        if not text_clean and image is None:
            return "<div style='color:#dc2626; font-size:13px; padding:12px; border:1px solid #fecaca; background:#fef2f2; border-radius:6px;'>Please provide text, upload an image, or both for multimodal analysis.</div>"

        res = pipeline.predict(text=text_clean or "Visual forensic evaluation", image=image, ling_analyzer=ling_analyzer, ppl_estimator=ppl_engine)
        if "error" in res:
            return f"<div style='color:#dc2626; font-size:13px; padding:12px; border:1px solid #fecaca; background:#fef2f2; border-radius:6px;'>{res['error']}</div>"
        
        v_code = res.get("verdict_code", "human")
        color = "#dc2626" if v_code == "pure_ai" else ("#d97706" if v_code == "humanized_ai" else "#16a34a")
        bg = "#fef2f2" if v_code == "pure_ai" else ("#fffbeb" if v_code == "humanized_ai" else "#f0fdf4")
        
        overall_ai = res.get("overall_ai_probability", 0.0)
        t_prob = f"{res.get('text_ai_probability', 0.0)}%" if text_clean else "N/A (No text provided)"
        i_prob = f"{res.get('image_ai_probability', 0.0)}%" if image is not None else "N/A (No image provided)"
        align = f"{res.get('cross_modal_alignment', 0.0)}" if (text_clean and image is not None) else "N/A (Requires paired text and image)"

        html = f"""
        <div style='background:{bg}; border:2px solid {color}; border-radius:8px; padding:16px; text-align:center;'>
            <div style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; color:{color};'>Multimodal Authenticity Verdict</div>
            <div style='font-size:20px; font-weight:700; color:{color}; margin:4px 0;'>{res["verdict"].upper()}</div>
            <p style='font-size:14px; margin:4px 0 0 0; color:#334155;'><b>Overall Multimodal AI Probability:</b> {overall_ai}%</p>
        </div>
        <div style='margin-top:14px; display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
            <div style='background:#ffffff; border:1px solid #e2e8f0; padding:14px; border-radius:6px;'>
                <div style='font-size:11px; color:#64748b; font-weight:600; text-transform:uppercase;'>Text Modality Probability</div>
                <div style='font-size:20px; font-weight:700; color:#2563eb; margin-top:4px;'>{t_prob}</div>
            </div>
            <div style='background:#ffffff; border:1px solid #e2e8f0; padding:14px; border-radius:6px;'>
                <div style='font-size:11px; color:#64748b; font-weight:600; text-transform:uppercase;'>Visual Forensic FFT Artifacts</div>
                <div style='font-size:20px; font-weight:700; color:#7c3aed; margin-top:4px;'>{i_prob}</div>
            </div>
        </div>
        <div style='background:#ffffff; border:1px solid #e2e8f0; padding:14px; border-radius:6px; margin-top:12px;'>
            <div style='font-size:12px; font-weight:600; color:#0f172a;'>Cross-Modal Cosine Semantic Alignment: {align}</div>
            <div style='font-size:11px; color:#64748b; margin-top:2px;'>Evaluates contrastive cosine alignment between image spatial representations and textual caption semantics. Low alignment indicates synthetic generative hallucination.</div>
        </div>
        """
        return html
    except Exception as e:
        import traceback
        return f"<div style='color:#dc2626; padding:12px;'>Execution Error: {traceback.format_exc()}</div>"

app_css = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background-color: #f8fafc !important;
}
.main-title-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    padding: 24px 28px !important;
    border-radius: 10px !important;
    margin-bottom: 22px !important;
    border-left: 6px solid #3b82f6 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15) !important;
}
.main-title-text {
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.3 !important;
    margin: 0 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}
"""

def build_app():
    import gradio as gr
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=app_css, title="A Multimodal and Linguistic Framework for AI-Generated Content Detection") as demo:
        gr.HTML("""
        <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px 28px; border-radius: 10px; margin-bottom: 22px; border-left: 6px solid #3b82f6; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);'>
            <div style='color: #ffffff !important; font-size: 24px !important; font-weight: 800 !important; letter-spacing: -0.02em; line-height: 1.3; margin: 0;'>
                A Multimodal and Linguistic Framework for AI-Generated Content Detection
            </div>
        </div>
        """)

        with gr.Tabs():
            with gr.TabItem("Text Forensics Studio"):
                with gr.Row():
                    with gr.Column(scale=5):
                        domain_selector = gr.Radio(
                            choices=["Standard Calibration", "Academic Publication Mode (False-Positive Control)"],
                            value="Standard Calibration",
                            label="Domain Sensitivity Calibration",
                            info="Academic Mode prevents false positives on formal research and scientific publications."
                        )
                        text_input = gr.Textbox(
                            label="Input Text for Forensic Analysis",
                            placeholder="Paste article, student paper, or document excerpt here...",
                            lines=7
                        )
                        gr.Markdown("**Benchmark Preloaded Samples (Click to test):**")
                        with gr.Row():
                            btn_sample_ai = gr.Button("Pure AI (ChatGPT-4)", variant="secondary", size="sm")
                            btn_sample_quillbot = gr.Button("Humanized AI (QuillBot)", variant="secondary", size="sm")
                            btn_sample_stealth = gr.Button("Humanized AI (StealthGPT)", variant="secondary", size="sm")
                        with gr.Row():
                            btn_sample_academic = gr.Button("Formal Academic (Human)", variant="secondary", size="sm")
                            btn_sample_human = gr.Button("Essay Writing (Human)", variant="secondary", size="sm")
                        with gr.Row():
                            btn_clear = gr.Button("Clear Text", size="sm")
                            btn_analyze = gr.Button("Run Forensic Analysis", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        verdict_output = gr.HTML(label="Verdict")
                        radar_output = gr.Image(type="pil", label="Stylometric Vector Profile")

                with gr.Row():
                    with gr.Column(scale=6):
                        heatmap_output = gr.HTML(label="Sentence Perplexity Heatmap")
                    with gr.Column(scale=4):
                        metrics_output = gr.HTML(label="Linguistic Indicators Table")

                s_ai = "Artificial intelligence is a transformative branch of computer science dedicated to developing computational systems capable of performing cognitive tasks. These tasks encompass natural language processing, visual object recognition, automated decision-making, and translation between human languages. In recent years, deep learning algorithms and transformer architectures have achieved unprecedented performance across diverse empirical benchmarks, fundamentally revolutionizing the technological and industrial landscape."
                s_quillbot = "Undeniably, artificial intelligence stands as a revolutionary facet of computation focused on crafting mechanisms proficient in executing intellectual operations. These operations incorporate textual interpretation, optical recognition, and structured deduction. Recent evolutions within deep learning frameworks have established unrivaled outcomes over assorted testing baselines."
                s_stealth = "It is generally acknowledged that global warming represents an acute ecological crisis requiring prompt multilateral coordination. Amplified greenhouse gas accumulations across the troposphere continue to instigate unpredictable meteorological anomalies. As a result, municipal authorities and legislative entities must proactively deploy comprehensive carbon offset strategies."
                s_academic = "In this investigation, we evaluated the statistical significance of regional demographic variance across five hundred surveyed Midwestern households. The empirical results demonstrate a pronounced divergence in post-2020 economic mobility. Contrary to prevailing hypotheses, urban industrial clusters exhibited lower household savings rates than rural agricultural communities."
                s_human = "I've been thinking a lot lately about how much technology has changed since I was a kid. Back then, we didn't have smartphones or high-speed internet in our pockets all day long. If you wanted to talk to a friend, you actually called their home landline and hoped their parents didn't answer first. Honestly, while it's amazing that we can look up anything in two seconds now, I kind of miss the quiet times when people weren't constantly staring at little glass screens during dinner."

                def load_and_analyze(text_val, domain_val):
                    v_h, hm_h, r_i, m_h = analyze_text_tab(text_val, domain_val)
                    return text_val, domain_val, v_h, hm_h, r_i, m_h

                def clear_all():
                    return "", "Standard Calibration", "", "", None, ""

                btn_sample_ai.click(
                    fn=lambda: load_and_analyze(s_ai, "Standard Calibration"),
                    outputs=[text_input, domain_selector, verdict_output, heatmap_output, radar_output, metrics_output]
                )
                btn_sample_quillbot.click(
                    fn=lambda: load_and_analyze(s_quillbot, "Standard Calibration"),
                    outputs=[text_input, domain_selector, verdict_output, heatmap_output, radar_output, metrics_output]
                )
                btn_sample_stealth.click(
                    fn=lambda: load_and_analyze(s_stealth, "Standard Calibration"),
                    outputs=[text_input, domain_selector, verdict_output, heatmap_output, radar_output, metrics_output]
                )
                btn_sample_academic.click(
                    fn=lambda: load_and_analyze(s_academic, "Academic Publication Mode (False-Positive Control)"),
                    outputs=[text_input, domain_selector, verdict_output, heatmap_output, radar_output, metrics_output]
                )
                btn_sample_human.click(
                    fn=lambda: load_and_analyze(s_human, "Standard Calibration"),
                    outputs=[text_input, domain_selector, verdict_output, heatmap_output, radar_output, metrics_output]
                )
                btn_clear.click(
                    fn=clear_all,
                    outputs=[text_input, domain_selector, verdict_output, heatmap_output, radar_output, metrics_output]
                )

                btn_analyze.click(
                    analyze_text_tab,
                    inputs=[text_input, domain_selector],
                    outputs=[verdict_output, heatmap_output, radar_output, metrics_output]
                )

            with gr.TabItem("Multimodal Forensics (Image & Text)"):
                gr.Markdown("Inspect paired media content to detect generative visual artifacts and cross-modal semantic inconsistency.")
                with gr.Row():
                    with gr.Column(scale=5):
                        mm_image_input = gr.Image(type="pil", label="Associated Image (Visual Forensic Input)")
                        mm_text_input = gr.Textbox(label="Accompanying Caption / Text Article", lines=4)
                        btn_mm_analyze = gr.Button("Execute Multimodal Forensic Analysis", variant="primary")
                    with gr.Column(scale=5):
                        mm_output = gr.HTML(label="Multimodal Results")

                btn_mm_analyze.click(
                    analyze_multimodal_tab,
                    inputs=[mm_text_input, mm_image_input],
                    outputs=[mm_output]
                )

            with gr.TabItem("Humanizers & False-Positive Analysis"):
                gr.Markdown("""
                ### Scientific Investigation: Adversarial Humanizers and Plagiarism False Positives

                #### 1. Why Humanized Text Appears & How GMLF-Net Exposes It
                * **Mechanism of Humanizers (QuillBot, StealthGPT):** These tools substitute high-frequency words with rare, high-syllable synonyms and insert punctuation jitter to artificially inflate scalar perplexity ($PPL$).
                * **Why Baseline Detectors Fail:** Simple detectors assume high perplexity indicates human composition, producing false negatives.
                * **Our Countermeasure:** Humanizers create an abnormal mathematical divergence: **Elevated Lexical Diversity ($TTR > 0.80$) paired with a depressed Burstiness Index ($CV < 0.22$)**. Dense contextual transformer embeddings remain invariant under surface synonym substitutions.

                #### 2. Why AI Is Falsely Detected in Plagiarism Checks & How to Control It
                * **Scientific Cause:** Peer-reviewed academic writing is concise, formal, and utilizes standardized domain terminology (*"empirical methodology"*, *"statistically significant"*). Because these phrases are predictable in academic corpora, naive perplexity detectors misclassify them as AI.
                * **Non-Native Writer Bias:** ESL students naturally employ cleaner, less idiosyncratic phrasing, which simple detectors penalize.
                * **Control Mechanism:** GMLF-Net implements **Domain-Adaptive Baseline Thresholding** (calibrating the baseline from $PPL < 22$ to $PPL < 16$ in academic mode) and mandates multi-feature corroboration across burstiness and syntactic uniformity before classifying content as synthetic.
                """)

            with gr.TabItem("Mathematical Formulations"):
                gr.Markdown("""
                ### Mathematical Formulation of Forensic Indicators

                | Metric | Mathematical Expression | Forensic Role |
                | :--- | :--- | :--- |
                | **Perplexity ($PPL$)** | $$PPL(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i \mid w_{<i})\right)$$ | Quantifies token predictability under causal autoregressive modeling. |
                | **Burstiness ($CV$)** | $$CV = \frac{\sigma_{PPL}}{\mu_{PPL}}$$ | Measures sentence-level variance. Low $CV$ indicates mechanical cadence. |
                | **Type-Token Ratio ($TTR$)** | $$TTR = \frac{|V|}{N}, \quad RTTR = \frac{|V|}{\sqrt{N}}$$ | Evaluates vocabulary richness across token length. |
                | **Cross-Modal Alignment** | $$S_{align} = \frac{\mathbf{z}_{text} \cdot \mathbf{z}_{vis}}{\|\mathbf{z}_{text}\|_2 \|\mathbf{z}_{vis}\|_2}$$ | Computes cosine alignment between visual and textual semantics. |
                | **Gated Fusion** | $$\mathbf{h}_{fused} = [\mathbf{g} \odot \mathbf{h}_v \,\|\, (1-\mathbf{g}) \odot \mathbf{h}_t \,\|\, \mathbf{h}_{ling} \,\|\, S_{align}]$$ | Dynamically weights visual and linguistic evidence based on modality confidence. |
                """)
    return demo

if __name__ == "__main__":
    try:
        import gradio as gr
        demo = build_app()
        demo.launch(share=False)
    except ImportError:
        print("Gradio is not installed in the local environment. Install via: pip install gradio")
