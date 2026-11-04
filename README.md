# Evaluating AI Debate as Oversight by a Weaker Judge Using Open-Source LLMs

Using only open-weight models on free hardware, we tested whether letting two AI "experts" debate helps a weaker AI judge pick the right answer to reading-comprehension questions it can't verify on its own. It didn't. Debate scored 57.5%, below the judge working alone (65%). A single expert allowed to choose and defend its own answer scored 82.5%. Debate corrected 0 of 14 of the judge's initial mistakes, while open consultancy corrected 9.

Judge accuracy by condition

Why this matters

As AI systems get more capable, the people (or models) overseeing them will increasingly be unable to check answers directly. Debate is one proposed fix for this scalable oversight problem: two experts argue opposite sides, and a weaker judge decides which argument holds up. Prior work (Khan et al., 2024; Kenton et al., 2024) found promising results under certain conditions. We ran a small, fully reproducible version using open-source models (Qwen2.5 7B and 1.5B) on free hardware to see whether a simple prompted debate protocol helps a weak judge. Because the weights are open, anyone can rerun the full experiment at no cost, with no paid APIs.

Research question

Does adversarial debate improve a weak judge's reading-comprehension accuracy compared with unassisted judging and consultancy, under a matched expert-call budget?

Setup
	
Expert	Qwen2.5-7B-Instruct (sees the full passage)
Judge	Qwen2.5-1.5B-Instruct (sees only the question, two answer choices, and the arguments)
Data	40 questions from the QuALITY dev split, one per article, each reduced to the gold answer plus one random distractor
Pilot	10 training-split questions used to revise prompts before freezing the protocol
Models	Open-weight, Apache 2.0 licensed, pinned to exact Hugging Face commit hashes
Hardware	Free Google Colab Tesla T4, 4-bit NF4 quantization, no fine-tuning
Cost	$0 (no paid APIs or closed models)
Conditions

Every question was run through all four conditions:

Unassisted: the judge answers with no help.
Assigned consultancy: one expert argues for a preassigned answer (correct on 20 questions, incorrect on 20), with an opening plus three self-revisions.
Open consultancy: one expert chooses its own answer and can revise it across four messages.
Debate: two isolated expert instances argue opposite answers, each giving an opening and then a rebuttal. Speaking order is balanced.

All assisted conditions used four expert calls, with an 80-word instruction and a 256-token cap per call. The judge saw each transcript in a fresh context and had to answer only "A" or "B".

Results
Condition	Correct	Accuracy	95% bootstrap CI
Unassisted	26/40	65.0%	50.0–80.0%
Assigned consultancy	27/40	67.5%	52.5–82.5%
Open consultancy	33/40	82.5%	70.0–92.5%
Debate	23/40	57.5%	42.5–72.5%

Paired comparisons (debate minus baseline):

Comparison	Difference	95% CI
Debate − Assigned consultancy (primary)	−10.0 pts	−22.5 to 0
Debate − Unassisted	−7.5 pts	−17.5 to 0
Debate − Open consultancy	−25.0 pts	−42.5 to −7.5

Answer changes relative to the unassisted judge:

Condition	Fixed (of 14 initially wrong)	Broke (of 26 initially correct)
Assigned consultancy	3	2
Open consultancy	9	2
Debate	0	3

Intervals use 10,000 paired bootstrap resamples over questions (seed 42). There were no malformed judge outputs, and only 1 of 160 debate messages hit the token cap, so truncation doesn't explain the result.

What went wrong with debate?

We inspected all three cases where debate flipped a correct judge answer to an incorrect one. In each, the advocate's argument didn't match its assigned label:

On a ranking question, an advocate defended a "most liked to least liked" ordering while describing scores going from low to high.
On a question asking which location did not have a Maglev car, an advocate admitted its assigned location had one, yet kept the same answer label.

In other words, the 7B experts often failed to argue the side they were assigned. That leaves the weak judge with contradictory transcripts it can't resolve. Open consultancy avoids this because the expert argues for what it actually believes.

Takeaways
For this model pair and task, an expert choosing and refining its own answer gave the judge more useful information than forced adversarial advocacy.
This does not show that debate fails in general. It shows that assigning opposing roles is not enough when advocates contradict themselves or drift from their positions.
Likely next steps: systematically measure role adherence across all transcripts, add quote verification (a key ingredient in Khan et al.), and test whether the judge is swayed by surface cues such as speaking order or argument length.
Limitations
Small sample: 40 questions, one model pair, one generation seed. Bootstrap intervals capture variation across questions, not across models or reruns.
Protocol comparison, not a mechanism test: debate requires one advocate to defend a wrong answer, while open consultancy does not, so we compare whole protocols rather than isolating the adversarial part.
Unequal realized output: call counts and token caps were matched, but debate produced 11,146 expert tokens vs. 7,629 for open consultancy.
Quotes were not verified for the judge.
Length-filtered sample: only 42 dev articles were 6,000 tokens or shorter, so the sample skews toward shorter passages.
Role adherence was inspected qualitatively, not measured across every response.
Shared model-family biases, possible benchmark contamination, and an artificial information gap limit generalization to real human oversight.
Repository structure
.
├── README.md
├── CS1998_AI_Debate_Final_Report.pdf   # full write-up
├── AI_Debate.ipynb                      # complete pilot + main pipeline (Colab)
├── figures/
│   └── accuracy.png                     # Figure 1
└── results/                             # main_v2 outputs
    ├── results.csv                      # per-question, per-condition judge answers
    ├── summary.csv                      # accuracy + bootstrap CIs
    ├── paired_comparisons.csv
    ├── reversals.csv
    ├── usage.csv                        # tokens and GPU time per call
    ├── transcripts.json                 # all expert arguments
    └── review_cases.json                # randomly sampled disagreements
Reproducing the results

The notebook is built to be reproducible and resumable on a free Colab GPU.

Open AI_Debate.ipynb in Google Colab and select a T4 GPU runtime.
Run cell 0A (installs pinned packages), then Runtime → Restart session, then run 0B. Don't continue until 0B prints PASS.
Run with MODE = 'pilot' first. Review the transcripts, then run the freeze cell (Section 7).
Set MODE = 'main' and run Sections 1–6. Every generation is checkpointed to Google Drive, so interrupted sessions resume where they stopped.

Reproducibility safeguards built into the pipeline:

Model revisions are pinned by commit hash, and package versions are recorded and checked on resume.
The QuALITY archive is hashed, and question sampling is seeded and never depends on model outputs.
Each generation call gets a deterministic seed (SHA-256(str(42) + call key)).
Prompts are hashed, so resuming with changed prompts raises an error instead of silently reusing stale outputs.
Main settings are checked against the frozen pilot configuration before the main run.

Environment: Python 3.13, torch 2.11.0+cu130, transformers 4.51.3, accelerate 1.6.0, bitsandbytes 0.50.2, huggingface_hub 0.30.2. Runtime: about 59 minutes of generation for the main run (480 expert calls, 160 judge calls), excluding model downloads and loading.

Contributions
Griffins Lelgut: [fill in]
Addis Sahle: [fill in]
AI use

We used ChatGPT to assist with experimental design, code development, troubleshooting, analysis, and drafting. We chose the research question, ran the experiments, and checked the reported results against the saved outputs.

References
Khan, A., et al. (2024). Debating with More Persuasive LLMs Leads to More Truthful Answers. ICML. arXiv:2402.06782
Kenton, Z., et al. (2024). On scalable oversight with weak LLMs judging strong LLMs. arXiv:2407.04622
Pang, R. Y., et al. (2022). QuALITY: Question Answering with Long Input Texts, Yes! NAACL. arXiv:2112.08608
Qwen Team. Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct model cards.
