import json
import os
import re

html_file = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\CausalCrash_2\index.html'

gt_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Final_2\ZZ_294_ZZ_4fvWtH-C3Cs.json'
intern_inf_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Chaitanya_Inferences\InternVL3_5-14B\ZZ_294_ZZ_4fvWtH-C3Cs.json'

if not os.path.exists(intern_inf_path):
    intern_inf_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Chaitanya_Inferences\InternVL3-14B\ZZ_294_ZZ_4fvWtH-C3Cs.json'

gpt_eval_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Results_gpt-4.1-mini\InternVL3_5-14B\ZZ_294_ZZ_4fvWtH-C3Cs_eval.json'
gemini_eval_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Results_Gemini-2.5-flash\InternVL3_5-14B\ZZ_294_ZZ_4fvWtH-C3Cs_eval.json'

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.dumps(json.load(f), indent=2)

def syntax_highlight(json_str):
    json_str = json_str.replace('<', '&lt;').replace('>', '&gt;')
    return json_str

gt_json = syntax_highlight(load_json(gt_path))
intern_inf_json = syntax_highlight(load_json(intern_inf_path))
gpt_eval_json = syntax_highlight(load_json(gpt_eval_path))
gemini_eval_json = syntax_highlight(load_json(gemini_eval_path))

# The HTML string for the entire Case Study block
new_case_study = f"""        <!-- Case Study -->
        <section class="mb-16">
            <h2 class="text-3xl font-bold mb-6 text-gray-800 border-b pb-2">Full Example: The Dual-Judge Framework</h2>
            
            <div class="glass-panel p-6 rounded-xl mb-8">
                <p class="text-gray-700 mb-4 text-lg">
                    This example features an overly speedy cement mixer truck entering an intersection, leading to a severe rollover and crushing a silver van. 
                    Below, we demonstrate the full evaluation pipeline on a single model's prediction <strong>(InternVL3.5-14B)</strong>, evaluated by two independent LLM Judges.
                </p>
                <div class="rounded-xl overflow-hidden shadow-lg bg-black flex justify-center mb-6">
                    <video controls muted autoplay loop class="max-h-[500px] w-full object-contain">
                        <source src="assets/example.mp4" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>

            <!-- 1. Prediction Tabbed -->
            <h3 class="text-2xl font-bold mb-4 text-gray-800">1. Ground Truth vs Prediction</h3>
            <p class="text-gray-600 mb-4 italic">Comparing the human expert annotation against InternVL3.5-14B's raw prediction.</p>
            <div class="bg-gray-800 rounded-xl overflow-hidden shadow-lg flex flex-col mb-12 border border-gray-700">
                <div class="flex border-b border-gray-700 bg-gray-900">
                    <button id="inf-tab-gt" onclick="showTab('inf', 'gt')" class="inf-tab flex-1 py-3 px-4 font-bold text-sm bg-gray-700 text-white transition">✅ Ground Truth</button>
                    <button id="inf-tab-int" onclick="showTab('inf', 'int')" class="inf-tab flex-1 py-3 px-4 font-bold text-sm text-gray-400 hover:text-white transition">🤖 InternVL3.5-14B Prediction</button>
                </div>
                
                <div id="inf-content-gt" class="inf-content bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{gt_json}</pre>
                </div>
                
                <div id="inf-content-int" class="inf-content hidden bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{intern_inf_json}</pre>
                </div>
            </div>

            <!-- 2. Dual Judge Tabbed -->
            <h3 class="text-2xl font-bold mb-4 text-gray-800">2. Dual LLM-as-a-Judge Evaluation</h3>
            <p class="text-gray-600 mb-4 italic">How GPT-4.1-mini and Gemini-2.5-Flash independently scored the InternVL3.5-14B prediction.</p>
            <div class="bg-gray-800 rounded-xl overflow-hidden shadow-lg flex flex-col mb-8 border border-gray-700">
                <div class="flex border-b border-gray-700 bg-gray-900">
                    <button id="eval-tab-gpt" onclick="showTab('eval', 'gpt')" class="eval-tab flex-1 py-3 px-4 font-bold text-sm bg-gray-700 text-white transition">⚖️ Evaluated by GPT-4.1-mini</button>
                    <button id="eval-tab-gem" onclick="showTab('eval', 'gem')" class="eval-tab flex-1 py-3 px-4 font-bold text-sm text-gray-400 hover:text-white transition">⚖️ Evaluated by Gemini 2.5 Flash</button>
                </div>
                
                <div id="eval-content-gpt" class="eval-content bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{gpt_eval_json}</pre>
                </div>

                <div id="eval-content-gem" class="eval-content hidden bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{gemini_eval_json}</pre>
                </div>
            </div>
            
            <!-- 3. Conclusions -->
            <div class="glass-panel p-6 rounded-xl border-l-4 border-indigo-500 bg-indigo-50/50">
                <h4 class="text-xl font-bold text-indigo-900 mb-2">Conclusions & Insights</h4>
                <p class="text-gray-800 leading-relaxed mb-3">
                    The dual-judge evaluation exposes crucial blind spots in current state-of-the-art vision models and emphasizes the necessity of using multiple independent judges:
                </p>
                <ul class="list-disc pl-5 text-gray-700 space-y-2">
                    <li><strong>Temporal Failure:</strong> InternVL completely missed the critical timing sequence (predicting 7.5s instead of 3.0s) and falsely categorized the collision as a generic crash rather than a <em>rollover</em>.</li>
                    <li><strong>Inter-Judge Disagreement:</strong> While <strong>GPT-4.1-mini</strong> gave a somewhat lenient Level 1 score (7) and marked <code>physical_logic_valid: true</code>, <strong>Gemini 2.5 Flash</strong> was much stricter, assigning a Level 1 score of 5 and flagging the physical logic as <code>false</code> due to the fundamental misunderstanding of the rollover dynamics.</li>
                    <li><strong>The Value of CausalCrash:</strong> This discrepancy highlights why dual-judge protocols are essential to prevent automated leniency bias when evaluating complex counterfactual reasoning.</li>
                </ul>
            </div>
        </section>"""

html_content = open(html_file, 'r', encoding='utf-8').read()

# Replace the existing Case Study section entirely
parts = html_content.split('<!-- Case Study -->')
after = parts[1].split('<!-- Evaluation Pipeline -->')
html_content = parts[0] + new_case_study + '\n\n        <!-- Evaluation Pipeline -->' + after[1]

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print('Updated build_html.py execution successful')
