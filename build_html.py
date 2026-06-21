import json
import os
import re

# File Paths
html_file = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\CausalCrash_2\index.html'

gt_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Final_2\ZZ_294_ZZ_4fvWtH-C3Cs.json'
gemini_inf_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Chaitanya_Inferences\gemini-2.5-flash\ZZ_294_ZZ_4fvWtH-C3Cs.json'
intern_inf_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Chaitanya_Inferences\InternVL3-14B\ZZ_294_ZZ_4fvWtH-C3Cs.json'
gemini_eval_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Results_Gemini-2.5-flash\gemini-2.5-flash\ZZ_294_ZZ_4fvWtH-C3Cs_eval.json'
intern_eval_path = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\Dataset_Curations\Results_Gemini-2.5-flash\InternVL3_5-14B\ZZ_294_ZZ_4fvWtH-C3Cs_eval.json'

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.dumps(json.load(f), indent=2)

def syntax_highlight(json_str):
    json_str = json_str.replace('<', '&lt;').replace('>', '&gt;')
    return json_str

gt_json = syntax_highlight(load_json(gt_path))
gemini_inf_json = syntax_highlight(load_json(gemini_inf_path))
intern_inf_json = syntax_highlight(load_json(intern_inf_path))
gemini_eval_json = syntax_highlight(load_json(gemini_eval_path))
intern_eval_json = syntax_highlight(load_json(intern_eval_path))

# --- HTML BLOCKS ---

js_script = """
<script>
function showTab(group, tabId) {
    const tabs = document.querySelectorAll(`.${group}-tab`);
    const contents = document.querySelectorAll(`.${group}-content`);
    tabs.forEach(t => {
        t.classList.remove('bg-gray-700', 'text-white');
        t.classList.add('bg-gray-900', 'text-gray-400');
    });
    contents.forEach(c => c.classList.add('hidden'));
    
    document.getElementById(`${group}-tab-${tabId}`).classList.remove('bg-gray-900', 'text-gray-400');
    document.getElementById(`${group}-tab-${tabId}`).classList.add('bg-gray-700', 'text-white');
    document.getElementById(`${group}-content-${tabId}`).classList.remove('hidden');
}
</script>
"""

metrics_html = """
        <!-- Human Baseline Section -->
        <section class="mb-16">
            <h2 class="text-3xl font-bold mb-6 text-gray-800 border-b pb-2">Human Baseline & Inter-Annotator Agreement</h2>
            <p class="text-lg text-gray-700 mb-6 glass-panel p-6 rounded-xl">
                To determine the structural objectivity of the CausalCrash annotation scheme, 44 videos were double-annotated independently by two human experts. The validation results demonstrate highly robust agreement across all reasoning dimensions:
            </p>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
                <div class="glass-panel p-6 rounded-xl border-l-4 border-blue-500">
                    <div class="text-3xl font-extrabold text-blue-600 mb-2">0.882</div>
                    <div class="text-sm uppercase tracking-wide font-semibold text-gray-600">Cohen's &kappa; (Vehicle Type)</div>
                    <div class="text-xs text-gray-500 mt-2">Near-perfect categorization of initiating agents.</div>
                </div>
                <div class="glass-panel p-6 rounded-xl border-l-4 border-green-500">
                    <div class="text-3xl font-extrabold text-green-600 mb-2">100%</div>
                    <div class="text-sm uppercase tracking-wide font-semibold text-gray-600">Contextual Hazard Overlap</div>
                    <div class="text-xs text-gray-500 mt-2">Experts agreed on the physical hazard in every situation.</div>
                </div>
                <div class="glass-panel p-6 rounded-xl border-l-4 border-purple-500">
                    <div class="text-3xl font-extrabold text-purple-600 mb-2">1.29s</div>
                    <div class="text-sm uppercase tracking-wide font-semibold text-gray-600">Temporal CP MAE</div>
                    <div class="text-xs text-gray-500 mt-2">Critical Point Mean Absolute Error variation.</div>
                </div>
            </div>
        </section>
"""

case_study_html = f"""
        <!-- Case Study -->
        <section class="mb-16">
            <h2 class="text-3xl font-bold mb-6 text-gray-800 border-b pb-2">Full Example: Multi-Vehicle Complex Crash</h2>
            
            <div class="glass-panel p-6 rounded-xl mb-8">
                <p class="text-gray-700 mb-4 text-lg">
                    This example features an overly speedy cement mixer truck entering an intersection, leading to a severe rollover and crushing a silver van. 
                    Below are the full raw JSON predictions from the models alongside the Human Ground Truth, followed by the rigorous LLM-as-a-judge evaluation results.
                </p>
                <div class="rounded-xl overflow-hidden shadow-lg bg-black flex justify-center mb-6">
                    <video controls muted autoplay loop class="max-h-[500px] w-full object-contain">
                        <source src="assets/example.mp4" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>

            <!-- Inferences Tabbed -->
            <h3 class="text-2xl font-bold mb-4 text-gray-800">1. Full Model Inferences</h3>
            <div class="bg-gray-800 rounded-xl overflow-hidden shadow-lg flex flex-col mb-12">
                <div class="flex border-b border-gray-700 bg-gray-900">
                    <button id="inf-tab-gt" onclick="showTab('inf', 'gt')" class="inf-tab flex-1 py-3 px-4 font-bold text-sm bg-gray-700 text-white transition">✅ Ground Truth</button>
                    <button id="inf-tab-gem" onclick="showTab('inf', 'gem')" class="inf-tab flex-1 py-3 px-4 font-bold text-sm text-gray-400 hover:text-white transition">🤖 Gemini 2.5 Flash</button>
                    <button id="inf-tab-int" onclick="showTab('inf', 'int')" class="inf-tab flex-1 py-3 px-4 font-bold text-sm text-gray-400 hover:text-white transition">🤖 InternVL3-14B</button>
                </div>
                
                <div id="inf-content-gt" class="inf-content bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{gt_json}</pre>
                </div>
                
                <div id="inf-content-gem" class="inf-content hidden bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{gemini_inf_json}</pre>
                </div>

                <div id="inf-content-int" class="inf-content hidden bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{intern_inf_json}</pre>
                </div>
            </div>

            <!-- Judge Evals Tabbed -->
            <h3 class="text-2xl font-bold mb-4 text-gray-800">2. LLM Judge Evaluation Reports</h3>
            <p class="text-gray-600 mb-4 italic">The judge script strictly evaluates the predicted JSONs against the Ground Truth across Level 1, 2, 4, and 5.</p>
            <div class="bg-gray-800 rounded-xl overflow-hidden shadow-lg flex flex-col mb-10">
                <div class="flex border-b border-gray-700 bg-gray-900">
                    <button id="eval-tab-gem" onclick="showTab('eval', 'gem')" class="eval-tab flex-1 py-3 px-4 font-bold text-sm bg-gray-700 text-white transition">⚖️ Gemini Evaluation</button>
                    <button id="eval-tab-int" onclick="showTab('eval', 'int')" class="eval-tab flex-1 py-3 px-4 font-bold text-sm text-gray-400 hover:text-white transition">⚖️ InternVL Evaluation</button>
                </div>
                
                <div id="eval-content-gem" class="eval-content bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{gemini_eval_json}</pre>
                </div>

                <div id="eval-content-int" class="eval-content hidden bg-[#1e1e1e] p-4 overflow-auto max-h-[600px]">
                    <pre class="m-0 text-gray-300 text-xs leading-relaxed font-mono whitespace-pre-wrap">{intern_eval_json}</pre>
                </div>
            </div>
            
        </section>
"""

html_content = open(html_file, 'r', encoding='utf-8').read()

# Remove the old case study
if '<!-- Case Study -->' in html_content:
    parts = html_content.split('<!-- Case Study -->')
    after = parts[1].split('<!-- Evaluation Pipeline -->')
    html_content = parts[0] + '<!-- Evaluation Pipeline -->' + after[1]

# Insert JS script in head
if '</head>' in html_content and 'function showTab' not in html_content:
    html_content = html_content.replace('</head>', js_script + '\n</head>')

# Insert metrics and case study before Evaluation Pipeline
html_content = html_content.replace('<!-- Evaluation Pipeline -->', metrics_html + case_study_html + '\n        <!-- Evaluation Pipeline -->')

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print('Successfully injected fully detailed JSONs and metrics into index.html')
