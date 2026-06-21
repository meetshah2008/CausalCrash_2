import os

html_file = r'C:\Users\MEET\Desktop\IIIT_D\Thesis\CausalCrash_2\index.html'

benchmark_html = """
        <!-- Benchmark Results -->
        <section class="mb-16">
            <h2 class="text-3xl font-bold mb-6 text-gray-800 border-b pb-2">Benchmark Results (7 State-of-the-Art VLMs)</h2>
            <p class="text-lg text-gray-700 mb-6 glass-panel p-6 rounded-xl">
                We evaluated seven leading Vision-Language Models in a zero-shot setting across the 273 CausalCrash videos. The results expose a significant gap in temporal and causal reasoning capabilities.
            </p>
            
            <div class="glass-panel rounded-xl overflow-hidden shadow-lg mb-8 overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-gray-800 text-white">
                            <th class="py-4 px-6 font-bold text-sm uppercase tracking-wide">Model</th>
                            <th class="py-4 px-6 font-bold text-sm uppercase tracking-wide text-center">L1 Perception</th>
                            <th class="py-4 px-6 font-bold text-sm uppercase tracking-wide text-center">L2 Temporal</th>
                            <th class="py-4 px-6 font-bold text-sm uppercase tracking-wide text-center">L4 Preventive</th>
                            <th class="py-4 px-6 font-bold text-sm uppercase tracking-wide text-center">L5 Counterfactual</th>
                            <th class="py-4 px-6 font-bold text-sm uppercase tracking-wide text-center bg-blue-900 text-blue-100">CausalCrash Score (CCS)</th>
                        </tr>
                    </thead>
                    <tbody class="text-gray-700">
                        <tr class="border-b border-gray-200 hover:bg-gray-50">
                            <td class="py-4 px-6 font-semibold">Qwen3.5-plus</td>
                            <td class="py-4 px-6 text-center">5.04</td>
                            <td class="py-4 px-6 text-center">4.51</td>
                            <td class="py-4 px-6 text-center">5.67</td>
                            <td class="py-4 px-6 text-center">5.93</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-50">5.28</td>
                        </tr>
                        <tr class="border-b border-gray-200 hover:bg-gray-50 bg-gray-50">
                            <td class="py-4 px-6 font-semibold">MiMo-VL-7B-RL</td>
                            <td class="py-4 px-6 text-center">5.00</td>
                            <td class="py-4 px-6 text-center">4.81</td>
                            <td class="py-4 px-6 text-center">5.42</td>
                            <td class="py-4 px-6 text-center">5.63</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-100">5.22</td>
                        </tr>
                        <tr class="border-b border-gray-200 hover:bg-gray-50">
                            <td class="py-4 px-6 font-semibold">Qwen3.5-VL-27B</td>
                            <td class="py-4 px-6 text-center">4.06</td>
                            <td class="py-4 px-6 text-center">3.73</td>
                            <td class="py-4 px-6 text-center">4.61</td>
                            <td class="py-4 px-6 text-center">4.77</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-50">4.29</td>
                        </tr>
                        <tr class="border-b border-gray-200 hover:bg-gray-50 bg-gray-50">
                            <td class="py-4 px-6 font-semibold">Gemini 2.5 Flash</td>
                            <td class="py-4 px-6 text-center">3.99</td>
                            <td class="py-4 px-6 text-center">3.42</td>
                            <td class="py-4 px-6 text-center">4.37</td>
                            <td class="py-4 px-6 text-center">4.59</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-100">4.09</td>
                        </tr>
                        <tr class="border-b border-gray-200 hover:bg-gray-50">
                            <td class="py-4 px-6 font-semibold">GLM-4.5V</td>
                            <td class="py-4 px-6 text-center">3.18</td>
                            <td class="py-4 px-6 text-center">2.69</td>
                            <td class="py-4 px-6 text-center">3.85</td>
                            <td class="py-4 px-6 text-center">4.03</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-50">3.44</td>
                        </tr>
                        <tr class="border-b border-gray-200 hover:bg-gray-50 bg-gray-50">
                            <td class="py-4 px-6 font-semibold">InternVL3.5-14B</td>
                            <td class="py-4 px-6 text-center">3.01</td>
                            <td class="py-4 px-6 text-center">2.31</td>
                            <td class="py-4 px-6 text-center">3.65</td>
                            <td class="py-4 px-6 text-center">3.72</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-100">3.17</td>
                        </tr>
                        <tr class="border-b border-gray-200 hover:bg-gray-50">
                            <td class="py-4 px-6 font-semibold">InternVL3-14B</td>
                            <td class="py-4 px-6 text-center">2.63</td>
                            <td class="py-4 px-6 text-center">2.22</td>
                            <td class="py-4 px-6 text-center">3.00</td>
                            <td class="py-4 px-6 text-center">3.08</td>
                            <td class="py-4 px-6 text-center font-bold text-blue-800 bg-blue-50">2.73</td>
                        </tr>
                    </tbody>
                </table>
                <div class="bg-gray-100 p-4 text-xs text-gray-500 italic text-center">
                    * Scores are on a 0-10 scale. Evaluated via the Dual LLM-as-a-Judge protocol using GPT-4.1-mini and Gemini 2.5 Flash.
                </div>
            </div>

            <!-- Key Takeaways -->
            <div class="glass-panel p-6 rounded-xl border-l-4 border-red-500 bg-red-50/50">
                <h4 class="text-xl font-bold text-red-900 mb-2">Key Conclusions</h4>
                <ul class="list-disc pl-5 text-gray-800 space-y-2">
                    <li><strong>Significant Performance Degradation:</strong> We observed a consistent drop in scores from perception (L1) down to temporal reasoning (L2) across all models. Current VLMs struggle heavily with precise spatial-temporal dynamics.</li>
                    <li><strong>The CausalCrash Challenge:</strong> Even the top-scoring frontier model (Qwen3.5-plus) achieves only a <strong>5.28 / 10</strong> on the composite CausalCrash Score. This proves that deep counterfactual reasoning in crash scenarios remains largely unsolved.</li>
                    <li><strong>Judge Leniency Bias:</strong> Throughout the evaluation, GPT-4.1-mini exhibited a persistent leniency bias compared to Gemini 2.5 Flash, which underscores the absolute necessity of our Dual-Judge Evaluation Framework.</li>
                </ul>
            </div>
        </section>
"""

with open(html_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Only insert if not already present
if '<!-- Benchmark Results -->' not in content:
    # Insert right before the Citation section
    parts = content.split('<!-- Citation -->')
    if len(parts) == 2:
        new_content = parts[0] + benchmark_html + '\n        <!-- Citation -->' + parts[1]
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Successfully added Benchmark Results to index.html")
    else:
        print("Could not find Citation section to anchor the insertion.")
else:
    print("Benchmark Results already present in index.html.")
