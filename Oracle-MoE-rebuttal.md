Review 1(2)
Thank you for your professional review comments. Figures mentioned in the reply can be accessed at: https://anonymous.4open.science/r/ICML2025REBUTTAL-E158/README.md
Q1
__Q1__ It would be better to measure the temporal inconsistencies for the whole dataset and different layers. It would be more convincing to provide a metric to quantitatively measure the temporal inconsistency and show how Oracle-MOE reduces this inconsistency.
__A1__ We propose temporal activation inconsistency, defined as the average number of inconsistent expert activation per 100 consecutive tokens per expert. Results over the entire dataset and across different models and layers are listed below. Existing MoEs show strong temporal activation inconsistency within all layers, while Oracle-MoE reduces this.
||DeepSeek-16B|Qwen-14B|Switch(Our pretrained)|Oracle(Our pretrained)|
|---|---|---|---|---|
|1st quarter of layers avg|80.84|81.56|69.20|6.03|
|2nd quarter of layers avg|65.35|71.04|64.87|4.82|
|3rd quarter of layers avg|70.68|75.37|53.36|4.20|
|4th quarter of layers avg|76.61|77.16|75.44|5.11|

DeepSeek-16B
Qwen-14B
Switch(Our pretrained)
Oracle(Our pretrained)
1st quarter of layers avg
80.84
81.56
69.20
6.03
2nd quarter of layers avg
65.35
71.04
64.87
4.82
3rd quarter of layers avg
70.68
75.37
53.36
4.20
4th quarter of layers avg
76.61
77.16
75.44
5.11
Q2
__Q2__ It would be better to provide some preliminary experiments to show the evidence of semantic locality in real datasets, .. across different models/layers/samples.
__A2__ Experiments with DeepSeekMoE-16B and Qwen1.5-MoE-A2.7B on real chat datasets(Wizard-of-Wikipedia and Synthetic-Persona-Chat) are shown in Figure 1-5 in the link. Clear semantic locality is observed in all these models and datasets across different layers, indicating the potential of Oracle-MoE being a general-purpose solution.
Q3
__Q3__ Can the authors explain more about why the mapping of Q/K/attention score will group consecutive tokens with similar semantics? Why does this happen for different layers/samples?
__A3__ Previous studies [A][B] on representation space have shown that semantically similar samples exhibit higher embedding similarity than semantically dissimilar ones, which is also widely validated in experiments with general-purpose large-scale models. We corroborate this observation and further identify a more fine-grained pattern: token representations encapsulate high-level and token identity semantics. Among tokens with the same identity, embeddings of those that share the same high-level semantics tend to be more similar. This pattern is consistently observed in models including DeepSeek-16B-2.8B and Qwen1.5-MoE-A2.7B, as illustrated in Figure 2 in our paper and Figure 1 in the link.
Theoretical insights are also well-supported by [C][D]. Computing attention scores involves first assessing token correlations through inner products of Q and K vectors, followed by normalization via softmax, and finally allocating contextual information through V weighted by the normalized scores. Among these, the Q-K inner product effectively captures token similarity and reflects high-level semantic alignment, as visualized in Figure 2,4 in the link.
[A] A Survey on Word Embeddings: From Shallow Models to Deep Learning" (Goldberg, 2017)
[B] Deep Learning for NLP and Speech Recognition" (Hinton et al., 2012)
[C]Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
[D]Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
Q4
__Q4__ There are some approximations in the derivation of oracle-MOE. It would be better to validate such approximation in the experimental design.
__A4__ Experiments validating this approximation are listed below: for token pairs with distance between [x,y), we count on average how many inconsistent expert activation is triggered. There is a statistically positive correlation that the higher the distance between tokens, the more likely they activate different experts. More results are in Figure 6 in the link.
|EmbeddingDistance|[0.0,22)|[22,33)|[33,44)|[44,55)|[55,66)|[66,77)|[77,88)|[88,+∞)|
|---|---|---|---|---|---|---|---|---|
|Inconsistentactivationpertoken|0.0|0.083|0.222|0.377|0.576|0.770|0.878|0.903|
Embedding Distance
[0.0, 22)
[22, 33)
[33, 44)
[44,55)
[55,66)
[66,77)
[77, 88)
[88,+∞)
Inconsistent activation per token
0.0
0.083
0.222
0.377
0.576
0.770
0.878
0.903
Q5
__Q5__ Ablation study on CSD and hyperparameter study of γ are missing. How does γ influence the proposed algorithm?
__A5__ CSD and task performance lower-bound γ are key metrics we defined for the optimization problem of minimizing the model's CSD while maintaining its task performance above γ. We found that the expert number can influence γ, which is listed in the table below. On our training data, 8 experts seem to be the optimal configuration; however, when the number of experts increases, the performance of our method does not degrade compared to a standard MoE, demonstrating the robustness to the hyperparameters. Additionally, the fact that  using 8 experts is optimal in these experiments is also related to the dataset we used. We believe that as the model scale becomes larger and the dataset more diverse, the optimal number of experts will increase, and our model will still maintain robustness to hyperparameters.
|Expert num|4|8|16|24|
|--|--|--|--|--|
|$$\Delta \gamma$$|+0.02|+0.60|+0.49|+0.36|
Expert num
4
8
16
24
$$\Delta\gamma$$
+0.02
+0.60
+0.49
+0.36
Review 2 (4)
Thank you for your professional review comments. Figures mentioned in the reply can be accessed at: https://anonymous.4open.science/r/ICML2025REBUTTAL-E158/README.md
Q1
__Q1__ The paper could benefit from further discussions on practical deployment considerations and real-world constraints, such as more diverse hardware scenarios beyond the NVIDIA Jetson platform, like A100s and H100s.
__A1__ Results on A100s are listed below. Our method still outperforms existing MoE by 50%~350%.
Results on A100s are listed below. Our method still outperforms existing MoE by 50%~350%.
|Model size|Memory budget|Switch(FIFO)|Switch(LRU)|Switch(SwapMoE)|Ours(FIFO)|Full Memory|
|---|---|---|---|---|---|---|
|9*24(2.06B)|1GB|83.798s|84.766s|87.776s|18.472s|14.290s|
||2GB|66.536s|80.163s|72.094s|16.369s||
||4GB|57.810s|62.255s|50.055s|15.813s||
||7GB|33.190s|41.038s|30.146s|15.160s||
Model size
Memory budget
Switch(FIFO)
Switch(LRU)
Switch(SwapMoE)
Ours(FIFO)
Full Memory
9*24(2.06B)
1GB
83.798s
84.766s
87.776s
18.472s
14.290s

2GB
66.536s
80.163s
72.094s
16.369s


4GB
57.810s
62.255s
50.055s
15.813s


7GB
33.190s
41.038s
30.146s
15.160s

Q2
__Q2__ The concept of predicting expert activations in deeper layers based on shallow layers is promising, but the rationale and robustness of the 85%-95% prediction accuracy are left for future exploration. More details on this approach would strengthen the work significantly.
__A2__ We predict by training a linear classifier with representations in shallow layers as input and routing results in deeplayers as target labels. It turns out that such a linear classifier can reach an accuracy of 85%~95% in our structure and 40%~60% in existing MoE structures. More fine-grained observations show that layers that are closer predict each other well, e.g., representations in layer 10 predict routing results in layer 12 better than that of layer 5, and layers that predict each other well show a grouped pattern. We have been studying this phenomenon and believe this can be attributed to residual connections maintaining semantics across layers to some extent.
|AvgPredictionAccfrom|Qwen|Switch(Ourtrained)|Oracle(Ourtrained)|
|---|---|---|---|
|Layer0|48.23|50.98|89.66|
|Halfofthemodellayers|55.59|59.72|92.82|
|3/4ofthemodellayers|63.27|68.68|95.51|
Avg Prediction Acc from
Qwen
Switch(Our trained)
Oracle(Our trained)
Layer 0
48.23
50.98
89.66
Half of the model layers
55.59
59.72
92.82
3/4 of the model layers
63.27
68.68
95.51
Q3 
__Q3__ While semantic locality is effectively leveraged, the paper does not deeply investigate scenarios where semantic locality is minimal (highly diverse or abrupt topic changes), potentially limiting generalizability.
__A3__ We tested scenarios where topic changes frequently. We randomly sample sentences from different datasets and combine them into a whole sequence. We observed that our proposed oracle space can still distinguish semantic groups efficiently, both in our models and public large MOE models (DeepSeek-16B-2.8B\Qwen1.5-MoE-A2.7B),as shown in Figure 7,8,9,10 in the link. We also tested the expert activation variation of such highly diverse data with Oracle-MoE and switch-transformer. On average, in every 100 consecutive token generation, Oracle-MoE only changes 12.20 times while switch transformer changes 90.54 times. This is because that in human narual language, it takes at least dozens of tokens to express a complete meaning, so that our method still benefits from such "abrupt" semantic locality.
Q4 (= Review3 Q1)
__Q4__ The paper did not discuss cases of using fine-grained experts ( e.g., number of experts > 128, like the ones in DeepSeek's model).
__A4__ Considering limited training resources and time, we train a model following the setting of DeepSeekMoE-16B but with fewer layers: 12 MoE layers with 64 experts each, where each expert contains about 17.20% parameters as a normal expert(FFN) does, and the top 6 experts are selected for each token. Such setting still achieved xx% latency reduction while maintaining performance.
Review 3 (3)
Thank you for your professional review comments. Figures mentioned in the reply can be accessed at: https://anonymous.4open.science/r/ICML2025REBUTTAL-E158/README.md
Q1 
__Q1__ The main concern I have with this paper is the issues of scalability. In the paper, the largest MoE models are 2B with 32 experts. However, the existing models like Mistral-8x7B [1] and DeepSeekMoE-16B [2] are much larger. It would be better if the authors could provide results on these models to ensure the scale-up and scale-out ability of the proposed method.
__A1__ Considering limited training resources and time, we train a model following the setting of DeepSeekMoE-16B but with fewer layers: 12 MoE layers with 64 experts each, where each expert contains about 17.20% parameters as a normal expert(FFN) does, and the top 6 experts are selected for each token. Such setting still achieved xx% latency reduction while maintaining performance.
We've also explored how well oracle space and semantic locality work in widely used MOE models(DeepSeekMoE-16B\Qwen1.5-MoE-A2.7B). Experimental results reveal that semantic group embedding changes smoothly in the oracle space. Therefore, we believe that our method can yield low latency on a larger scale.
Q2
__Q2__ How would the number of sampled data in oracle space initialization impact the accuracy performance? Would this cause a potentially high overhead for larger models?
__A2__ As for the sampling overhead, the largest batch size we sampled in our experiments was 16382, and it only took 20 minutes for our 8*GTX3090 platform to process the sampling. Compared with 32.94 hours training time, the sampling and oracle space construction contributes to only 1% of the overall wall-clock time, which is neglectible. At the inference stage, our method introduces no inference overhead, thus there is no need to worry about larger models.
Q3 
__Q3__ In terms of figure location, I suggest putting all figures on the top of the page instead of in between the texts (see Figure 2/3/4).
__A3__ Thank you for your suggestions. We will make such modifications to the layout of these pictures in the updated versions.
Review 4 (3)
Thank you for your professional review comments. Figures mentioned in the reply can be accessed at: https://anonymous.4open.science/r/ICML2025REBUTTAL-E158/README.md
Q1
__Q1__ Semantic locality: While the paper claims that tokens with higher mutual attention scores share similar high-level semantics, I am not convinced there is strong evidence of this.
__A1__ Previous studies [A][B] on representation space analysis have shown that semantically similar samples exhibit higher similarity in their embeddings compared to semantically dissimilar ones, which is also widely validated in experiments with general-purpose large-scale models We corroborate this observation and further identify a more fine-grained similarity pattern: token representations encapsulate both high-level semantics and token identity semantics. Among tokens with the same identity, the embeddings of those that share the same high-level semantic meaning tend to be more similar. This pattern is consistently observed in various models, including widely used large models like DeepSeek-16B-2.8B and Qwen1.5-MoE-A2.7B, which are illustrated in Figure 2 in our paper and Figure 1 in the link.
Theoretical insights of how attention mechanisms compute correlations between tokens using the inner product of query (Q) and key (K) vectors are also well-supported by existing studies[C][D].The computation of attention scores involves first assessing token correlations through inner products of query (Q) and key (K) vectors, followed by normalization of these correlations via softmax, and finally allocating contextual information through value (V) vectors weighted by the normalized scores. Among which, the Q-K inner product effectively captures token similarity and reflects high-level semantic alignment, as visualized in Figure 2,4 in the link.
[A] A Survey on Word Embeddings: From Shallow Models to Deep Learning" (Goldberg, 2017)
[B] Deep Learning for NLP and Speech Recognition" (Hinton et al., 2012)
[C]Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
[D]Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2020)
Q2
__Q2__ Effectiveness of oracle space: The paper claims that the oracle space efficiently describes various high-level semantics and that routing in this space preserves semantic locality; again, there is now strong evidence of this.
__A2__ As is mentioned above, the Self-Attention, especially attention score, can reflect and amplify the high-level semantics across tokens in the same context. Therefore, oracle space, which is constructed by grouping tokens with high attention scores and extracting high-level semantics from them, can capture semantic locality well. 
To further demonstrate the generalizability of this conclusion, we constructed oracle space with DeepSeekMoE-16B\Qwen1.5-MoE-A2.7B\Llama-3.1-8B on several datasets, as is illustrated in Figure 2-5 in the link. We found that semantic group embeddings in these oracle spaces all preserve semantic locality well, with embeddings of each semantic group vary slowly and smoothly, showing the potential of scaling-up and generalization.
Q3
__Q3__ Narrow scope of application If you're in a memory-constrained LLM inference setup, maybe you shouldn't be using MoEs.
__A3__ Pursuing the trade-off between performance and latency is a key topic in the field of LLM inference. The MoE structure, wich scales up model performance without increasing the number of activated parameters, is the most edge-friendly model architecture. Recently, models like DeepSeek and QwQ exhibit impressive ability, demonstrating the value of MoE edge deployment. Companies like Qualcomm have also started research concerning MOE edge deployment, which was presented at the NeurIPS 2024 enterprise section, illustrating the huge market value of this problem. It is believed that this technology will further increase the accessibility to LLMs at edge devices, and give full play to the advantages of MOE's natural adaptation to edge side deployment.
