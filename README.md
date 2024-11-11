# Exploring Different Imputation Techniques in the Reconstruction of CT Lung Images

Welcome to the official repository of our project, **Exploring Different Imputation Techniques in the Reconstruction of CT Lung Images**. This repository provides a comprehensive research guide and additional resources, including detailed explanations, images, and notebooks. **The associated paper has been submitted to the European Symposium on Artificial Neural Networks, Computational Intelligence, and Machine Learning (ESANN 2025)**. Please note that the below results section reflects the detailed outcomes of the project, with all relevant information included for further exploration.

---

## Abstract
Despite significant technological advancements, Computed Tomography (CT) scans remain vulnerable to artefacts and errors, leading to the loss of critical medical information and clinical efficacy. This study investigates the reconstruction of missing or corrupted regions in lung CT images by comparing five established image imputation models &mdash; *Context Encoder* (CE), *Global and Local Consistency Image Completion* (GLCIC), *Contextual Attention* (CA), *Edge-Connected* (EC) and *Edge and Structure Information for Medical Image Inpainting* (ESMII). These learning-based algorithms are evaluated using a 10-fold cross-validation approach on a dataset comprising 5,350 transverse slices from 31 chest CT volumes of patients with non-small cell lung cancer. The assessment encompasses four levels of missing data &mdash; 10%, 20%, 30% and 40% &mdash; which correspond to the proportion of missing pixels relative to the total number of lung tissue pixels in each image. The ESMII algorithm demonstrates superior overall performance across all percentages of missing data, consistently achieving accurate structural and textural reconstructions. However, in the context of tissue typology analysis and tumour mass reconstruction, the EC model exhibits the highest imputation performance across nearly all scenarios, even if, from a broader perspective, the effectiveness of all models in reconstructing these noisy patterns remains limited. In conclusion, the findings underscore the promising potential of advanced reconstruction techniques to reduce the need for image reacquisition, lower patient radiation exposure and decrease healthcare costs. Given the novelty of these methods in medical imaging, future research should focus on refining the existing models, exploring new types of missing data, and incorporating expert validation to further enhance their clinical applicability.

---
## Project Overview
This project is organized into three primary procedural steps &mdash; Preprocessing Steps, Missing Data Generation and Missing Data Reconstruction &mdash; as illustrated in Figure 1. The entire methodology, from image processing tasks to Machine Learning (ML) algorithms, was implemented using *Python*.

<p align="center">
    <img src="/imgs/experimental_setup.png" alt="Figure 1" width="750"/>
    <br>
    <em><strong>Figure 1:</strong> Diagram of the experimental procedure developed.</em>
</p>

---

## Key Results
This study’s results capture the performance of the compared image imputation techniques for reconstructing missing regions in lung CT images. Through quantitative and qualitative analyses, the study assessed each model's effectiveness in reconstructing essential tissue characteristics and handling complex structural and textural challenges.

- ### Qualitative Results
Although these conclusions have not yet been medically validated, the qualitative analysis offers insights into the realism and coherence of the reconstructions, particularly concerning the characteristics of the missing data, as shown in Figure 8, 9 and 10.

<p>
    <img src="./imgs/results_structure.png" alt="Figure 8" width="1240"/>
    <br>
    <em><strong>Figure 8:</strong> Qualitative results of inpainting models on pulmonary tissue structures, with rows showing input images with 10%, 20%, 30% and 40% of missing data. The final column presents ground truth data; blue arrows mark expected reconstructions and green arrows indicate consistent results.</em>
</p>
<br />
<p>
    <img src="./imgs/results_texture.png" alt="Figure 9" width="1240"/>
    <br>
    <em><strong>Figure 9:</strong> Qualitative results of inpainting models on pulmonary parenchyma missings, with rows showing input images with 10%, 20%, 30% and 40% of missing data. The final column presents ground truth data; blue arrows mark expected reconstructions and green arrows indicate consistent results.</em>
</p>
<br />
<p>
    <img src="./imgs/results_omission.png" alt="Figure 10" width="1240"/>
    <br>
    <em><strong>Figure 10:</strong> Qualitative results of inpainting models on images with total missing structures, with rows showing input images with 10%, 20%, 30% and 40% of missing data. The final column presents ground truth data; blue arrows mark expected reconstructions and green arrows indicate consistent results.</em>
</p> 

- ### Quantitative Results
Tables 1 and 2 evaluate the models' performance based on pixel accuracy and image consistency across healthy tissues (HT), tumor lesions (LT), and combined tissues (ALL). Table 3 shows performance across Lung, External-Lung, and Tumor tissues, while Table 4 measures tumor overlap between the generated and original data. All tables account for the standard deviation from the 10-fold cross-validation procedure.
     
- #### Overall Analysis
<p>
    <em><strong>Table 1:</strong> Quantitative results obtained after testing all the inpainting models compared during this study, based on a pixel-based approach. This approach was designed to assess the algorithms’ performance in healthy tissues (HT), tissues with tumour lesions (LT), and their combination (ALL). Note that the up and down arrows next to the metrics indicate the optimal direction for the models’ test parameters evolution. The values in bold highlight the best values for each condition.</em>
</p>
<p align="center">
    <img src="./tabs/overall_pixel.png" alt="Table 1" width="750"/>
</p>
<br />
<p>
    <em><strong>Table 2:</strong> Quantitative results obtained after testing all the inpainting models compared during this study, based on an image consistency evaluation. This approach was designed to assess the algorithms’ performance in healthy tissues (HT), tissues with tumour lesions (LT), and their combination (ALL). Note that the up and down arrows next to the metrics indicate the optimal direction for the models’ test parameters evolution. The values in bold highlight the best values for each condition.</em>
</p>
<p align="center">
    <img src="./tabs/overall_highlevel.png" alt="Table 2" width="750"/>
</p>
    
- #### Tissues-Based Analysis
<p>
  <em><strong>Table 3:</strong> Qualitative results from testing the top-performing inpainting models across Lung, External-Lung, and Tumour tissues, with bold values marking the best performance and arrows indicating the optimal direction of metric evolution.</em>
</p>
<p align="center">
  <img src="./tabs/tissuebased.png" alt="Table 3" width="750"/>
</p>


- #### Tumoural Omission Analysis
<p>
  <em><strong>Table 4:</strong> Qualitative results focusing on tumour reconstruction by measuring the tumour overlap between generated and original data. Bold values indicate the best performance and arrows show the optimal metric direction.</em>
</p>
<p align="center">
  <img src="./tabs/tumourbased.png" alt="Table 4" width="750"/>
</p>

---

## Contact

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/hugo-tomas-alexandre/) or reach out via email at [hugotomas.2001@outlook.pt](mailto:hugotomas.2001@outlook.pt.com) for any inquiries.

**Please note that some parts of the code related to dataset manipulation, such as compression, uniformization and other preprocessing steps, have been omitted. This is due to the fact that these parts of the code are not yet well-organized for direct inclusion here. Additionally, some of the statistical analysis code is also missing for similar reasons, but I would be happy to provide these upon request.**
