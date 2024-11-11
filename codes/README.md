# Exploring Different Imputation Techniques in the Reconstruction of CT Lung Images

This methodology forms the backbone of the codebase, covering everything from data preparation and missing-data creation to the setup of imputation models for lung CT scans. Each section of the code aligns with the steps described here, from data normalization to generating targeted missing data zones and configuring the inpainting models. With a 10-fold cross-validation setup built in, users can easily test and compare model performance across different levels of missing data.

The codebase, makes it straightforward to follow along and apply the methods outlined, allowing users to dive into model training, preprocessing, and reconstruction techniques on this well-prepared dataset.

---

## Methodology
This chapter is organized into three primary procedural sections &mdash; Preprocessing Steps, Missing Data Generation and Missing Data Reconstruction &mdash; as illustrated in Figure 1. The entire methodology, from image processing tasks to Machine Learning (ML) algorithms, was implemented using *Python*.

<p align="center">
    <img src="/imgs/experimental_setup.png" alt="Figure 1" width="750"/>
    <br>
    <em><strong>Figure 1:</strong> Diagram of the experimental procedure developed.</em>
</p>

   - ### Preprocessing Steps
The dataset for this study was sourced from the Medical Segmentation Decathlon (MSD) and comprised 95 chest CT volumes from The Cancer Imaging Archive (TCIA) [1,2]. Of these, only 63 scans included 3D annotations identifying non-small cell lung tumours, leading to the remaining volumes' exclusion from further analysis.

To ensure consistency across the dataset, variations in the field of view (FOV) acquisition required standardization, with only scans having slice counts within the first and third quartiles retained. This process resulted in a final selection of 31 volumes. A lung identification step was then applied, selecting only slices that contained pulmonary structures. Together, these steps yielded a set of 5,350 transverse slices, with an average of 172.58 slices per scan and a standard deviation of 18.52, which were used to train and evaluate the imputation models.

Pixel values, originally measured in *Hounsfield* Units (HU), were then normalized to a range of $[0, 1]$ following a windowing process from $[-600, 1000]$ HU, as illustrated in Figure 2. This normalization accommodated the density ranges of both thoracic and carcinogenic tissues, ultimately supporting more accurate algorithmic comparisons during specific evaluation stages.

<p align="center">
    <img src="/imgs/hounsdfield_graph.png" alt="Figure 2" width="500"/>
    <br>
    <em><strong>Figure 2:</strong> Windowing process and respective conversion from *Hounsfield* to Pixel Intensity scale.</em>
</p>
    
   - ### Missing Generation
Unlike traditional methods that estimate missing pixels across the entire image, this approach focused exclusively on the lung area, weighting the missing data relative to the amount of omitted pulmonary tissue. For example, consider a slice with a resolution of $256 \times 256$ pixels and a missing data level of $10\%$. Conventional methods would calculate missing pixels as $256 \times 256 \times 0.1 \approx 6,554$ pixels. In contrast, assuming the lung region comprises $50\%$ of the image area, or $256 \times 256 \times 0.5 \approx 32,768$ pixels, this method calculates missing pixels as $(256 \times 256 \times 0.5) \times 0.1 \approx 3,277$ pixels. This lung-focused approach resulted in variable missing region sizes, even for a fixed percentage of missing data, ensuring that images with smaller lung areas had fewer omitted pixels than those with larger lung regions.

To comprehensively assess the models, four levels of missing data &mdash; **10%, 20%, 30% and 40%** &mdash; were applied using a square pattern that densely removed information, allowing models' performance analysis under progressive data loss, as outlined in Figure 3. To maintain focus on the lung tissue, the centre of the missing region was assumed to fall within the lung mask. However, due to the variable shapes of lung slices &mdash; particularly in the narrower, elongated inferior lobes &mdash; it was not always feasible to fully confine high-percentage missing data areas within the lung boundaries. To address this, as represented in Figure 4, a missing mask was deemed valid if at least $60\%$ of its pixels were within the lung area and $100\%$ were within the region of interest (ROI). This condition prevented the models trained on higher missing-data percentages from being restricted to medial lung slices with larger pulmonary cavities, enhancing generalizability across different lung regions.

<p align="center">
    <img src="/imgs/missing_creation.png" alt="Figure 3" width="500"/>
    <br>
    <em><strong>Figure 3:</strong> Minimalist schematic representative of the algorithm behind missing mask creation.</em>
</p>
<br>
<p align="center">
    <img src="/imgs/mask_validation.png" alt="Figure 4" width="500"/>
</p>

<p>
    <em><strong>Figure 4:</strong> Visual representation of the process for creating and validating missing zones, using the masks from lung region segmentation and images of ROIs, previously extracted.</em>
</p>

In summary, this targeted approach provided a deeper understanding of how imputation models perform with tissues of distinct characteristics, especially in cases where missing data could compromise medical assessment accuracy. By refining the precision of reconstructed regions, this method aimed to yield more reliable clinical insights, improve ML model specificity and minimize potential errors or inconsistencies during reconstruction.

   - ### Missing Reconstruction
This study compared several imputation models to evaluate their performance on different thoracic tissue types, each with distinct characteristics. The approach was based on the hypothesis that certain models may excel with specific tissue structures, even if their overall performance varied. **Two adapted versions of the CE model [3], each focused on local and global discrimination, were highlighted alongside comparisons with other established models such as GLCIC [4], CA [5], EC [6] and ESMII [7]**.

A 10-fold cross-validation protocol was applied for training and testing phases across all imputation models to ensure robust and consistent results. This approach was adopted to address an issue observed during traditional dataset division, where a random split into training and test sets, as commonly practised in the literature, inadvertently led to slices from the same patient scan appearing in both sets, as demonstrated in Figure 5. This overlap allowed the reconstruction models to benefit from adjacent training slices, potentially inflating performance metrics by relying on neighbouring data rather than the intended independent slices.

<p align="center">
    <img src="/imgs/overfitting.png" alt="Figure 5" width="750"/>
</p>

<p>
    <em><strong>Figure 5:</strong> A potential overfitting scenario caused by the commonly used split approach between training and test sets. In this situation, slices 100 and 101 from the same CT volume are included in the test and training phases, respectively, leading the model to rely on memorization rather than learning during the training procedure. This particular overfitting effect is highlighted by a mask that covers 90% of the total image area, demonstrating an effective failure in the model’s generalization capacity.</em>
</p>


To provide fair comparison conditions, all models were trained and evaluated under standardized settings, including uniform data preprocessing, consistent training parameters **(100 epochs with a batch size of 8 slices)** and controlled testing environments. The experiments were conducted on a virtual machine equipped with an **Intel Xeon Silver 4314 CPU** at **2.40 GHz** and an **NVIDIA RTX A6000 GPU** with **47.95 GiB** of memory, ensuring consistent computational conditions across all techniques.

---

## References

[1] Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette Kopp-
Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, and
Ronald M. Summers et al. The medical segmentation decathlon. Nature Communications,
13, 2022.

[2] The Cancer Imaging Archive. The Cancer Imaging Archive (TCIA), 2024.

[3] Deepak Pathak, Philipp Kr&auml;henb&uuml;hl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros.
Context encoders: Feature learning by inpainting. CoRR, abs/1604.07379, 2016.

[4] Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent
image completion. ACM Trans. Graph., 36(4), 2017.

[5] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Generative
image inpainting with contextual attention. In 2018 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 5505–5514. IEEE Computer Society,
2018.

[6] Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Qureshi, and Mehran Ebrahimi. Edge-
connect: Structure guided image inpainting using edge prediction. In 2019 IEEE/CVF
International Conference on Computer Vision Workshop (ICCVW), pages 3265–3274,
2019.

[7] Qianna Wang, Yi Chen, Nan Zhang, and Yanhui Gu. Medical image inpainting with
edge and structure priors. Measurement: Journal of the International Measurement
Confederation, 185, 2021.

