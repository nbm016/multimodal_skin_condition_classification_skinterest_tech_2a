# Multi-Modal Skin Condition Classification - Skinterest Tech 2A

---

## 👥 **Team Members**


| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Nivi Munjal    | @nbm016 | Data Cleaning and Exploration, Image Similarity Analysis, Data Visualization, Model Research, Building, Training, and Evaluation            |
| Sarah Shafiq   | @sshafiq1     | Data preprocessing, Model selection, model training and optimization       |
| Mahek Patel     | @mahekp05  | Data preprocessing, Image Analysis, Image Cleaning, Model training, Database Configurations, Metrics Research                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using computer vision and supervised learning to address skin classification problems for different population subgroups
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `Skinterest Tech`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

* View the notebook files in this repository for data exploration, data cleaning, image analysis, and model building/training of this dataset.
* Place the notebook files in Google Colab to run the code. Google Colab will ask for authorization when running the code.
* The SCIN dataset for this project is accessed from the code.
* To save the trained models, adjust the filepaths in the code to desired filepaths/locations. 

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

* “47% of dermatologists felt that their training was inadequate to diagnose skin disease in SOC (Skin of Color) patients.” (Narla et al., 2022)

<br>
<img src="./images/BTT logo.png" alt="Similar images 1" width="200"/>
<img src="./images/Skinterest logo.png" alt="Similar images 1" width="200"/>

---

## 📊 **Data Exploration**

**Dataset: Skin Condition Image Network (SCIN) Google Dataset**
* Crowdsourced from Google Search users to increase the diversity of dermatology images available for public health education and research.
* Images in dataset are paired with self-reported metadata dermatologist expert skin condition labels.
* Emphasis of diversity and fairness in this dataset compared to other skin condition datasets.
* Format: Google Cloud Storage (GCS) - User Metadata and Images
* Relevant Features: Dermatologist condition labels and confidence scores, race and ethnicity, monk skin tone scale, image paths
* Size: 5,000+ user contributions and 10,000+ total images (up to 3 images per user/case)
* Types of Data: Images, Text, Categorical
<br>
  
**Exploration and Preprocessing Approaches**
* Removed images from dataset with low quality, such as low brightness.
* Identified similar images to capture a variety of angles for certain skin condition images. 
* Conducted a diversity analysis of submitted photos using race/ethnicity and monk skin tone scale.
* Representation of common skin conditions in dataset (eczema).
* Metadata about users and their submitted skin condition photos (age, location of skin condition, relevant symptoms, texture of conditions, and more).
<br>

**Challenges in Dataset**
* Fair representation of diversity in dermatology.
* Many images did not have an associated skin condition label and confidence score.
* Other missing data for certain metadata columns (race/ethnicity, age).
<br>

**Data and Image Visualizations** <br><br>
Race Distribution of Dataset's Users:
<br><br>
<img src="./images/SCIN_Dataset_Race_Feature.png" alt="Race distribution of dataset users" width="350"/>
<br><br>

Examples of User-Submitted Skin Conditions (Multiple Angles Included):
<br><br>
<img src="./images/Similar_Images_1.png" alt="Similar images 1" width="400"/>
<img src="./images/Similar_Images_2.png" alt="Similar images 1" width="400"/>
<img src="./images/Similar_Images_3.png" alt="Similar images 1" width="400"/>
<img src="./images/Similar_Images_4.png" alt="Similar images 1" width="400"/>

<br>


---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

* Experiment with additional feature combinations for model building and training
* Apply both supervised and unsupervised machine learning methods to learn hidden trends and patterns, especially with unlabeled data.
* Web application of skin condition classification: Users can submit their own skin condition images and have them evaluated in real-time by models. 

---

## 📝 **License**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References**

* Hartanto, D., & Herawati, R. (2024). COMPARATIVE ANALYSIS OF EFFICIENTNET AND RESNET MODELS IN THE CLASSIFICATION OF SKIN CANCER. Proxies : Jurnal Informatika, 7(2), 69–84. https://doi.org/10.24167/proxies.v7i2.12468
* Narla, S., Heath, C. R., Alexis, A., & Silverberg, J. I. (2022). Racial disparities in dermatology. Archives of Dermatological Research, 315(5). https://doi.org/10.1007/s00403-022-02507-z (https://pmc.ncbi.nlm.nih.gov/articles/PMC9743121/)
* Randellini, E. (2023, January 5). Image classification: ResNet vs EfficientNet vs EfficientNet_v2 vs Compact Convolutional…. Medium. https://medium.com/@enrico.randellini/image-classification-resnet-vs-efficientnet-vs-efficientnet-v2-vs-compact-convolutional-c205838bbf49 
---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
