# BEC Phishing Detection Using Machine Learning & NLP

A Cyber Security case study investigating the **2013--2015
Google--Facebook Business Email Compromise (BEC) fraud** and evaluating
machine-learning approaches for detecting phishing and fraudulent
invoice emails.

## Project Overview

Business Email Compromise (BEC) is a targeted social-engineering attack
in which a fraudster impersonates a trusted supplier, executive, or
business partner to manipulate an organisation into transferring money
or changing payment details. Unlike conventional phishing, BEC emails
may contain **no malicious links or malware**, making them difficult for
traditional filters to identify.

Our group studied the Google--Facebook fraud associated with the
impersonation of Quanta Computer, then implemented an ML/NLP experiment
to investigate the strengths and limitations of generic phishing
classifiers on BEC-style invoice emails.

## Objectives

-   Analyse the Google--Facebook BEC fraud, its attack chain, and the
    reasons it succeeded.
-   Train and compare machine-learning models on a labelled email
    dataset.
-   Evaluate whether models trained on generic phishing data can
    identify BEC-style invoice fraud.
-   Test BEC-specific indicators such as look-alike sender domains,
    bank-detail changes, mismatched bank countries, and urgency.
-   Identify practical controls that combine ML detection with
    business-process verification.

## Case Study: Google--Facebook BEC Fraud

Between 2013 and 2015, the attacker impersonated Quanta Computer, a
genuine supplier to both companies, using a similarly named company and
fraudulent payment requests. The case involved approximately **US\$121
million** in transfers. The attacker was later prosecuted in the United
States and sentenced to five years in prison in December 2019.

The case demonstrates that an email can look like ordinary business
correspondence while exploiting supplier trust, routine invoice
workflows, and weak verification of payment-detail changes.

## Experimental Methodology

### 1. Dataset

-   **CEAS-2008:** 39,154 labelled emails used for generic phishing/spam
    classification.
-   **Train/test split:** 80/20, stratified, with random seed `42`.
-   **Synthetic BEC test set:** 120 generated invoice emails---60
    legitimate and 60 BEC-style---created to represent patterns
    discussed in the case study.

The synthetic set is a controlled proof-of-concept, not a substitute for
evaluation on real enterprise BEC emails.

### 2. Feature Engineering

**Text features** - Subject and body combined for text analysis. -
TF-IDF representation using unigrams and bigrams. - Up to 50,000 text
features, English stop-word removal, and sublinear term frequency.

**Email metadata** - URL presence - Log-transformed body length -
Exclamation-mark ratio - Uppercase-character ratio - Urgency-term rate -
Payment-term rate - Bank-change-term rate - Presence of a monetary
amount

**BEC-specific indicators** - Sender domain resembles a known vendor
domain but does not match it. - Email requests a bank-account or
payment-detail change. - Bank country differs from the vendor's
country. - Urgency or deadline wording is present.

### 3. Models Compared

The experiment evaluates five generic classifiers on CEAS-2008:

1.  Multinomial Naive Bayes using TF-IDF
2.  Logistic Regression using TF-IDF
3.  Linear Support Vector Machine (SVM) using TF-IDF
4.  Random Forest using metadata features
5.  Hybrid Logistic Regression using TF-IDF and metadata features

A separate Logistic Regression model uses the four BEC-specific
indicators and is evaluated on the synthetic set using stratified 5-fold
cross-validation.

### 4. Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   False-positive rate (FPR)
-   Training time
-   BEC detection rate and false-flag rate for legitimate synthetic
    invoices

## Key Findings

### Generic phishing dataset (CEAS-2008 test set)

  ----------------------------------------------------------------------------------
  Model             Accuracy    Precision       Recall     F1-score   False-positive
                                                                                rate
  ------------- ------------ ------------ ------------ ------------ ----------------
  Multinomial         0.9890       0.9991       0.9812       0.9901            0.12%
  Naive Bayes                                                       
  (TF-IDF)                                                          

  Logistic            0.9971       0.9975       0.9973       0.9974            0.32%
  Regression                                                        
  (TF-IDF)                                                          

  **Linear SVM    **0.9982**   **0.9984**   **0.9984**   **0.9984**        **0.20%**
  (TF-IDF)**                                                        

  Random Forest       0.9588       0.9602       0.9661       0.9631            5.05%
  (metadata                                                         
  only)                                                             

  Hybrid              0.9972       0.9975       0.9975       0.9975            0.32%
  Logistic                                                          
  Regression                                                        
  ----------------------------------------------------------------------------------

The Linear SVM achieved the best reported F1-score on the CEAS-2008 test
set. However, high performance on a generic phishing dataset did **not**
reliably translate to BEC-style invoice detection.

### Synthetic BEC stress test

The report found that the generic models were unstable on the synthetic
invoice set: some missed BEC emails, while others flagged legitimate
invoices. The BEC-specific feature model separated the two classes on
this small synthetic dataset.

**Important limitation:** this result is specific to the generated
examples and their engineered signals. It should not be interpreted as
proof of perfect performance on real-world BEC attacks.

## Conclusion

The study indicates that generic email text classification alone is not
sufficient for reliable BEC detection. A stronger approach combines
message text with sender-domain checks, vendor and payment context, and
behavioural indicators. High-risk payment requests---especially changes
to bank details---should be held for **independent out-of-band
verification**, such as calling a previously verified vendor contact.

## Tech Stack

-   Python 3
-   pandas and NumPy
-   scikit-learn
-   SciPy
-   Matplotlib
-   TF-IDF and classical machine learning

## Repository Contents

A suggested repository layout:

``` text
.
├── README.md
├── Case_Study_Report.pdf
├── BEC_Phishing_Case_Study.pptx
├── phishing bec experiment(1).py
└── ds/
    └── Phishing-Email-Dataset-HEAD/
        └── CEAS_08.csv
```

The dataset file is not included here; place the CEAS-2008 CSV at the
path expected by the script, or update the path in the script to match
your local dataset location.

## How to Run

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install dependencies

``` bash
python -m pip install pandas numpy scikit-learn scipy matplotlib
```

### 3. Add the dataset

Download or obtain the CEAS-2008 dataset separately and place
`CEAS_08.csv` at:

``` text
ds/Phishing-Email-Dataset-HEAD/CEAS_08.csv
```

Alternatively, edit the CSV path in the Python script.

### 4. Run the experiment

``` bash
python "phishing bec experiment(1).py"
```

The script prints model metrics and BEC stress-test results. It also
saves plots (`fig_models.png`, `fig_cm.png`, and `fig_bec.png`) and a
`results.json` file in the current working directory.

## Team

-   **Tarang** --- PRN: 23070123140
-   **Vivaan** --- PRN: 23070123151
-   **Yaman Ansari** --- PRN: 23070123155

**Programme:** B.Tech, Electronics & Telecommunication Engineering\
**Course:** Cyber Security --- Continuous Assessment II, Unit 4\
**Batch:** 2023--2027 \| Semester VII\
**Faculty:** Dr. Mangal Singh

## Disclaimer

This repository is for academic study and defensive cybersecurity
research. The BEC messages are synthetic examples created for controlled
evaluation. Results depend on the dataset, feature design, and test
setup; they do not guarantee detection of real attacks.

## References

The case study report cites U.S. Department of Justice releases
concerning the charges, guilty plea, and sentencing; FBI Internet Crime
Complaint Center reports; and published research on phishing detection,
email authentication, and BEC detection. Refer to
`Case_Study_Report.pdf` for the full bibliography and source details.
