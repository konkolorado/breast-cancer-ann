Supplementary information for the TRANSBIG2006AFFY series
----------------------------------------------------------

This validation series use a subset of patients from experiment E-TABM-77 in ArrayExpress Repository

transbig2006affy_demo.txt
--------------------------

Demographics file containing the clinical information for 198 untreated patients of the TRANSBIG validation study. The clinical information are the following:

samplename: unique id for the patient

id: original id for the patients (id given by the institution, useful to look for intersection between multiple datasets)
hospital	Center (the combination of subject_id and hospital uniquely identifies the patient)

age	Age

size	Diameter of tumor (in mm)

Surgery_type	0 = breast conserving therapy, 1 = mastectomy

Histtype	Histopathological tumor type
	1=IDC, invasive ductal carcinoma
	2=ILC, invasive lobular carcinoma
	3=IDC/ILC
	4=mucinous
	5=metaplastic
	6=medular
	7=tubular/lobular
	
Angioinv	Angioinvasion (lymphvascular invasion)
	1 = -
	2 = +/-
	3 = +

Lymp_infil	Lymphocytic infiltrate
	1 = -
	2 = +/-
	3 = +
	
grade	Histopathological grading
	1=well differentiated
	2=intermediate
	3=poorly differentiated
	
er	Estrogen receptors
	0 = negative
	1 = positive
	
t.tdm	Time to distant metastasis (days)

e.tdm	indicator for time to distant metastasis
	1 = event
	0 = censoring
	
t.rfs	Disease-free survival (days)

e.rfs	indicator for disease-free survival
	1 = event
	0 = censoring
	

t.os	Overall survival (days)

e.os	indicator for overall survival
	0 = event
	1 = censoring
	
t.dmfs	Distant metastasis-free survival (days)

e.dmfs	indicator for distant metastasis-free survival
	1 = event
	0 = censoring
	
risksg	Clinical risk group according to St Gallen criteria
	Good = low risk
	Poor = high risk
	
NPI	NPI score

risknpi	Clinical risk group according to NPI criteria
	Good = low risk (NPI score <= 3.3)
	Poor = high risk (NPI score > 3.3)
	
AOL_os_10y	10-year overall survival probability predicted by Adjuvant online

risk_AOL	Clinical risk group according to Adjuvant online
	Good = low risk (AOL_os_10y>=0.88 for ER+ patients and AOL_os_10y>=0.92 for ER- patients)
	Poor = high risk (AOL_os_10y<0.88 for ER+ patients and AOL_os_10y<0.92 for ER- patients)
	
veridex_risk	Clinical risk group according to Veridex signature
	Good = low risk (good prognosis)
	Poor = high risk (poor prognosis)


transbig2006affy.RData
------------------------

R workspace containing the following objrcts:

 data: matrix of gene expressions with the 198 untreated patients in rows from the TRANSBIG validation study and the affy probesets in columns (22283). The probesets are all the probsets from the chip hgu133a MAS5 normalization was performed with a target value of 600.
 
 demo: matrix with clinical information concerning the untreated patients (198).
 
 annot: annotations of all the probesets (22283).

