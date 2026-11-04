# Error analysis (lora_r16, val)

56 of 206 criterion predictions have a wrong label. Counts per category: A false_certainty: 28, B missed_info: 14, C wrong_polarity: 2, D unsupported: 29, E wrong_evidence: 15, F other: 4

## A false_certainty  (28 cases)

### Case 1: annotation 8 (patient sigir-20141, NCT01397994, exclusion)
- **Criterion:** Unstable angina
- **Gold:** not enough information | evidence [0, 1, 2]: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.; [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.; [2] The latest episode of pain ended half an hour prior to her arrival.
- **Predicted:** not excluded | evidence [0, 1, 2]: [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.; [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.; [2] The latest episode of pain ended half an hour prior to her arrival.
- **Note:** [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. [2] The latest episode of pain ended half an hour prior to her arrival. [3] She is known to have hypertension and obesity. [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. [5] She currently takes no medications. [6] Physical examination is normal. [7] The EKG shows nonspecific changes. [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 2: annotation 35 (patient sigir-20142, NCT02618655, exclusion)
- **Criterion:** we find that the patient selected does not meet the selection criteria within the observation period
- **Gold:** not enough information | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. [1] He has just returned from a 5 day vacation in Colorado. [2] Parents report that prior to the onset of fever and cough, he had loose stools. [3] He denies upper respiratory tract symptoms. [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left. [5] A chest x-ray shows bilateral lung infiltrates. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 3: annotation 253 (patient sigir-201415, NCT00180739, exclusion)
- **Criterion:** 3. Patients who are considered high risk pregnancy due to uterine factors (e.g. abnormal uterus, uterine scars, cerclage) except fibroids.
- **Gold:** not enough information | evidence [4]: [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- **Predicted:** not excluded | evidence [0, 1, 4]: [0] A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting.; [1] The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm.; [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- **Note:** [0] A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting. [1] The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm. [2] The complete blood count and biochemical profile are normal. [3] Point of care pregnancy test with cut-off sensitivity of 25 mIU/ml Beta-HCG is negative. [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior. [5] The differential diagnosis includes vesicular mole vs fibroid degeneration. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## B missed_info  (14 cases)

### Case 4: annotation 21 (patient sigir-20141, NCT00149227, inclusion)
- **Criterion:** Clinical diagnosis of hypertension
- **Gold:** included | evidence [3]: [3] She is known to have hypertension and obesity.
- **Predicted:** not enough information | evidence [3]: [3] She is known to have hypertension and obesity.
- **Note:** [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. [2] The latest episode of pain ended half an hour prior to her arrival. [3] She is known to have hypertension and obesity. [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. [5] She currently takes no medications. [6] Physical examination is normal. [7] The EKG shows nonspecific changes. [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 5: annotation 31 (patient sigir-20142, NCT02618655, inclusion)
- **Criterion:** patients have fever more than one week
- **Gold:** not included | evidence [0]: [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
- **Predicted:** not enough information | evidence [0]: [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
- **Note:** [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. [1] He has just returned from a 5 day vacation in Colorado. [2] Parents report that prior to the onset of fever and cough, he had loose stools. [3] He denies upper respiratory tract symptoms. [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left. [5] A chest x-ray shows bilateral lung infiltrates. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 6: annotation 366 (patient sigir-201420, NCT02255487, inclusion)
- **Criterion:** Has experienced abdominal trauma, blunt or penetrating, requiring open abdominal laparotomy with primary closure
- **Gold:** included | evidence [0, 4, 5]: [0] A 32-year-old woman is admitted to the ER following a car accident.; [4] Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants.; [5] She has no bowel sounds.
- **Predicted:** not enough information | evidence [0, 1, 4, 5]: [0] A 32-year-old woman is admitted to the ER following a car accident.; [1] She has sustained multiple injuries including upper and lower extremity fractures.; [4] Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants.; [5] She has no bowel sounds.
- **Note:** [0] A 32-year-old woman is admitted to the ER following a car accident. [1] She has sustained multiple injuries including upper and lower extremity fractures. [2] She is fully awake and alert, and she reports that she was not wearing a seat belt. [3] Her blood pressure is 134/74 mm Hg, and her pulse is 87/min. [4] Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants. [5] She has no bowel sounds. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## C wrong_polarity  (2 cases)

### Case 7: annotation 263 (patient sigir-201415, NCT00180739, exclusion)
- **Criterion:** 13. Evidence of uterine pathology other than leiomyoma
- **Gold:** excluded | evidence [5]: [5] The differential diagnosis includes vesicular mole vs fibroid degeneration.
- **Predicted:** not excluded | evidence [4]: [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- **Note:** [0] A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting. [1] The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm. [2] The complete blood count and biochemical profile are normal. [3] Point of care pregnancy test with cut-off sensitivity of 25 mIU/ml Beta-HCG is negative. [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior. [5] The differential diagnosis includes vesicular mole vs fibroid degeneration. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 8: annotation 935 (patient sigir-201522, NCT02534727, inclusion)
- **Criterion:** Thought likely to be Mycobacterium culture positive (including NTM infected for the NIH CC) by enrolling physician
- **Gold:** not included | evidence [3]: [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
- **Predicted:** included | evidence [3]: [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.
- **Note:** [0] A 65-year-old male with a history of tuberculosis has started to complain of productive cough with tinges of blood. [1] Chest X-ray reveals a round opaque mass within a cavity in his left upper lobe. [2] The spherical mass moved in the cavity during supine and prone CT imaging. [3] Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls. [4] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## D unsupported  (29 cases)

### Case 9: annotation 24 (patient sigir-20141, NCT00149227, exclusion)
- **Criterion:** Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)
- **Gold:** not applicable | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. [2] The latest episode of pain ended half an hour prior to her arrival. [3] She is known to have hypertension and obesity. [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. [5] She currently takes no medications. [6] Physical examination is normal. [7] The EKG shows nonspecific changes. [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 10: annotation 42 (patient sigir-20142, NCT00711399, exclusion)
- **Criterion:** Pregnant women
- **Gold:** not applicable | evidence [0]: [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. [1] He has just returned from a 5 day vacation in Colorado. [2] Parents report that prior to the onset of fever and cough, he had loose stools. [3] He denies upper respiratory tract symptoms. [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left. [5] A chest x-ray shows bilateral lung infiltrates. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 11: annotation 379 (patient sigir-201420, NCT01594385, exclusion)
- **Criterion:** Pregnant patients
- **Gold:** not enough information | evidence []: (none)
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 32-year-old woman is admitted to the ER following a car accident. [1] She has sustained multiple injuries including upper and lower extremity fractures. [2] She is fully awake and alert, and she reports that she was not wearing a seat belt. [3] Her blood pressure is 134/74 mm Hg, and her pulse is 87/min. [4] Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants. [5] She has no bowel sounds. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## E wrong_evidence  (15 cases)

### Case 12: annotation 13 (patient sigir-20141, NCT01397994, exclusion)
- **Criterion:** IDDM (Type-1 diabetes mellitus)
- **Gold:** not excluded | evidence [4]: [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. [1] The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. [2] The latest episode of pain ended half an hour prior to her arrival. [3] She is known to have hypertension and obesity. [4] She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. [5] She currently takes no medications. [6] Physical examination is normal. [7] The EKG shows nonspecific changes. [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 13: annotation 238 (patient sigir-201415, NCT00277680, exclusion)
- **Criterion:** Small submucous fibroids suitable for hysteroscopic resection
- **Gold:** not excluded | evidence [4]: [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting. [1] The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm. [2] The complete blood count and biochemical profile are normal. [3] Point of care pregnancy test with cut-off sensitivity of 25 mIU/ml Beta-HCG is negative. [4] The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior. [5] The differential diagnosis includes vesicular mole vs fibroid degeneration. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 14: annotation 369 (patient sigir-201420, NCT02255487, exclusion)
- **Criterion:** Estimated Abbreviated Injury Scale (AIS) score of six (6) at the time of surgery, for all trauma patients
- **Gold:** not excluded | evidence [2]: [2] She is fully awake and alert, and she reports that she was not wearing a seat belt.
- **Predicted:** not excluded | evidence []: (none)
- **Note:** [0] A 32-year-old woman is admitted to the ER following a car accident. [1] She has sustained multiple injuries including upper and lower extremity fractures. [2] She is fully awake and alert, and she reports that she was not wearing a seat belt. [3] Her blood pressure is 134/74 mm Hg, and her pulse is 87/min. [4] Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants. [5] She has no bowel sounds. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

## F other  (4 cases)

### Case 15: annotation 33 (patient sigir-20142, NCT02618655, inclusion)
- **Criterion:** full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis
- **Gold:** not applicable | evidence []: (none)
- **Predicted:** not enough information | evidence [0, 4, 5]: [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.; [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left.; [5] A chest x-ray shows bilateral lung infiltrates.
- **Note:** [0] An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. [1] He has just returned from a 5 day vacation in Colorado. [2] Parents report that prior to the onset of fever and cough, he had loose stools. [3] He denies upper respiratory tract symptoms. [4] On examination he is in respiratory distress and has bronchial respiratory sounds on the left. [5] A chest x-ray shows bilateral lung infiltrates. [6] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 16: annotation 558 (patient sigir-201430, NCT00721006, exclusion)
- **Criterion:** Inability to provide informed consent.
- **Gold:** not applicable | evidence [8]: [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Predicted:** not excluded | evidence [8]: [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Note:** [0] A 72-year-old man complains of increasing calf pain when walking uphill. [1] The symptoms have gradually increased over the past 3 months. [2] The patient had an uncomplicated myocardial infarction 2 years earlier and a transient ischemic attack 6 months ago. [3] Over the past month, his blood pressure has worsened despite previous control with diltiazem, hydrochlorothiazide, and propranolol. [4] His is currently taking isosorbide dinitrate, hydrochlorothiazide, and aspirin. [5] On physical examination, his blood pressure is 151/91 mm Hg, and his pulse is 67/min. [6] There is a right carotid bruit. [7] His lower extremities are slightly cool to the touch and have diminished pulses at the dorsalis pedis. [8] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_

### Case 17: annotation 801 (patient sigir-201515, NCT00932425, exclusion)
- **Criterion:** Definite small-vessel etiology by history or imaging
- **Gold:** not applicable | evidence []: (none)
- **Predicted:** not excluded | evidence [0, 1, 2]: [0] Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.; [1] At the time, computed tomography was negative for brain hemorrhage and she was given thrombolytic therapy with resolution of her symptoms.; [2] Transesophageal echocardiogram and magnetic resonance angiogram of brain and great vessels found no evidence of abnormalities.
- **Note:** [0] Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago. [1] At the time, computed tomography was negative for brain hemorrhage and she was given thrombolytic therapy with resolution of her symptoms. [2] Transesophageal echocardiogram and magnetic resonance angiogram of brain and great vessels found no evidence of abnormalities. [3] She presents currently with a blood pressure of 120/70 mm Hg, normal glucose, and normal sinus rhythm on a 12-lead electrocardiogram. [4] She reports history of occasional palpitations, shortness of breath and chest pain lasting for a few minutes and then stopping on their own. [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Why it failed:** _(to be filled in by hand in the report)_
